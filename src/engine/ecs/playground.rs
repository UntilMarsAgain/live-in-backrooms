//! Playground：把加载期 [`Scene`](crate::engine::scene::Scene) 模板
//! 生成进 ECS `World`（`Playground::spawn`），以及整场景卸载（`despawn`）。
//!
//! Playground 登记了根实体与引用的资产句柄——关卡切换时先 `unpin` 资产再
//! 级联 `despawn`（bevy `ChildOf` 关系自动清理整棵子树），资产回收链随之闭环。

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;

use super::components::{
    CameraC, Collider, LightC, LocalTransform, MainCamera, MaterialC, MeshHandle, WorldMatrix,
};
use crate::engine::core::asset::{AssetManager, Handle, MeshSource};
use crate::engine::core::data::aabb::Aabb;
use crate::engine::core::data::mesh::Mesh;
use crate::engine::core::data::texture::Texture;
use crate::engine::scene::{ObjectKey, Scene, SceneObjectKind};

/// 一次正在运行的场景：顶层实体 + 引用的资产句柄（卸载时 unpin 用）。
#[derive(Debug, Default)]
pub struct Playground {
    /// 顶层实体（每个 root 的 `ChildOf` 子树在 `despawn` 时级联清理）。
    pub roots: Vec<Entity>,
    /// 主相机实体（模板指定；App 兜底补默认相机后也会登记）。
    pub main_camera: Option<Entity>,
    /// 该场景引用的网格句柄。
    pub mesh_handles: Vec<Handle<Mesh>>,
    /// 该场景引用的贴图句柄。
    pub texture_handles: Vec<Handle<Texture>>,
}

impl Playground {
    /// 把场景模板生成进 `world`，返回 Playground（根实体、主相机、资产句柄）。
    ///
    /// `meshes` 用于在生成时派生 `Collider`（网格局部 AABB），系统运行时不再需要
    /// 资产库。父子关系用 bevy `ChildOf` 建立（自动维护父的 `Children`），
    /// `WorldMatrix` 初始为局部矩阵，下一物理刻由传播系统校正。
    pub fn spawn(scene: &Scene, world: &mut World, meshes: &dyn MeshSource) -> Playground {
        let mut playground = Playground::default();
        for (key, _) in scene.roots() {
            let root = spawn_node(scene, key, None, world, meshes, &mut playground);
            playground.roots.push(root);
        }
        playground
    }

    /// 卸载一个 Playground：先 `unpin` 引用的资产，再级联 `despawn` 各根实体。
    pub fn despawn(&self, world: &mut World, assets: &mut AssetManager) {
        for handle in &self.mesh_handles {
            assets.unpin(*handle);
        }
        for handle in &self.texture_handles {
            assets.unpin(*handle);
        }
        for root in &self.roots {
            if let Ok(entity) = world.get_entity_mut(*root) {
                // bevy 层级：despawn 父实体时级联 despawn 整棵子树。
                entity.despawn();
            }
        }
    }
}

/// 递归生成单个节点（父先于子，父实体已存在时建立 `ChildOf`）。
fn spawn_node(
    scene: &Scene,
    key: ObjectKey,
    parent: Option<Entity>,
    world: &mut World,
    meshes: &dyn MeshSource,
    playground: &mut Playground,
) -> Entity {
    let object = scene.object(key).expect("存活节点");
    let local = LocalTransform(object.transform);
    let world_matrix = WorldMatrix(object.transform.to_mat4());
    let entity = match object.kind {
        SceneObjectKind::Empty => world.spawn((local, world_matrix)).id(),
        SceneObjectKind::Mesh(handle) => {
            let collider = meshes
                .mesh(handle)
                .map(|mesh| Collider(mesh.bounds()))
                .unwrap_or(Collider(Aabb::EMPTY));
            playground.mesh_handles.push(handle);
            playground
                .texture_handles
                .extend(object.material.texture_handles());
            world
                .spawn((
                    local,
                    world_matrix,
                    MeshHandle(handle),
                    MaterialC(object.material.clone()),
                    collider,
                ))
                .id()
        }
        SceneObjectKind::Light(light) => world.spawn((local, world_matrix, LightC(light))).id(),
        SceneObjectKind::Camera(camera) => {
            let mut entity_world_mut = world.spawn((local, world_matrix, CameraC(camera)));
            if scene.main_camera() == Some(key) {
                entity_world_mut.insert(MainCamera);
            }
            let entity = entity_world_mut.id();
            if scene.main_camera() == Some(key) {
                playground.main_camera = Some(entity);
            }
            entity
        }
    };
    if let Some(parent) = parent {
        // 插入 ChildOf：bevy 自动把子实体加进父的 Children。
        world.entity_mut(entity).insert(ChildOf(parent));
    }
    for child_key in scene.children_of(key) {
        spawn_node(scene, child_key, Some(entity), world, meshes, playground);
    }
    entity
}
