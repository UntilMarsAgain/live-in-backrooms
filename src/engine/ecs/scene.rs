//! 场景模板 → ECS `World`：把加载期 [`Scene`](crate::engine::scene::Scene)
//! 生成成实体（组件化，含层级链接与主相机标记）。

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;

use super::components::{
    CameraC, Children, Collider, LightC, LocalTransform, MainCamera, MaterialC, MeshHandle, Parent,
    WorldMatrix,
};
use crate::engine::core::asset::MeshSource;
use crate::engine::core::data::aabb::Aabb;
use crate::engine::scene::{ObjectKey, Scene, SceneObjectKind};

/// 把场景模板生成进 `world`，返回主相机实体（未指定主相机时返回 `None`）。
///
/// `meshes` 用于在生成时派生 `Collider`（网格局部 AABB），系统运行时不再需要
/// 资产库。层级按模板树建立 `Parent` / `Children`，`WorldMatrix` 初始为局部
/// 矩阵，下一物理刻由传播系统校正。
pub fn spawn_scene(scene: &Scene, world: &mut World, meshes: &dyn MeshSource) -> Option<Entity> {
    // 1. 父先子后的节点顺序（层序 + 访问集合，防环）。
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: Vec<ObjectKey> = scene.roots().map(|(key, _)| key).collect();
    while let Some(key) = queue.pop() {
        if !visited.insert(key) {
            continue;
        }
        order.push(key);
        queue.extend(scene.children_of(key));
    }

    let main_camera = scene.main_camera();
    let mut entity_of: HashMap<ObjectKey, Entity> = HashMap::new();

    // 2. 创建实体（类型决定组件束）。
    for key in &order {
        let object = scene.object(*key).expect("存活节点");
        let local = LocalTransform(object.transform);
        let world_matrix = WorldMatrix(object.transform.to_mat4());
        let entity = match object.kind {
            SceneObjectKind::Empty => world.spawn((local, world_matrix)).id(),
            SceneObjectKind::Mesh(handle) => {
                let collider = meshes
                    .mesh(handle)
                    .map(|mesh| Collider(mesh.bounds()))
                    .unwrap_or(Collider(Aabb::EMPTY));
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
                if main_camera == Some(*key) {
                    entity_world_mut.insert(MainCamera);
                }
                entity_world_mut.id()
            }
        };
        entity_of.insert(*key, entity);
    }

    // 3. 链接层级。
    for key in &order {
        let entity = entity_of[key];
        let children: Vec<Entity> = scene
            .children_of(*key)
            .filter_map(|child| entity_of.get(&child).copied())
            .collect();
        if !children.is_empty() {
            world.entity_mut(entity).insert(Children(children.clone()));
            for child in children {
                world.entity_mut(child).insert(Parent(entity));
            }
        }
    }

    main_camera.and_then(|key| entity_of.get(&key).copied())
}
