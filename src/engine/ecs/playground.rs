//! Playground：**运行中的 ECS 世界**封装。
//!
//! [`Scene`](crate::engine::scene::Scene) 是加载期静态模板（glTF / 演示构建、
//! 合并、环境），Playground 才是"场景跑起来"的地方：持有 `World`、物理刻
//! 与渲染刻两个调度器、固定步长累加器、输入快照，以及当前场景的实例登记。
//!
//! 对使用方（App）来说只有一个入口：`Playground::load_scene`——传入一个
//! `Scene` 模板就生成实体，自动卸载上一实例（unpin 资产 + 级联 despawn）。
//! 输入事件写 `set_key` / `add_look_delta` / `add_scroll`，每帧 `advance`
//! （物理刻）+ `prepare_frame`（渲染刻）+ `render_frame`（取帧数据），ECS
//! 的存储与调度细节都收在内部。

use std::time::Duration;

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;

use super::components::{
    CameraC, Collider, LightC, LocalTransform, MainCamera, MaterialC, MeshHandle, WorldMatrix,
};
use super::frame::render_schedule;
use super::{DebugFlags, FixedStep, FixedTimestep, InputSnapshot, tick_schedule};
use crate::engine::asset::MeshView;
use crate::engine::core::asset::{AssetManager, Handle, MeshSource};
use crate::engine::core::data::aabb::Aabb;
use crate::engine::core::data::mesh::Mesh;
use crate::engine::core::data::texture::Texture;
use crate::engine::core::frame::RenderCommand;
use crate::engine::scene::{ObjectKey, Scene, SceneObjectKind};
use winit::keyboard::KeyCode;

/// 运行中的 ECS 世界：实体、输入、渲染帧、场景实例都在这里。
pub struct Playground {
    world: World,
    tick_schedule: Schedule,
    render_schedule: Schedule,
    timestep: FixedTimestep,
    /// 当前场景实例（`load_scene` 切换时先卸载它再生成新场景）。
    scene: Option<SceneInstance>,
}

/// 一次场景实例的登记：顶层实体 + 引用的资产句柄（卸载时 unpin 用）。
#[derive(Debug, Default)]
struct SceneInstance {
    /// 顶层实体（每个 root 的 `ChildOf` 子树在 `despawn` 时级联清理）。
    roots: Vec<Entity>,
    /// 主相机实体（模板指定；App 兜底补默认相机后也会登记）。
    main_camera: Option<Entity>,
    /// 该场景引用的网格句柄。
    mesh_handles: Vec<Handle<Mesh>>,
    /// 该场景引用的贴图句柄。
    texture_handles: Vec<Handle<Texture>>,
}

impl Playground {
    /// 新建运行中的 ECS 世界：注册固定步长 / 输入快照 / 渲染帧资源，
    /// 装配物理刻与渲染刻两个调度器。
    pub fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(FixedStep(Duration::from_secs_f64(1.0 / 60.0)));
        world.insert_resource(InputSnapshot::default());
        world.insert_resource(DebugFlags::default());
        world.insert_resource(RenderCommand::default());
        Self {
            world,
            tick_schedule: tick_schedule(),
            render_schedule: render_schedule(),
            timestep: FixedTimestep::new(Duration::from_secs_f64(1.0 / 60.0)),
            scene: None,
        }
    }

    /// 记录按键状态（winit 事件回调写入，系统在物理刻消费）。
    pub fn set_key(&mut self, code: KeyCode, pressed: bool) {
        let mut input = self.world.resource_mut::<InputSnapshot>();
        if pressed {
            input.keys.insert(code);
        } else {
            input.keys.remove(&code);
        }
    }

    /// 累积鼠标旋转量（像素；系统消费后清零）。
    pub fn add_look_delta(&mut self, dx: f32, dy: f32) {
        let mut input = self.world.resource_mut::<InputSnapshot>();
        input.look_delta.0 += dx;
        input.look_delta.1 += dy;
    }

    /// 累积滚轮位移（格数；系统消费后清零）。
    pub fn add_scroll(&mut self, delta: f32) {
        self.world.resource_mut::<InputSnapshot>().scroll_delta += delta;
    }

    /// 切换灯光调试可视化，返回切换后的状态。
    pub fn toggle_light_debug(&mut self) -> bool {
        let mut flags = self.world.resource_mut::<DebugFlags>();
        flags.show_light_debug = !flags.show_light_debug;
        flags.show_light_debug
    }

    /// 切换碰撞箱调试可视化，返回切换后的状态。
    pub fn toggle_collision_debug(&mut self) -> bool {
        let mut flags = self.world.resource_mut::<DebugFlags>();
        flags.show_collision_debug = !flags.show_collision_debug;
        flags.show_collision_debug
    }

    /// 累加真实时间并跑固定步长物理刻；返回本帧实际执行的物理刻数
    /// （调用方可据此做每刻一次的收尾工作，如资产回收）。
    pub fn advance(&mut self, frame_dt: Duration) -> u32 {
        let (ticks, _alpha) = self.timestep.advance(frame_dt);
        for _ in 0..ticks {
            self.tick_schedule.run(&mut self.world);
        }
        ticks
    }

    /// 渲染刻：把当前组件状态收集成 [`RenderCommand`]（渲染指令）。
    pub fn prepare_frame(&mut self) {
        self.render_schedule.run(&mut self.world);
    }

    /// 上一渲染刻准备好的帧数据（交给渲染器）。
    pub fn render_frame(&self) -> &RenderCommand {
        self.world.resource::<RenderCommand>()
    }

    /// 当前场景的主相机实体（没有加载场景时为 `None`）。
    pub fn main_camera(&self) -> Option<Entity> {
        self.scene.as_ref().and_then(|scene| scene.main_camera)
    }

    /// 窗口尺寸变化时同步主相机宽高比。
    pub fn set_aspect(&mut self, aspect: f32) {
        let mut query = self.world.query::<(&mut CameraC, &MainCamera)>();
        if let Ok((mut camera, _)) = query.single_mut(&mut self.world) {
            camera.0.set_aspect(aspect);
        }
    }

    /// 加载场景模板：卸载上一实例（unpin 资产 + 级联 despawn），再把新
    /// 场景生成进世界。资产驻留（pin）与 GPU 上传由调用方在调用前后负责，
    /// 这里只通过 `assets` 完成旧实例的 unpin 与生成时的碰撞盒派生。
    pub fn load_scene(&mut self, scene: &Scene, assets: &mut AssetManager) {
        if let Some(old) = self.scene.take() {
            old.despawn(&mut self.world, assets);
        }
        let view = MeshView::new(assets);
        let mut instance = SceneInstance::default();
        for (key, _) in scene.roots() {
            let root = spawn_node(scene, key, None, &mut self.world, &view, &mut instance);
            instance.roots.push(root);
        }
        self.scene = Some(instance);
    }
}

impl SceneInstance {
    /// 卸载一个场景实例：先 `unpin` 引用的资产，再级联 `despawn` 各根实体。
    fn despawn(&self, world: &mut World, assets: &mut AssetManager) {
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
    instance: &mut SceneInstance,
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
            instance.mesh_handles.push(handle);
            instance
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
                instance.main_camera = Some(entity);
            }
            entity
        }
    };
    if let Some(parent) = parent {
        // 插入 ChildOf：bevy 自动把子实体加进父的 Children。
        world.entity_mut(entity).insert(ChildOf(parent));
    }
    for child_key in scene.children_of(key) {
        spawn_node(scene, child_key, Some(entity), world, meshes, instance);
    }
    entity
}
