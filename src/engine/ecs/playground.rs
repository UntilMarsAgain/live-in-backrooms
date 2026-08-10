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
use std::sync::{Arc, Mutex};

use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::ScheduleSystem;
use bevy_trait_query::{RegisterExt, TraitQuery, TraitQueryMarker};

use super::components::{
    CameraC, Collider, LightC, LocalTransform, MainCamera, MeshObject, WorldMatrix,
};
use super::frame::{render_schedule, RenderExtract};
use super::{tick_schedule, DebugFlags, FixedStep, FixedTimestep, InputSnapshot};
use crate::engine::asset::MeshView;
use crate::engine::core::asset::{AssetManager, Handle, MeshSource, PinToken};
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
#[derive(Debug)]
struct SceneInstance {
    /// 顶层实体（每个 root 的 `ChildOf` 子树在 `despawn` 时级联清理）。
    roots: Vec<Entity>,
    /// 主相机实体（模板指定；App 兜底补默认相机后也会登记）。
    main_camera: Option<Entity>,
    /// 该场景的资产驻留令牌：与实例同生命周期，覆盖/卸载时自动 unpin。
    pin_token: PinToken,
}

impl Playground {
    /// 新建运行中的 ECS 世界：注册固定步长 / 输入快照 / 渲染帧资源，
    /// 装配物理刻与渲染刻两个调度器。
    pub fn new() -> Self {
        let mut world = World::new();
        // 注册 trait 查询：把可渲染组件登记到 `RenderExtract` 注册表。
        // 必须在首次运行调度（查询初始化）之前调用。
        world
            .register_component_as::<dyn RenderExtract, MeshObject>()
            .register_component_as::<dyn RenderExtract, LightC>()
            .register_component_as::<dyn RenderExtract, Collider>();
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

    /// 注册一个**物理刻**系统（在固定步长循环里运行）。
    ///
    /// 系统可以在任意文件里定义，只有注册到这个调度器才会被执行；
    /// 可在首次 `advance` 之前或之后调用（bevy 会在下一次运行前完成初始化）。
    pub fn add_tick_system<M>(
        &mut self,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.tick_schedule.add_systems(systems);
        self
    }

    /// 注册一个**渲染刻**系统（每帧运行一次）。
    ///
    /// 同样支持任意文件里定义的系统；可在首次 `prepare_frame` 之前或之后调用。
    pub fn add_render_system<M>(
        &mut self,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.render_schedule.add_systems(systems);
        self
    }

    /// 把组件登记到 trait 查询注册表（如 `dyn RenderExtract`）。
    ///
    /// **必须在首次运行任何调度之前调用**，否则 bevy-trait-query 会 panic。
    pub fn register_component_as<Trait: ?Sized + TraitQuery, C: Component>(&mut self) -> &mut Self
    where
        (C,): TraitQueryMarker<Trait, Covered = C>,
    {
        self.world.register_component_as::<Trait, C>();
        self
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

    /// 加载场景模板：**旧实例被覆盖（drop）→ 其 `PinToken` 自动 unpin 资产**，
    /// 再把新场景生成进世界。`pin_token` 是调用方对新场景资产做的一次性驻留
    /// 声明；生成时的碰撞盒派生需要 `assets`（只读借用）。
    ///
    /// 顺序很关键：先 take 旧实例——despawn 其 ECS 实体（不需要 assets），
    /// 随后旧实例 drop → 旧 PinToken drop → 自动 unpin（此刻**不持锁**，
    /// 内部自行 lock/unlock）；再锁住生成新场景，否则 Mutex 重入死锁。
    pub fn load_scene(
        &mut self,
        scene: &Scene,
        pin_token: PinToken,
        assets: &Arc<Mutex<AssetManager>>,
    ) {
        if let Some(old) = self.scene.take() {
            old.despawn(&mut self.world);
        }
        let assets = assets.lock().expect("资产库锁中毒");
        let view = MeshView::new(&*assets);
        let mut instance = SceneInstance {
            roots: Vec::new(),
            main_camera: None,
            pin_token,
        };
        for (key, _) in scene.roots() {
            let root = spawn_node(scene, key, None, &mut self.world, &view, &mut instance);
            instance.roots.push(root);
        }
        self.scene = Some(instance);
    }
}

impl SceneInstance {
    /// 级联 despawn 各根实体（资产驻留由 `PinToken` 的 drop 负责，这里不管）。
    fn despawn(&self, world: &mut World) {
        for root in &self.roots {
            if let Ok(entity) = world.get_entity_mut(*root) {
                // bevy 层级：despawn 父实体时级联 despawn 整棵子树。
                entity.despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::prelude::*;
    use glam::{Mat4, Quat, Vec3};

    use super::*;
    use crate::engine::core::data::light::LightKind;
    use crate::engine::core::frame::LightData;

    /// 外部自定义的可渲染组件：定义在测试里，不碰引擎内部文件。
    #[derive(Component, Debug)]
    struct ExternalBeacon(f32);

    impl RenderExtract for ExternalBeacon {
        fn extract(&self, _world: &Mat4, frame: &mut RenderCommand) {
            frame.lights.push(LightData {
                kind: LightKind::Point,
                position: Vec3::splat(self.0),
                rotation: Quat::IDENTITY,
                color: Vec3::ONE,
                intensity: self.0,
            });
        }
    }

    /// 外部系统：给自定义组件做动画（物理刻跑）。
    fn bob_beacon(mut beacons: Query<&mut ExternalBeacon>) {
        for mut beacon in &mut beacons {
            beacon.0 += 1.0;
        }
    }

    /// 外部渲染刻系统：往独立资源里记录"跑过一次"（避免与提取系统争指令写入顺序）。
    fn stamp_frame(mut stamps: ResMut<FrameStamps>) {
        stamps.0.push(42.0);
    }

    #[derive(Resource, Default)]
    struct FrameStamps(Vec<f32>);

    /// 组件/系统都可以定义在任何文件，只有注册到 Playground 才生效。
    #[test]
    fn external_component_and_systems_register_into_playground() {
        let mut playground = Playground::new();
        // 外部注册 trait 组件 + 两个外部系统（物理刻 / 渲染刻各一个）。
        playground.world.insert_resource(FrameStamps::default());
        playground
            .register_component_as::<dyn RenderExtract, ExternalBeacon>()
            .add_tick_system(bob_beacon)
            .add_render_system(stamp_frame);

        playground
            .world
            .spawn((ExternalBeacon(1.0), WorldMatrix(Mat4::IDENTITY)));

        playground.advance(Duration::from_secs_f64(1.0 / 60.0));
        playground.prepare_frame();

        let frame = playground.render_frame();
        // 物理刻系统把组件从 1.0 加到 2.0，trait 提取自动收集。
        assert_eq!(frame.lights.len(), 1);
        assert_eq!(frame.lights[0].intensity, 2.0);
        // 渲染刻外部系统确实每帧跑了一次。
        assert_eq!(playground.world.resource::<FrameStamps>().0, vec![42.0]);
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
            world
                .spawn((
                    local,
                    world_matrix,
                    MeshObject {
                        mesh: handle,
                        material: object.material.clone(),
                    },
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
