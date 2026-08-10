//! 渲染指令准备：**单个提取系统 + trait 查询**。
//!
//! 可渲染组件实现 [`RenderExtract`]，把自己的语义数据写进 [`RenderCommand`]
//! （拍扁的场景）；系统用 `Query<&dyn RenderExtract>` 一次性提取所有类型。
//! 指令只带语义数据（世界矩阵 / 资源库句柄 / 灯光类型+位置 / 碰撞箱）；
//! uniform 打包与线框生成是渲染侧的工作（`render::uniform` / `render::debug`）。
//!
//! **新增可渲染组件 = 实现 [`RenderExtract`] + 注册（`register_component_as`），
//! 不用改提取系统。**（相机是带 `MainCamera` 标记的唯一主相机，保持固定查询；
//! 调试开关是资源而非组件，也不走 trait 查询。）

use bevy_ecs::prelude::*;
use bevy_trait_query::queryable;
use glam::Mat4;

use super::DebugFlags;
use super::components::{CameraC, Collider, LightC, MainCamera, MeshObject, WorldMatrix};
use crate::engine::core::frame::{ColliderData, LightData, RenderCommand};

/// 渲染指令作为 ECS 资源使用（core 层保持 bevy 无关）。
impl Resource for RenderCommand {}

/// 可渲染组件的提取契约：每个可渲染组件实现它，把自己的语义数据写进渲染指令。
///
/// `world` 是世界矩阵——组件不自存矩阵，避免与层级传播维护的
/// `WorldMatrix` 组件重复。
#[queryable]
pub trait RenderExtract {
    fn extract(&self, world: &Mat4, frame: &mut RenderCommand);
}

/// 渲染刻调度（按帧）：单个提取系统。
pub fn render_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(extract_frame);
    schedule
}

/// 渲染提取系统：主相机（固定查询）+ 所有 [`RenderExtract`] 组件 + 调试开关。
fn extract_frame(
    camera: Query<&CameraC, With<MainCamera>>,
    renderables: Query<(&dyn RenderExtract, &WorldMatrix)>,
    flags: Res<DebugFlags>,
    mut frame: ResMut<RenderCommand>,
) {
    frame.camera = camera.iter().next().map(|c| c.0);
    // 指令资源跨帧复用：先清空再按当前组件状态重建。
    frame.meshes.clear();
    frame.lights.clear();
    frame.colliders.clear();
    for (traits, world) in &renderables {
        for extract in traits {
            extract.extract(&world.0, &mut frame);
        }
    }
    frame.show_light_debug = flags.show_light_debug;
    frame.show_collision_debug = flags.show_collision_debug;
}

/// 网格物体：网格句柄 + 材质 → 指令物体（并入同网格同材质的组）。
impl RenderExtract for MeshObject {
    fn extract(&self, world: &Mat4, frame: &mut RenderCommand) {
        frame.push_mesh_instance(self.mesh, self.material.clone(), *world);
    }
}

/// 灯光：类型 + 位置/朝向 + 光参数（打包 uniform 是渲染侧的事）。
impl RenderExtract for LightC {
    fn extract(&self, world: &Mat4, frame: &mut RenderCommand) {
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        frame.lights.push(LightData {
            kind: self.0.kind,
            position: translation,
            rotation,
            color: self.0.color,
            intensity: self.0.intensity,
        });
    }
}

/// 碰撞箱：局部 AABB + 世界矩阵（生成线框是渲染侧的事）。
impl RenderExtract for Collider {
    fn extract(&self, world: &Mat4, frame: &mut RenderCommand) {
        frame.colliders.push(ColliderData {
            aabb: self.0,
            world: *world,
        });
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::prelude::*;
    use bevy_trait_query::RegisterExt;
    use glam::{Quat, Vec3};

    use super::*;
    use crate::engine::core::data::light::LightKind;

    /// 测试专用"新可渲染组件"：只要实现 [`RenderExtract`] 并注册，
    /// 提取系统无需任何改动就会自动收集它。
    #[derive(Component, Debug)]
    struct TestBeacon(f32);

    impl RenderExtract for TestBeacon {
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

    /// 新增可渲染组件 = 实现 trait + 注册，不改提取系统（这正是 trait 查询的意义）。
    #[test]
    fn new_renderable_component_needs_no_system_change() {
        let mut world = World::new();
        world
            .register_component_as::<dyn RenderExtract, TestBeacon>()
            .register_component_as::<dyn RenderExtract, LightC>()
            .register_component_as::<dyn RenderExtract, MeshObject>()
            .register_component_as::<dyn RenderExtract, Collider>();
        world.insert_resource(RenderCommand::default());
        world.insert_resource(DebugFlags::default());

        // 只有新组件 + 世界矩阵，没有改任何系统。
        world.spawn((TestBeacon(3.0), WorldMatrix(Mat4::IDENTITY)));

        render_schedule().run(&mut world);

        let frame = world.resource::<RenderCommand>();
        assert_eq!(frame.lights.len(), 1);
        assert_eq!(frame.lights[0].intensity, 3.0);
        assert_eq!(frame.lights[0].position, Vec3::splat(3.0));
    }
}
