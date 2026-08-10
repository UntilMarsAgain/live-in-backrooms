//! 渲染指令准备：把 ECS 组件拆成**多个提取系统**，每个渲染关注点一个系统，
//! 各自查询自己的组件、填 [`RenderCommand`]（拍扁的场景）。
//!
//! 系统只提取语义数据（世界矩阵 / 资源库句柄 / 灯光类型+位置 / 碰撞箱）；
//! uniform 打包与线框生成是渲染侧的工作（`render::uniform` / `render::debug`）。
//! 指令类型定义在 [`crate::engine::core::frame`]。
//!
//! **新增可渲染组件 = 新增一个提取系统并加入 [`render_schedule`]**，
//! 不用改任何已有系统。

use bevy_ecs::prelude::*;

use super::DebugFlags;
use super::components::{
    CameraC, Collider, LightC, MainCamera, MaterialC, MeshHandle, WorldMatrix,
};
use crate::engine::core::frame::{ColliderData, LightData, RenderCommand, RenderObject};

/// 渲染指令作为 ECS 资源使用（core 层保持 bevy 无关）。
impl Resource for RenderCommand {}

/// 渲染刻调度（按帧）：各渲染关注点各自一个提取系统。
///
/// 所有系统都写 `RenderCommand`，bevy 检测到同一资源的写冲突后自动串行执行，
/// 顺序即注册顺序。
pub fn render_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems((
        extract_camera,
        extract_objects,
        extract_lights,
        extract_colliders,
        extract_debug_flags,
    ));
    schedule
}

/// 相机：主相机 → 指令相机。
fn extract_camera(camera: Query<&CameraC, With<MainCamera>>, mut frame: ResMut<RenderCommand>) {
    frame.camera = camera.iter().next().map(|c| c.0);
}

/// 网格物体：(世界矩阵, 网格句柄, 材质) → 指令物体。
fn extract_objects(
    objects: Query<(&WorldMatrix, &MeshHandle, &MaterialC)>,
    mut frame: ResMut<RenderCommand>,
) {
    frame.objects = objects
        .iter()
        .map(|(world, mesh, material)| RenderObject {
            world_matrix: world.0,
            material: material.0.clone(),
            mesh: mesh.0,
        })
        .collect();
}

/// 灯光：类型 + 位置/朝向 + 光参数（打包 uniform 是渲染侧的事）。
fn extract_lights(lights: Query<(&LightC, &WorldMatrix)>, mut frame: ResMut<RenderCommand>) {
    frame.lights = lights
        .iter()
        .map(|(light, world)| {
            let (_, rotation, translation) = world.0.to_scale_rotation_translation();
            LightData {
                kind: light.0.kind,
                position: translation,
                rotation,
                color: light.0.color,
                intensity: light.0.intensity,
            }
        })
        .collect();
}

/// 碰撞箱：局部 AABB + 世界矩阵（生成线框是渲染侧的事）。
fn extract_colliders(
    colliders: Query<(&WorldMatrix, &Collider)>,
    mut frame: ResMut<RenderCommand>,
) {
    frame.colliders = colliders
        .iter()
        .map(|(world, collider)| ColliderData {
            aabb: collider.0,
            world: world.0,
        })
        .collect();
}

/// 调试开关：`DebugFlags` 资源 → 指令。
fn extract_debug_flags(flags: Res<DebugFlags>, mut frame: ResMut<RenderCommand>) {
    frame.show_light_debug = flags.show_light_debug;
    frame.show_collision_debug = flags.show_collision_debug;
}
