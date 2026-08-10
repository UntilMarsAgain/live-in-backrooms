//! 渲染指令准备：查询 ECS 组件 → 填 core 的 [`RenderCommand`]（拍扁的场景）。
//!
//! 系统只提取语义数据（世界矩阵 / 资源库句柄 / 灯光类型+位置 / 碰撞箱）；
//! uniform 打包与线框生成是渲染侧的工作（`render::uniform` / `render::debug`）。
//! 指令类型定义在 [`crate::engine::core::frame`]，本模块只负责填充。

use bevy_ecs::prelude::*;

use super::DebugFlags;
use super::components::{
    CameraC, Collider, LightC, MainCamera, MaterialC, MeshHandle, WorldMatrix,
};
use crate::engine::core::frame::{ColliderData, LightData, RenderCommand, RenderObject};

/// 渲染指令作为 ECS 资源使用（core 层保持 bevy 无关）。
impl Resource for RenderCommand {}

/// 渲染刻调度（按帧）：准备 [`RenderCommand`]。
pub fn render_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(prepare_frame);
    schedule
}

/// 渲染准备系统：查询 ECS 组件 → 填 [`RenderCommand`]。
pub fn prepare_frame(
    camera: Query<&CameraC, With<MainCamera>>,
    objects: Query<(&WorldMatrix, &MeshHandle, &MaterialC)>,
    lights: Query<(&LightC, &WorldMatrix)>,
    colliders: Query<(&WorldMatrix, &Collider)>,
    flags: Res<DebugFlags>,
    mut frame: ResMut<RenderCommand>,
) {
    frame.camera = camera.iter().next().map(|c| c.0);
    frame.objects = objects
        .iter()
        .map(|(world, mesh, material)| RenderObject {
            world_matrix: world.0,
            material: material.0.clone(),
            mesh: mesh.0,
        })
        .collect();
    // 灯光：只提取语义数据（类型 + 位置/朝向 + 光参数），
    // 打包 uniform / 生成线框留给渲染侧。
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
    // 碰撞箱：只提取语义数据（局部 AABB + 世界矩阵），生成线框留给渲染侧。
    frame.colliders = colliders
        .iter()
        .map(|(world, collider)| ColliderData {
            aabb: collider.0,
            world: world.0,
        })
        .collect();
    frame.show_light_debug = flags.show_light_debug;
    frame.show_collision_debug = flags.show_collision_debug;
}
