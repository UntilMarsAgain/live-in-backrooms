//! 渲染帧准备：把 ECS `World` 里的组件收集成每帧可绘制的 [`RenderFrame`]。
//!
//! 这是"渲染逻辑 = 系统"的一环：查询组件（世界矩阵 / 网格 / 材质 / 灯光 /
//! 碰撞盒），产出渲染层直接消费的纯数据帧；实际绘制仍由 [`Renderer`] 完成。

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};

use super::debug::{self, DebugVertex};
use super::uniform::{LIGHT_CAPACITY, LightUniform, MAX_NEARBY_LIGHTS};
use crate::engine::core::asset::Handle;
use crate::engine::core::camera::Camera;
use crate::engine::core::data::light::LightKind;
use crate::engine::core::data::material::Material;
use crate::engine::core::data::mesh::Mesh;
use crate::engine::ecs::components::{
    CameraC, Collider, LightC, MainCamera, MaterialC, MeshHandle, WorldMatrix,
};

/// 一帧可绘制的物体（只含网格实体；实例下标 = ObjectData 数组下标）。
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub world_matrix: Mat4,
    pub material: Material,
    pub mesh: Handle<Mesh>,
}

/// 每帧的渲染数据：相机 + 物体 + 灯光 + 调试线框。
///
/// 作为资源放进 ECS `World`，由 [`prepare_frame`] 系统每帧刷新，
/// App 渲染时读取后交给 [`Renderer::render`]。
#[derive(Resource, Debug, Default)]
pub struct RenderFrame {
    pub camera: Option<Camera>,
    pub objects: Vec<RenderObject>,
    pub lights: Vec<LightUniform>,
    pub light_gizmos: Vec<DebugVertex>,
    pub collision_gizmos: Vec<DebugVertex>,
}

/// 渲染刻调度（按帧）：准备 [`RenderFrame`]。
pub fn render_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(prepare_frame);
    schedule
}

/// 渲染准备系统：查询 ECS 组件 → 填 [`RenderFrame`]。
pub fn prepare_frame(
    camera: Query<&CameraC, With<MainCamera>>,
    objects: Query<(&WorldMatrix, &MeshHandle, &MaterialC)>,
    lights: Query<(&LightC, &WorldMatrix)>,
    colliders: Query<(&WorldMatrix, &Collider)>,
    mut frame: ResMut<RenderFrame>,
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
    let camera_position = frame.camera.map(|c| c.position()).unwrap_or_default();
    frame.lights = collect_lights(lights.iter(), camera_position);
    frame.light_gizmos = debug::build_light_gizmos_ecs(lights.iter());
    frame.collision_gizmos = debug::build_collision_gizmos_ecs(colliders.iter());
}

/// 灯光收集（移植自 `uniform::collect_lights`，输入改为 ECS 查询迭代器）：
/// 所有方向光 + 离相机最近的 [`MAX_NEARBY_LIGHTS`] 盏局部光。
fn collect_lights<'a>(
    lights: impl Iterator<Item = (&'a LightC, &'a WorldMatrix)>,
    camera_position: Vec3,
) -> Vec<LightUniform> {
    let mut directional = Vec::new();
    let mut nearby: Vec<(f32, LightUniform)> = Vec::new();

    for (light, world) in lights {
        let (_, rotation, translation) = world.0.to_scale_rotation_translation();
        if matches!(light.0.kind, LightKind::Directional) {
            directional.push(LightUniform {
                kind: 0,
                _pad: [0; 3],
                direction: (rotation * Vec3::NEG_Z).to_array(),
                _pad_direction: 0.0,
                position: [0.0; 3],
                _pad_position: 0.0,
                color: light.0.color.to_array(),
                intensity: light.0.intensity,
                size: [0.0; 2],
                _pad_size: [0.0; 2],
            });
            continue;
        }
        let entry = match light.0.kind {
            LightKind::Point => LightUniform {
                kind: 1,
                _pad: [0; 3],
                direction: [0.0; 3],
                _pad_direction: 0.0,
                position: translation.to_array(),
                _pad_position: 0.0,
                color: light.0.color.to_array(),
                intensity: light.0.intensity,
                size: [0.0; 2],
                _pad_size: [0.0; 2],
            },
            LightKind::Area { width, height } => LightUniform {
                kind: 2,
                _pad: [0; 3],
                direction: (rotation * Vec3::NEG_Z).to_array(),
                _pad_direction: 0.0,
                position: translation.to_array(),
                _pad_position: 0.0,
                color: light.0.color.to_array(),
                intensity: light.0.intensity,
                size: [width, height],
                _pad_size: [0.0; 2],
            },
            LightKind::Directional => unreachable!("方向光已在上方处理"),
        };
        nearby.push((translation.distance(camera_position), entry));
    }

    nearby.sort_by(|a, b| a.0.total_cmp(&b.0));
    nearby.truncate(MAX_NEARBY_LIGHTS);
    directional.extend(nearby.into_iter().map(|(_, light)| light));
    directional.truncate(LIGHT_CAPACITY);
    directional
}
