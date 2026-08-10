//! 自由视角相机系统：取代旧的 `FreeCameraController`。
//!
//! 输入来自 [`super::InputSnapshot`] 资源（App 事件回调累积），系统在固定刻
//! 直接修改主相机组件，碰撞用 `Collider` 组件做分轴滑动（不再需要场景模板查询）。

use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3};

use super::components::{CameraC, Collider, LocalTransform, MainCamera, WorldMatrix};
use super::input::{ActionState, InputAction};
use super::{FixedStep, InputSnapshot};
use crate::engine::core::data::aabb::Aabb;

/// 鼠标灵敏度（弧度 / 像素）。
const SENSITIVITY: f32 = 0.003;
/// 速度上下限与滚轮步进。
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 50.0;
const SPEED_STEP: f32 = 1.25;
/// 相机碰撞体半尺寸（"人"的占位）。
const COLLIDER_HALF: Vec3 = Vec3::new(0.3, 0.9, 0.3);

/// 相机系统私有状态（移动速度等）。
#[derive(Debug)]
pub struct FreeCamState {
    pub speed: f32,
}

impl Default for FreeCamState {
    fn default() -> Self {
        Self { speed: 5.0 }
    }
}

/// 自由视角相机：读取输入快照，旋转 / 移动主相机，分轴碰撞滑动。
pub fn free_camera(
    mut input: ResMut<InputSnapshot>,
    step: Res<FixedStep>,
    actions: Res<ActionState>,
    mut state: Local<FreeCamState>,
    mut camera: Query<(&mut CameraC, &mut LocalTransform), With<MainCamera>>,
    colliders: Query<(&WorldMatrix, &Collider)>,
) {
    let Ok((mut cam, _local)) = camera.single_mut() else {
        return;
    };

    // 1. 鼠标旋转（向下拖动鼠标 → 俯视）。
    let yaw_delta = input.look_delta.0 * SENSITIVITY;
    let pitch_delta = -input.look_delta.1 * SENSITIVITY;
    input.look_delta = (0.0, 0.0);

    // 2. 滚轮调整移动速度（每格 ×1.25，clamp 到 [min, max]）。
    if input.scroll_delta != 0.0 {
        state.speed =
            (state.speed * SPEED_STEP.powf(input.scroll_delta)).clamp(MIN_SPEED, MAX_SPEED);
        input.scroll_delta = 0.0;
    }

    cam.0.rotate(yaw_delta, pitch_delta);

    // 3. 键盘移动（语义动作：W/↑ 等绑定在 InputBindings，系统不关心物理键）。
    let mut movement = Vec3::ZERO;
    if actions.pressed(InputAction::MoveForward) {
        movement += cam.0.forward_horizontal();
    }
    if actions.pressed(InputAction::MoveBackward) {
        movement -= cam.0.forward_horizontal();
    }
    if actions.pressed(InputAction::MoveRight) {
        movement += cam.0.right();
    }
    if actions.pressed(InputAction::MoveLeft) {
        movement -= cam.0.right();
    }
    if actions.pressed(InputAction::MoveUp) {
        movement += Vec3::Y;
    }
    if actions.pressed(InputAction::MoveDown) {
        movement -= Vec3::Y;
    }

    let delta = movement.normalize_or_zero() * state.speed * step.0.as_secs_f32();

    // 4. 分轴滑动：先水平（X、Z）再垂直（Y），每轴单独测碰撞。
    let mut translate = Vec3::ZERO;
    for axis in [Vec3::X, Vec3::Z, Vec3::Y] {
        let axis_step = delta * axis;
        if axis_step == Vec3::ZERO {
            continue;
        }
        let probe =
            Aabb::from_half_extents(cam.0.position() + translate + axis_step, COLLIDER_HALF);
        if !collides(&colliders, &probe) {
            translate += axis_step;
        }
    }
    cam.0.translate(translate);
}

/// 探测盒是否与任一 `Collider` 的世界 AABB 相交。
fn collides(colliders: &Query<(&WorldMatrix, &Collider)>, probe: &Aabb) -> bool {
    colliders.iter().any(|(world, collider)| {
        let world_aabb = transform_aabb(collider.0, world.0);
        probe.intersects(&world_aabb)
    })
}

/// AABB 经矩阵变换：8 个角点变换后重取包围盒。
fn transform_aabb(aabb: Aabb, matrix: Mat4) -> Aabb {
    let (min, max) = (aabb.min, aabb.max);
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    Aabb::from_points(
        corners
            .iter()
            .map(|corner| matrix.transform_point3(*corner)),
    )
}
