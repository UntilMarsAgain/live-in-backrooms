//! 自由相机控制器：第一人称式输入 → 相机。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::{Quat, Vec3};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use super::InputController;
use crate::engine::core::aabb::Aabb;
use crate::engine::core::camera::{Camera, CameraAction};
use crate::engine::core::mesh::MeshLibrary;
use crate::engine::core::transform::Transform;
use crate::engine::scene::Scene;

/// 第一人称式相机控制器。
///
/// 操作：
/// - WASD / 方向键：水平移动
/// - Space / Ctrl：上升 / 下降
/// - 点击窗口：捕获鼠标，之后移动鼠标直接旋转视角（自由视角）
/// - Esc：释放鼠标（返回系统光标）
/// - 滚轮：调整移动速度（乘法步进，带上下限）
/// - L：切换灯光调试可视化（灯泡 + 射线的显示/隐藏）
/// - B：切换碰撞箱调试可视化（世界 AABB 线框的显示/隐藏）
pub struct FreeCameraController {
    /// 移动速度（单位 / 秒）。
    speed: f32,
    /// 速度下限。
    min_speed: f32,
    /// 速度上限。
    max_speed: f32,
    /// 鼠标灵敏度（弧度 / 像素）。
    sensitivity: f32,
    /// 当前按下的键。
    keys: HashSet<KeyCode>,
    /// 是否按住左键拖动。
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    /// 待应用的鼠标旋转量（偏航 / 俯仰，像素）。
    look_delta: (f32, f32),
    /// 待应用的滚轮位移（格数）。
    scroll_delta: f32,
    /// 鼠标是否已捕获（隐藏并锁定，用于自由视角）。
    mouse_captured: bool,
    /// 灯光调试可视化开关（与 App/渲染器共享，L 键翻转）。
    ///
    /// 开关不属于相机状态，用共享原子标志传回 App，保持
    /// "输入控制器只驱动目标"的抽象不被破坏。
    show_light_debug: Arc<AtomicBool>,
    /// 碰撞箱调试可视化开关（与 App/渲染器共享，B 键翻转）。
    show_collision_debug: Arc<AtomicBool>,
    /// 相机碰撞体（局部空间，以相机位置为中心）。移动时与世界障碍
    /// AABB 做相交测试，撞到就放弃该轴分量（贴墙滑动）。
    collider: Aabb,
}

impl FreeCameraController {
    /// 新建控制器；两个 `Arc<AtomicBool>` 是与 App 共享的调试开关，
    /// 分别由 L / B 键翻转，App 侧每帧读取决定是否绘制对应线框。
    pub fn new(
        show_light_debug: Arc<AtomicBool>,
        show_collision_debug: Arc<AtomicBool>,
    ) -> Self {
        Self {
            speed: 5.0,
            min_speed: 0.5,
            max_speed: 50.0,
            sensitivity: 0.003,
            keys: HashSet::new(),
            dragging: false,
            last_cursor: None,
            look_delta: (0.0, 0.0),
            scroll_delta: 0.0,
            mouse_captured: false,
            show_light_debug,
            show_collision_debug,
            // 默认：半尺寸 (0.3, 0.9, 0.3) 的小盒子，代表"人"的占位。
            collider: Aabb::from_half_extents(Vec3::ZERO, Vec3::new(0.3, 0.9, 0.3)),
        }
    }

    /// 设置相机碰撞体（局部空间，以相机位置为中心）。
    #[allow(dead_code)] // 公共配置 API：demo/模组可调整碰撞盒大小，暂无调用方
    pub fn with_collider(mut self, collider: Aabb) -> Self {
        self.collider = collider;
        self
    }

    fn pressed(&self, code: KeyCode) -> bool {
        self.keys.contains(&code)
    }

    fn capture_mouse(&mut self, window: &Window) {
        if window.set_cursor_grab(CursorGrabMode::Locked).is_ok() {
            window.set_cursor_visible(false);
            self.mouse_captured = true;
        } else if window.set_cursor_grab(CursorGrabMode::Confined).is_ok() {
            // 部分平台不支持锁定光标，退而求其次限制在窗口内。
            window.set_cursor_visible(false);
            self.mouse_captured = true;
        }
    }

    fn release_mouse(&mut self, window: &Window) {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        self.mouse_captured = false;
        self.last_cursor = None;
    }
}

impl InputController<Camera> for FreeCameraController {
    type Action = CameraAction;

    fn handle_event(&mut self, event: &WindowEvent, window: &Window) {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let PhysicalKey::Code(code) = key_event.physical_key else {
                    return;
                };
                match key_event.state {
                    ElementState::Pressed => {
                        self.keys.insert(code);
                        // L：切换灯光调试可视化（长按不重复触发）。
                        if code == KeyCode::KeyL && !key_event.repeat {
                            let on = self.show_light_debug.fetch_xor(true, Ordering::Relaxed);
                            eprintln!("灯光调试可视化：{}", if on { "关" } else { "开" });
                        }
                        // B：切换碰撞箱调试可视化（长按不重复触发）。
                        if code == KeyCode::KeyB && !key_event.repeat {
                            let on = self
                                .show_collision_debug
                                .fetch_xor(true, Ordering::Relaxed);
                            eprintln!("碰撞箱调试可视化：{}", if on { "关" } else { "开" });
                        }
                        // Esc 释放鼠标，回到系统光标。
                        if code == KeyCode::Escape && self.mouse_captured {
                            self.release_mouse(window);
                        }
                    }
                    ElementState::Released => {
                        self.keys.remove(&code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.dragging = *state == ElementState::Pressed;
                    if !self.dragging {
                        self.last_cursor = None;
                    }
                }
                // 点击窗口后捕获鼠标，进入自由视角。
                if *state == ElementState::Pressed && !self.mouse_captured {
                    self.capture_mouse(window);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // 未捕获鼠标时（例如平台不支持捕获），左键拖动作为兜底。
                if self.dragging && !self.mouse_captured {
                    if let Some((last_x, last_y)) = self.last_cursor {
                        self.look_delta.0 += (position.x - last_x) as f32;
                        self.look_delta.1 += (position.y - last_y) as f32;
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                };
                self.scroll_delta += scroll;
            }
            _ => {}
        }
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured || self.dragging {
                self.look_delta.0 += delta.0 as f32;
                self.look_delta.1 += delta.1 as f32;
            }
        }
    }

    fn update(
        &mut self,
        target: &Camera,
        dt: f32,
        scene: &Scene,
        meshes: &MeshLibrary,
    ) -> CameraAction {
        // 1. 鼠标旋转（向下拖动鼠标 → 俯视）。
        let yaw_delta = self.look_delta.0 * self.sensitivity;
        let pitch_delta = -self.look_delta.1 * self.sensitivity;
        self.look_delta = (0.0, 0.0);

        // 2. 滚轮调整移动速度（每格 ×1.25，clamp 到 [min_speed, max_speed]）。
        if self.scroll_delta != 0.0 {
            self.speed = (self.speed * 1.25_f32.powf(self.scroll_delta))
                .clamp(self.min_speed, self.max_speed);
            self.scroll_delta = 0.0;
        }

        // 3. 键盘移动。
        let mut movement = Vec3::ZERO;
        if self.pressed(KeyCode::KeyW) || self.pressed(KeyCode::ArrowUp) {
            movement += target.forward_horizontal();
        }
        if self.pressed(KeyCode::KeyS) || self.pressed(KeyCode::ArrowDown) {
            movement -= target.forward_horizontal();
        }
        if self.pressed(KeyCode::KeyD) || self.pressed(KeyCode::ArrowRight) {
            movement += target.right();
        }
        if self.pressed(KeyCode::KeyA) || self.pressed(KeyCode::ArrowLeft) {
            movement -= target.right();
        }
        if self.pressed(KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.pressed(KeyCode::ShiftLeft) {
            movement -= Vec3::Y;
        }

        let delta = movement.normalize_or_zero() * self.speed * dt;
        // 分轴滑动：先水平（X、Z）再垂直（Y），每轴单独测碰撞。
        // 撞到障碍就放弃该轴分量，其余轴继续——贴墙移动不会卡死，
        // 斜向撞墙会自然沿墙滑过去。探测基于"目标当前位置 + 已通过分量 +
        // 本轴分量"，控制器不修改目标，最终平移量由返回的操作携带。
        let mut translate = Vec3::ZERO;
        for axis in [Vec3::X, Vec3::Z, Vec3::Y] {
            let step = delta * axis;
            if step == Vec3::ZERO {
                continue;
            }
            let probe_center = target.position() + translate + step;
            let transform = Transform::new(probe_center, Quat::IDENTITY, Vec3::ONE);
            if scene.collides_with(meshes, &transform, self.collider, &[]).is_none() {
                translate += step;
            }
        }

        CameraAction {
            translate,
            yaw_delta,
            pitch_delta,
        }
    }
}

impl Default for FreeCameraController {
    fn default() -> Self {
        Self::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Mesh, SceneObject, SceneObjectKind};

    /// 默认碰撞盒：半尺寸 (0.3, 0.9, 0.3)，以相机位置为中心。
    fn controller() -> FreeCameraController {
        let mut c = FreeCameraController::default();
        c.keys.insert(KeyCode::KeyW);
        c
    }

    /// 场景：把若干边长 1 的立方体（`Mesh::cube`）摆在 `centers` 处作为障碍，
    /// 障碍 AABB 即 [center-0.5, center+0.5]³。
    fn obstacle_scene(centers: &[Vec3]) -> (Scene, MeshLibrary) {
        let mut scene = Scene::new();
        let mut meshes = MeshLibrary::new();
        for center in centers {
            let key = meshes.register(Mesh::cube());
            scene.add_object(SceneObject::new(
                SceneObjectKind::Mesh(key),
                Transform::new(*center, Quat::IDENTITY, Vec3::ONE),
            ));
        }
        (scene, meshes)
    }

    fn camera() -> Camera {
        Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0)
    }

    /// 正前方有墙：W 移动被挡，操作平移量为零。
    #[test]
    fn forward_movement_blocked_by_wall() {
        let mut c = controller();
        let cam = camera();
        // yaw=0 时 W = 朝 +X；墙在 (1,0,0)，中心距 1 < 0.5+0.3。
        let (scene, meshes) = obstacle_scene(&[Vec3::new(1.0, 0.0, 0.0)]);
        let action = c.update(&cam, 0.1, &scene, &meshes);
        assert_eq!(action.translate, Vec3::ZERO, "撞墙应被挡在原地");
        assert_eq!(cam.position(), Vec3::ZERO, "控制器不应修改目标");
    }

    /// 无障碍时正常移动：W 走 speed × dt。
    #[test]
    fn forward_movement_free_when_no_obstacle() {
        let mut c = controller();
        let cam = camera();
        let (scene, meshes) = obstacle_scene(&[]);
        let action = c.update(&cam, 0.1, &scene, &meshes);
        assert_eq!(action.translate, Vec3::new(0.5, 0.0, 0.0));
    }

    /// 斜向撞墙：X 分量被挡，Z 分量继续 → 沿墙滑动，不卡死。
    #[test]
    fn diagonal_movement_slides_along_wall() {
        let mut c = controller();
        c.keys.insert(KeyCode::KeyD);
        let cam = camera();
        // W+D：归一化后 (0.707, 0, 0.707)，speed 5 × dt 0.1 = 0.354/轴。
        // X 轴被墙 (1,0,0) 挡下，Z 轴照常移动。
        let (scene, meshes) = obstacle_scene(&[Vec3::new(1.0, 0.0, 0.0)]);
        let action = c.update(&cam, 0.1, &scene, &meshes);
        assert!(action.translate.x.abs() < 1e-6, "X 应被挡：{:?}", action);
        assert!(
            (action.translate.z - std::f32::consts::FRAC_1_SQRT_2 * 0.5).abs() < 1e-6,
            "Z 应滑动：{:?}",
            action
        );
    }

    /// 移动速度不会被碰撞测试改变（speed 是乘性调参，碰撞只回退位置）。
    #[test]
    fn collision_does_not_change_speed() {
        let mut c = controller();
        let cam = camera();
        let (scene, meshes) = obstacle_scene(&[Vec3::new(1.0, 0.0, 0.0)]);
        c.update(&cam, 0.1, &scene, &meshes);
        assert_eq!(c.speed, 5.0);
    }
}
