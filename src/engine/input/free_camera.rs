//! 自由相机控制器：第一人称式输入 → 相机。

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec3;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use super::InputController;
use crate::engine::core::camera::Camera;

/// 第一人称式相机控制器。
///
/// 操作：
/// - WASD / 方向键：水平移动
/// - Space / Ctrl：上升 / 下降
/// - 点击窗口：捕获鼠标，之后移动鼠标直接旋转视角（自由视角）
/// - Esc：释放鼠标（返回系统光标）
/// - 滚轮：调整移动速度（乘法步进，带上下限）
/// - L：切换灯光调试可视化（灯泡 + 射线的显示/隐藏）
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
}

impl FreeCameraController {
    /// 新建控制器；`show_light_debug` 是与 App 共享的灯光调试开关，
    /// 由 L 键翻转，App 侧每帧读取决定是否绘制调试线框。
    pub fn new(show_light_debug: Arc<AtomicBool>) -> Self {
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
        }
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

    fn update(&mut self, target: &mut Camera, dt: f32) {
        // 1. 鼠标旋转（向下拖动鼠标 → 俯视）。
        target.rotate(
            self.look_delta.0 * self.sensitivity,
            -self.look_delta.1 * self.sensitivity,
        );
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

        if movement != Vec3::ZERO {
            target.translate(movement.normalize_or_zero() * self.speed * dt);
        }
    }
}

impl Default for FreeCameraController {
    fn default() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)))
    }
}
