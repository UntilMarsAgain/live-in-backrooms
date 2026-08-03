//! 相机控制器：把键盘/鼠标输入翻译成相机的移动与旋转。
//!
//! 事件回调里只记录输入状态，每帧 [`FreeCameraController::update`] 时统一
//! 应用到相机，避免在事件回调中直接修改相机造成顺序不一致。

use std::collections::HashSet;

use glam::Vec3;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use super::Camera;

/// 第一人称式相机控制器。
///
/// 操作：
/// - WASD / 方向键：水平移动
/// - Space / Ctrl：上升 / 下降
/// - 点击窗口：捕获鼠标，之后移动鼠标直接旋转视角（自由视角）
/// - Esc：释放鼠标（返回系统光标）
/// - 滚轮：沿视线前后推进
pub struct FreeCameraController {
    /// 移动速度（单位 / 秒）。
    speed: f32,
    /// 鼠标灵敏度（弧度 / 像素）。
    sensitivity: f32,
    /// 滚轮推进速度（单位 / 格）。
    scroll_speed: f32,
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
}

impl FreeCameraController {
    pub fn new() -> Self {
        Self {
            speed: 5.0,
            sensitivity: 0.003,
            scroll_speed: 1.5,
            keys: HashSet::new(),
            dragging: false,
            last_cursor: None,
            look_delta: (0.0, 0.0),
            scroll_delta: 0.0,
            mouse_captured: false,
        }
    }

    /// 处理 winit 窗口事件（只关心输入类事件，其余忽略）。
    pub fn handle_event(&mut self, event: &WindowEvent, window: &Window) {
        match event {
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                let PhysicalKey::Code(code) = key_event.physical_key else {
                    return;
                };
                match key_event.state {
                    ElementState::Pressed => {
                        self.keys.insert(code);
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
            WindowEvent::MouseInput {
                state, button, ..
            } => {
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

    /// 处理设备事件（鼠标相对位移等），捕获状态下提供自由视角。
    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured || self.dragging {
                self.look_delta.0 += delta.0 as f32;
                self.look_delta.1 += delta.1 as f32;
            }
        }
    }

    /// 每帧调用：把积累的输入应用到相机。
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        // 1. 鼠标旋转（向下拖动鼠标 → 俯视）。
        camera.rotate(
            self.look_delta.0 * self.sensitivity,
            -self.look_delta.1 * self.sensitivity,
        );
        self.look_delta = (0.0, 0.0);

        // 2. 滚轮沿视线推进。
        if self.scroll_delta != 0.0 {
            camera.move_forward(self.scroll_delta * self.scroll_speed);
            self.scroll_delta = 0.0;
        }

        // 3. 键盘移动。
        let mut movement = Vec3::ZERO;
        if self.pressed(KeyCode::KeyW) || self.pressed(KeyCode::ArrowUp) {
            movement += camera.forward_horizontal();
        }
        if self.pressed(KeyCode::KeyS) || self.pressed(KeyCode::ArrowDown) {
            movement -= camera.forward_horizontal();
        }
        if self.pressed(KeyCode::KeyD) || self.pressed(KeyCode::ArrowRight) {
            movement += camera.right();
        }
        if self.pressed(KeyCode::KeyA) || self.pressed(KeyCode::ArrowLeft) {
            movement -= camera.right();
        }
        if self.pressed(KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.pressed(KeyCode::ControlLeft) || self.pressed(KeyCode::ControlRight) {
            movement -= Vec3::Y;
        }

        if movement != Vec3::ZERO {
            camera.translate(movement.normalize_or_zero() * self.speed * dt);
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

impl Default for FreeCameraController {
    fn default() -> Self {
        Self::new()
    }
}
