//! 输入子系统：定义输入控制器接口。
//!
//! [`InputController`] 是"输入 → 目标"的抽象（目前目标是相机）：事件回调只积累
//! 输入状态，`update` 每帧统一应用到目标，避免在事件回调中直接修改目标造成
//! 顺序不一致。未来的角色控制器、轨道相机控制器都实现同一接口，App 可以替换。
//!
//! 与 [`super::core::camera`] 解耦：相机模块只负责纯数学与 GPU 数据，不依赖输入。

mod free_camera;

pub use free_camera::FreeCameraController;

use winit::event::{DeviceEvent, WindowEvent};
use winit::window::Window;

/// 输入控制器接口：处理输入事件，并在每帧把积累的输入应用到目标。
pub trait InputController<T> {
    /// 处理窗口事件（键盘/鼠标/滚轮）。
    fn handle_event(&mut self, event: &WindowEvent, window: &Window);

    /// 处理设备事件（鼠标相对位移等）。
    fn handle_device_event(&mut self, event: &DeviceEvent);

    /// 每帧调用：把积累的输入应用到目标。
    fn update(&mut self, target: &mut T, dt: f32);
}
