mod app;
mod camera;
mod controller;
mod mesh;
mod render;
mod scene;

use std::error::Error;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{Window, WindowId};

use app::App;

/// main.rs 里的薄壳：创建窗口，并把窗口事件原样转发给 App。
struct WindowedApp {
    app: Option<App>,
}

impl WindowedApp {
    fn new() -> Self {
        Self { app: None }
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
        let attributes = Window::default_attributes()
            .with_title("Live in Backrooms")
            .with_inner_size(LogicalSize::new(1280.0, 720.0));
        Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create window"),
        )
    }
}

impl ApplicationHandler for WindowedApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // 应用恢复时（首次启动、从后台回到前台）创建窗口并装配 App。
        if self.app.is_none() {
            let window = Self::create_window(event_loop);
            let display = Box::new(event_loop.owned_display_handle());
            self.app = Some(App::new(window, display).expect("failed to initialize app"));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // 只做转发：窗口事件交给 App 处理。
        if let Some(app) = &mut self.app {
            app.handle_window_event(event_loop, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = &mut self.app {
            app.update();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // 只做转发：设备事件交给 App 处理。
        if let Some(app) = &mut self.app {
            app.handle_device_event(event);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    // 使用 Wait：事件循环无事可做时真正休眠，渲染节奏由 update() 里的
    // request_redraw() 驱动；需要按帧动画时用 WaitUntil 即可。
    event_loop.set_control_flow(ControlFlow::Wait);
    // 始终接收设备事件（自由视角需要鼠标相对位移）。
    event_loop.listen_device_events(DeviceEvents::Always);

    let mut windowed_app = WindowedApp::new();
    event_loop.run_app(&mut windowed_app)?;
    Ok(())
}
