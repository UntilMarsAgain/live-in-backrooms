mod app;
mod engine;
mod game;

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{Window, WindowId};

use app::App;
use tracing_subscriber::layer::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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
            match App::new(window, display) {
                Ok(app) => self.app = Some(app),
                Err(e) => {
                    // 启动期错误（资源包依赖/冲突/环等）：打印后退出，不 panic。
                    tracing::error!("应用启动失败：{e:#}");
                    event_loop.exit();
                }
            }
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = &mut self.app {
            app.update();
        }
    }
}

fn main() -> anyhow::Result<()> {
    // 日志：终端与文件挂**各自的** EnvFilter，等级互不影响。
    // - 终端：`RUST_LOG` 控制（默认 info）；
    // - 文件：`BACKROOMS_FILE_LOG` 控制（默认 info；debug 会引入
    //   wgpu/bevy 等第三方库的大量日志，需要排查时再临时调高）。
    // 文件每次启动**覆盖**写入 logs/live-in-backrooms.log，避免长期积累
    // 把硬盘写满。
    let stdout_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let file_filter = std::env::var("BACKROOMS_FILE_LOG")
        .map(tracing_subscriber::EnvFilter::new)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(stdout_filter);
    let _ = std::fs::create_dir_all("logs");
    match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("logs/live-in-backrooms.log")
    {
        Ok(file) => {
            tracing_subscriber::registry()
                .with(stdout_layer)
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_writer(file)
                        .with_ansi(false)
                        .with_filter(file_filter),
                )
                .init();
        }
        Err(e) => {
            // 文件打开失败时降级为仅终端输出，不阻断启动。
            eprintln!("无法打开日志文件 logs/live-in-backrooms.log：{e}，仅输出到终端");
            tracing_subscriber::registry().with(stdout_layer).init();
        }
    }

    let event_loop = EventLoop::new()?;
    // 使用 Wait：事件循环无事可做时真正休眠，渲染节奏由 update() 里的
    // request_redraw() 驱动；需要按帧动画时用 WaitUntil 即可。
    event_loop.set_control_flow(ControlFlow::Wait);
    // 始终接收设备事件（自由视角需要鼠标相对位移）。
    event_loop.listen_device_events(DeviceEvents::WhenFocused);

    let mut windowed_app = WindowedApp::new();
    event_loop.run_app(&mut windowed_app)?;
    Ok(())
}
