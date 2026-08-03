//! App：各功能的集成点。
//!
//! 这里持有窗口与渲染器等子系统，把 winit 事件翻译成上层逻辑。
//! 后续的输入、游戏逻辑、场景管理都可以从这里接入。

use std::sync::Arc;
use std::time::Instant;

use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::camera::{Camera, FreeCameraController};
use crate::render::{DisplayHandle, Renderer, RendererError};
use crate::scene::Scene;

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    camera: Camera,
    controller: FreeCameraController,
    last_frame: Instant,
    scene: Scene,
}

impl App {
    /// 装配各子系统。窗口已由 main.rs 创建好，这里初始化渲染器。
    pub fn new(window: Arc<Window>, display: DisplayHandle) -> Result<Self, RendererError> {
        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = Camera::new(
            glam::Vec3::new(0.0, 1.0, 3.0),
            -std::f32::consts::FRAC_PI_2, // 初始朝向 -Z，正好看向原点附近的三角形
            -0.15,                        // 稍微俯视
            std::f32::consts::FRAC_PI_4,  // 45° 垂直视野
            aspect,
            0.1,
            100.0,
        );
        let controller = FreeCameraController::new();
        let renderer = Renderer::new(&window, display)?;
        let mut app = Self {
            window,
            renderer,
            camera,
            controller,
            last_frame: Instant::now(),
            scene: Scene::default(),
        };
        // 启动时加载演示场景。
        app.load_scene(Scene::demo());
        Ok(app)
    }

    /// App 级别的场景切换 API：渲染器（GPU 数据）与后续游戏逻辑统一从这里换场景。
    pub fn load_scene(&mut self, scene: Scene) {
        self.renderer.load_scene(&scene);
        self.scene = scene;
    }

    /// main.rs 转发的窗口事件统一在这里处理。
    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.camera
                    .set_aspect(size.width as f32 / size.height.max(1) as f32);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.renderer.render(&self.camera, &self.scene),
            // 其余（键盘、鼠标等）交给控制器处理。
            _ => self.controller.handle_event(&event, &self.window),
        }
    }

    /// main.rs 转发的设备事件（鼠标相对位移等）统一在这里处理。
    pub fn handle_device_event(&mut self, event: DeviceEvent) {
        self.controller.handle_device_event(&event);
    }

    /// 每帧更新入口：后续的游戏逻辑（输入、物理、场景）从这里扩展。
    pub fn update(&mut self) {
        // 计算帧间隔并交给控制器。
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.25);
        self.last_frame = now;

        self.controller.update(&mut self.camera, dt);
        // 请求下一帧，驱动持续渲染。
        self.window.request_redraw();
    }
}
