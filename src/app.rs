//! App：各功能的集成点。
//!
//! 持有 [`Playground`]（运行中的 ECS 世界：实体 + 双调度器 + 输入快照），
//! winit 事件只负责**累积输入快照**，不直接改游戏状态；每帧 `advance`
//! 跑物理刻（世界变换传播、自由相机），`prepare_frame` 跑渲染刻
//! （查询组件 → 填渲染指令），再把指令交给渲染器。

use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use crate::engine::ecs::playground::Playground;
use crate::engine::render::Renderer;
use crate::engine::{
    AssetManager, Camera, DisplayHandle, Environment, GamePath, GcPolicy, GpuManager, Handle,
    MergedResourceSpace, Mesh, PackConfig, Scene, Texture,
};

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    /// 运行中的 ECS 世界（场景实例、输入快照、渲染帧数据）。
    playground: Playground,
    last_frame: Instant,
    /// CPU 资产管理器：内存副本、磁盘读取与内存卸载（无 GPU 依赖）。
    assets: AssetManager,
    /// GPU 资产管理器：句柄 → 显存表示的驻留表（上传/回收）。
    gpu: GpuManager,
    /// 鼠标是否已捕获（自由视角）。
    mouse_captured: bool,
}

impl App {
    /// 装配各子系统。窗口已由 main.rs 创建好，这里初始化渲染器与 ECS 世界。
    pub fn new(window: Arc<Window>, display: DisplayHandle) -> anyhow::Result<Self> {
        let renderer = Renderer::new(&window, display)?;
        // 启动主流程：扫描有效包 → 生成/更新 packs.toml 顺序（环 → 报错），
        // 再按 order 校验依赖与冲突（不满足 → 报错退出）。
        let (pack_config, packages) = PackConfig::discover_and_update("game-data")?;
        pack_config.validate(&packages)?;
        eprintln!("资源包加载顺序：{}", pack_config.order().join(" → "));
        let assets = AssetManager::new(MergedResourceSpace::from_pack_roots(
            pack_config.pack_roots("game-data"),
        ));
        let gpu = GpuManager::new(
            std::sync::Arc::new(renderer.device()),
            std::sync::Arc::new(renderer.queue().clone()),
        );

        let mut app = Self {
            window,
            renderer,
            playground: Playground::new(),
            last_frame: Instant::now(),
            assets,
            gpu,
            mouse_captured: false,
        };
        app.load_startup_scene();
        Ok(app)
    }

    /// 启动场景：优先加载 `BACKROOMS_GLTF` 环境变量指定的 glTF 文件，
    /// 其次尝试 `assets/scene.glb`；都不可用时回退到内置演示场景。
    fn load_startup_scene(&mut self) {
        // 环境（天空盒 + IBL）是关卡数据的一部分：解码后绑定到场景上，
        // 由 `load_scene` 统一上传，而不是单独加载。
        let environment = self.load_example_environment();

        let keys = self.register_meshes(vec![Mesh::triangle(), Mesh::quad(), Mesh::cube()]);
        let [triangle, quad, cube] = keys.as_slice() else {
            unreachable!("demo 注册了 3 个网格")
        };
        // 演示贴图：棋盘格贴到立方体上，验证纹理采样。
        let checker = self
            .register_textures(vec![Texture::checkerboard(64, 8)])
            .pop()
            .expect("注册了 1 张贴图");
        let mut scene = Scene::demo(*triangle, *quad, *cube, Some(checker));
        if let Some(env) = &environment {
            scene = scene
                .with_environment(env.clone())
                .with_environment_intensity(1.0);
        }
        // 通过游戏路径加载测试模型：走合并资源空间 → 解析 → 资产库。
        let test_path: GamePath = match "test:test.glb".parse() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("测试模型路径无效：{e}");
                return;
            }
        };
        // 场景加载（节点摆放/材质）走 GamePath 一次解析，并进 demo。
        match self.assets.load_scene(&test_path) {
            Ok(gltf_scene) => {
                // 把测试模型放大 5 倍（等比），并挪到演示物体右前方，
                // 避免和原点处的三角形重叠。
                for key in scene.merge(&gltf_scene) {
                    if let Some(object) = scene.object_mut(key) {
                        object.transform.scale *= 5.0;
                        object.transform.position += glam::Vec3::new(1.8, 0.0, -1.2);
                    }
                }
            }
            Err(e) => eprintln!("加载 {} 场景失败：{e}", test_path),
        }
        self.load_scene(scene);
    }

    /// 加载环境贴图：`BACKROOMS_ENV`（GamePath 字符串）优先，否则 `test:test.hdr`。
    fn load_example_environment(&mut self) -> Option<Arc<Environment>> {
        let path: GamePath = std::env::var("BACKROOMS_ENV")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| "test:test.hdr".parse().expect("内置环境路径合法"));
        let bytes = match self.assets.space().read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("环境贴图读取失败 {path}：{e}");
                return None;
            }
        };
        match Environment::from_bytes(&bytes) {
            Ok(env) => {
                eprintln!("环境贴图 {path} 加载成功（{}×{}）", env.width, env.height);
                Some(Arc::new(env))
            }
            Err(e) => {
                eprintln!("环境贴图解码失败 {path}：{e}");
                None
            }
        }
    }

    /// 尝试从 glTF 文件加载场景；成功返回 `true`，失败打印原因并返回 `false`。
    #[allow(dead_code)] // 预留：BACKROOMS_GLTF / game-data 场景加载路径，demo 阶段未启用
    fn try_load_gltf(&mut self, path: &GamePath, environment: Option<&Arc<Environment>>) -> bool {
        match self.assets.load_scene(path) {
            Ok(scene) => {
                let scene = match environment {
                    Some(env) => scene.with_environment(env.clone()),
                    None => scene,
                };
                self.load_scene(scene);
                true
            }
            Err(e) => {
                eprintln!("加载 glTF {path} 失败：{e}");
                false
            }
        }
    }

    /// 批量注册网格：追加进全局资产库，返回句柄列表。
    pub fn register_meshes(&mut self, meshes: Vec<Mesh>) -> Vec<Handle<Mesh>> {
        meshes
            .into_iter()
            .map(|mesh| self.assets.register(mesh))
            .collect()
    }

    /// 批量注册贴图：追加进全局纹理库，返回句柄列表。
    pub fn register_textures(&mut self, textures: Vec<Texture>) -> Vec<Handle<Texture>> {
        textures
            .into_iter()
            .map(|texture| self.assets.register(texture))
            .collect()
    }

    /// App 级别的场景切换 API：模板场景 → 资产驻留 → 交给 Playground 生成。
    pub fn load_scene(&mut self, mut scene: Scene) {
        // 每个场景必须有一个主相机（出生点视角）；缺省时补一个默认相机。
        self.ensure_main_camera(&mut scene);
        // 环境跟随场景：场景自带环境（天空盒 + IBL）时一并上传；
        // 不带环境时切回默认黑环境，避免残留上一关卡的天空盒。
        match scene.environment() {
            Some(env) => {
                self.renderer.set_environment(env);
                self.renderer
                    .set_environment_intensity(scene.environment_intensity());
                self.renderer
                    .set_environment_agx_ev(scene.agx_min_ev(), scene.agx_max_ev());
            }
            None => self.renderer.reset_environment(),
        }
        // 收集场景引用的资产并 pin（CPU 侧标记驻留），随后 GPU 层上传。
        self.pin_scene_assets(&scene);
        self.gpu.sync(&mut self.assets);
        // 场景模板 → ECS 实体（旧实例自动卸载：unpin 资产 + 级联 despawn；
        // 碰撞盒在生成时从网格 AABB 派生）。
        self.playground.load_scene(&scene, &mut self.assets);
        if self.playground.main_camera().is_none() {
            eprintln!("场景生成后没有主相机（不应发生：ensure_main_camera 已兜底）");
        }
    }

    /// 场景引用的所有网格/贴图句柄标记为驻留（pin）。
    fn pin_scene_assets(&mut self, scene: &Scene) {
        for (_, object) in scene.objects() {
            if let Some(handle) = object.mesh_handle() {
                self.assets.pin(handle);
            }
            for handle in object.material.texture_handles() {
                self.assets.pin(handle);
            }
        }
    }

    /// 场景没有主相机时补一个默认相机（与早期硬编码相机相同的位置/朝向）。
    fn ensure_main_camera(&self, scene: &mut Scene) {
        if scene.main_camera().is_none() {
            let size = self.window.inner_size();
            let aspect = size.width as f32 / size.height.max(1) as f32;
            let camera = scene.add_camera(Camera::new(
                glam::Vec3::new(0.0, 1.0, 3.0),
                -std::f32::consts::FRAC_PI_2, // 初始朝向 -Z，看向原点附近
                -0.15,                        // 稍微俯视
                std::f32::consts::FRAC_PI_4,  // 45° 垂直视野
                aspect,
                0.1,
                100.0,
            ));
            assert!(
                scene.set_main_camera(camera),
                "新添加的相机节点必然能设为主相机"
            );
        }
    }

    /// main.rs 转发的窗口事件统一在这里处理：只累积输入快照 / 切换调试开关。
    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                let aspect = size.width as f32 / size.height.max(1) as f32;
                self.playground.set_aspect(aspect);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let PhysicalKey::Code(code) = key_event.physical_key else {
                    return;
                };
                self.playground
                    .set_key(code, key_event.state == ElementState::Pressed);
                if key_event.state == ElementState::Pressed {
                    // L：切换灯光调试可视化（长按不重复触发）。
                    if code == KeyCode::KeyL && !key_event.repeat {
                        let on = self.playground.toggle_light_debug();
                        eprintln!("灯光调试可视化：{}", if on { "关" } else { "开" });
                    }
                    // B：切换碰撞箱调试可视化（长按不重复触发）。
                    if code == KeyCode::KeyB && !key_event.repeat {
                        let on = self.playground.toggle_collision_debug();
                        eprintln!("碰撞箱调试可视化：{}", if on { "关" } else { "开" });
                    }
                }
                // Esc 释放鼠标，回到系统光标。
                if code == KeyCode::Escape && self.mouse_captured {
                    self.release_mouse();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                // 点击窗口后捕获鼠标，进入自由视角。
                if *button == MouseButton::Left
                    && *state == ElementState::Pressed
                    && !self.mouse_captured
                {
                    self.capture_mouse();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                };
                self.playground.add_scroll(scroll);
            }
            _ => {}
        }
    }

    /// main.rs 转发的设备事件（鼠标相对位移等）。
    pub fn handle_device_event(&mut self, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured {
                self.playground
                    .add_look_delta(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn capture_mouse(&mut self) {
        if self.window.set_cursor_grab(CursorGrabMode::Locked).is_ok() {
            self.window.set_cursor_visible(false);
            self.mouse_captured = true;
        } else if self
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .is_ok()
        {
            // 部分平台不支持锁定光标，退而求其次限制在窗口内。
            self.window.set_cursor_visible(false);
            self.mouse_captured = true;
        }
    }

    fn release_mouse(&mut self) {
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.mouse_captured = false;
    }

    /// 每帧更新入口：固定步长物理刻（世界变换 + 相机 + 资产回收），
    /// 然后渲染刻准备帧数据，请求下一帧。
    pub fn update(&mut self) {
        let now = Instant::now();
        let frame_dt = (now - self.last_frame).min(Duration::from_secs_f32(0.25));
        self.last_frame = now;

        let ticks = self.playground.advance(frame_dt);
        for _ in 0..ticks {
            // 资产回收挂到物理刻（统一 GC 策略）。
            let policy = GcPolicy::default();
            self.assets.gc(&policy);
            self.gpu.gc(&policy);
        }

        self.playground.prepare_frame();
        self.window.request_redraw();
    }

    /// 渲染一帧：把 Playground 里的渲染指令交给渲染器执行（指令只带资源库
    /// 句柄，渲染器绘制时拿句柄向库取 GPU 数据）。
    fn draw(&mut self) {
        let command = self.playground.render_frame();
        self.renderer
            .render(command, &mut self.gpu, &mut self.assets);
    }
}
