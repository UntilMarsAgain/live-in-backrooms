//! App：各功能的集成点。
//!
//! 这里持有窗口与渲染器等子系统，把 winit 事件翻译成上层逻辑。
//! 后续的输入、游戏逻辑、场景管理都可以从这里接入。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::{
    AssetManager, ClientScene, DisplayHandle, FreeCameraController, InputController, Renderer,
};
use crate::sync::SnapshotPlayer;
use lbr_server::Server;
use lbr_shared::WorldSnapshot;
use lbr_shared::{Camera, CameraAction, Environment, GamePath, MergedResourceSpace};

/// 单机集成模式下服务端的 tick 频率（Hz）。
const SERVER_TICK_HZ: f64 = 30.0;

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    controller: Box<dyn InputController<Camera, Action = CameraAction>>,
    last_frame: Instant,
    /// 统一资产管理器：网格/贴图等资源的句柄与 CPU/GPU 双持有。
    assets: AssetManager,
    /// 客户端视图场景：共享场景树 + 渲染信息（材质/相机/环境）。
    scene: ClientScene,
    /// 服务端快照接收端：服务端跑在独立线程，最新状态优先。
    snapshots: std::sync::mpsc::Receiver<WorldSnapshot>,
    /// 客户端同步层：快照 → 视图场景（实体模板缓存 + 变换更新）。
    sync: SnapshotPlayer,
    /// 灯光调试可视化开关（控制器按 L 翻转，渲染时读取）。
    show_light_debug: Arc<AtomicBool>,
    /// 碰撞箱调试可视化开关（控制器按 B 翻转，渲染时读取）。
    show_collision_debug: Arc<AtomicBool>,
}

impl App {
    /// 装配各子系统。窗口已由 main.rs 创建好，这里初始化渲染器。
    pub fn new(window: Arc<Window>, display: DisplayHandle) -> anyhow::Result<Self> {
        // 灯光调试开关由控制器独占翻转，App 只读；用 Arc 共享给两边。
        let show_light_debug = Arc::new(AtomicBool::new(false));
        let show_collision_debug = Arc::new(AtomicBool::new(false));
        let controller: Box<dyn InputController<Camera, Action = CameraAction>> = Box::new(
            FreeCameraController::new(show_light_debug.clone(), show_collision_debug.clone()),
        );
        let renderer = Renderer::new(&window, display)?;
        let assets = AssetManager::new(
            std::sync::Arc::new(renderer.device()),
            std::sync::Arc::new(renderer.queue().clone()),
            MergedResourceSpace::new("game-data/vanilla/".into()),
        );
        // 服务端权威模拟跑在独立线程，快照经内存队列进入本线程（无序列化）。
        let snapshots = Server::spawn_thread(SERVER_TICK_HZ);
        let mut app = Self {
            window,
            renderer,
            controller,
            last_frame: Instant::now(),
            assets,
            scene: ClientScene::new(),
            snapshots,
            sync: SnapshotPlayer::new(),
            show_light_debug,
            show_collision_debug,
        };
        app.load_startup_scene();
        Ok(app)
    }

    /// 启动场景：只装配客户端渲染信息（环境/相机），实体由服务端快照驱动。
    fn load_startup_scene(&mut self) {
        // 环境（天空盒 + IBL）是关卡数据的一部分：解码后绑定到场景上，
        // 由 `load_scene` 统一上传，而不是单独加载。
        let environment = self.load_example_environment();
        let mut scene = ClientScene::new();
        if let Some(env) = &environment {
            scene = scene
                .with_environment(env.clone())
                .with_environment_intensity(1.0);
        }
        // 首个服务端快照会在 update() 里被消费并自动装配实体。
        self.load_scene(scene);
    }

    /// 加载环境贴图：`BACKROOMS_ENV`（GamePath 字符串）优先，否则 `test:test.hdr`。
    /// 经合并资源空间读取，按内容自动识别 HDR / LDR。
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
                eprintln!(
                    "环境贴图 {path} 加载成功（{}×{}）",
                    env.width, env.height
                );
                Some(Arc::new(env))
            }
            Err(e) => {
                eprintln!("环境贴图解码失败 {path}：{e}");
                None
            }
        }
    }

    /// App 级别的场景切换 API：渲染器（GPU 数据）与后续游戏逻辑统一从这里换场景。
    pub fn load_scene(&mut self, scene: ClientScene) {
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
        // 收集场景引用的资产并 pin（预上传，预分配语义；不 pin 也有渲染兜底）。
        pin_scene_assets(&mut self.assets, &scene);
        self.renderer.load_scene(&scene, &self.assets);
        self.scene = scene;
    }

    /// main.rs 转发的窗口事件统一在这里处理。
    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.scene
                    .camera_mut()
                    .set_aspect(size.width as f32 / size.height.max(1) as f32);
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // 主相机由客户端视图场景持有：切换场景即切换出生点视角。
                self.renderer.render(
                    self.scene.camera(),
                    &self.scene,
                    &mut self.assets,
                    self.show_light_debug.load(Ordering::Relaxed),
                    self.show_collision_debug.load(Ordering::Relaxed),
                );
            }
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

        // 资产 GPU 同步：pinned 且未上传的上传，非 pinned 的回收。
        self.assets.sync_gpu();

        // 消费服务端最新快照（丢弃积压：只留最新一帧，"最新状态优先"）。
        let mut latest: Option<WorldSnapshot> = None;
        while let Ok(snap) = self.snapshots.try_recv() {
            latest = Some(snap);
        }
        if let Some(snap) = latest {
            match self.sync.apply(&mut self.assets, &snap, &mut self.scene) {
                Ok(true) => {
                    // 实体结构变化：pin 新资产、上传、重建 GPU 静态部分。
                    pin_scene_assets(&mut self.assets, &self.scene);
                    self.assets.sync_gpu();
                    self.renderer.load_scene(&self.scene, &self.assets);
                }
                Ok(false) => {}
                Err(e) => eprintln!("应用服务端快照失败：{e:#}"),
            }
        }

        // 控制器只读相机 + 查询共享场景树碰撞，输出一帧操作；操作由视图场景
        // 统一应用（不可变借用先结束，再可变借用应用，无冲突）。
        let action = self
            .controller
            .update(self.scene.camera(), dt, self.scene.scene(), &self.assets);
        self.scene.apply_camera_action(action);
        // 请求下一帧，驱动持续渲染。
        self.window.request_redraw();
    }
}

/// 场景引用的所有网格/贴图句柄标记为 GPU 驻留（pin）。
fn pin_scene_assets(assets: &mut AssetManager, scene: &ClientScene) {
    for (key, object) in scene.scene().objects() {
        if let Some(handle) = object.mesh_handle() {
            assets.pin_meshes(handle);
        }
        if let Some(material) = scene.material(key) {
            for handle in material.texture_handles() {
                assets.pin_textures(handle);
            }
        }
    }
}
