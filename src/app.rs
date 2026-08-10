//! App：各功能的集成点。
//!
//! 持有 [`Playground`]（运行中的 ECS 世界：实体 + 双调度器 + 输入快照），
//! winit 事件只负责**累积输入快照**，不直接改游戏状态；每帧 `advance`
//! 跑物理刻（世界变换传播、自由相机），`prepare_frame` 跑渲染刻
//! （查询组件 → 填渲染指令），再把指令交给渲染器。

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Quat, Vec3};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

use crate::engine::ecs::playground::Playground;
use crate::engine::render::Renderer;
use crate::engine::{
    AssetManager, Camera, DisplayHandle, Environment, GamePath, GcPolicy, GpuManager, Handle,
    MergedResourceSpace, Mesh, PackConfig, PinToken, Scene, SceneObject, SceneObjectKind, Texture,
    Transform,
};

/// 资产 GC 的最小间隔：每真实时间 5 秒跑一次（不是每个物理刻）。
const GC_INTERVAL: Duration = Duration::from_secs(5);

/// CPU 资产逐出窗口：**放宽**——逐出后重载要读磁盘 + 完整重解析
/// （glTF 几十~几百毫秒），且内存容量通常比显存宽裕。
const CPU_GC_STALE_WINDOW: Duration = Duration::from_mins(5);

/// GPU 资产逐出窗口：**收紧**——显存更宝贵；重传只需 CPU 数据在内存时
/// 再上传（毫秒级），窗口短一点能及时腾显存。
const GPU_GC_STALE_WINDOW: Duration = Duration::from_secs(10);

/// 未配置环境变量时，CPU 预算 = 系统物理内存的 25%（保守：操作系统、
/// 其他程序也要内存，别把机器吃满）。
const CPU_MEMORY_RATIO: f64 = 0.25;

/// 未配置环境变量时，GPU 预算的兜底值（字节；显存总量跨平台不可靠查询，
/// 用 2 GiB 起步，可在环境变量里按机器调）。
const DEFAULT_GPU_MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// 内存预算配置：CPU / GPU 各自的驻留上限（字节）。
#[derive(Debug, Clone, Copy)]
struct MemoryBudget {
    cpu_limit: u64,
    gpu_limit: u64,
}

impl MemoryBudget {
    /// 从环境变量读取（单位 MiB），未设置时按玩家机器推算：
    /// - CPU：系统物理内存的 [`CPU_MEMORY_RATIO`]；
    /// - GPU：固定兜底 [`DEFAULT_GPU_MEMORY_LIMIT`]。
    ///
    /// 环境变量优先级最高，便于按机器覆盖：
    /// `BACKROOMS_CPU_MEMORY_MB` / `BACKROOMS_GPU_MEMORY_MB`。
    fn load() -> Self {
        let cpu_limit = env_mib("BACKROOMS_CPU_MEMORY_MB")
            .unwrap_or_else(|| {
                total_system_memory()
                    .map(|total| (total as f64 * CPU_MEMORY_RATIO) as u64)
                    .unwrap_or(DEFAULT_GPU_MEMORY_LIMIT)
            });
        let gpu_limit = env_mib("BACKROOMS_GPU_MEMORY_MB")
            .unwrap_or(DEFAULT_GPU_MEMORY_LIMIT);
        Self {
            cpu_limit,
            gpu_limit,
        }
    }
}

/// 读取环境变量（单位 MiB → 字节）；未设置/非法返回 `None`。
fn env_mib(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    let mib: u64 = value.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

/// 系统物理内存总量（字节）。用 sysinfo 跨平台读取（Linux/Windows/macOS）；
/// 读取失败或为 0 时返回 `None`（预算回退到默认值）。
fn total_system_memory() -> Option<u64> {
    let system = sysinfo::System::new_all();
    let total = system.total_memory();
    (total > 0).then_some(total)
}

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    /// 运行中的 ECS 世界（场景实例、输入快照、渲染帧数据）。
    playground: Playground,
    last_frame: Instant,
    /// CPU 资产管理器：内存副本、磁盘读取与内存卸载（无 GPU 依赖）。
    /// 用 `Arc<Mutex<>>` 共享：`PinToken` 持弱引用，drop 时经锁自动 unpin。
    assets: Arc<std::sync::Mutex<AssetManager>>,
    /// GPU 资产管理器：句柄 → 显存表示的驻留表（上传/回收）。
    gpu: GpuManager,
    /// 鼠标是否已捕获（自由视角）。
    mouse_captured: bool,
    /// 距上次资产 GC 的时间累积（按真实时间，与物理刻解耦）。
    gc_accum: Duration,
    /// CPU / GPU 驻留预算（字节）：内存压力检测的阈值。
    memory_budget: MemoryBudget,
    /// 预构建的演示场景模板（F1/F2 切换）。
    demo_scenes: [Scene; 2],
    /// 当前激活的 demo 下标（0 或 1）。
    current_demo: usize,
}

impl App {
    /// 装配各子系统。窗口已由 main.rs 创建好，这里初始化渲染器与 ECS 世界。
    pub fn new(window: Arc<Window>, display: DisplayHandle) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(&window, display)?;
        // Bloom 默认参数：阈值 2.0（金属反射环境光不至于误触发辉光）、
        // 强度 0.3。后续可做成关卡/场景配置。
        renderer.set_bloom_threshold(2.0);
        renderer.set_bloom_intensity(0.3);
        // 启动主流程：扫描有效包 → 生成/更新 packs.toml 顺序（环 → 报错），
        // 再按 order 校验依赖与冲突（不满足 → 报错退出）。
        let (pack_config, packages) = PackConfig::discover_and_update("game-data")?;
        pack_config.validate(&packages)?;
        tracing::info!("资源包加载顺序：{}", pack_config.order().join(" → "));
        let assets = Arc::new(std::sync::Mutex::new(AssetManager::new(
            MergedResourceSpace::from_pack_roots(pack_config.pack_roots("game-data")),
        )));
        let gpu = GpuManager::new(
            std::sync::Arc::new(renderer.device()),
            std::sync::Arc::new(renderer.queue().clone()),
        );
        let memory_budget = MemoryBudget::load();
        tracing::info!(
            "内存预算：CPU {:.1} MiB / GPU {:.1} MiB",
            memory_budget.cpu_limit as f64 / (1024.0 * 1024.0),
            memory_budget.gpu_limit as f64 / (1024.0 * 1024.0),
        );

        let mut app = Self {
            window,
            renderer,
            playground: Playground::new(),
            last_frame: Instant::now(),
            assets,
            gpu,
            mouse_captured: false,
            gc_accum: Duration::ZERO,
            memory_budget,
            demo_scenes: [Scene::new(), Scene::new()],
            current_demo: 0,
        };
        app.load_startup_scene();
        Ok(app)
    }

    /// 启动场景：预构建两个演示模板（demo1：全套资产；demo2：精简资产），
    /// 激活 demo1。切换用 F1/F2（见 [`Self::switch_demo`]）。
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
        let demo1 = self.build_demo1(*triangle, *quad, *cube, checker, environment.as_ref());
        // demo2：精简资产（无 glb、无贴图），切换时对比 PinToken 生命周期。
        let mut demo2 = Scene::demo2(*triangle, *cube);
        if let Some(env) = &environment {
            demo2 = demo2
                .with_environment(env.clone())
                .with_environment_intensity(1.0);
        }
        self.demo_scenes = [demo1, demo2];
        self.current_demo = 0;
        tracing::info!("演示场景：F1=demo1（全套资产） F2=demo2（精简资产）");
        self.load_scene(self.demo_scenes[0].clone());
    }

    /// 构建 demo1：内置三角形/四边形/立方体 + 棋盘贴图 + glb 测试模型 +
    /// 50 个实例阵列（实例化合并验证）。资产多，用于对比场景切换的 pin/unpin。
    fn build_demo1(
        &mut self,
        triangle: Handle<Mesh>,
        quad: Handle<Mesh>,
        cube: Handle<Mesh>,
        checker: Handle<Texture>,
        environment: Option<&Arc<Environment>>,
    ) -> Scene {
        let mut scene = Scene::demo(triangle, quad, cube, Some(checker));
        if let Some(env) = environment {
            scene = scene
                .with_environment(env.clone())
                .with_environment_intensity(1.0);
        }
        // 通过游戏路径加载测试模型：走合并资源空间 → 解析 → 资产库。
        let test_path: GamePath = match "test:test.glb".parse() {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("测试模型路径无效：{e}");
                return scene;
            }
        };
        // 场景加载（节点摆放/材质）走 GamePath 一次解析，并进 demo。
        match self.assets.lock().unwrap().load_scene(&test_path) {
            Ok(gltf_scene) => {
                // 把测试模型放大 5 倍（等比），并挪到演示物体右前方，
                // 避免和原点处的三角形重叠。
                for key in scene.merge(&gltf_scene) {
                    if let Some(object) = scene.object_mut(key) {
                        object.transform.scale *= 5.0;
                        object.transform.position += glam::Vec3::new(1.8, 0.0, -1.2);
                    }
                }
                // 实例化测试：取 glb 里第一个网格，铺一大片**同网格同材质**
                // 的副本，验证渲染指令把它们合并成一个绘制组、一次
                // draw_indexed 画完（multi-draw 的前置）。
                // 实例化测试：找 glb 里第一个网格（可能在子节点，用全量遍历），
                // 铺一大片**同网格同材质**的副本，验证渲染指令把它们合并成
                // 一个绘制组、一次 draw_indexed 画完（multi-draw 的前置）。
                if let Some((wrench_mesh, wrench_material)) = gltf_scene
                    .objects()
                    .find_map(|(_, object)| object.mesh_handle().map(|h| (h, object.material)))
                {
                    for row in 0..5 {
                        for col in 0..10 {
                            scene.add_object(
                                SceneObject::new(
                                    SceneObjectKind::Mesh(wrench_mesh),
                                    Transform::new(
                                        Vec3::new(
                                            -4.5 + col as f32 * 1.2,
                                            0.0,
                                            -3.5 - row as f32 * 1.2,
                                        ),
                                        Quat::from_rotation_y(row as f32 * 0.4 + col as f32 * 0.25),
                                        Vec3::splat(5.0),
                                    ),
                                )
                                .with_material(wrench_material),
                            );
                        }
                    }
                } else {
                    tracing::warn!("test.glb 没有网格节点，跳过实例化测试");
                }
            }
            Err(e) => tracing::error!("加载 {} 场景失败：{e}", test_path),
        }
        scene
    }

    /// 切换演示场景：F1 → demo1，F2 → demo2。
    ///
    /// `load_scene` 里新资产 pin 成新令牌；Playground 覆盖旧实例 → 旧
    /// `PinToken` drop → 自动 unpin（debug 日志可见完整生命周期）。
    fn switch_demo(&mut self, index: usize) {
        if index == self.current_demo || index >= self.demo_scenes.len() {
            return;
        }
        tracing::info!(
            "切换演示：demo{}（资产 {} 网格/贴图句柄） → demo{}",
            self.current_demo + 1,
            self.demo_scenes[self.current_demo]
                .objects()
                .filter_map(|(_, o)| o.mesh_handle())
                .count(),
            index + 1,
        );
        self.current_demo = index;
        self.load_scene(self.demo_scenes[index].clone());
    }

    /// 加载环境贴图：`BACKROOMS_ENV`（GamePath 字符串）优先，否则 `test:test.hdr`。
    fn load_example_environment(&mut self) -> Option<Arc<Environment>> {
        let path: GamePath = std::env::var("BACKROOMS_ENV")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| "test:test.hdr".parse().expect("内置环境路径合法"));
        let bytes = match self.assets.lock().unwrap().space().read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("环境贴图读取失败 {path}：{e}");
                return None;
            }
        };
        match Environment::from_bytes(&bytes) {
            Ok(env) => {
                tracing::info!("环境贴图 {path} 加载成功（{}×{}）", env.width, env.height);
                Some(Arc::new(env))
            }
            Err(e) => {
                tracing::error!("环境贴图解码失败 {path}：{e}");
                None
            }
        }
    }

    /// 尝试从 glTF 文件加载场景；成功返回 `true`，失败打印原因并返回 `false`。
    #[allow(dead_code)] // 预留：BACKROOMS_GLTF / game-data 场景加载路径，demo 阶段未启用
    fn try_load_gltf(&mut self, path: &GamePath, environment: Option<&Arc<Environment>>) -> bool {
        let scene = self.assets.lock().unwrap().load_scene(path);
        let mut scene = match scene {
            Ok(scene) => scene,
            Err(e) => {
                tracing::error!("加载 glTF {path} 失败：{e}");
                return false;
            }
        };
        if let Some(env) = environment {
            scene = scene.with_environment(env.clone());
        }
        self.load_scene(scene);
        true
    }

    /// 批量注册网格：追加进全局资产库，返回句柄列表。
    pub fn register_meshes(&mut self, meshes: Vec<Mesh>) -> Vec<Handle<Mesh>> {
        meshes
            .into_iter()
            .map(|mesh| self.assets.lock().unwrap().register(mesh))
            .collect()
    }

    /// 批量注册贴图：追加进全局纹理库，返回句柄列表。
    pub fn register_textures(&mut self, textures: Vec<Texture>) -> Vec<Handle<Texture>> {
        textures
            .into_iter()
            .map(|texture| self.assets.lock().unwrap().register(texture))
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
        // 收集场景引用的资产并 pin 成令牌（CPU 侧标记驻留），随后 GPU 层上传。
        // 令牌随 Playground 的场景实例持有：下次 load_scene 覆盖旧实例时自动
        // unpin，无需手动配对。
        let pin_token = self.pin_scene_assets(&scene);
        self.gpu.sync(&mut self.assets.lock().unwrap());
        // 场景模板 → ECS 实体（旧实例自动卸载：unpin 资产 + 级联 despawn；
        // 碰撞盒在生成时从网格 AABB 派生）。
        self.playground.load_scene(&scene, pin_token, &self.assets);
        if self.playground.main_camera().is_none() {
            tracing::warn!("场景生成后没有主相机（不应发生：ensure_main_camera 已兜底）");
        }
    }

    /// 场景引用的所有网格/贴图句柄标记为驻留（pin），返回驻留令牌。
    fn pin_scene_assets(&mut self, scene: &Scene) -> PinToken {
        // 去重：同一句柄被多个物体引用（如 50 个实例共用同一网格）只 pin 一次，
        // 引用计数不随实例数膨胀。
        let mut keys: std::collections::HashSet<slotmap::DefaultKey> = Default::default();
        for (_, object) in scene.objects() {
            if let Some(handle) = object.mesh_handle() {
                keys.insert(handle.key());
            }
            for handle in object.material.texture_handles() {
                keys.insert(handle.key());
            }
        }
        let keys: Vec<_> = keys.into_iter().collect();
        PinToken::pin(&self.assets, &keys)
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
                        tracing::info!("灯光调试可视化：{}", if on { "开" } else { "关" });
                    }
                    // B：切换碰撞箱调试可视化（长按不重复触发）。
                    if code == KeyCode::KeyB && !key_event.repeat {
                        let on = self.playground.toggle_collision_debug();
                        tracing::info!("碰撞箱调试可视化：{}", if on { "开" } else { "关" });
                    }
                    // F1/F2：切换 demo1 / demo2（长按不重复触发）。
                    if code == KeyCode::F1 && !key_event.repeat {
                        self.switch_demo(0);
                    }
                    if code == KeyCode::F2 && !key_event.repeat {
                        self.switch_demo(1);
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

        let _ticks = self.playground.advance(frame_dt);

        // 资产回收按真实时间间隔触发（统一 GC 策略，两侧同窗口参数）。
        self.gc_accum += frame_dt;
        if self.gc_accum >= GC_INTERVAL {
            self.gc_accum = Duration::ZERO;

            // 内存压力检测：占用超预算时**强制全量清扫**（忽略闲置窗口，
            // 只保留 pinned），否则按各自的闲置窗口常规回收。
            // 两侧预算与窗口互相独立，代价不同分开调参。
            let cpu_usage = self.assets.lock().unwrap().memory_usage();
            let gpu_usage = self.gpu.memory_usage();
            let cpu_pressure = cpu_usage > self.memory_budget.cpu_limit;
            let gpu_pressure = gpu_usage > self.memory_budget.gpu_limit;
            tracing::debug!(
                "资产占用：CPU {:.1} MiB / 预算 {:.1} MiB（{}），GPU {:.1} MiB / 预算 {:.1} MiB（{}）",
                cpu_usage as f64 / (1024.0 * 1024.0),
                self.memory_budget.cpu_limit as f64 / (1024.0 * 1024.0),
                if cpu_pressure { "超限" } else { "正常" },
                gpu_usage as f64 / (1024.0 * 1024.0),
                self.memory_budget.gpu_limit as f64 / (1024.0 * 1024.0),
                if gpu_pressure { "超限" } else { "正常" },
            );

            self.assets.lock().unwrap().gc(&GcPolicy {
                stale_window: CPU_GC_STALE_WINDOW,
                ignore_stale_window: cpu_pressure,
                ..GcPolicy::default()
            });
            self.gpu.gc(&GcPolicy {
                stale_window: GPU_GC_STALE_WINDOW,
                ignore_stale_window: gpu_pressure,
                ..GcPolicy::default()
            });
        }

        self.playground.prepare_frame();
        self.window.request_redraw();
    }

    /// 渲染一帧：把 Playground 里的渲染指令交给渲染器执行（指令只带资源库
    /// 句柄，渲染器绘制时拿句柄向库取 GPU 数据）。
    fn draw(&mut self) {
        let command = self.playground.render_frame();
        self.renderer
            .render(command, &mut self.gpu, &mut self.assets.lock().unwrap());
    }
}
