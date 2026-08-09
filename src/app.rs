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

use crate::engine::{
    AssetManager, Camera, CameraAction, DisplayHandle, Environment, FreeCameraController, GamePath,
    GpuManager, Handle, InputController, MergedResourceSpace, Mesh, MeshView, PackConfig, Renderer,
    Scene, Texture,
};

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    controller: Box<dyn InputController<Camera, Action = CameraAction>>,
    last_frame: Instant,
    /// CPU 资产管理器：内存副本、磁盘读取与内存卸载（无 GPU 依赖）。
    assets: AssetManager,
    /// GPU 资产管理器：句柄 → 显存表示的驻留表（上传/回收）。
    gpu: GpuManager,
    scene: Scene,
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
        // 启动主流程：扫描有效包 → 生成/更新 packs.toml 顺序（环 → 报错），
        // 再按 order 校验依赖与冲突（不满足 → 报错退出）。
        let (pack_config, packages) = PackConfig::discover_and_update("game-data")?;
        pack_config.validate(&packages)?;
        eprintln!("资源包加载顺序：{}", pack_config.order().join(" → "));
        let assets =
            AssetManager::new(MergedResourceSpace::from_pack_roots(pack_config.pack_roots("game-data")));
        let gpu = GpuManager::new(
            std::sync::Arc::new(renderer.device()),
            std::sync::Arc::new(renderer.queue().clone()),
        );
        let mut app = Self {
            window,
            renderer,
            controller,
            last_frame: Instant::now(),
            assets,
            gpu,
            scene: Scene::default(),
            show_light_debug,
            show_collision_debug,
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
        // "路径 → 句柄"的 File 条目链路（load_file/load_file_async）由测试覆盖。
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

    /// 尝试从 glTF 文件加载场景；成功返回 `true`，失败打印原因并返回 `false`。
    #[allow(dead_code)] // 预留：BACKROOMS_GLTF / game-data 场景加载路径，demo 阶段未启用
    fn try_load_gltf(
        &mut self,
        path: &GamePath,
        environment: Option<&Arc<Environment>>,
    ) -> bool {
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

    /// 批量注册网格：追加进全局资产库并整体重传 GPU 合并缓冲，返回句柄列表。
    pub fn register_meshes(&mut self, meshes: Vec<Mesh>) -> Vec<Handle<Mesh>> {
        meshes
            .into_iter()
            .map(|mesh| self.assets.register(mesh))
            .collect()
    }

    /// 批量注册贴图：追加进全局纹理库并增量上传 GPU 纹理，返回句柄列表。
    pub fn register_textures(&mut self, textures: Vec<Texture>) -> Vec<Handle<Texture>> {
        textures
            .into_iter()
            .map(|texture| self.assets.register(texture))
            .collect()
    }

    /// App 级别的场景切换 API：渲染器（GPU 数据）与后续游戏逻辑统一从这里换场景。
    pub fn load_scene(&mut self, mut scene: Scene) {
        // 每个场景必须有一个主相机（出生点视角）；外部内容（glTF）通常不带相机，
        // 缺省时补一个默认相机。模组作者也可以在场景里显式摆放相机并切换。
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
        // 收集场景引用的资产并 pin（CPU 侧标记驻留），随后 GPU 层上传，
        // 再让渲染器构建绑定组（贴图视图需要已上传）。
        self.pin_scene_assets(&scene);
        self.gpu.sync(&mut self.assets);
        self.renderer.load_scene(&scene, &mut self.gpu, &mut self.assets);
        self.scene = scene;
    }

    /// 场景引用的所有网格/贴图句柄标记为 GPU 驻留（pin）。
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
    ///
    /// 只有"未指定"（`None`）才会兜底；若场景声明了主相机但指向已删除的节点或
    /// 非相机节点，属于运行违例，[`Scene::main_camera`] 会直接 panic，加载被拒绝。
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

    /// main.rs 转发的窗口事件统一在这里处理。
    pub fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                if let Some(camera) = self.scene.main_camera_mut() {
                    camera.set_aspect(size.width as f32 / size.height.max(1) as f32);
                }
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // 主相机来自场景树：场景自带出生点视角，切换场景即切换相机。
                if let Some(camera) = self.scene.main_camera_ref() {
                    self.renderer.render(
                        camera,
                        &self.scene,
                        &mut self.gpu,
                        &mut self.assets,
                        self.show_light_debug.load(Ordering::Relaxed),
                        self.show_collision_debug.load(Ordering::Relaxed),
                    );
                }
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

        // GPU 驻留调和：pinned 且未上传的上传，非 pinned 的回收。
        self.gpu.sync(&mut self.assets);

        // 控制器只读相机 + 查询场景碰撞，输出一帧操作；操作由场景统一应用
        // （不可变借用先结束，再可变借用应用，无冲突）。
        let mesh_view = MeshView::new(&self.assets);
        let action = match self.scene.main_camera_ref() {
            Some(camera) => self.controller.update(camera, dt, &self.scene, &mesh_view),
            None => CameraAction::default(),
        };
        self.scene.apply_main_camera_action(action);
        // 请求下一帧，驱动持续渲染。
        self.window.request_redraw();
    }
}
