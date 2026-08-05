//! App：各功能的集成点。
//!
//! 这里持有窗口与渲染器等子系统，把 winit 事件翻译成上层逻辑。
//! 后续的输入、游戏逻辑、场景管理都可以从这里接入。

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::engine::asset;
use crate::engine::{
    Camera, DisplayHandle, Environment, FreeCameraController, InputController, Mesh, MeshKey,
    MeshLibrary, Renderer, RendererError, Scene, Texture, TextureKey, TextureLibrary,
};

/// 应用的集成层：main.rs 只负责创建窗口，其余都在这里装配。
pub struct App {
    window: Arc<Window>,
    renderer: Renderer,
    controller: Box<dyn InputController<Camera>>,
    last_frame: Instant,
    /// 全局网格资产库（永久驻留，跨场景共享）。
    mesh_library: MeshLibrary,
    /// 全局纹理资产库（永久驻留，跨场景共享）。
    texture_library: TextureLibrary,
    scene: Scene,
    /// 灯光调试可视化开关（控制器按 L 翻转，渲染时读取）。
    show_light_debug: Arc<AtomicBool>,
}

impl App {
    /// 装配各子系统。窗口已由 main.rs 创建好，这里初始化渲染器。
    pub fn new(window: Arc<Window>, display: DisplayHandle) -> Result<Self, RendererError> {
        // 灯光调试开关由控制器独占翻转，App 只读；用 Arc 共享给两边。
        let show_light_debug = Arc::new(AtomicBool::new(false));
        let controller: Box<dyn InputController<Camera>> =
            Box::new(FreeCameraController::new(show_light_debug.clone()));
        let renderer = Renderer::new(&window, display)?;
        let mut app = Self {
            window,
            renderer,
            controller,
            last_frame: Instant::now(),
            mesh_library: MeshLibrary::new(),
            texture_library: TextureLibrary::new(),
            scene: Scene::default(),
            show_light_debug,
        };
        app.load_startup_scene();
        Ok(app)
    }

    /// 启动场景：优先加载 `BACKROOMS_GLTF` 环境变量指定的 glTF 文件，
    /// 其次尝试 `assets/scene.glb`；都不可用时回退到内置演示场景。
    fn load_startup_scene(&mut self) {
        // 环境（天空盒 + IBL）是关卡数据的一部分：解码后绑定到场景上，
        // 由 `load_scene` 统一上传，而不是单独加载。
        let environment = self.load_environment();

        if let Some(path) = std::env::var_os("BACKROOMS_GLTF") {
            if self.try_load_gltf(Path::new(&path), environment.as_ref()) {
                return;
            }
            eprintln!("回退到演示场景");
        } else {
            let default = Path::new("assets/scene.glb");
            if default.is_file() && self.try_load_gltf(default, environment.as_ref()) {
                return;
            }
        }
        // 回退：内置演示场景；仓库内的 src/asset/test2.glb（全套 PBR 样例）一并并入。
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
            scene = scene.with_environment(env.clone()).with_environment_intensity(1.0);
        }
        let test_glb = Path::new("src/engine/asset/test.glb");
        if test_glb.is_file() {
            match asset::load_scene(
                test_glb,
                &mut self.mesh_library,
                &mut self.texture_library,
            ) {
                Ok(gltf_scene) => {
                    // 把测试模型放大 5 倍（等比），并挪到演示物体右前方，
                    // 避免和原点处的三角形重叠。
                    for key in scene.merge(&gltf_scene) {
                        if let Some(object) = scene.object_mut(key) {
                            object.transform.scale *=5.0;
                            object.transform.position += glam::Vec3::new(1.8, 0.0, -1.2);
                        }
                    }
                    self.renderer.upload_meshes(&self.mesh_library);
                    self.renderer.upload_textures(&self.texture_library);
                }
                Err(e) => eprintln!("加载 {} 失败：{e}", test_glb.display()),
            }
        }
        self.load_scene(scene);
    }

    /// 加载环境贴图：`BACKROOMS_ENV` 环境变量优先，否则尝试 `assets/environments/test.hdr`。
    /// 按文件内容自动识别 HDR / LDR（PNG/JPEG 等）。
    fn load_environment(&mut self) -> Option<Arc<Environment>> {
        let path = std::env::var_os("BACKROOMS_ENV")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("assets/environments/test.hdr"));
        match Environment::from_file(&path) {
            Ok(env) => {
                eprintln!(
                    "环境贴图 {} 加载成功（{}×{}）",
                    path.display(),
                    env.width,
                    env.height
                );
                Some(Arc::new(env))
            }
            Err(e) => {
                eprintln!("环境贴图加载失败 {}：{e}", path.display());
                None
            }
        }
    }

    /// 尝试从 glTF 文件加载场景；成功返回 `true`，失败打印原因并返回 `false`。
    fn try_load_gltf(&mut self, path: &Path, environment: Option<&Arc<Environment>>) -> bool {
        match asset::load_scene(path, &mut self.mesh_library, &mut self.texture_library) {
            Ok(scene) => {
                self.renderer.upload_meshes(&self.mesh_library);
                self.renderer.upload_textures(&self.texture_library);
                let scene = match environment {
                    Some(env) => scene.with_environment(env.clone()),
                    None => scene,
                };
                self.load_scene(scene);
                true
            }
            Err(e) => {
                eprintln!("加载 glTF {} 失败：{e}", path.display());
                false
            }
        }
    }

    /// 批量注册网格：追加进全局资产库并整体重传 GPU 合并缓冲，返回句柄列表。
    pub fn register_meshes(&mut self, meshes: Vec<Mesh>) -> Vec<MeshKey> {
        let keys = self.mesh_library.register_many(meshes);
        self.renderer.upload_meshes(&self.mesh_library);
        keys
    }

    /// 批量注册贴图：追加进全局纹理库并增量上传 GPU 纹理，返回句柄列表。
    pub fn register_textures(&mut self, textures: Vec<Texture>) -> Vec<TextureKey> {
        let keys = self.texture_library.register_many(textures);
        self.renderer.upload_textures(&self.texture_library);
        keys
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
        self.renderer.load_scene(&scene);
        self.scene = scene;
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
                        self.show_light_debug.load(Ordering::Relaxed),
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

        if let Some(camera) = self.scene.main_camera_mut() {
            self.controller.update(camera, dt);
        }
        // 请求下一帧，驱动持续渲染。
        self.window.request_redraw();
    }
}
