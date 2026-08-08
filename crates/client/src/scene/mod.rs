//! 客户端视图场景：共享场景树 + 渲染所需信息。
//!
//! 服务端只持有 [`lbr_shared::scene::Scene`]（纯场景树：节点、局部变换、
//! 网格/灯光引用）；客户端在这棵树之上叠加渲染状态：
//! - 材质：按节点句柄映射（仅网格节点有值）；
//! - 主相机：视图位置/朝向（场景切换即切换出生点视角）；
//! - 环境（天空盒 + IBL）与环境强度；
//! - AgX 色调映射的 EV 窗口（层级风格配置）。
//!
//! C/S 同步时传输的是实体 ID（合并资源空间里的描述文件），客户端在本地
//! 解析并重建这棵视图树，因此树本身可以完全复用共享场景，这里只负责
//! 渲染侧数据。

use std::collections::HashMap;
use std::sync::Arc;

use glam::Vec3;

use crate::render::uniform::{AGX_DEFAULT_EV_MAX, AGX_DEFAULT_EV_MIN};
use lbr_shared::core::camera::{Camera, CameraAction};
use lbr_shared::core::environment::Environment;
use lbr_shared::core::material::Material;
use lbr_shared::scene::{ObjectKey, Scene};

/// 客户端视图场景：树（共享）+ 渲染信息（客户端）。
#[derive(Debug, Clone)]
pub struct ClientScene {
    /// 共享场景树（与服务端同构，服务端视图的纯场景树）。
    scene: Scene,
    /// 节点 → 材质（仅网格节点；缺失时渲染按默认材质处理）。
    materials: HashMap<ObjectKey, Material>,
    /// 主相机（出生点视角）。
    camera: Camera,
    /// 关卡环境（天空盒 + IBL）。`None` = 纯手动布光 / 保持默认黑环境。
    environment: Option<Arc<Environment>>,
    /// 环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    environment_intensity: f32,
    /// AgX 色调映射的 EV 窗口（相对中间灰 0.18 的 EV 档位），默认与 Blender 一致。
    agx_min_ev: f32,
    agx_max_ev: f32,
}

impl Default for ClientScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ClientScene {
    pub fn new() -> Self {
        Self {
            scene: Scene::new(),
            materials: HashMap::new(),
            camera: default_camera(),
            environment: None,
            // 默认满环境光：不显式设置强度时保持"环境图参与光照"的既有行为。
            environment_intensity: 1.0,
            // 默认 EV 窗口 = Blender AgX（中间灰上下 -10 ~ +6.5 EV）。
            agx_min_ev: AGX_DEFAULT_EV_MIN,
            agx_max_ev: AGX_DEFAULT_EV_MAX,
        }
    }

    /// 由一棵共享场景树构建视图场景（材质/相机/环境取默认值）。
    pub fn from_scene(scene: Scene) -> Self {
        let mut out = Self::new();
        out.scene = scene;
        out
    }

    /// 共享场景树（只读）。
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// 共享场景树（可变）。
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// 设置节点材质（仅网格节点有意义；覆盖已有值）。
    pub fn set_material(&mut self, key: ObjectKey, material: Material) {
        self.materials.insert(key, material);
    }

    /// 节点材质；未设置或句柄失效返回 `None`。
    pub fn material(&self, key: ObjectKey) -> Option<&Material> {
        self.materials.get(&key)
    }

    /// 主相机（只读）。
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// 主相机（可变；窗口尺寸变化、输入控制等）。
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// 替换主相机（场景切换/模组摆放出生点视角）。
    #[allow(dead_code)] // 公共配置 API：模组/关卡数据摆放出生点视角时使用
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    /// 应用相机操作（先旋转再平移）。相机由视图场景持有与控制，
    /// 输入控制器只产出 [`CameraAction`]，由这里统一应用。
    pub fn apply_camera_action(&mut self, action: CameraAction) {
        self.camera.rotate(action.yaw_delta, action.pitch_delta);
        self.camera.translate(action.translate);
    }

    /// 给场景绑定环境（天空盒 + IBL）；`load_scene` 时自动上传。
    ///
    /// `Arc` 避免环境像素（MB 级）在场景间复制；多个关卡可共享同一环境。
    pub fn with_environment(mut self, environment: Arc<Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// 场景绑定的环境（无则 `None`）。
    pub fn environment(&self) -> Option<&Environment> {
        self.environment.as_deref()
    }

    /// 设置环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    #[allow(dead_code)] // 公共配置 API：关卡数据构建场景时使用
    pub fn with_environment_intensity(mut self, intensity: f32) -> Self {
        self.environment_intensity = intensity;
        self
    }

    /// 场景的环境强度（默认 1.0）。
    pub fn environment_intensity(&self) -> f32 {
        self.environment_intensity
    }

    /// 覆盖 AgX 色调映射的 EV 窗口（场景级风格配置）。
    ///
    /// 参数是**相对中间灰 0.18 的 EV 档位**：默认 -10 ~ +6.5 EV（Blender 一致），
    /// 一般无需修改；想让某个层级更亮/更暗或动态范围更宽/更窄时再调。
    /// 窗口越宽曲线越平缓（整体偏灰），越窄对比越强；要求 `min_ev < max_ev`。
    #[allow(dead_code)] // 公共配置 API：关卡数据构建场景时使用
    pub fn with_environment_agx_ev(mut self, min_ev: f32, max_ev: f32) -> Self {
        debug_assert!(min_ev < max_ev, "AgX EV 窗口要求 min_ev < max_ev");
        self.agx_min_ev = min_ev;
        self.agx_max_ev = max_ev;
        self
    }

    /// 场景的 AgX EV 窗口下界（默认 -10，相对中间灰的 EV）。
    pub fn agx_min_ev(&self) -> f32 {
        self.agx_min_ev
    }

    /// 场景的 AgX EV 窗口上界（默认 +6.5，相对中间灰的 EV）。
    pub fn agx_max_ev(&self) -> f32 {
        self.agx_max_ev
    }

    /// 把 `other` 的视图场景并入本场景（树 + 材质一起并入），返回新复制的
    /// 根节点句柄。用于把 glTF 等外部场景并入现有视图场景。
    pub fn merge(&mut self, other: &ClientScene) -> Vec<ObjectKey> {
        // 树合并返回完整重映射：材质按旧句柄 → 新句柄复制。
        let remap = self.scene.merge(&other.scene);
        for (old_key, new_key) in &remap {
            if let Some(material) = other.materials.get(old_key) {
                self.materials.insert(*new_key, material.clone());
            }
        }
        other.scene.roots().map(|(key, _)| remap[&key]).collect()
    }

    /// 清空实体（树 + 材质表），保留相机/环境等渲染信息。
    ///
    /// 供同步层在"服务端实体结构变化"时重建实体用；相机/环境不受影响。
    pub fn clear_entities(&mut self) {
        let roots: Vec<ObjectKey> = self.scene.roots().map(|(key, _)| key).collect();
        for root in roots {
            self.scene.remove_object(root);
        }
        self.materials.clear();
    }
}

/// 默认主相机：与 App 缺省相机一致的位置/朝向，宽高比按默认窗口 16:9。
fn default_camera() -> Camera {
    Camera::new(
        Vec3::new(0.0, 1.0, 3.0),
        -std::f32::consts::FRAC_PI_2,
        -0.15,
        std::f32::consts::FRAC_PI_4,
        16.0 / 9.0,
        0.1,
        100.0,
    )
}

mod demo;
#[cfg(test)]
mod tests;
