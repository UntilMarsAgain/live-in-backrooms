//! 场景级渲染设置：环境（天空盒 + IBL）与 AgX 色调映射窗口。

use crate::engine::core::environment::Environment;
use crate::engine::render::uniform::{
    AGX_DEFAULT_EV_MAX, AGX_DEFAULT_EV_MIN, AGX_MIDDLE_GRAY_LOG2,
};
use crate::engine::render::Renderer;

impl Renderer {
    /// 上传环境贴图（HDRI 等距矩形图）并转换成环境立方体贴图 + 辐照度图。
    ///
    /// 转换由两个计算着色器在启动时一次性完成，之后每帧只采样；
    /// 关卡切换换环境时重建纹理与绑定组，旧资源随替换自动释放。
    pub fn set_environment(&mut self, environment: &Environment) {
        self.environment =
            self.environment_resources
                .convert(&self.device, &self.queue, environment);
    }

    /// 设置环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    /// 只写 uniform，不重建环境资源。
    pub fn set_environment_intensity(&self, intensity: f32) {
        self.environment_resources
            .set_intensity(&self.queue, intensity);
    }

    /// 覆盖 AgX 色调映射的 EV 窗口（场景级风格配置，默认与 Blender 一致）。
    ///
    /// 参数是**相对中间灰 0.18 的 EV 档位**（如 -10 ~ +6.5），内部换算成
    /// shader 需要的绝对 log2 锚点；只写 uniform，不重建任何资源。
    pub fn set_environment_agx_ev(&self, ev_min: f32, ev_max: f32) {
        self.environment_resources.set_agx_range(
            &self.queue,
            ev_min + AGX_MIDDLE_GRAY_LOG2,
            ev_max + AGX_MIDDLE_GRAY_LOG2,
        );
    }

    /// 清除环境：切回默认的 1×1 黑环境（无天空盒、无 IBL），并把环境强度与
    /// AgX 窗口恢复默认。用于加载不带环境的场景时，避免残留上一关卡的天空盒。
    pub fn reset_environment(&mut self) {
        self.environment = self.environment_resources.default_environment.clone();
        self.set_environment_intensity(1.0);
        self.set_environment_agx_ev(AGX_DEFAULT_EV_MIN, AGX_DEFAULT_EV_MAX);
    }
}
