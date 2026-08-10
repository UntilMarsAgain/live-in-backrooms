//! 环境子系统：环境贴图（天空盒 + IBL）的 GPU 资源与转换。
//!
//! 按启动时决定的后端路径转换 HDRI：Vulkan/Metal 用 GPU 计算着色器，
//! GL 等后端回退 CPU 转换 + 逐层上传（见 docs/BUG.md）。
//!
//! 子模块分工：
//! - [`textures`]：纹理创建与上传、默认占位绑定组；
//! - [`pipelines`]：绑定组布局与计算/天空盒管线的装配；
//! - [`convert`]：HDRI → 环境图/辐照度/预过滤/BRDF LUT 的转换编排。

mod convert;
mod pipelines;
mod textures;

/// 环境立方体贴图每面尺寸。
pub(super) const ENV_CUBEMAP_SIZE: u32 = 256;
/// 辐照度图（漫反射 IBL）每面尺寸。
pub(super) const IRRADIANCE_SIZE: u32 = 32;
/// 辐照度图余弦加权采样数：启动时一次性计算，取大一些换取平滑。
pub(super) const IRRADIANCE_SAMPLES: u32 = 1024;
/// 镜面预过滤图（Specular IBL）每面尺寸。
pub(super) const PREFILTERED_SIZE: u32 = 128;
/// 预过滤图 mip 层数：128 → 1（8 层，roughness 均匀映射到各层）。
pub(super) const PREFILTER_MIP_COUNT: u32 = 8;
/// 预过滤每纹素 GGX 采样数。
pub(super) const PREFILTER_SAMPLES: u32 = 1024;
/// BRDF 积分查找表尺寸（x = NdotV，y = roughness）。
pub(super) const BRDF_LUT_SIZE: u32 = 128;

/// 环境贴图的 GPU 表示：环境立方体贴图 + 辐照度图 + 镜面预过滤图 + BRDF LUT。
///
/// 纹理由视图持有引用，`set_environment` 重建时旧资源自动随引用释放。
#[derive(Clone)]
pub(super) struct EnvironmentGpu {
    /// 环境立方体贴图（天空盒采样；镜面预过滤的输入）。
    #[allow(dead_code)] // 资源所有权显式化；readback 诊断与镜面 IBL（Phase 2）会使用
    pub(super) environment_texture: wgpu::Texture,
    /// 环境立方体贴图视图（天空盒采样）。
    #[allow(dead_code)]
    pub(super) environment_view: wgpu::TextureView,
    /// 辐照度图纹理（漫反射 IBL）。
    #[allow(dead_code)]
    pub(super) irradiance_texture: wgpu::Texture,
    /// 辐照度图视图（漫反射 IBL）。
    #[allow(dead_code)]
    pub(super) irradiance_view: wgpu::TextureView,
    /// 镜面预过滤图纹理（mip 链，按粗糙度采样）。
    #[allow(dead_code)]
    pub(super) prefiltered_texture: wgpu::Texture,
    /// 镜面预过滤图视图（mesh 着色器反射采样）。
    #[allow(dead_code)]
    pub(super) prefiltered_view: wgpu::TextureView,
    /// BRDF 积分查找表纹理。
    #[allow(dead_code)]
    pub(super) brdf_lut_texture: wgpu::Texture,
    /// BRDF 查找表视图。
    #[allow(dead_code)]
    pub(super) brdf_lut_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) sampler: wgpu::Sampler,
    /// mesh 管线 @group(4) 绑定组。
    pub(super) mesh_bind_group: wgpu::BindGroup,
    /// 天空盒管线绑定组。
    pub(super) skybox_bind_group: wgpu::BindGroup,
}

/// 环境子系统的 GPU 资源：绑定组布局、计算管线、天空盒管线、默认绑定组。
///
/// 从 `Renderer::new` 中独立出来，便于无窗口的 headless 测试直接复用同一套
/// 资源创建与转换逻辑（见 `tests::environment_headless_smoke`）。
pub(super) struct EnvironmentResources {
    /// mesh 管线 @group(4) 环境绑定组布局。
    pub(super) environment_bind_group_layout: wgpu::BindGroupLayout,
    /// 天空盒管线绑定组布局。
    pub(super) skybox_bind_group_layout: wgpu::BindGroupLayout,
    /// 环境采样器（ClampToEdge；过滤能力取决于设备）。
    pub(super) env_sampler: wgpu::Sampler,
    /// 环境强度 uniform（IBL 系数，mesh 管线 @group(4) binding 3）。
    pub(super) env_params_buffer: wgpu::Buffer,
    /// equirect → cubemap 计算管线的绑定组布局。
    pub(super) env_convert_layout: wgpu::BindGroupLayout,
    /// cubemap → 辐照度图计算管线的绑定组布局。
    pub(super) irradiance_layout: wgpu::BindGroupLayout,
    /// equirect → cubemap 参数 uniform 缓冲。
    ///
    /// 两个计算 pass 必须用**独立**参数缓冲：`queue.write_buffer` 是即时入队
    /// 操作，会先于 `submit()` 里的 compute pass 执行；若分时复用同一个缓冲，
    /// 第二个 write 会先覆盖，导致第一个 pass 读到错误参数。
    pub(super) env_convert_params: wgpu::Buffer,
    /// cubemap → 辐照度图参数 uniform 缓冲。
    pub(super) irradiance_params: wgpu::Buffer,
    /// equirect → cubemap 计算管线。
    pub(super) env_convert_pipeline: wgpu::ComputePipeline,
    /// cubemap → 辐照度图计算管线。
    pub(super) irradiance_pipeline: wgpu::ComputePipeline,
    /// 预过滤（镜面 IBL）计算管线的绑定组布局。
    pub(super) prefilter_layout: wgpu::BindGroupLayout,
    /// BRDF LUT 计算管线的绑定组布局。
    pub(super) brdf_lut_layout: wgpu::BindGroupLayout,
    /// 预过滤参数 uniform 缓冲（每个 mip 一个）。
    ///
    /// 复用单个缓冲会踩 docs/BUG.md 记录过的坑：`queue.write_buffer` 先于
    /// `submit()` 里的 pass 执行，循环里多次写同一缓冲会让所有 pass 读到
    /// 最后一次的参数（mip 0..6 全部提前 return，预过滤图基本全黑）。
    pub(super) prefilter_params: Vec<wgpu::Buffer>,
    /// 预过滤计算管线（GGX 重要性采样 mip 链）。
    pub(super) prefilter_pipeline: wgpu::ComputePipeline,
    /// BRDF LUT 计算管线（split-sum 第二项）。
    pub(super) brdf_lut_pipeline: wgpu::ComputePipeline,
    /// 天空盒渲染管线。
    pub(super) skybox_pipeline: wgpu::RenderPipeline,
    /// 无环境时的默认绑定组（1×1 黑环境）。
    pub(super) default_environment: EnvironmentGpu,
}

// 再导出：render/mod.rs 与测试仍按原路径使用，拆分不改变对外 API。
#[allow(unused_imports)] // 仅测试使用；保留导出路径供无头测试引用
pub(super) use textures::{create_cube_texture, create_default_environment};
