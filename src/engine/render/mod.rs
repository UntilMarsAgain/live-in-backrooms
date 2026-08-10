//! 渲染模块：wgpu 初始化、surface 管理与每帧绘制。
//!
//! 本文件只做**集合点**：类型定义与再导出，具体实现拆到子模块：
//! - [`init`]：渲染器初始化（设备/交换链/绑定组/管线装配）；
//! - [`scene`]：环境设置（天空盒/IBL）；
//! - [`frame`]：每帧渲染与窗口尺寸变化；
//! - [`uniform`]：GPU uniform 布局与灯光收集；
//! - [`environment`]：环境贴图（天空盒 + IBL）的 GPU 资源与转换；
//! - [`debug`]：灯光调试可视化；
//! - [`blit`]：色调映射 blit（HDR 中间目标 → 交换链）；
//! - [`asset`]：资产管理器（GPU 表示类型、上传器、统一句柄资源管理）；
//! - [`tests`]：WGSL 校验 + 无头冒烟测试（仅测试构建）。
//!
//! 渲染指令（[`crate::engine::core::frame::RenderCommand`]）是 ECS 与渲染器
//! 之间的数据契约，定义在 core：本模块消费它，不反向依赖 ECS。

mod asset;
mod blit;
mod bloom;
mod debug;
mod environment;
mod frame;
mod init;
mod scene;
#[cfg(test)]
mod tests;
pub(crate) mod uniform;

use self::blit::BlitResources;
use self::bloom::BloomResources;
pub use asset::{GpuManager, MeshGpu, TextureGpu};
use wgpu::Color;

use self::debug::LineGizmos;
use self::environment::{EnvironmentGpu, EnvironmentResources};

/// 窗口的显示句柄，用于创建 wgpu 实例。
pub type DisplayHandle = Box<dyn wgpu::wgt::WgpuHasDisplayHandle>;

/// 初始背景色：暗黄绿的"后室"氛围色，后续可改为可配置。
pub const CLEAR_COLOR: Color = Color {
    r: 0.06,
    g: 0.07,
    b: 0.05,
    a: 1.0,
};

/// HDR 中间目标的纹理格式：16 位浮点，可在 WebGPU 全后端作为渲染目标
/// 且默认支持过滤（不像 RGBA32F 需要 FLOAT32_FILTERABLE）。
pub(super) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// 场景 pass 的 MSAA 采样数：渲染到 4x 附件后 resolve 到 HDR 纹理，
/// bloom / blit 消费的是 resolve 后的单采样结果（FXAA 不做，避免糊纹理）。
pub(super) const MSAA_SAMPLES: u32 = 4;

/// wgpu 渲染器：持有 surface / device / queue，负责清屏渲染。
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
    /// 相机 uniform 缓冲区与绑定组。
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    /// 物体数据（全部物体一个只读 storage 数组，按实例索引取）。
    object_bind_group_layout: wgpu::BindGroupLayout,
    object_data_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    /// 灯光：数量 uniform + 只读 storage 数组（每帧写入收集结果）。
    light_count_buffer: wgpu::Buffer,
    light_storage_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    /// 纹理（材质贴图）绑定组相关。
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_sampler: wgpu::Sampler,
    /// 默认 1×1 白纹理视图（基础色/金属度粗糙度贴图兜底）。
    default_white_view: wgpu::TextureView,
    /// 默认 1×1 中性法线纹理视图。
    default_normal_view: wgpu::TextureView,
    /// 网格渲染管线。
    pipeline: wgpu::RenderPipeline,
    /// 深度缓冲（纹理 + 视图），随窗口尺寸重建。
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// HDR 中间目标：场景 pass（网格 + 天空盒 + 调试线框）渲染到这里，
    /// blit pass 采样它做色调映射后写交换链。随窗口尺寸重建。
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    /// 场景 pass 的 MSAA 附件：4x 采样渲染，随后 resolve 到 [`Self::hdr_view`]。
    hdr_msaa_texture: wgpu::Texture,
    hdr_msaa_view: wgpu::TextureView,
    /// 色调映射 blit 资源（管线 + 绑定组布局 + 采样器）。
    blit_resources: BlitResources,
    /// blit 绑定组：引用 HDR 视图与环境参数 uniform，resize 时重建。
    blit_bind_group: wgpu::BindGroup,
    /// Bloom（辉光）后处理：提取高亮 + 多级下采样/上采样合并。
    bloom: BloomResources,
    /// 环境贴图（天空盒 + IBL）。始终存在：未加载时为 1×1 黑环境。
    environment: EnvironmentGpu,
    /// 环境子系统资源（布局、计算管线、天空盒管线等）。
    environment_resources: EnvironmentResources,
    /// 灯光调试可视化（灯泡 + 射线线框；顶点在 load_scene 时上传一次）。
    light_gizmos: LineGizmos,
    /// 碰撞箱调试可视化（世界 AABB 线框；顶点在 load_scene 时上传一次）。
    collision_gizmos: LineGizmos,
}
