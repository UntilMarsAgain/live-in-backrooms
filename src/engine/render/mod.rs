//! 渲染模块：wgpu 初始化、surface 管理与每帧绘制。
//!
//! 本文件只做**集合点**：类型定义与再导出，具体实现拆到子模块：
//! - [`init`]：渲染器初始化（设备/交换链/绑定组/管线装配）；
//! - [`scene`]：场景数据上传（网格/纹理/环境/灯光调试）；
//! - [`frame`]：每帧渲染与窗口尺寸变化；
//! - [`uniform`]：GPU uniform 布局与灯光收集；
//! - [`environment`]：环境贴图（天空盒 + IBL）的 GPU 资源与转换；
//! - [`debug`]：灯光调试可视化；
//! - [`tests`]：WGSL 校验 + 无头冒烟测试（仅测试构建）。

mod debug;
mod environment;
mod frame;
mod init;
mod scene;
#[cfg(test)]
mod tests;
pub(crate) mod uniform;

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
    /// 物体数据（动态 uniform，模型矩阵）相关。
    object_bind_group_layout: wgpu::BindGroupLayout,
    object_data_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,
    /// 物体数据步长：每个物体的矩阵在缓冲中的间隔（满足设备对齐要求）。
    object_stride: u32,
    /// 灯光：数量 uniform + 只读 storage 数组（每帧写入收集结果）。
    light_count_buffer: wgpu::Buffer,
    light_storage_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    /// 纹理（材质贴图）绑定组相关。
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_sampler: wgpu::Sampler,
    /// 按 [`TextureKey`](super::core::texture::TextureKey) 稠密编号索引的纹理视图。
    texture_views: Vec<wgpu::TextureView>,
    /// 默认 1×1 白纹理视图（基础色/金属度粗糙度贴图兜底）。
    default_white_view: wgpu::TextureView,
    /// 默认 1×1 中性法线纹理视图。
    default_normal_view: wgpu::TextureView,
    /// 全默认材质的绑定组（非网格节点占位用）。
    default_material_bind_group: wgpu::BindGroup,
    /// 每个物体的材质绑定组（load_scene 时按 objects() 顺序构建）。
    material_bind_groups: Vec<wgpu::BindGroup>,
    /// 已上传的纹理库版本。
    uploaded_texture_version: u64,
    /// 网格渲染管线。
    pipeline: wgpu::RenderPipeline,
    /// 深度缓冲（纹理 + 视图），随窗口尺寸重建。
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// 全局网格缓冲（永久驻留，所有注册网格合并），由 upload_meshes 维护。
    mesh_buffer: Option<MeshGpu>,
    /// 已上传的网格库版本，避免重复上传。
    mesh_uploaded_version: u64,
    /// 环境贴图（天空盒 + IBL）。始终存在：未加载时为 1×1 黑环境。
    environment: EnvironmentGpu,
    /// 环境子系统资源（布局、计算管线、天空盒管线等）。
    environment_resources: EnvironmentResources,
    /// 灯光调试可视化（灯泡 + 射线线框；顶点在 load_scene 时上传一次）。
    light_gizmos: LineGizmos,
    /// 碰撞箱调试可视化（世界 AABB 线框；顶点在 load_scene 时上传一次）。
    collision_gizmos: LineGizmos,
}

/// 资产库中单个网格在合并缓冲里的区间。
#[derive(Debug, Clone, Copy)]
struct MeshRange {
    index_offset: u32,
    index_count: u32,
}

/// 全局网格的 GPU 表示：合并后的顶点/索引缓冲，以及每个网格的区间。
/// 区间按 [`MeshKey`](super::core::mesh::MeshKey) 的稠密编号索引（句柄即下标）。
struct MeshGpu {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    mesh_ranges: Vec<MeshRange>,
}
