//! 渲染模块：负责 wgpu 初始化、surface 管理和每帧绘制。

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, ComputePipelineDescriptor, CurrentSurfaceTexture,
    DeviceDescriptor, FragmentState, InstanceDescriptor, LoadOp, Operations,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, TextureViewDescriptor, VertexState,
};
use winit::window::Window;

use super::core::camera::{Camera, CameraUniform};
use super::core::environment::Environment;
use super::core::light::LightKind;
use super::core::mesh::{MeshLibrary, Vertex};
use super::scene::{Scene, SceneObjectKind};
use super::core::texture::{Texture, TextureLibrary};

/// 窗口的显示句柄，用于创建 wgpu 实例。
pub type DisplayHandle = Box<dyn wgpu::wgt::WgpuHasDisplayHandle>;

/// 从场景收集方向光（世界方向由物体的世界旋转决定）。
///
/// 在 `load_scene` 时调用一次：静态光源不需要每帧重新推导。
/// 将来出现会动的光源（移动/闪烁/玩家手电）时，再从这里加刷新入口。
fn collect_lights(scene: &Scene) -> LightsUniform {
    let mut light_uniform = LightsUniform {
        count: 0,
        _pad: [0; 3],
        lights: [LightUniform {
            kind: 0,
            _pad: [0; 3],
            direction: [0.0; 3],
            _pad_direction: 0.0,
            position: [0.0; 3],
            _pad_position: 0.0,
            color: [0.0; 3],
            intensity: 0.0,
            size: [0.0; 2],
            _pad_size: [0.0; 2],
        }; MAX_LIGHTS],
    };
    for (key, object) in scene.objects() {
        if light_uniform.count as usize >= MAX_LIGHTS {
            break;
        }
        let SceneObjectKind::Light(light) = object.kind else {
            continue;
        };
        let world = scene
            .world_transform(key)
            .expect("objects() 只产出存活节点");
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        let entry = &mut light_uniform.lights[light_uniform.count as usize];
        match light.kind {
            LightKind::Directional => {
                entry.kind = 0;
                entry.direction = (rotation * Vec3::NEG_Z).to_array();
            }
            LightKind::Point => {
                entry.kind = 1;
                entry.position = translation.to_array();
            }
            LightKind::Area { width, height } => {
                entry.kind = 2;
                entry.direction = (rotation * Vec3::NEG_Z).to_array();
                entry.position = translation.to_array();
                entry.size = [width, height];
            }
        }
        entry.intensity = light.intensity;
        entry.color = light.color.to_array();
        light_uniform.count += 1;
    }
    light_uniform
}

/// 初始背景色：暗黄绿的"后室"氛围色，后续可改为可配置。
pub const CLEAR_COLOR: Color = Color {
    r: 0.06,
    g: 0.07,
    b: 0.05,
    a: 1.0,
};

/// 最多同时支持的方向光数量（与 WGSL 中 `MAX_LIGHTS` 一致）。
const MAX_LIGHTS: usize = 8;

/// 环境立方体贴图每面尺寸。
const ENV_CUBEMAP_SIZE: u32 = 256;
/// 辐照度图（漫反射 IBL）每面尺寸。
const IRRADIANCE_SIZE: u32 = 32;
/// 辐照度图余弦加权采样数：启动时一次性计算，取大一些换取平滑。
const IRRADIANCE_SAMPLES: u32 = 1024;

/// 每物体 uniform：模型矩阵 + 法线矩阵（逆转置，正确处理非等比缩放）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectData {
    model: Mat4,
    /// 法线矩阵（WGSL `mat3x3<f32>` 布局：每列 16 字节，含填充）。
    normal_matrix: [[f32; 4]; 3],
    /// 材质基础色因子（RGBA）。
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    _pad: [f32; 2],
}

/// 单个光源在 uniform 缓冲里的布局（80 字节，std140 兼容）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    /// 0=方向光 1=点光 2=面光。
    kind: u32,
    _pad: [u32; 3],
    /// 方向光/面光：世界空间光照方向（局部 -Z 经旋转）。
    direction: [f32; 3],
    _pad_direction: f32,
    /// 点光/面光：世界位置。
    position: [f32; 3],
    _pad_position: f32,
    color: [f32; 3],
    intensity: f32,
    /// 面光：面板尺寸（当前近似未直接使用，为 LTC 预留）。
    size: [f32; 2],
    _pad_size: [f32; 2],
}

/// 灯光 uniform：数量 + 固定大小数组。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightsUniform {
    count: u32,
    _pad: [u32; 3],
    lights: [LightUniform; MAX_LIGHTS],
}

const _: () = assert!(std::mem::size_of::<LightsUniform>() == 16 + 80 * MAX_LIGHTS);

/// 环境计算着色器参数 uniform：`size` = 每面尺寸，`sample_count` = 辐照度采样数。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EnvParams {
    size: u32,
    sample_count: u32,
    _pad: [u32; 2],
}

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
    /// 灯光 uniform（方向光数组，场景级数据）。
    light_buffer: wgpu::Buffer,
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

/// 环境贴图的 GPU 表示：环境立方体贴图 + 辐照度图 + 绑定组。
///
/// 纹理由视图持有引用，`set_environment` 重建时旧资源自动随引用释放。
#[derive(Clone)]
struct EnvironmentGpu {
    /// 环境立方体贴图（天空盒采样；未来的镜面预过滤也以此为输入）。
    #[allow(dead_code)] // 资源所有权显式化；readback 诊断与镜面 IBL（Phase 2）会使用
    environment_texture: wgpu::Texture,
    /// 环境立方体贴图视图（天空盒与未来的镜面反射采样）。
    #[allow(dead_code)]
    environment_view: wgpu::TextureView,
    /// 辐照度图纹理（漫反射 IBL）。
    #[allow(dead_code)]
    irradiance_texture: wgpu::Texture,
    /// 辐照度图视图（漫反射 IBL）。
    #[allow(dead_code)]
    irradiance_view: wgpu::TextureView,
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    /// mesh 管线 @group(4) 绑定组。
    mesh_bind_group: wgpu::BindGroup,
    /// 天空盒管线绑定组。
    skybox_bind_group: wgpu::BindGroup,
}

/// 创建与窗口尺寸一致的深度纹理。
fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// 把 CPU 侧 RGBA8 纹理上传为 GPU 纹理并返回视图。
///
/// `write_texture` 没有行字节 256 对齐的要求（那是 `copy_buffer_to_texture` 的限制），
/// 因此可以直接整块上传。
fn create_texture_view(device: &wgpu::Device, queue: &wgpu::Queue, texture: &Texture) -> wgpu::TextureView {
    let gpu_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("texture"),
        size: wgpu::Extent3d {
            width: texture.width,
            height: texture.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texture.rgba8,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width * 4),
            rows_per_image: Some(texture.height),
        },
        wgpu::Extent3d {
            width: texture.width,
            height: texture.height,
            depth_or_array_layers: 1,
        },
    );

    gpu_texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// 无环境贴图时的默认绑定组：1×1×6 黑色立方体贴图。
///
/// 保证 mesh 管线 @group(4) 与天空盒管线始终有可绑定的资源。
fn create_default_environment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    environment_layout: &wgpu::BindGroupLayout,
    skybox_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> EnvironmentGpu {
    let black = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("default black environment"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &[0u8; 6 * 16],
    );
    let view = black.create_view(&TextureViewDescriptor {
        label: Some("default black environment view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });
    let mesh_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("default environment mesh bind group"),
        layout: environment_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    let skybox_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("default skybox bind group"),
        layout: skybox_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    EnvironmentGpu {
        environment_texture: black.clone(),
        environment_view: view.clone(),
        irradiance_texture: black,
        irradiance_view: view,
        sampler: sampler.clone(),
        mesh_bind_group,
        skybox_bind_group,
    }
}

/// 创建 6 层 RGBA32F 立方体贴图并逐层上传（层序 +X,-X,+Y,-Y,+Z,-Z）。
///
/// 逐层写而非整块写：wgpu 的 GL 后端对"一次 write_texture 上传整个
/// 2D 数组纹理"的实现不可靠（实测读出全零），逐层上传与单层纹理同样稳定。
fn create_cube_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face_size: u32,
    rgba32f: &[[f32; 4]],
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let layer_pixels = (face_size * face_size) as usize;
    for layer in 0..6u32 {
        let layer_data =
            &rgba32f[(layer as usize * layer_pixels)..((layer as usize + 1) * layer_pixels)];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(layer_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(face_size * 16),
                rows_per_image: Some(face_size),
            },
            wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// 环境转换路径：Vulkan/Metal 用 GPU 计算，其余后端回退 CPU。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvConversionPath {
    /// GPU 计算着色器（storage 数组纹理可靠的后端）。
    Gpu,
    /// CPU 转换 + 逐层上传（兼容性兜底）。
    Cpu,
}

/// 环境子系统的 GPU 资源：绑定组布局、计算管线、天空盒管线、默认绑定组。
///
/// 从 `Renderer::new` 中独立出来，便于无窗口的 headless 测试直接复用同一套
/// 资源创建与转换逻辑（见 `tests::environment_headless_smoke`）。
struct EnvironmentResources {
    /// 环境转换路径（启动时按后端决定，日志可见）。
    conversion_path: EnvConversionPath,
    /// mesh 管线 @group(4) 环境绑定组布局。
    environment_bind_group_layout: wgpu::BindGroupLayout,
    /// 天空盒管线绑定组布局。
    skybox_bind_group_layout: wgpu::BindGroupLayout,
    /// 环境采样器（ClampToEdge；过滤能力取决于设备）。
    env_sampler: wgpu::Sampler,
    /// equirect → cubemap 计算管线的绑定组布局。
    env_convert_layout: wgpu::BindGroupLayout,
    /// cubemap → 辐照度图计算管线的绑定组布局。
    irradiance_layout: wgpu::BindGroupLayout,
    /// equirect → cubemap 参数 uniform 缓冲。
    ///
    /// 两个计算 pass 必须用**独立**参数缓冲：`queue.write_buffer` 是即时入队
    /// 操作，会先于 `submit()` 里的 compute pass 执行；若分时复用同一个缓冲，
    /// 第二个 write 会先覆盖，导致第一个 pass 读到错误参数。
    env_convert_params: wgpu::Buffer,
    /// cubemap → 辐照度图参数 uniform 缓冲。
    irradiance_params: wgpu::Buffer,
    /// equirect → cubemap 计算管线。
    env_convert_pipeline: wgpu::ComputePipeline,
    /// cubemap → 辐照度图计算管线。
    irradiance_pipeline: wgpu::ComputePipeline,
    /// 天空盒渲染管线。
    skybox_pipeline: wgpu::RenderPipeline,
    /// 无环境时的默认绑定组（1×1 黑环境）。
    default_environment: EnvironmentGpu,
}

impl EnvironmentResources {
    /// 创建环境子系统的全部 GPU 资源。
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        float32_filterable: bool,
        conversion_path: EnvConversionPath,
    ) -> Self {
        // mesh 管线 @group(4)：辐照度图 + 环境图 + 采样器。
        let environment_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("environment bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: float32_filterable,
                            },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: float32_filterable,
                            },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(if float32_filterable {
                            wgpu::SamplerBindingType::Filtering
                        } else {
                            wgpu::SamplerBindingType::NonFiltering
                        }),
                        count: None,
                    },
                ],
            });

        // 天空盒：相机 + 环境立方体贴图 + 采样器。
        let skybox_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("skybox bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: float32_filterable,
                            },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(if float32_filterable {
                            wgpu::SamplerBindingType::Filtering
                        } else {
                            wgpu::SamplerBindingType::NonFiltering
                        }),
                        count: None,
                    },
                ],
            });


        // 环境采样器：ClampToEdge；支持 float32 过滤时用双线性，否则点采样。
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: if float32_filterable {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            min_filter: if float32_filterable {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // 默认环境（黑色）：保证 @group(4) 与天空盒绑定组恒可用。
        let default_environment = create_default_environment(
            device,
            queue,
            &environment_bind_group_layout,
            &skybox_bind_group_layout,
            &env_sampler,
        );

        // 环境着色器：天空盒 + 计算转换入口。
        let env_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("environment shader"),
            source: ShaderSource::Wgsl(include_str!("environment.wgsl").into()),
        });

        // 计算转换的资源（GPU 路径执行；CPU 回退时同样创建，只是不运行）。
        let env_convert_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("equirect convert bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: float32_filterable,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(if float32_filterable {
                        wgpu::SamplerBindingType::Filtering
                    } else {
                        wgpu::SamplerBindingType::NonFiltering
                    }),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let irradiance_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("irradiance bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: float32_filterable,
                        },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(if float32_filterable {
                        wgpu::SamplerBindingType::Filtering
                    } else {
                        wgpu::SamplerBindingType::NonFiltering
                    }),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let env_convert_params = device.create_buffer(&BufferDescriptor {
            label: Some("environment convert params buffer"),
            size: std::mem::size_of::<EnvParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let irradiance_params = device.create_buffer(&BufferDescriptor {
            label: Some("irradiance params buffer"),
            size: std::mem::size_of::<EnvParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let env_convert_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("equirect convert pipeline layout"),
            bind_group_layouts: &[Some(&env_convert_layout)],
            immediate_size: 0,
        });
        let irradiance_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("irradiance pipeline layout"),
            bind_group_layouts: &[Some(&irradiance_layout)],
            immediate_size: 0,
        });
        let env_convert_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("equirect to cubemap pipeline"),
            layout: Some(&env_convert_pipeline_layout),
            module: &env_shader,
            entry_point: Some("equirect_to_cubemap"),
            compilation_options: Default::default(),
            cache: None,
        });
        let irradiance_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("irradiance pipeline"),
            layout: Some(&irradiance_pipeline_layout),
            module: &env_shader,
            entry_point: Some("irradiance"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 天空盒管线：全屏三角形，深度写关 + LessEqual（先画，网格随后正常遮挡）。
        let skybox_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("skybox pipeline layout"),
            bind_group_layouts: &[Some(camera_bind_group_layout), Some(&skybox_bind_group_layout)],
            immediate_size: 0,
        });
        let skybox_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("skybox pipeline"),
            layout: Some(&skybox_pipeline_layout),
            vertex: VertexState {
                module: &env_shader,
                entry_point: Some("skybox_vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &env_shader,
                entry_point: Some("skybox_fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            conversion_path,
            environment_bind_group_layout,
            skybox_bind_group_layout,
            env_sampler,
            env_convert_layout,
            irradiance_layout,
            env_convert_params,
            irradiance_params,
            env_convert_pipeline,
            irradiance_pipeline,
            skybox_pipeline,
            default_environment,
        }
    }

    /// 上传环境贴图（HDRI 等距矩形图）并转换成环境立方体贴图 + 辐照度图。
    ///
    /// 按启动时决定的路径转换：Vulkan/Metal 用 GPU 计算着色器，其余后端
    /// （GL 等 storage 数组纹理不可靠）回退 CPU 转换 + 逐层上传。
    fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &Environment,
    ) -> EnvironmentGpu {
        let face_size = ENV_CUBEMAP_SIZE;
        let irradiance_size = IRRADIANCE_SIZE;

        // 1. 按路径生成两张 6 层 RGBA32F 立方体贴图。
        let (env_texture, irradiance_texture) = match self.conversion_path {
            EnvConversionPath::Gpu => self.convert_gpu(device, queue, environment),
            EnvConversionPath::Cpu => {
                // CPU 转换 + 逐层上传（write_texture 无 256 对齐要求）。
                let cube_pixels = environment.to_cubemap(face_size);
                let irradiance_pixels = Environment::irradiance_map(
                    &cube_pixels,
                    face_size,
                    irradiance_size,
                    IRRADIANCE_SAMPLES,
                );
                (
                    create_cube_texture(
                        device,
                        queue,
                        face_size,
                        &cube_pixels,
                        "environment cubemap",
                    ),
                    create_cube_texture(
                        device,
                        queue,
                        irradiance_size,
                        &irradiance_pixels,
                        "irradiance cubemap",
                    ),
                )
            }
        };

        // 2. 立方体视图（采样用）。
        let env_cube_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let irradiance_cube_view = irradiance_texture.create_view(&TextureViewDescriptor {
            label: Some("irradiance cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });

        // 3. 构建 mesh 管线 @group(4) 与天空盒的绑定组。
        let mesh_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("environment mesh bind group"),
            layout: &self.environment_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&irradiance_cube_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&env_cube_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
            ],
        });
        let skybox_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skybox bind group"),
            layout: &self.skybox_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&env_cube_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
            ],
        });

        EnvironmentGpu {
            environment_texture: env_texture,
            environment_view: env_cube_view,
            irradiance_texture,
            irradiance_view: irradiance_cube_view,
            sampler: self.env_sampler.clone(),
            mesh_bind_group,
            skybox_bind_group,
        }
    }

    /// GPU 路径：上传等距矩形源，两个计算 pass 产出环境图与辐照度图。
    ///
    /// 只在 storage 数组纹理可靠的后端（Vulkan/Metal）调用；GL 后端在这里
    /// 会写入全零（见 BUG.md），因此由调用方按 `conversion_path` 分流。
    fn convert_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &Environment,
    ) -> (wgpu::Texture, wgpu::Texture) {
        let face_size = ENV_CUBEMAP_SIZE;
        let irradiance_size = IRRADIANCE_SIZE;

        // 1. 等距矩形源纹理（RGBA32F 单层）。
        let src_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment equirect source"),
            size: wgpu::Extent3d {
                width: environment.width,
                height: environment.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut rgba = Vec::with_capacity((environment.width * environment.height * 4) as usize);
        for rgb in &environment.rgb {
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(environment.width * 16),
                rows_per_image: Some(environment.height),
            },
            wgpu::Extent3d {
                width: environment.width,
                height: environment.height,
                depth_or_array_layers: 1,
            },
        );
        let src_view = src_texture.create_view(&TextureViewDescriptor::default());

        // 2. 输出纹理：环境图 + 辐照度图（存储写入 + 采样双用途）。
        let env_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment cubemap"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let env_storage_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap storage view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let env_cube_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let irradiance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance cubemap"),
            size: wgpu::Extent3d {
                width: irradiance_size,
                height: irradiance_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let irradiance_storage_view = irradiance_texture.create_view(&TextureViewDescriptor {
            label: Some("irradiance storage view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });

        // 3. 两个计算 pass（拆开，保证"存储写入 → 采样读取"在 pass 边界同步）。
        {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("environment conversion encoder"),
            });

            // 3.1 equirect → cubemap。
            queue.write_buffer(
                &self.env_convert_params,
                0,
                bytemuck::bytes_of(&EnvParams {
                    size: face_size,
                    sample_count: 0,
                    _pad: [0; 2],
                }),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let convert_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("equirect convert bind group"),
                    layout: &self.env_convert_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.env_convert_params.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&src_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&env_storage_view),
                        },
                    ],
                });
                pass.set_pipeline(&self.env_convert_pipeline);
                pass.set_bind_group(0, &convert_bind_group, &[]);
                pass.dispatch_workgroups(face_size.div_ceil(8), face_size.div_ceil(8), 6);
            }

            // 3.2 cubemap → 辐照度图。
            queue.write_buffer(
                &self.irradiance_params,
                0,
                bytemuck::bytes_of(&EnvParams {
                    size: irradiance_size,
                    sample_count: IRRADIANCE_SAMPLES,
                    _pad: [0; 2],
                }),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let irradiance_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("irradiance bind group"),
                    layout: &self.irradiance_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.irradiance_params.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&env_cube_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&irradiance_storage_view),
                        },
                    ],
                });
                pass.set_pipeline(&self.irradiance_pipeline);
                pass.set_bind_group(0, &irradiance_bind_group, &[]);
                pass.dispatch_workgroups(
                    irradiance_size.div_ceil(8),
                    irradiance_size.div_ceil(8),
                    6,
                );
            }
            queue.submit([encoder.finish()]);
        }

        (env_texture, irradiance_texture)
    }
}

impl Renderer {
    /// 创建 wgpu 实例、占住窗口的 surface，请求适配器与设备，并配置交换链。
    pub fn new(window: &Arc<Window>, display: DisplayHandle) -> Result<Self, RendererError> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        // 1. 创建 wgpu 实例（携带事件循环的显示句柄），并接管窗口 surface。
        let instance = wgpu::Instance::new(InstanceDescriptor::new_with_display_handle(display));
        let surface = instance.create_surface(window.clone())?;

        // 2. 请求与 surface 兼容的适配器，并创建逻辑设备与队列。
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))?;
        eprintln!(
            "渲染后端：{}（{:?}）",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        // 环境转换路径：Vulkan/Metal 的 storage 数组纹理可靠，用 GPU 计算；
        // 其余后端（GL 等）回退 CPU，见 BUG.md。
        let conversion_path = match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => EnvConversionPath::Gpu,
            _ => EnvConversionPath::Cpu,
        };
        eprintln!(
            "环境转换：{}",
            match conversion_path {
                EnvConversionPath::Gpu => "GPU 计算（Vulkan/Metal）",
                EnvConversionPath::Cpu => "CPU 回退（GL 等后端）",
            }
        );

        // 环境贴图用 RGBA32F 存储 HDR 数据，线性过滤需要显式请求该特性；
        // 不可用时回退为非过滤采样（环境转换与采样都会点采样）。
        let float32_filterable = adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        if !float32_filterable {
            eprintln!(
                "警告：设备不支持 RGBA32F 线性过滤（FLOAT32_FILTERABLE），\
                 环境贴图将使用点采样（天空盒与 IBL 会偏颗粒感）"
            );
        }
        let requested_features = if float32_filterable {
            wgpu::Features::FLOAT32_FILTERABLE
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("main device"),
            required_features: requested_features,
            // 绑定组约定 0-4（相机/物体/灯光/纹理/环境），默认上限 4 不够用；
            // 提到 8 给未来的阴影、额外环境参数等留余量（桌面后端普遍支持 8+）。
            required_limits: wgpu::Limits {
                max_bind_groups: 8,
                ..Default::default()
            },
            ..Default::default()
        }))?;

        // 3. 用 surface 的首选格式配置交换链。
        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererError::UnsupportedSurface)?;
        surface.configure(&device, &config);

        // 3.5 深度缓冲：管线与渲染通道都要用它来做正确的遮挡关系。
        let (depth_texture, depth_view) = create_depth_texture(&device, width, height);

        // 绑定组与着色器阶段的约定（改布局/着色器时两边都要对账）：
        //   group 0 相机    ：VERTEX 用 view_proj，FRAGMENT 用 position
        //   group 1 物体    ：VERTEX 用 model/normal_matrix，FRAGMENT 用 base_color/metallic/roughness
        //   group 2 灯光    ：仅 FRAGMENT
        //   group 3 纹理    ：仅 FRAGMENT（基础色 / 采样器 / 金属度粗糙度 / 法线）

        // 4. 相机 uniform：缓冲区 + 绑定组。
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("camera uniform buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("camera bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    // 顶点用 view_proj，片元用 position（PBR 视线方向）。
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 5. 物体数据：动态 uniform 缓冲布局（每个物体一个模型矩阵）。
        let object_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("object bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    // 顶点用模型/法线矩阵，片元用 base_color 因子。
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ObjectData>() as u64),
                    },
                    count: None,
                }],
            });

        // 物体数据至少一个 ObjectData 大小，且动态偏移必须是设备对齐值的整数倍。
        let object_stride = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(std::mem::size_of::<ObjectData>() as u32);
        let object_data_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("object data buffer"),
            size: object_stride as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("object data bind group"),
            layout: &object_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                // 动态偏移的校验以这里声明的绑定范围为准：只声明一个矩阵（64 字节），
                // 这样每个物体的偏移（i * stride）才不会"顶穿"整个缓冲。
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_data_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ObjectData>() as u64),
                }),
            }],
        });

        // 5.5 灯光 uniform：方向光数组（场景级，每帧写入）。
        let light_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("light bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let light_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("light uniform buffer"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("light bind group"),
            layout: &light_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });

        // 5.6 纹理绑定组：基础色贴图 + 采样器（材质级，@group(3)）。
        let texture_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("texture bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("texture sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // 默认 1×1 纹理（白 / 中性法线）：无贴图材质直接采样它们，着色器无需分支。
        let default_white_view = create_texture_view(&device, &queue, &Texture::white());
        let default_normal_view = create_texture_view(&device, &queue, &Texture::neutral_normal());
        let default_material_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("default material bind group"),
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_white_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&default_white_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&default_normal_view),
                },
            ],
        });

        // 5.7 环境子系统：布局、计算管线、天空盒管线与默认绑定组。
        let environment_resources = EnvironmentResources::new(
            &device,
            &queue,
            &camera_bind_group_layout,
            config.format,
            float32_filterable,
            conversion_path,
        );

        // 6. 渲染管线：网格 + 相机/物体/灯光/纹理/环境。
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh shader"),
            source: ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&object_bind_group_layout),
                Some(&light_bind_group_layout),
                Some(&texture_bind_group_layout),
                Some(&environment_resources.environment_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("mesh pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::layout())],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // 约定逆时针（CCW）为正面，并剔除背面。
                // 后续所有网格的顶点绕序都应按此标准：从外侧看为逆时针。
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size: (width, height),
            camera_buffer,
            camera_bind_group,
            object_bind_group_layout,
            object_data_buffer,
            object_bind_group,
            object_stride: object_stride as u32,
            light_buffer,
            light_bind_group,
            texture_bind_group_layout,
            texture_sampler,
            texture_views: Vec::new(),
            default_white_view,
            default_normal_view,
            default_material_bind_group,
            material_bind_groups: Vec::new(),
            uploaded_texture_version: 0,
            pipeline,
            depth_texture,
            depth_view,
            mesh_buffer: None,
            mesh_uploaded_version: 0,
            environment: environment_resources.default_environment.clone(),
            environment_resources,
        })
    }

    /// 把网格库中的全部资产合并成一份顶点/索引缓冲，永久驻留。
    ///
    /// 版本没变则跳过；新增资产后整体重传（前面的数据保持不变）。
    pub fn upload_meshes(&mut self, library: &MeshLibrary) {
        if library.version() == self.mesh_uploaded_version {
            return;
        }

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut mesh_ranges = Vec::with_capacity(library.len());
        for mesh in library.meshes() {
            let vertex_offset = vertices.len() as u32;
            mesh_ranges.push(MeshRange {
                index_offset: indices.len() as u32,
                index_count: mesh.indices().len() as u32,
            });
            vertices.extend_from_slice(mesh.vertices());
            // 合并时索引已按该网格的顶点起始偏移平移，因此绘制时 base_vertex 必须为 0。
            indices.extend(mesh.indices().iter().map(|i| i + vertex_offset));
        }

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh library vertex buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh library index buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::INDEX,
            });

        self.mesh_buffer = Some(MeshGpu {
            vertex_buffer,
            index_buffer,
            mesh_ranges,
        });
        self.mesh_uploaded_version = library.version();
    }

    /// 把纹理库中新增的贴图上传为 GPU 纹理并建好绑定组（只追加，增量上传）。
    pub fn upload_textures(&mut self, library: &TextureLibrary) {
        if library.version() == self.uploaded_texture_version {
            return;
        }
        for texture in library.textures().iter().skip(self.texture_views.len()) {
            let view = create_texture_view(&self.device, &self.queue, texture);
            self.texture_views.push(view);
        }
        self.uploaded_texture_version = library.version();
    }

    /// 上传环境贴图（HDRI 等距矩形图）并转换成环境立方体贴图 + 辐照度图。
    ///
    /// 转换由两个计算着色器在启动时一次性完成，之后每帧只采样；
    /// 关卡切换换环境时重建纹理与绑定组，旧资源随替换自动释放。
    pub fn set_environment(&mut self, environment: &Environment) {
        self.environment = self
            .environment_resources
            .convert(&self.device, &self.queue, environment);
    }

    /// 加载场景：按物体数量重建动态 uniform 缓冲（网格资产已在 `upload_meshes` 中常驻）。
    pub fn load_scene(&mut self, scene: &Scene) {
        // 按物体数量重建动态 uniform 缓冲与绑定组。
        let stride = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(std::mem::size_of::<ObjectData>() as u32);
        let object_data_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("object data buffer"),
            size: (scene.object_count() as u64).max(1) * stride as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("object data bind group"),
            layout: &self.object_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_data_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ObjectData>() as u64),
                }),
            }],
        });

        self.object_data_buffer = object_data_buffer;
        self.object_bind_group = object_bind_group;
        self.object_stride = stride as u32;

        // 灯光是静态场景数据：加载时收集一次并上传，渲染时只绑定。
        let light_uniform = collect_lights(scene);
        self.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(&light_uniform));

        // 每个物体的材质绑定组（与 objects() 迭代顺序一致，渲染时按同一下标取用）。
        let mut material_bind_groups = Vec::with_capacity(scene.object_count());
        for (_, object) in scene.objects() {
            if object.mesh_key().is_none() {
                material_bind_groups.push(self.default_material_bind_group.clone());
                continue;
            }
            let mat = &object.material;
            let base_view = mat
                .base_color_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_white_view);
            let mr_view = mat
                .metallic_roughness_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_white_view);
            let normal_view = mat
                .normal_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_normal_view);
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("material bind group"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(base_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(mr_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                ],
            });
            material_bind_groups.push(bind_group);
        }
        self.material_bind_groups = material_bind_groups;
    }

    /// 窗口尺寸变化时重建交换链。
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        // 深度缓冲必须与交换链尺寸保持一致。
        (self.depth_texture, self.depth_view) = create_depth_texture(&self.device, width, height);
    }

    /// 渲染一帧：写入相机与物体 uniform，清屏，绘制场景中所有物体并呈现。
    pub fn render(&mut self, camera: &Camera, scene: &Scene) {
        // 每帧把相机数据写入 uniform 缓冲区。
        let uniform = CameraUniform::from_camera(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // 每帧把物体世界矩阵 + 法线矩阵写入动态 uniform 缓冲（步长 = object_stride）。
        if scene.object_count() > 0 {
            let stride = self.object_stride as usize;
            let entry_size = std::mem::size_of::<ObjectData>();
            let mut bytes = vec![0u8; scene.object_count() * stride];
            for (i, (key, object)) in scene.objects().enumerate() {
                let model = scene
                    .world_transform(key)
                    .expect("objects() 只产出存活节点，world_transform 必然有值");
                // 法线矩阵 = 模型上三角的逆转置，非等比缩放下法线方向才正确。
                let m = Mat3::from_mat4(model).inverse().transpose();
                let cols = m.to_cols_array();
                let normal_matrix = [
                    [cols[0], cols[1], cols[2], 0.0],
                    [cols[3], cols[4], cols[5], 0.0],
                    [cols[6], cols[7], cols[8], 0.0],
                ];
                let data = ObjectData {
                    model,
                    normal_matrix,
                    base_color: object.material.base_color,
                    metallic: object.material.metallic_factor,
                    roughness: object.material.roughness_factor,
                    _pad: [0.0; 2],
                };
                bytes[i * stride..i * stride + entry_size]
                    .copy_from_slice(bytemuck::bytes_of(&data));
            }
            self.queue.write_buffer(&self.object_data_buffer, 0, &bytes);
        }

        // 获取当前帧；surface 状态异常时跳过或重建交换链。
        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame) | CurrentSurfaceTexture::Suboptimal(frame) => {
                frame
            }
            CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => return,
            CurrentSurfaceTexture::Outdated
            | CurrentSurfaceTexture::Lost
            | CurrentSurfaceTexture::Validation => {
                self.resize(self.size.0, self.size.1);
                return;
            }
        };

        {
            let view = frame.texture.create_view(&TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("frame encoder"),
                });

            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("main pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(CLEAR_COLOR),
                            store: StoreOp::Store,
                        },
                    })],
                    // 每帧清空深度为 1.0（远），只保留真正更近的片元。
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: LoadOp::Clear(1.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                // 天空盒：深度写关 + LessEqual，先画（深度已清为 1.0，
                // 网格随后用 Less 正常遮挡天空）。
                pass.set_pipeline(&self.environment_resources.skybox_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, &self.environment.skybox_bind_group, &[]);
                pass.draw(0..3, 0..1);

                if let Some(mesh_buffer) = &self.mesh_buffer {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    pass.set_bind_group(2, &self.light_bind_group, &[]);
                    pass.set_bind_group(4, &self.environment.mesh_bind_group, &[]);
                    pass.set_vertex_buffer(0, mesh_buffer.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        mesh_buffer.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );

                    // 每个物体：绑定它的世界矩阵（动态偏移），按句柄直取网格区间；
                    // 非网格节点（分组、未来的灯光/相机等）跳过。
                    for (i, (_, object)) in scene.objects().enumerate() {
                        let Some(mesh_key) = object.mesh_key() else { continue; };
                        let range = mesh_buffer.mesh_ranges[mesh_key.index()];
                        let offset = (i * self.object_stride as usize) as u32;
                        pass.set_bind_group(1, &self.object_bind_group, &[offset]);
                        pass.set_bind_group(3, &self.material_bind_groups[i], &[]);
                        pass.draw_indexed(
                            range.index_offset..range.index_offset + range.index_count,
                            0,
                            0..1,
                        );
                    }
                }
            }

            self.queue.submit([encoder.finish()]);
        }

        self.queue.present(frame);
    }
}

/// 渲染器初始化过程中的错误。
#[derive(Debug)]
pub enum RendererError {
    Surface(wgpu::CreateSurfaceError),
    Adapter(wgpu::RequestAdapterError),
    Device(wgpu::RequestDeviceError),
    UnsupportedSurface,
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Surface(e) => write!(f, "failed to create surface: {e}"),
            Self::Adapter(e) => write!(f, "failed to find a suitable adapter: {e}"),
            Self::Device(e) => write!(f, "failed to create device: {e}"),
            Self::UnsupportedSurface => write!(f, "surface is not supported by the adapter"),
        }
    }
}

impl Error for RendererError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Surface(e) => Some(e),
            Self::Adapter(e) => Some(e),
            Self::Device(e) => Some(e),
            Self::UnsupportedSurface => None,
        }
    }
}

impl From<wgpu::CreateSurfaceError> for RendererError {
    fn from(error: wgpu::CreateSurfaceError) -> Self {
        Self::Surface(error)
    }
}

impl From<wgpu::RequestAdapterError> for RendererError {
    fn from(error: wgpu::RequestAdapterError) -> Self {
        Self::Adapter(error)
    }
}

impl From<wgpu::RequestDeviceError> for RendererError {
    fn from(error: wgpu::RequestDeviceError) -> Self {
        Self::Device(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cargo build 不编译 WGSL，运行时错误会晚暴露；这里用 naga 提前校验。
    fn validate_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL 应能解析");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("WGSL 应通过校验");
    }

    #[test]
    fn mesh_shader_compiles() {
        validate_wgsl(include_str!("mesh.wgsl"));
    }

    #[test]
    fn environment_shader_compiles() {
        validate_wgsl(include_str!("environment.wgsl"));
    }

    /// 无窗口设备：请求适配器并创建设备（含 max_bind_groups 8 与
    /// FLOAT32_FILTERABLE 特性）。失败时打印原因并返回 `None`（CI 无 GPU 可跳过）。
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue, bool, EnvConversionPath)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .inspect_err(|e| eprintln!("headless 测试：请求适配器失败（{e}），跳过"))
        .ok()?;
        let float32_filterable = adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("smoke test device"),
            required_features: if float32_filterable {
                wgpu::Features::FLOAT32_FILTERABLE
            } else {
                wgpu::Features::empty()
            },
            required_limits: wgpu::Limits {
                max_bind_groups: 8,
                ..Default::default()
            },
            ..Default::default()
        }))
        .inspect_err(|e| eprintln!("headless 测试：设备创建失败（{e}），跳过"))
        .ok()?;
        let conversion_path = match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => EnvConversionPath::Gpu,
            _ => EnvConversionPath::Cpu,
        };
        Some((device, queue, float32_filterable, conversion_path))
    }


    /// 无窗口冒烟测试：不创建 surface，直接请求适配器/设备，验证环境资源创建、
    /// 计算转换与天空盒渲染不触发 wgpu 校验错误；无 GPU 环境（如 CI）则跳过。
    #[test]
    fn environment_headless_smoke() {
        let Some((device, queue, float32_filterable, conversion_path)) = headless_device() else {
            return;
        };

        // mesh 着色器声明了 @group(4)：校验它不超出 max_bind_groups 限制。
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("smoke mesh shader"),
            source: ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });

        // 相机绑定组布局 + uniform（天空盒管线需要）。
        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("smoke camera layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("smoke camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY,
                position: glam::Vec3::ZERO,
                _padding: 0,
                inverse_view_proj: glam::Mat4::IDENTITY,
            }),
        );
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("smoke camera bind group"),
            layout: &camera_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 环境资源：布局、计算管线、天空盒管线、默认绑定组。
        let resources = EnvironmentResources::new(
            &device,
            &queue,
            &camera_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            float32_filterable,
            conversion_path,
        );

        // 转换一个 2×1 的微型 HDR（左红右绿），验证计算管线与绑定组创建。
        let env = super::Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        };
        let gpu_env = resources.convert(&device, &queue, &env);

        // 天空盒渲染到离屏纹理，验证渲染管线 + 绑定组 + 实际绘制。
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke color texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = create_depth_texture(&device, 4, 4);
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("smoke encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("smoke pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(CLEAR_COLOR),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.1,
                    depth_ops: Some(wgpu::Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&resources.skybox_pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_bind_group(1, &gpu_env.skybox_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        // 等待 GPU 完成，确保编码/提交阶段没有触发校验错误。
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll 应成功");
    }


    /// 采样验证：已知全红立方体贴图经天空盒管线渲染到离屏，读回应偏红。
    /// （绕开 copy_texture_to_buffer 拷数组纹理的路径，直接验证"上传→采样"。）
    #[test]
    fn skybox_sampling_verifies_texture_content() {
        let Some((device, queue, float32_filterable, conversion_path)) = headless_device() else {
            return;
        };

        // 相机绑定组（天空盒需要 camera uniform）。
        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("verify camera layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("verify camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY,
                position: glam::Vec3::ZERO,
                _padding: 0,
                inverse_view_proj: glam::Mat4::IDENTITY,
            }),
        );
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("verify camera bind group"),
            layout: &camera_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let resources = EnvironmentResources::new(
            &device,
            &queue,
            &camera_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            float32_filterable,
            conversion_path,
        );

        // 1) 已知全红 cube（4×4×6）→ 天空盒渲染 → 应偏红。
        let known: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 1.0]; (4 * 4 * 6) as usize];
        let known_tex = create_cube_texture(&device, &queue, 4, &known, "known red cube");
        let known_view = known_tex.create_view(&TextureViewDescriptor {
            label: Some("known red cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let known_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("verify skybox bind group"),
            layout: &resources.skybox_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&known_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&resources.env_sampler),
                },
            ],
        });
        let data = render_skybox_rgb(&device, &queue, &resources, &camera_bind_group, &known_bind_group);
        let mut max_r = 0u8;
        for chunk in data.chunks_exact(4) {
            max_r = max_r.max(chunk[0]);
        }
        eprintln!("天空盒渲染读回最大 R 分量：{max_r}");
        assert!(
            max_r > 128,
            "已知全红 cube 经天空盒渲染后 R 分量过低（上传或采样失败）"
        );

        // 2) 真实 HDR → convert（CPU 转换 + 逐层上传）→ 天空盒渲染 → 非黑。
        let env = match Environment::from_hdr_file(std::path::Path::new("assets/environments/test.hdr"))
        {
            Ok(env) => env,
            Err(_) => Environment {
                width: 2,
                height: 1,
                rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
        };
        let gpu_env = resources.convert(&device, &queue, &env);
        let data = render_skybox_rgb(
            &device,
            &queue,
            &resources,
            &camera_bind_group,
            &gpu_env.skybox_bind_group,
        );
        let mut sum = 0u32;
        for chunk in data.chunks_exact(4) {
            sum += chunk[0] as u32 + chunk[1] as u32 + chunk[2] as u32;
        }
        let avg = sum as f32 / (data.len() / 4) as f32;
        eprintln!("真实 HDR 天空盒渲染平均 RGB：{avg:.1}");
        assert!(
            avg > 20.0,
            "真实 HDR 环境转换后天空盒渲染仍接近全黑（端到端链路失败）"
        );
    }

    /// 把天空盒渲染到 4×4 离屏 Rgba8UnormSrgb 并读回像素字节。
    fn render_skybox_rgb(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &EnvironmentResources,
        camera_bind_group: &wgpu::BindGroup,
        skybox_bind_group: &wgpu::BindGroup,
    ) -> Vec<u8> {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("verify color"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = create_depth_texture(device, 4, 4);
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("verify encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("verify pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.1,
                    depth_ops: Some(wgpu::Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&resources.skybox_pipeline);
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_bind_group(1, skybox_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        let aligned_row = 256u32; // 4 像素 × 4 字节 = 16，按 copy 要求对齐到 256
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("verify readback"),
            size: (aligned_row * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("verify readback encoder"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_row),
                    rows_per_image: Some(4),
                },
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll 应成功");
        rx.recv().expect("map 回调应触发").expect("map 应成功");
        let data = slice.get_mapped_range().expect("取范围应成功");
        data[..(4 * 4 * 4) as usize].to_vec()
    }
}
