//! 渲染模块：负责 wgpu 初始化、surface 管理和每帧绘制。
//!
//! 子模块分工：
//! - [`uniform`]：GPU uniform 布局与灯光收集；
//! - [`environment`]：环境贴图（天空盒 + IBL）的 GPU 资源与转换；
//! - [`tests`]：WGSL 校验 + 无头冒烟测试（仅测试构建）。

mod environment;
pub(crate) mod uniform;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use glam::Mat3;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, CurrentSurfaceTexture, DeviceDescriptor, FragmentState,
    InstanceDescriptor, LoadOp, Operations, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp,
    TextureViewDescriptor, VertexState,
};
use winit::window::Window;

use super::core::camera::{Camera, CameraUniform};
use super::core::environment::Environment;
use super::core::mesh::{MeshLibrary, Vertex};
use super::core::texture::{Texture, TextureLibrary};
use super::scene::Scene;

use self::environment::{EnvConversionPath, EnvironmentGpu, EnvironmentResources};
use self::uniform::{collect_lights, AGX_MIDDLE_GRAY_LOG2, LightsUniform, ObjectData};

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
        // 其余后端（GL 等）回退 CPU，见 docs/BUG.md。
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
