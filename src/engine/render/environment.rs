//! 环境子系统：环境贴图（天空盒 + IBL）的 GPU 资源与转换。
//!
//! 按启动时决定的后端路径转换 HDRI：Vulkan/Metal 用 GPU 计算着色器，
//! GL 等后端回退 CPU 转换 + 逐层上传（见 docs/BUG.md）。

use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, ComputePipelineDescriptor, FragmentState, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, TextureViewDescriptor, VertexState,
};

use crate::engine::core::environment::Environment;
use super::uniform::{EnvParams, EnvironmentParams};

/// 环境立方体贴图每面尺寸。
pub(super) const ENV_CUBEMAP_SIZE: u32 = 256;
/// 辐照度图（漫反射 IBL）每面尺寸。
pub(super) const IRRADIANCE_SIZE: u32 = 32;
/// 辐照度图余弦加权采样数：启动时一次性计算，取大一些换取平滑。
pub(super) const IRRADIANCE_SAMPLES: u32 = 1024;

/// 环境贴图的 GPU 表示：环境立方体贴图 + 辐照度图 + 绑定组。
///
/// 纹理由视图持有引用，`set_environment` 重建时旧资源自动随引用释放。
#[derive(Clone)]
pub(super) struct EnvironmentGpu {
    /// 环境立方体贴图（天空盒采样；未来的镜面预过滤也以此为输入）。
    #[allow(dead_code)] // 资源所有权显式化；readback 诊断与镜面 IBL（Phase 2）会使用
    pub(super) environment_texture: wgpu::Texture,
    /// 环境立方体贴图视图（天空盒与未来的镜面反射采样）。
    #[allow(dead_code)]
    pub(super) environment_view: wgpu::TextureView,
    /// 辐照度图纹理（漫反射 IBL）。
    #[allow(dead_code)]
    pub(super) irradiance_texture: wgpu::Texture,
    /// 辐照度图视图（漫反射 IBL）。
    #[allow(dead_code)]
    pub(super) irradiance_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) sampler: wgpu::Sampler,
    /// mesh 管线 @group(4) 绑定组。
    pub(super) mesh_bind_group: wgpu::BindGroup,
    /// 天空盒管线绑定组。
    pub(super) skybox_bind_group: wgpu::BindGroup,
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
    intensity_buffer: &wgpu::Buffer,
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
            BindGroupEntry {
                binding: 3,
                resource: intensity_buffer.as_entire_binding(),
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
            BindGroupEntry {
                binding: 2,
                resource: intensity_buffer.as_entire_binding(),
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
pub(super) fn create_cube_texture(
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
pub(super) enum EnvConversionPath {
    /// GPU 计算着色器（storage 数组纹理可靠的后端）。
    Gpu,
    /// CPU 转换 + 逐层上传（兼容性兜底）。
    Cpu,
}

/// 环境子系统的 GPU 资源：绑定组布局、计算管线、天空盒管线、默认绑定组。
///
/// 从 `Renderer::new` 中独立出来，便于无窗口的 headless 测试直接复用同一套
/// 资源创建与转换逻辑（见 `tests::environment_headless_smoke`）。
pub(super) struct EnvironmentResources {
    /// 环境转换路径（启动时按后端决定，日志可见）。
    pub(super) conversion_path: EnvConversionPath,
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
    /// 天空盒渲染管线。
    pub(super) skybox_pipeline: wgpu::RenderPipeline,
    /// 无环境时的默认绑定组（1×1 黑环境）。
    pub(super) default_environment: EnvironmentGpu,
}

impl EnvironmentResources {
    /// 创建环境子系统的全部 GPU 资源。
    pub(super) fn new(
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
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
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

        // 环境参数 uniform（mesh 管线 @group(4) binding 3、天空盒 @group(1) binding 2）。
        let env_params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("environment params buffer"),
            size: std::mem::size_of::<EnvironmentParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &env_params_buffer,
            0,
            bytemuck::bytes_of(&EnvironmentParams::default()),
        );

        // 默认环境（黑色）：保证 @group(4) 与天空盒绑定组恒可用。
        let default_environment = create_default_environment(
            device,
            queue,
            &environment_bind_group_layout,
            &skybox_bind_group_layout,
            &env_sampler,
            &env_params_buffer,
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
            env_params_buffer,
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
    pub(super) fn convert(
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
                BindGroupEntry {
                    binding: 3,
                    resource: self.env_params_buffer.as_entire_binding(),
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
                BindGroupEntry {
                    binding: 2,
                    resource: self.env_params_buffer.as_entire_binding(),
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

    /// 设置环境强度（IBL 系数）：`0` = 纯手动布光（环境图只当背景天空盒），
    /// `1` = 满环境光；可超 1 补亮。只写 intensity 字段，不重建任何资源。
    pub(super) fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.env_params_buffer, 0, bytemuck::bytes_of(&intensity));
    }

    /// 设置 AgX 色调映射的 EV 窗口（场景级配置，默认与 Blender 一致）。
    /// 只写 min/max 两个字段，不重建任何资源。
    pub(super) fn set_agx_range(&self, queue: &wgpu::Queue, min_ev: f32, max_ev: f32) {
        debug_assert!(min_ev < max_ev, "AgX EV 窗口要求 min_ev < max_ev");
        queue.write_buffer(
            &self.env_params_buffer,
            4,
            bytemuck::bytes_of(&[min_ev, max_ev]),
        );
    }

    /// GPU 路径：上传等距矩形源，两个计算 pass 产出环境图与辐照度图。
    ///
    /// 只在 storage 数组纹理可靠的后端（Vulkan/Metal）调用；GL 后端在这里
    /// 会写入全零（见 docs/BUG.md），因此由调用方按 `conversion_path` 分流。
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
