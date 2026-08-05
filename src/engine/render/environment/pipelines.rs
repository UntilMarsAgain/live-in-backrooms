//! 环境子系统：GPU 资源装配。
//!
//! 绑定组布局、计算管线（环境转换 / 辐照度 / 预过滤 / BRDF LUT）与天空盒管线的创建。

use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, ComputePipelineDescriptor,
    FragmentState, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages, VertexState,
};

use super::{
    EnvConversionPath, EnvironmentResources, PREFILTER_MIP_COUNT, create_default_environment,
};
use crate::engine::render::uniform::{EnvParams, EnvironmentParams, PrefilterParams};

impl EnvironmentResources {
    /// 创建环境子系统的全部 GPU 资源。
    pub(crate) fn new(
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
                    BindGroupLayoutEntry {
                        binding: 4,
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
                        binding: 5,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: float32_filterable,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
            // 镜面预过滤图需要三线性（mip 间插值）；辐照度/天空盒只采 mip 0，不受影响。
            mipmap_filter: if float32_filterable {
                wgpu::MipmapFilterMode::Linear
            } else {
                wgpu::MipmapFilterMode::Nearest
            },
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
        let env_convert_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
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

        // 镜面预过滤：GGX 重要性采样写 mip 链（每个 mip 一次 dispatch）。
        let prefilter_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prefilter bind group layout"),
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
        let brdf_lut_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("brdf lut bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let prefilter_params = (0..PREFILTER_MIP_COUNT)
            .map(|mip| {
                device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("prefilter params buffer (mip {mip})")),
                    size: std::mem::size_of::<PrefilterParams>() as u64,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let prefilter_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prefilter pipeline layout"),
            bind_group_layouts: &[Some(&prefilter_layout)],
            immediate_size: 0,
        });
        let brdf_lut_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brdf lut pipeline layout"),
            bind_group_layouts: &[Some(&brdf_lut_layout)],
            immediate_size: 0,
        });
        let prefilter_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prefilter pipeline"),
            layout: Some(&prefilter_pipeline_layout),
            module: &env_shader,
            entry_point: Some("prefilter"),
            compilation_options: Default::default(),
            cache: None,
        });
        let brdf_lut_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("brdf lut pipeline"),
            layout: Some(&brdf_lut_pipeline_layout),
            module: &env_shader,
            entry_point: Some("brdf_lut"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 天空盒管线：全屏三角形，深度写关 + LessEqual（先画，网格随后正常遮挡）。
        let skybox_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("skybox pipeline layout"),
            bind_group_layouts: &[
                Some(camera_bind_group_layout),
                Some(&skybox_bind_group_layout),
            ],
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
            prefilter_layout,
            brdf_lut_layout,
            prefilter_params,
            prefilter_pipeline,
            brdf_lut_pipeline,
            skybox_pipeline,
            default_environment,
        }
    }
}
