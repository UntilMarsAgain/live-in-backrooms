//! 渲染器初始化：wgpu 实例 / surface / 适配器 / 设备、交换链、绑定组与管线装配。

use std::sync::Arc;

use anyhow::Context;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    DeviceDescriptor, FragmentState, InstanceDescriptor, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, VertexState,
};
use winit::window::Window;

use crate::engine::core::camera::CameraUniform;
use crate::engine::core::mesh::Vertex;
use crate::engine::core::texture::Texture;
use crate::engine::render::debug::LineGizmos;
use crate::engine::render::environment::EnvironmentResources;
use crate::engine::render::uniform::{
    LIGHT_CAPACITY, LightCountUniform, LightUniform, ObjectDataUniform,
};
use crate::engine::render::{DisplayHandle, Renderer};

/// 创建与窗口尺寸一致的深度纹理。
pub(super) fn create_depth_texture(
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
pub(super) fn create_texture_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &Texture,
) -> wgpu::TextureView {
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
    pub fn new(window: &Arc<Window>, display: DisplayHandle) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let (width, height) = (size.width.max(1), size.height.max(1));

        // 1. 创建 wgpu 实例（携带事件循环的显示句柄），并接管窗口 surface。
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..InstanceDescriptor::new_with_display_handle(display)
        });
        let surface = instance.create_surface(window.clone())?;

        // 2. 请求与 surface 兼容的适配器，并创建逻辑设备与队列。
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))?;
        eprintln!(
            "渲染后端：{}（{:?}）",
            adapter.get_info().name,
            adapter.get_info().backend
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
            .context("surface 不被适配器支持")?;
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
            size: size_of::<CameraUniform>() as u64,
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
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ObjectDataUniform>() as u64,
                        ),
                    },
                    count: None,
                }],
            });

        // 物体数据至少一个 ObjectData 大小，且动态偏移必须是设备对齐值的整数倍。
        let object_stride = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(size_of::<ObjectDataUniform>() as u32);
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
                    size: wgpu::BufferSize::new(size_of::<ObjectDataUniform>() as u64),
                }),
            }],
        });

        // 5.5 灯光：数量 uniform + 只读 storage 数组（动态，每帧写入收集结果）。
        let light_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("light bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let light_count_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("light count uniform buffer"),
            size: std::mem::size_of::<LightCountUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_storage_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("light storage buffer"),
            size: LIGHT_CAPACITY as u64 * std::mem::size_of::<LightUniform>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("light bind group"),
            layout: &light_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: light_count_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: light_storage_buffer.as_entire_binding(),
                },
            ],
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
        );

        // 5.8 调试线框：灯光与碰撞箱各一个实例，管线复用相机绑定组（@group(0)）。
        let light_gizmos = LineGizmos::new(&device, &camera_bind_group_layout, config.format);
        let collision_gizmos = LineGizmos::new(&device, &camera_bind_group_layout, config.format);

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
            light_count_buffer,
            light_storage_buffer,
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
            light_gizmos,
            collision_gizmos,
        })
    }
}
