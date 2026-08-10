//! 色调映射 blit 子系统：把 HDR 中间目标采样出来做 AgX 色调映射，写交换链。
//!
//! 这是"HDR 多 pass"管线的最后一环：场景 pass（网格 + 天空盒 + 调试线框）
//! 输出原始辐射值到 [`super::HDR_FORMAT`] 离屏纹理，这里用全屏三角形把
//! 它映射成可显示的线性 sRGB 交给交换链。绑定组引用 HDR 视图与
//! 环境参数 uniform（AgX EV 窗口），窗口尺寸变化时由 Renderer 重建绑定组。

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, ColorTargetState, ColorWrites, FragmentState,
    PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, VertexState,
};

/// 色调映射 blit 的 GPU 资源：管线 + 绑定组布局 + 采样器。
///
/// 绑定组随 HDR 视图重建（resize 时），因此这里只持有不变的部分；
/// 见 [`BlitResources::create_bind_group`]。
pub(super) struct BlitResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl BlitResources {
    /// 创建 blit 管线（目标格式 = 交换链格式）与绑定组布局。
    pub(super) fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("blit shader"),
            source: ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("blit bind group layout"),
            entries: &[
                // HDR 中间目标（Rgba16Float，默认可过滤）。
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
                // 环境参数 uniform：只读 AgX EV 窗口（布局与 EnvironmentParams 对齐）。
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
                // Bloom 结果（blit 把它加回 HDR；无 bloom 时绑黑色纹理）。
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

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // 全屏三角形无正反面无意义，关闭剔除。
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        // HDR 纹理是单 mip，最近/双线性即可；边界不可能越出（UV ∈ [0,1]）。
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// 为给定的 HDR 视图创建绑定组（resize 重建 HDR 纹理后重新调用）。
    pub(super) fn create_bind_group(
        &self,
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        env_params_buffer: &wgpu::Buffer,
        bloom_view: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        // 无 bloom 时绑定黑色纹理，贡献为 0（blit 本身不区分是否有 bloom）。
        let black = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blit black bloom fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::HDR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let black_view = black.create_view(&wgpu::TextureViewDescriptor::default());
        let bloom_view = bloom_view.unwrap_or(&black_view);
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: env_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
            ],
        })
    }

    /// 绑定管线 + 绑定组，画全屏三角形。
    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, bind_group: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
