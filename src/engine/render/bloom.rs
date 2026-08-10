//! Bloom（辉光）后处理：从 HDR 目标提取高亮，多级下采样 + 逐级上采样合并。
//!
//! 插在"场景 pass → 色调映射 blit"之间（Blender 合成器 Glare 节点的位置）。
//! 场景 pass 输出未曝光的原始辐射值，本模块提取并扩散亮区；blit pass 把
//! bloom 结果加回 HDR 再统一曝光 + AgX 色调映射（`blit.wgsl` 采样两张纹理）。

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferBindingType,
    BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, FragmentState,
    LoadOp, Operations, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp, TextureDescriptor,
    TextureSampleType, TextureUsages, TextureViewDimension, VertexState,
};

use super::HDR_FORMAT;

/// Bloom mip 链级数（0 为全分辨率提取层，逐级减半）。
const LEVELS: usize = 5;

/// 参数 uniform（与 bloom.wgsl `BloomParams` 对齐）：16 字节。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParams {
    threshold: f32,
    intensity: f32,
    _pad: [u32; 3],
}

impl BloomParams {
    fn new(threshold: f32, intensity: f32) -> Self {
        Self {
            threshold,
            intensity,
            _pad: [0; 3],
        }
    }
}

/// Bloom 资源：参数缓冲 + 绑定组布局 + mip 链纹理 + 三条全屏管线。
///
/// 管线共用同一绑定组布局（参数 + 源纹理 + 采样器）：
/// - `extract`：HDR → bloom[0]（阈值截断，blend 关）；
/// - `downsample`：bloom[i-1] → bloom[i]（2×2 平均，blend 关）；
/// - `upsample`：bloom[i+1] → bloom[i]（双线性放大，**blend add**——
///   把上一级的辉光叠加到本级的原始高亮）。
pub(super) struct BloomResources {
    params_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pub(super) mip_chain: Vec<(wgpu::Texture, wgpu::TextureView)>,
    extract_pipeline: wgpu::RenderPipeline,
    downsample_pipeline: wgpu::RenderPipeline,
    upsample_pipeline: wgpu::RenderPipeline,
    /// 当前参数（阈值 / 强度），每帧写入 uniform。
    params: BloomParams,
}

impl BloomResources {
    /// 创建 Bloom 资源。`width`/`height` 为 HDR 目标尺寸（bloom[0] 与之相同，
    /// 后续每级减半）。
    pub(super) fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // 默认阈值 2.0：金属反射环境光（辐射值 1~2）不至于动不动就辉光；
        // 强度 0.3 控制光晕亮度（过高会盖过主体）。两者都可经 setter 调整。
        let params = BloomParams::new(2.0, 0.3);
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom shader"),
            source: ShaderSource::Wgsl(include_str!("bloom.wgsl").into()),
        });

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("bloom params buffer"),
            size: std::mem::size_of::<BloomParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("bloom bind group layout"),
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("bloom pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label: &str, entry: &str, blend: Option<BlendState>| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format: HDR_FORMAT,
                        blend,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let extract_pipeline = make_pipeline("bloom extract", "extract_fs", None);
        let downsample_pipeline = make_pipeline("bloom downsample", "downsample_fs", None);
        let upsample_pipeline = make_pipeline(
            "bloom upsample",
            "upsample_fs",
            Some(BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::One,
                    operation: BlendOperation::Add,
                },
            }),
        );

        Self {
            params_buffer,
            bind_group_layout,
            sampler,
            mip_chain: Self::create_mip_chain(device, width, height),
            extract_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            params,
        }
    }

    /// 设置辉光阈值（辐射值超过它才被提取；调高可避免金属反射/亮区误触发）。
    pub(super) fn set_threshold(&mut self, threshold: f32) {
        self.params.threshold = threshold;
    }

    /// 设置辉光强度（提取后乘的系数，控制光晕亮度）。
    pub(super) fn set_intensity(&mut self, intensity: f32) {
        self.params.intensity = intensity;
    }

    /// 重建 mip 链（窗口 resize 后 HDR 尺寸变化时调用）。
    pub(super) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.mip_chain = Self::create_mip_chain(device, width, height);
    }

    fn create_mip_chain(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Vec<(wgpu::Texture, wgpu::TextureView)> {
        (0..LEVELS)
            .map(|i| {
                let w = (width >> i).max(1);
                let h = (height >> i).max(1);
                let texture = device.create_texture(&TextureDescriptor {
                    label: Some(&format!("bloom mip {i}")),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                (texture, view)
            })
            .collect()
    }

    /// 执行 Bloom：提取 → 下采样 → 上采样合并。`hdr_view` 是场景 pass 的输出。
    ///
    /// **必须插进主 encoder**（场景 pass 之后、blit 之前）——如果本方法自己
    /// submit，bloom 会采样到上一帧的 HDR，与当前帧 blit 错位（视角转动时
    /// 辉光滞后、跳变）。
    pub(super) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut CommandEncoder,
        hdr_view: &wgpu::TextureView,
    ) {
        // 参数（阈值 / 强度）每帧写入。
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));

        // pass 0：提取高亮（源 = HDR，目标 = bloom[0]）。
        self.draw_pass(
            device,
            encoder,
            &self.extract_pipeline,
            hdr_view,
            &self.mip_chain[0].1,
            LoadOp::Clear(wgpu::Color::BLACK),
        );

        // pass 1..LEVELS-1：下采样。
        for i in 1..LEVELS {
            self.draw_pass(
                device,
                encoder,
                &self.downsample_pipeline,
                &self.mip_chain[i - 1].1,
                &self.mip_chain[i].1,
                LoadOp::Clear(wgpu::Color::BLACK),
            );
        }

        // pass LEVELS-2..0：上采样合并（blend add 把上一级加到本级）。
        for i in (0..LEVELS - 1).rev() {
            self.draw_pass(
                device,
                encoder,
                &self.upsample_pipeline,
                &self.mip_chain[i + 1].1,
                &self.mip_chain[i].1,
                wgpu::LoadOp::Load, // 保留本级高亮，只叠加上一级的辉光。
            );
        }
    }

    fn draw_pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        source_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bloom bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("bloom pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
