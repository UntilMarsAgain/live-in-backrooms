//! 渲染模块：负责 wgpu 初始化、surface 管理和每帧绘制。

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, CurrentSurfaceTexture, DeviceDescriptor,
    FragmentState, InstanceDescriptor, LoadOp, Operations, PipelineLayoutDescriptor,
    PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StoreOp, TextureViewDescriptor, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState,
};
use winit::window::Window;

use crate::camera::{Camera, CameraUniform};

/// 窗口的显示句柄，用于创建 wgpu 实例。
pub type DisplayHandle = Box<dyn wgpu::wgt::WgpuHasDisplayHandle>;

/// 初始背景色：暗黄绿的"后室"氛围色，后续可改为可配置。
pub const CLEAR_COLOR: Color = Color {
    r: 0.06,
    g: 0.07,
    b: 0.05,
    a: 1.0,
};

/// 三角形顶点：位置 + 颜色。
///
/// 本阶段没有物体变换，顶点坐标直接写在（世界）空间里。
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

const VERTEX_ATTRIBUTES: [VertexAttribute; 2] = [
    VertexAttribute {
        format: VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    VertexAttribute {
        format: VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
];

impl Vertex {
    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }
    }
}

/// 示例三角形（世界坐标 == 物体坐标）。
const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-0.5, -0.5, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0],
    },
];

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
    /// 三角形渲染管线与顶点缓冲。
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
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

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("main device"),
            ..Default::default()
        }))?;

        // 3. 用 surface 的首选格式配置交换链。
        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(RendererError::UnsupportedSurface)?;
        surface.configure(&device, &config);

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
                    visibility: ShaderStages::VERTEX,
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

        // 5. 渲染管线：三角形 + 相机 uniform。
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("triangle shader"),
            source: ShaderSource::Wgsl(include_str!("triangle.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("triangle pipeline layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("triangle pipeline"),
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
            depth_stencil: None,
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

        // 6. 三角形顶点缓冲。
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("triangle vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let vertex_count = VERTICES.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size: (width, height),
            camera_buffer,
            camera_bind_group,
            pipeline,
            vertex_buffer,
            vertex_count,
        })
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
    }

    /// 渲染一帧：写入相机 uniform，清屏，绘制三角形并呈现。
    pub fn render(&mut self, camera: &Camera) {
        // 每帧把相机数据写入 uniform 缓冲区。
        let uniform = CameraUniform::from_camera(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // 获取当前帧；surface 状态异常时跳过或重建交换链。
        let frame = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(frame)
            | CurrentSurfaceTexture::Suboptimal(frame) => frame,
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
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
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
                    ..Default::default()
                });

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                pass.draw(0..self.vertex_count, 0..1);
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
