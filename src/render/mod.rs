//! 渲染模块：负责 wgpu 初始化、surface 管理和每帧绘制。

use std::error::Error;
use std::fmt;
use std::sync::Arc;

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

use crate::camera::{Camera, CameraUniform};
use crate::mesh::{MeshLibrary, Vertex};
use crate::scene::Scene;

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
    /// 网格渲染管线。
    pipeline: wgpu::RenderPipeline,
    /// 深度缓冲（纹理 + 视图），随窗口尺寸重建。
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// 全局网格缓冲（永久驻留，所有注册网格合并），由 upload_meshes 维护。
    mesh_buffer: Option<MeshGpu>,
    /// 已上传的网格库版本，避免重复上传。
    mesh_uploaded_version: u64,
}

/// 资产库中单个网格在合并缓冲里的区间。
#[derive(Debug, Clone, Copy)]
struct MeshRange {
    index_offset: u32,
    index_count: u32,
}

/// 全局网格的 GPU 表示：合并后的顶点/索引缓冲，以及每个网格的区间。
/// 区间按 [`MeshKey`](crate::mesh::MeshKey) 的稠密编号索引（句柄即下标）。
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

        // 3.5 深度缓冲：管线与渲染通道都要用它来做正确的遮挡关系。
        let (depth_texture, depth_view) = create_depth_texture(&device, width, height);

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

        // 5. 物体数据：动态 uniform 缓冲布局（每个物体一个模型矩阵）。
        let object_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("object bind group layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });

        // 物体矩阵至少 64 字节，且动态偏移必须是设备对齐值的整数倍。
        let object_stride = device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(64);
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
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });

        // 6. 渲染管线：网格 + 相机/物体 uniform。
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh shader"),
            source: ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&object_bind_group_layout),
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
            pipeline,
            depth_texture,
            depth_view,
            mesh_buffer: None,
            mesh_uploaded_version: 0,
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

    /// 加载场景：按物体数量重建动态 uniform 缓冲（网格资产已在 `upload_meshes` 中常驻）。
    pub fn load_scene(&mut self, scene: &Scene) {
        // 按物体数量重建动态 uniform 缓冲与绑定组。
        let stride = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(64);
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
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });

        self.object_data_buffer = object_data_buffer;
        self.object_bind_group = object_bind_group;
        self.object_stride = stride as u32;
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

        // 每帧把物体世界矩阵写入动态 uniform 缓冲（步长 = object_stride）。
        if scene.object_count() > 0 {
            let stride = self.object_stride as usize;
            let mut bytes = vec![0u8; scene.object_count() * stride];
            for (i, (key, _)) in scene.objects().enumerate() {
                // 层级场景：世界矩阵由 Scene 沿祖先链累乘得到。
                let model = scene
                    .world_transform(key)
                    .expect("objects() 只产出存活节点，world_transform 必然有值");
                bytes[i * stride..i * stride + 64].copy_from_slice(bytemuck::bytes_of(&model));
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

        let Some(mesh_buffer) = &self.mesh_buffer else {
            return;
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

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, mesh_buffer.vertex_buffer.slice(..));
                pass.set_index_buffer(
                    mesh_buffer.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                // 每个物体：绑定它的世界矩阵（动态偏移），按句柄直取网格区间；
                // 纯分组节点（无网格）跳过。
                for (i, (_, object)) in scene.objects().enumerate() {
                    let Some(mesh_key) = object.mesh else { continue; };
                    let range = mesh_buffer.mesh_ranges[mesh_key.index()];
                    let offset = (i * self.object_stride as usize) as u32;
                    pass.set_bind_group(1, &self.object_bind_group, &[offset]);
                    pass.draw_indexed(
                        range.index_offset..range.index_offset + range.index_count,
                        0,
                        0..1,
                    );
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
