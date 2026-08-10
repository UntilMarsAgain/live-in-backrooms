//! 每帧渲染：执行一条 [`RenderCommand`]（只带资源库句柄的渲染指令），
//! 绘制时拿句柄向 `GpuManager` 取 GPU 缓冲/视图（缺失自愈上传），
//! 场景 pass 绘制到 HDR 目标、色调映射 blit pass 写交换链并呈现。

use std::mem::size_of;

use glam::Mat3;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    CurrentSurfaceTexture, LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor,
    StoreOp, TextureViewDescriptor,
};

use crate::engine::core::asset::AssetManager;
use crate::engine::core::camera::CameraUniform;
use crate::engine::core::frame::RenderCommand;
use crate::engine::render::asset::GpuManager;
use crate::engine::render::debug;
use crate::engine::render::init::{create_depth_texture, create_hdr_texture};
use crate::engine::render::uniform::{LightCountUniform, ObjectData, collect_light_uniforms};
use crate::engine::render::{CLEAR_COLOR, Renderer};

impl Renderer {
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
        // HDR 中间目标同样随尺寸重建；blit 绑定组引用它的视图，需要一并重建。
        (self.hdr_texture, self.hdr_view) = create_hdr_texture(&self.device, width, height);
        self.blit_bind_group = self.blit_resources.create_bind_group(
            &self.device,
            &self.hdr_view,
            &self.environment_resources.env_params_buffer,
        );
    }

    /// 渲染一帧：执行一条渲染指令（只带资源库句柄），写入 uniform，绘制时
    /// 拿句柄向 `GpuManager` 取 GPU 数据（`gpu`/`assets` 是资源库本身，
    /// 不是指令内容），场景 pass 绘制到 HDR 中间目标，blit 色调映射后写
    /// 交换链并呈现。
    ///
    /// 物体数据与材质绑定组**每帧**从指令构建（物体数量小，可接受；
    /// 后续优化：按材质缓存绑定组）。
    pub fn render(
        &mut self,
        command: &RenderCommand,
        gpu: &mut GpuManager,
        assets: &mut AssetManager,
    ) {
        let Some(camera) = &command.camera else {
            return;
        };
        // 相机 uniform。
        let uniform = CameraUniform::from_camera(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // 灯光：把指令里的语义灯光打包成 uniform（方向光 + 就近局部光），
        // 数量 uniform + storage 数组。
        let camera_position = command.camera.map(|c| c.position()).unwrap_or_default();
        let light_uniforms = collect_light_uniforms(&command.lights, camera_position);
        let light_count = LightCountUniform {
            count: light_uniforms.len() as u32,
            _pad: [0; 3],
        };
        self.queue.write_buffer(
            &self.light_count_buffer,
            0,
            bytemuck::bytes_of(&light_count),
        );
        if !light_uniforms.is_empty() {
            self.queue.write_buffer(
                &self.light_storage_buffer,
                0,
                bytemuck::cast_slice(&light_uniforms),
            );
        }

        // 物体数据：容量不足时重建 storage 缓冲（紧凑数组，实例下标 = 数组下标）。
        let entry_size = size_of::<ObjectData>();
        let needed = (command.objects.len() as u64).max(1) * entry_size as u64;
        if self.object_data_buffer.size() < needed {
            self.object_data_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("object data buffer"),
                size: needed,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.object_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("object data bind group"),
                layout: &self.object_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    // 绑定整个缓冲：着色器按实例索引访问，无动态偏移。
                    resource: self.object_data_buffer.as_entire_binding(),
                }],
            });
        }
        if !command.objects.is_empty() {
            let mut bytes = vec![0u8; command.objects.len() * entry_size];
            for (i, object) in command.objects.iter().enumerate() {
                let model = object.world_matrix;
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
                bytes[i * entry_size..(i + 1) * entry_size]
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

            // ---- Pass 1：场景 pass，渲染到 HDR 中间目标 ----
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("scene pass (HDR)"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &self.hdr_view,
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

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                // 物体数据 storage 数组：整组绑定一次，逐物体不再切换。
                pass.set_bind_group(1, &self.object_bind_group, &[]);
                pass.set_bind_group(2, &self.light_bind_group, &[]);
                pass.set_bind_group(4, &self.environment.mesh_bind_group, &[]);

                // 每个物体：句柄 → 资源库取 GPU 缓冲/贴图视图（缺失自愈上传），
                // 用实例区间 i..i+1 编码物体索引（instance_index = i）。
                // 材质绑定组每帧从解析出的视图构建（后续优化：按材质缓存）。
                for (i, object) in command.objects.iter().enumerate() {
                    let base_view = object
                        .material
                        .base_color_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_white_view.clone());
                    let mr_view = object
                        .material
                        .metallic_roughness_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_white_view.clone());
                    let normal_view = object
                        .material
                        .normal_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_normal_view.clone());
                    let material_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                        label: Some("material bind group"),
                        layout: &self.texture_bind_group_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&base_view),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                            },
                            BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&mr_view),
                            },
                            BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::TextureView(&normal_view),
                            },
                        ],
                    });
                    let Some(mesh_gpu) = gpu.mesh_gpu(object.mesh, assets) else {
                        eprintln!(
                            "渲染违例：场景引用了无效的网格句柄 {:?}，跳过绘制",
                            object.mesh
                        );
                        continue;
                    };
                    pass.set_vertex_buffer(0, mesh_gpu.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        mesh_gpu.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_bind_group(3, &material_bind_group, &[]);
                    pass.draw_indexed(0..mesh_gpu.index_count, 0, i as u32..i as u32 + 1);
                }

                // 调试线框：从指令里的语义灯光/碰撞箱生成顶点并上传绘制
                //（深度 Always + 不写深度，被遮挡也可见；仅开启时构建）。
                if command.show_light_debug {
                    let vertices = debug::build_light_gizmos_data(&command.lights);
                    self.light_gizmos
                        .upload(&self.device, &self.queue, &vertices);
                    self.light_gizmos.draw(&mut pass, &self.camera_bind_group);
                }
                if command.show_collision_debug {
                    let vertices = debug::build_collision_gizmos_data(&command.colliders);
                    self.collision_gizmos
                        .upload(&self.device, &self.queue, &vertices);
                    self.collision_gizmos
                        .draw(&mut pass, &self.camera_bind_group);
                }
            }

            // ---- Pass 2：色调映射 blit，HDR 目标 → 交换链 ----
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("tone map blit pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            // 场景 pass 覆盖全屏，Load 值无所谓；Clear 保持确定性。
                            load: LoadOp::Clear(CLEAR_COLOR),
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                self.blit_resources.draw(&mut pass, &self.blit_bind_group);
            }

            self.queue.submit([encoder.finish()]);
        }

        self.queue.present(frame);
    }
}
