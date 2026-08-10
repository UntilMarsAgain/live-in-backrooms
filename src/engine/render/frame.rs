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
use crate::engine::render::init::{
    create_depth_texture, create_hdr_msaa_texture, create_hdr_texture,
};
use crate::engine::render::uniform::{LightCountUniform, ObjectData, collect_light_uniforms};
use crate::engine::render::{CLEAR_COLOR, MSAA_SAMPLES, Renderer};

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
        (self.depth_texture, self.depth_view) =
            create_depth_texture(&self.device, width, height, MSAA_SAMPLES);
        // HDR 中间目标与 MSAA 附件随尺寸重建；blit 绑定组引用 HDR 视图。
        (self.hdr_texture, self.hdr_view) = create_hdr_texture(&self.device, width, height);
        (self.hdr_msaa_texture, self.hdr_msaa_view) =
            create_hdr_msaa_texture(&self.device, width, height, MSAA_SAMPLES);
        self.bloom.resize(&self.device, width, height);
        self.blit_bind_group = self.blit_resources.create_bind_group(
            &self.device,
            &self.hdr_view,
            &self.environment_resources.env_params_buffer,
            Some(&self.bloom.mip_chain[0].1),
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
        // 实例按"组 → 组内实例"顺序连续编号，绘制时用组区间画一次。
        let entry_size = size_of::<ObjectData>();
        let instance_count: usize = command.meshes.iter().map(|g| g.instances.len()).sum();
        let needed = (instance_count as u64).max(1) * entry_size as u64;
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
        if instance_count > 0 {
            let mut bytes = vec![0u8; instance_count * entry_size];
            let mut i = 0;
            for group in &command.meshes {
                for model in &group.instances {
                    // 法线矩阵 = 模型上三角的逆转置，非等比缩放下法线方向才正确。
                    let m = Mat3::from_mat4(*model).inverse().transpose();
                    let cols = m.to_cols_array();
                    let normal_matrix = [
                        [cols[0], cols[1], cols[2], 0.0],
                        [cols[3], cols[4], cols[5], 0.0],
                        [cols[6], cols[7], cols[8], 0.0],
                    ];
                    let data = ObjectData {
                        model: *model,
                        normal_matrix,
                        base_color: group.material.base_color,
                        metallic: group.material.metallic_factor,
                        roughness: group.material.roughness_factor,
                        emissive: [
                            group.material.emissive_factor[0],
                            group.material.emissive_factor[1],
                            group.material.emissive_factor[2],
                            1.0,
                        ],
                        _pad: [0.0; 2],
                    };
                    bytes[i * entry_size..(i + 1) * entry_size]
                        .copy_from_slice(bytemuck::bytes_of(&data));
                    i += 1;
                }
            }
            self.queue.write_buffer(&self.object_data_buffer, 0, &bytes);
            // 一次性输出实例化分组统计，便于 demo 手动验证合并是否生效
            //（组数远小于实例数 = 同网格同材质合并成功）。
            static PRINTED_INSTANCE_STATS: std::sync::Once = std::sync::Once::new();
            PRINTED_INSTANCE_STATS.call_once(|| {
                tracing::debug!(
                    "渲染指令实例化分组：{} 个绘制组 / {} 个实例",
                    command.meshes.len(),
                    instance_count
                );
            });
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

            // ---- Pass 1：场景 pass，渲染到 MSAA 附件并 resolve 到 HDR ----
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("scene pass (HDR)"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        // 渲染到 4x 附件，resolve 到单采样 HDR（bloom/blit 消费）。
                        view: &self.hdr_msaa_view,
                        depth_slice: None,
                        resolve_target: Some(&self.hdr_view),
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

                // 每个绘制组：句柄 → 资源库取 GPU 缓冲/贴图视图（缺失自愈上传），
                // 组内所有实例一次 draw_indexed，实例区间 = 组在 object_data
                // 数组中的连续段（first_instance 随组累加）。
                // 材质绑定组每帧从解析出的视图构建（后续优化：按材质缓存）。
                let mut first_instance = 0u32;
                for group in &command.meshes {
                    let base_view = group
                        .material
                        .base_color_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_white_view.clone());
                    let mr_view = group
                        .material
                        .metallic_roughness_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_white_view.clone());
                    let normal_view = group
                        .material
                        .normal_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        .unwrap_or_else(|| self.default_normal_view.clone());
                    let emissive_view = group
                        .material
                        .emissive_texture
                        .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                        // 无 emissive 贴图时绑白色：因子直接生效
                        //（emissive_factor = 0 时仍不发光，因子本身控制开关）。
                        .unwrap_or_else(|| self.default_white_view.clone());
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
                            BindGroupEntry {
                                binding: 4,
                                resource: wgpu::BindingResource::TextureView(&emissive_view),
                            },
                        ],
                    });
                    let Some(mesh_gpu) = gpu.mesh_gpu(group.mesh, assets) else {
                        tracing::error!(
                            "渲染违例：场景引用了无效的网格句柄 {:?}，跳过绘制",
                            group.mesh
                        );
                        first_instance += group.instances.len() as u32;
                        continue;
                    };
                    pass.set_vertex_buffer(0, mesh_gpu.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        mesh_gpu.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_bind_group(3, &material_bind_group, &[]);
                    let instances = group.instances.len() as u32;
                    pass.draw_indexed(
                        0..mesh_gpu.index_count,
                        0,
                        first_instance..first_instance + instances,
                    );
                    first_instance += instances;
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

            // ---- Pass 1.5：Bloom（辉光）----
            // 场景 pass 已结束（HDR 目标可采样），提取高亮 + 多级下采样/
            // 上采样合并；blit pass 会把 bloom 结果加回 HDR。
            self.bloom
                .run(&self.device, &self.queue, &mut encoder, &self.hdr_view);

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
