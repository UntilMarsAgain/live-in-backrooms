//! 每帧渲染：相机/灯光/物体数据提交、主 pass 绘制与呈现。

use glam::Mat3;
use wgpu::{
    CommandEncoderDescriptor, CurrentSurfaceTexture, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureViewDescriptor,
};

use crate::engine::core::camera::{Camera, CameraUniform};
use crate::engine::render::init::create_depth_texture;
use crate::engine::render::uniform::{
    LIGHT_CAPACITY, LightCountUniform, ObjectDataUniform, collect_lights,
};
use crate::engine::render::{CLEAR_COLOR, Renderer};
use crate::engine::scene::Scene;

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
    }

    /// 渲染一帧：写入相机与物体 uniform，清屏，绘制场景中所有物体并呈现。
    ///
    /// `show_light_debug` / `show_collision_debug` 为 `true` 时，在网格之后
    /// 叠加对应的调试线框（顶点在 `load_scene` 时已上传，见 [`debug`] 模块）。
    pub fn render(
        &mut self,
        camera: &Camera,
        scene: &Scene,
        show_light_debug: bool,
        show_collision_debug: bool,
    ) {
        // 每帧把相机数据写入 uniform 缓冲区。
        let uniform = CameraUniform::from_camera(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // 每帧收集灯光：所有方向光 + 离相机最近的局部光（场景灯光缓存），
        // 写入数量 uniform 与 storage 数组。手电筒等动态光以后并入同一列表。
        let lights = collect_lights(scene, camera.position());
        debug_assert!(lights.len() <= LIGHT_CAPACITY);
        let light_count = LightCountUniform {
            count: lights.len() as u32,
            _pad: [0; 3],
        };
        self.queue.write_buffer(
            &self.light_count_buffer,
            0,
            bytemuck::bytes_of(&light_count),
        );
        if !lights.is_empty() {
            self.queue
                .write_buffer(&self.light_storage_buffer, 0, bytemuck::cast_slice(&lights));
        }

        // 每帧把物体世界矩阵 + 法线矩阵写入动态 uniform 缓冲（步长 = object_stride）。
        if scene.object_count() > 0 {
            let stride = self.object_stride as usize;
            let entry_size = std::mem::size_of::<ObjectDataUniform>();
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
                let data = ObjectDataUniform {
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
                        let Some(mesh_key) = object.mesh_key() else {
                            continue;
                        };
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

                // 调试线框：顶点已上传，这里只按开关绘制
                //（深度 Always + 不写深度，被遮挡也可见；关闭时跳过）。
                if show_light_debug {
                    self.light_gizmos.draw(&mut pass, &self.camera_bind_group);
                }
                if show_collision_debug {
                    self.collision_gizmos
                        .draw(&mut pass, &self.camera_bind_group);
                }
            }

            self.queue.submit([encoder.finish()]);
        }

        self.queue.present(frame);
    }
}
