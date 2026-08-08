//! 每帧渲染：相机/灯光/物体数据提交、场景 pass 绘制到 HDR 目标、
//! 色调映射 blit pass 写交换链并呈现。

use glam::Mat3;
use wgpu::{
    CommandEncoderDescriptor, CurrentSurfaceTexture, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDescriptor, StoreOp, TextureViewDescriptor,
};

use crate::engine::core::camera::{Camera, CameraUniform};
use crate::engine::core::asset::AssetManager;
use crate::engine::render::init::{create_depth_texture, create_hdr_texture};
use crate::engine::render::uniform::{
    LIGHT_CAPACITY, LightCountUniform, ObjectData, collect_lights,
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
        // HDR 中间目标同样随尺寸重建；blit 绑定组引用它的视图，需要一并重建。
        (self.hdr_texture, self.hdr_view) = create_hdr_texture(&self.device, width, height);
        self.blit_bind_group = self.blit_resources.create_bind_group(
            &self.device,
            &self.hdr_view,
            &self.environment_resources.env_params_buffer,
        );
    }

    /// 渲染一帧：写入相机与物体 uniform，清屏，绘制场景中所有物体并呈现。
    ///
    /// `show_light_debug` / `show_collision_debug` 为 `true` 时，在网格之后
    /// 叠加对应的调试线框（顶点在 `load_scene` 时已上传，见 [`debug`] 模块）。
    ///
    /// 两段式：场景 pass 把天空盒/网格/线框画进 HDR 中间目标（原始辐射值，
    /// 可 >1），blit pass 采样它做 AgX 色调映射后写交换链。色调映射全帧
    /// 只做一次，后处理（Bloom/SSAO/SSR 等）以后可以插在两步之间。
    pub fn render(
        &mut self,
        camera: &Camera,
        scene: &Scene,
        assets: &mut AssetManager,
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

        // 每帧把物体数据写入 storage 数组（紧凑布局，无对齐步长填充）。
        if scene.object_count() > 0 {
            let entry_size = size_of::<ObjectData>();
            let mut bytes = vec![0u8; scene.object_count() * entry_size];
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

                // 每个物体：句柄 → 资产管理器取独立 GPU 缓冲（每网格一份），
                // 用实例区间 i..i+1 编码物体索引（instance_index = i）。
                // 有效句柄未上传时**立即上传**（渲染违例，不能静默跳过）；
                // 句柄彻底无效（已卸载/不存在）属于运行错误，终端报错后跳过。
                // 非网格节点（分组、未来的灯光/相机等）同样跳过。
                for (i, (_, object)) in scene.objects().enumerate() {
                    let Some(handle) = object.mesh_handle() else {
                        continue;
                    };
                    let Some(mesh_gpu) = assets.ensure_meshes_gpu(handle) else {
                        eprintln!(
                            "渲染违例：场景引用了无效的网格句柄 {handle:?}，跳过绘制"
                        );
                        continue;
                    };
                    pass.set_vertex_buffer(0, mesh_gpu.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        mesh_gpu.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_bind_group(3, &self.material_bind_groups[i], &[]);
                    pass.draw_indexed(0..mesh_gpu.index_count, 0, i as u32..i as u32 + 1);
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
