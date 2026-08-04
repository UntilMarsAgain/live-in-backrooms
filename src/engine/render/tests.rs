//! 渲染模块测试：naga WGSL 校验 + 无头 GPU 冒烟 + 端到端像素验证。
//!
//! GPU 相关测试在 llvmpipe 软件渲染上并行跑会段错误，统一用
//! `cargo test -- --test-threads=1`；无 GPU 环境自动跳过并打印原因。

use super::*;
use super::environment::create_cube_texture;


    /// cargo build 不编译 WGSL，运行时错误会晚暴露；这里用 naga 提前校验。
    fn validate_wgsl(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL 应能解析");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator.validate(&module).expect("WGSL 应通过校验");
    }

    #[test]
    fn mesh_shader_compiles() {
        validate_wgsl(include_str!("mesh.wgsl"));
    }

    #[test]
    fn environment_shader_compiles() {
        validate_wgsl(include_str!("environment.wgsl"));
    }

    /// 无窗口设备：请求适配器并创建设备（含 max_bind_groups 8 与
    /// FLOAT32_FILTERABLE 特性）。失败时打印原因并返回 `None`（CI 无 GPU 可跳过）。
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue, bool, EnvConversionPath)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .inspect_err(|e| eprintln!("headless 测试：请求适配器失败（{e}），跳过"))
        .ok()?;
        let float32_filterable = adapter
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE);
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("smoke test device"),
            required_features: if float32_filterable {
                wgpu::Features::FLOAT32_FILTERABLE
            } else {
                wgpu::Features::empty()
            },
            required_limits: wgpu::Limits {
                max_bind_groups: 8,
                ..Default::default()
            },
            ..Default::default()
        }))
        .inspect_err(|e| eprintln!("headless 测试：设备创建失败（{e}），跳过"))
        .ok()?;
        let conversion_path = match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => EnvConversionPath::Gpu,
            _ => EnvConversionPath::Cpu,
        };
        Some((device, queue, float32_filterable, conversion_path))
    }


    /// 无窗口冒烟测试：不创建 surface，直接请求适配器/设备，验证环境资源创建、
    /// 计算转换与天空盒渲染不触发 wgpu 校验错误；无 GPU 环境（如 CI）则跳过。
    #[test]
    fn environment_headless_smoke() {
        let Some((device, queue, float32_filterable, conversion_path)) = headless_device() else {
            return;
        };

        // mesh 着色器声明了 @group(4)：校验它不超出 max_bind_groups 限制。
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some("smoke mesh shader"),
            source: ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });

        // 相机绑定组布局 + uniform（天空盒管线需要）。
        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("smoke camera layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("smoke camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY,
                position: glam::Vec3::ZERO,
                _padding: 0,
                inverse_view_proj: glam::Mat4::IDENTITY,
            }),
        );
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("smoke camera bind group"),
            layout: &camera_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // 环境资源：布局、计算管线、天空盒管线、默认绑定组。
        let resources = EnvironmentResources::new(
            &device,
            &queue,
            &camera_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            float32_filterable,
            conversion_path,
        );

        // 转换一个 2×1 的微型 HDR（左红右绿），验证计算管线与绑定组创建。
        let env = super::Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        };
        let gpu_env = resources.convert(&device, &queue, &env);

        // 天空盒渲染到离屏纹理，验证渲染管线 + 绑定组 + 实际绘制。
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke color texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = create_depth_texture(&device, 4, 4);
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("smoke encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("smoke pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(CLEAR_COLOR),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.1,
                    depth_ops: Some(wgpu::Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&resources.skybox_pipeline);
            pass.set_bind_group(0, &camera_bind_group, &[]);
            pass.set_bind_group(1, &gpu_env.skybox_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        // 等待 GPU 完成，确保编码/提交阶段没有触发校验错误。
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll 应成功");
    }


    /// 采样验证：已知全红立方体贴图经天空盒管线渲染到离屏，读回应偏红。
    /// （绕开 copy_texture_to_buffer 拷数组纹理的路径，直接验证"上传→采样"。）
    #[test]
    fn skybox_sampling_verifies_texture_content() {
        let Some((device, queue, float32_filterable, conversion_path)) = headless_device() else {
            return;
        };

        // 相机绑定组（天空盒需要 camera uniform）。
        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("verify camera layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("verify camera buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY,
                position: glam::Vec3::ZERO,
                _padding: 0,
                inverse_view_proj: glam::Mat4::IDENTITY,
            }),
        );
        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("verify camera bind group"),
            layout: &camera_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let resources = EnvironmentResources::new(
            &device,
            &queue,
            &camera_layout,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            float32_filterable,
            conversion_path,
        );

        // 1) 已知全红 cube（4×4×6）→ 天空盒渲染 → 应偏红。
        let known: Vec<[f32; 4]> = vec![[1.0, 0.0, 0.0, 1.0]; (4 * 4 * 6) as usize];
        let known_tex = create_cube_texture(&device, &queue, 4, &known, "known red cube");
        let known_view = known_tex.create_view(&TextureViewDescriptor {
            label: Some("known red cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let known_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("verify skybox bind group"),
            layout: &resources.skybox_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&known_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&resources.env_sampler),
                },
            ],
        });
        let data = render_skybox_rgb(&device, &queue, &resources, &camera_bind_group, &known_bind_group);
        let mut max_r = 0u8;
        for chunk in data.chunks_exact(4) {
            max_r = max_r.max(chunk[0]);
        }
        eprintln!("天空盒渲染读回最大 R 分量：{max_r}");
        assert!(
            max_r > 128,
            "已知全红 cube 经天空盒渲染后 R 分量过低（上传或采样失败）"
        );

        // 2) 真实 HDR → convert（CPU 转换 + 逐层上传）→ 天空盒渲染 → 非黑。
        let env = Environment::from_hdr_file(std::path::Path::new("assets/environments/test.hdr")).unwrap_or_else(|_| Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        });
        let gpu_env = resources.convert(&device, &queue, &env);
        let data = render_skybox_rgb(
            &device,
            &queue,
            &resources,
            &camera_bind_group,
            &gpu_env.skybox_bind_group,
        );
        let mut sum = 0u32;
        for chunk in data.chunks_exact(4) {
            sum += chunk[0] as u32 + chunk[1] as u32 + chunk[2] as u32;
        }
        let avg = sum as f32 / (data.len() / 4) as f32;
        eprintln!("真实 HDR 天空盒渲染平均 RGB：{avg:.1}");
        assert!(
            avg > 20.0,
            "真实 HDR 环境转换后天空盒渲染仍接近全黑（端到端链路失败）"
        );
    }

    /// 把天空盒渲染到 4×4 离屏 Rgba8UnormSrgb 并读回像素字节。
    fn render_skybox_rgb(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resources: &EnvironmentResources,
        camera_bind_group: &wgpu::BindGroup,
        skybox_bind_group: &wgpu::BindGroup,
    ) -> Vec<u8> {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("verify color"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = create_depth_texture(device, 4, 4);
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("verify encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("verify pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.1,
                    depth_ops: Some(wgpu::Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&resources.skybox_pipeline);
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_bind_group(1, skybox_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        let aligned_row = 256u32; // 4 像素 × 4 字节 = 16，按 copy 要求对齐到 256
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("verify readback"),
            size: (aligned_row * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("verify readback encoder"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_row),
                    rows_per_image: Some(4),
                },
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll 应成功");
        rx.recv().expect("map 回调应触发").expect("map 应成功");
        let data = slice.get_mapped_range().expect("取范围应成功");
        data[..(4 * 4 * 4) as usize].to_vec()
    }
