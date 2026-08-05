//! 环境子系统：转换编排。
//!
//! HDRI → 环境立方体贴图 + 辐照度图 + 镜面预过滤 mip 链 + BRDF LUT；
//! 按后端分流：Vulkan/Metal 走 GPU 计算，GL 等回退 CPU 转换 + 逐层上传。

use wgpu::{BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor, TextureViewDescriptor};

use super::{
    BRDF_LUT_SAMPLES, BRDF_LUT_SIZE, ENV_CUBEMAP_SIZE, EnvConversionPath, EnvironmentGpu,
    EnvironmentResources, IRRADIANCE_SAMPLES, IRRADIANCE_SIZE, PREFILTER_MIP_COUNT,
    PREFILTER_SAMPLES, PREFILTERED_SIZE, create_2d_texture, create_cube_texture,
    create_mip_cube_texture,
};
use crate::engine::core::environment::Environment;
use crate::engine::render::uniform::{EnvParams, PrefilterParams};

impl EnvironmentResources {
    /// 上传环境贴图（HDRI 等距矩形图）并转换成环境立方体贴图 + 辐照度图。
    ///
    /// 按启动时决定的路径转换：Vulkan/Metal 用 GPU 计算着色器，其余后端
    /// （GL 等 storage 数组纹理不可靠）回退 CPU 转换 + 逐层上传。
    pub(crate) fn convert(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &Environment,
    ) -> EnvironmentGpu {
        let face_size = ENV_CUBEMAP_SIZE;
        let irradiance_size = IRRADIANCE_SIZE;

        // 1. 按路径生成环境图、辐照度图、镜面预过滤图与 BRDF LUT。
        let (env_texture, irradiance_texture, prefiltered_texture, brdf_lut_texture) =
            match self.conversion_path {
                EnvConversionPath::Gpu => self.convert_gpu(device, queue, environment),
                EnvConversionPath::Cpu => {
                    // CPU 转换 + 逐层上传（write_texture 无 256 对齐要求）。
                    let cube_pixels = environment.to_cubemap(face_size);
                    let irradiance_pixels = Environment::irradiance_map(
                        &cube_pixels,
                        face_size,
                        irradiance_size,
                        IRRADIANCE_SAMPLES,
                    );
                    let prefiltered_mips = Environment::prefilter_map(
                        &cube_pixels,
                        face_size,
                        PREFILTERED_SIZE,
                        PREFILTER_MIP_COUNT,
                        PREFILTER_SAMPLES,
                    );
                    let brdf_pixels = Environment::brdf_lut(BRDF_LUT_SIZE, BRDF_LUT_SAMPLES);
                    (
                        create_cube_texture(
                            device,
                            queue,
                            face_size,
                            &cube_pixels,
                            "environment cubemap",
                        ),
                        create_cube_texture(
                            device,
                            queue,
                            irradiance_size,
                            &irradiance_pixels,
                            "irradiance cubemap",
                        ),
                        create_mip_cube_texture(
                            device,
                            queue,
                            PREFILTERED_SIZE,
                            PREFILTER_MIP_COUNT,
                            &prefiltered_mips,
                            "prefiltered cubemap",
                        ),
                        create_2d_texture(
                            device,
                            queue,
                            BRDF_LUT_SIZE,
                            BRDF_LUT_SIZE,
                            &brdf_pixels,
                            "brdf lut",
                        ),
                    )
                }
            };

        // 2. 立方体视图（采样用）。
        let env_cube_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let irradiance_cube_view = irradiance_texture.create_view(&TextureViewDescriptor {
            label: Some("irradiance cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let prefiltered_cube_view = prefiltered_texture.create_view(&TextureViewDescriptor {
            label: Some("prefiltered cubemap cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let brdf_lut_view = brdf_lut_texture.create_view(&TextureViewDescriptor {
            label: Some("brdf lut view"),
            ..Default::default()
        });

        // 3. 构建 mesh 管线 @group(4) 与天空盒的绑定组。
        let mesh_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("environment mesh bind group"),
            layout: &self.environment_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&irradiance_cube_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&env_cube_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.env_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&prefiltered_cube_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&brdf_lut_view),
                },
            ],
        });
        let skybox_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skybox bind group"),
            layout: &self.skybox_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&env_cube_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.env_params_buffer.as_entire_binding(),
                },
            ],
        });

        EnvironmentGpu {
            environment_texture: env_texture,
            environment_view: env_cube_view,
            irradiance_texture,
            irradiance_view: irradiance_cube_view,
            prefiltered_texture,
            prefiltered_view: prefiltered_cube_view,
            brdf_lut_texture,
            brdf_lut_view,
            sampler: self.env_sampler.clone(),
            mesh_bind_group,
            skybox_bind_group,
        }
    }

    /// 设置环境强度（IBL 系数）：`0` = 纯手动布光（环境图只当背景天空盒），
    /// `1` = 满环境光；可超 1 补亮。只写 intensity 字段，不重建任何资源。
    pub(crate) fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.env_params_buffer, 0, bytemuck::bytes_of(&intensity));
    }

    /// 设置 AgX 色调映射的 EV 窗口（场景级配置，默认与 Blender 一致）。
    /// 只写 min/max 两个字段，不重建任何资源。
    pub(crate) fn set_agx_range(&self, queue: &wgpu::Queue, min_ev: f32, max_ev: f32) {
        debug_assert!(min_ev < max_ev, "AgX EV 窗口要求 min_ev < max_ev");
        queue.write_buffer(
            &self.env_params_buffer,
            4,
            bytemuck::bytes_of(&[min_ev, max_ev]),
        );
    }

    /// GPU 路径：上传等距矩形源，计算 pass 产出环境图、辐照度图、
    /// 镜面预过滤 mip 链与 BRDF LUT。
    ///
    /// 只在 storage 数组纹理可靠的后端（Vulkan/Metal）调用；GL 后端在这里
    /// 会写入全零（见 docs/BUG.md），因此由调用方按 `conversion_path` 分流。
    fn convert_gpu(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environment: &Environment,
    ) -> (wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::Texture) {
        let face_size = ENV_CUBEMAP_SIZE;
        let irradiance_size = IRRADIANCE_SIZE;

        // 1. 等距矩形源纹理（RGBA32F 单层）。
        let src_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment equirect source"),
            size: wgpu::Extent3d {
                width: environment.width,
                height: environment.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut rgba = Vec::with_capacity((environment.width * environment.height * 4) as usize);
        for rgb in &environment.rgb {
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(environment.width * 16),
                rows_per_image: Some(environment.height),
            },
            wgpu::Extent3d {
                width: environment.width,
                height: environment.height,
                depth_or_array_layers: 1,
            },
        );
        let src_view = src_texture.create_view(&TextureViewDescriptor::default());

        // 2. 输出纹理：环境图 + 辐照度图（存储写入 + 采样双用途）。
        let env_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("environment cubemap"),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let env_storage_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap storage view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let env_cube_view = env_texture.create_view(&TextureViewDescriptor {
            label: Some("environment cubemap cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        let irradiance_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("irradiance cubemap"),
            size: wgpu::Extent3d {
                width: irradiance_size,
                height: irradiance_size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let irradiance_storage_view = irradiance_texture.create_view(&TextureViewDescriptor {
            label: Some("irradiance storage view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        });
        // 预过滤图（128×128×6，8 层 mip）与 BRDF LUT（128×128）。
        let prefiltered_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefiltered cubemap"),
            size: wgpu::Extent3d {
                width: PREFILTERED_SIZE,
                height: PREFILTERED_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: PREFILTER_MIP_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let brdf_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brdf lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let brdf_lut_storage_view = brdf_lut_texture.create_view(&TextureViewDescriptor::default());

        // 3. 四个计算 pass（拆开，保证"存储写入 → 采样读取"在 pass 边界同步）。
        {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("environment conversion encoder"),
            });

            // 3.1 equirect → cubemap。
            queue.write_buffer(
                &self.env_convert_params,
                0,
                bytemuck::bytes_of(&EnvParams {
                    size: face_size,
                    sample_count: 0,
                    _pad: [0; 2],
                }),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let convert_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("equirect convert bind group"),
                    layout: &self.env_convert_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.env_convert_params.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&src_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&env_storage_view),
                        },
                    ],
                });
                pass.set_pipeline(&self.env_convert_pipeline);
                pass.set_bind_group(0, &convert_bind_group, &[]);
                pass.dispatch_workgroups(face_size.div_ceil(8), face_size.div_ceil(8), 6);
            }

            // 3.2 cubemap → 辐照度图。
            queue.write_buffer(
                &self.irradiance_params,
                0,
                bytemuck::bytes_of(&EnvParams {
                    size: irradiance_size,
                    sample_count: IRRADIANCE_SAMPLES,
                    _pad: [0; 2],
                }),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let irradiance_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("irradiance bind group"),
                    layout: &self.irradiance_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.irradiance_params.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&env_cube_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&irradiance_storage_view),
                        },
                    ],
                });
                pass.set_pipeline(&self.irradiance_pipeline);
                pass.set_bind_group(0, &irradiance_bind_group, &[]);
                pass.dispatch_workgroups(
                    irradiance_size.div_ceil(8),
                    irradiance_size.div_ceil(8),
                    6,
                );
            }

            // 3.3 cubemap → 镜面预过滤 mip 链（每个 mip 一次 dispatch）。
            for mip in 0..PREFILTER_MIP_COUNT {
                let mip_size = PREFILTERED_SIZE >> mip;
                let prefiltered_storage_view =
                    prefiltered_texture.create_view(&TextureViewDescriptor {
                        label: Some("prefiltered storage view"),
                        dimension: Some(wgpu::TextureViewDimension::D2Array),
                        base_mip_level: mip,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(6),
                        ..Default::default()
                    });
                queue.write_buffer(
                    &self.prefilter_params[mip as usize],
                    0,
                    bytemuck::bytes_of(&PrefilterParams {
                        size: mip_size,
                        mip,
                        mip_count: PREFILTER_MIP_COUNT,
                        sample_count: PREFILTER_SAMPLES,
                    }),
                );
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let prefilter_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("prefilter bind group"),
                    layout: &self.prefilter_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: self.prefilter_params[mip as usize].as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&env_cube_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&prefiltered_storage_view),
                        },
                    ],
                });
                pass.set_pipeline(&self.prefilter_pipeline);
                pass.set_bind_group(0, &prefilter_bind_group, &[]);
                pass.dispatch_workgroups(mip_size.div_ceil(8), mip_size.div_ceil(8), 6);
            }

            // 3.4 BRDF 积分查找表。
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                let brdf_bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some("brdf lut bind group"),
                    layout: &self.brdf_lut_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut_storage_view),
                    }],
                });
                pass.set_pipeline(&self.brdf_lut_pipeline);
                pass.set_bind_group(0, &brdf_bind_group, &[]);
                pass.dispatch_workgroups(BRDF_LUT_SIZE.div_ceil(8), BRDF_LUT_SIZE.div_ceil(8), 1);
            }
            queue.submit([encoder.finish()]);
        }

        (
            env_texture,
            irradiance_texture,
            prefiltered_texture,
            brdf_lut_texture,
        )
    }
}
