//! 渲染模块测试：naga WGSL 校验 + 无头 GPU 冒烟 + 端到端像素验证。
//!
//! GPU 相关测试统一用 `gpu_` 前缀命名：默认全量跑，**必须有 GPU**——
//! 无可用适配器时直接断言失败（防止假绿），无 GPU 的机器必须显式用
//! `cargo test -- --skip gpu_` 跳过它们。

use super::blit::BlitResources;
use super::environment::create_cube_texture;
use super::init::{create_depth_texture, create_hdr_texture};
use super::uniform::EnvironmentParams;
use super::*;
use crate::engine::core::camera::CameraUniform;
use crate::engine::core::environment::Environment;
use std::mem::size_of;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    DeviceDescriptor, LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages, StoreOp,
    TextureViewDescriptor,
};

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
    validate_wgsl(include_str!("environment/environment.wgsl"));
}

#[test]
fn debug_shader_compiles() {
    validate_wgsl(include_str!("debug/debug.wgsl"));
}

#[test]
fn blit_shader_compiles() {
    validate_wgsl(include_str!("blit.wgsl"));
}

/// 无窗口设备：请求适配器并创建设备（含 max_bind_groups 8 与
/// FLOAT32_FILTERABLE 特性）。失败时打印原因并返回 `None`。
fn headless_device() -> Option<(wgpu::Device, wgpu::Queue, bool)> {
    // 与运行时一致：只启用 PRIMARY 后端（无 OpenGL），避免踩 GL 后端的
    // 数组纹理读回 bug（见 docs/BUG.md）。
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
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
    Some((device, queue, float32_filterable))
}

/// 取无头 GPU 设备；**无可用适配器直接断言失败**——GPU 测试必须在有 GPU 的
/// 环境跑，无 GPU 的机器应显式用 `cargo test -- --skip gpu_` 跳过。
/// 绝不静默"通过"（否则没跑也算绿，正是这种假绿掩盖过真实 bug）。
fn require_headless_device() -> (wgpu::Device, wgpu::Queue, bool) {
    headless_device().unwrap_or_else(|| {
        panic!(
            "需要 GPU：无可用适配器。无 GPU 的机器请显式用 \
             `cargo test -- --skip gpu_` 跳过 GPU 测试，而不是让它们假装通过"
        )
    })
}

/// 无窗口冒烟测试：不创建 surface，直接请求适配器/设备，验证环境资源创建、
/// 计算转换与天空盒渲染不触发 wgpu 校验错误；无 GPU 环境（如 CI）则跳过。
#[test]
fn gpu_environment_headless_smoke() {
    let (device, queue, float32_filterable) = require_headless_device();

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
        super::HDR_FORMAT,
        float32_filterable,
    );

    // 转换一个 2×1 的微型 HDR（左红右绿），验证计算管线与绑定组创建。
    let env = Environment {
        width: 2,
        height: 1,
        rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };
    let gpu_env = resources.convert(&device, &queue, &env);

    // 天空盒渲染到离屏纹理，验证渲染管线 + 绑定组 + 实际绘制。
    // 天空盒管线现在输出到 HDR 中间目标（原始辐射值）。
    let (_, color_view) = create_hdr_texture(&device, 4, 4);
    let depth = create_depth_texture(&device, 4, 4);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("smoke encoder"),
    });
    {
        // 只清屏不绘制：pass 存活到块尾以正确结束渲染通道。
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

/// 无窗口冒烟测试：灯光调试线框管线的创建与绘制不触发 wgpu 校验错误。
#[test]
fn gpu_light_debug_gizmos_headless_smoke() {
    let (device, queue, _) = require_headless_device();

    // 相机绑定组（调试管线复用 @group(0) 的相机 uniform 布局）。
    let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("debug smoke camera layout"),
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
        label: Some("debug smoke camera buffer"),
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
        label: Some("debug smoke camera bind group"),
        layout: &camera_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    // 调试管线 + 两条线段（4 个顶点）绘制到离屏纹理。
    let mut gizmos =
        super::debug::LineGizmos::new(&device, &camera_layout, wgpu::TextureFormat::Rgba8UnormSrgb);
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("debug smoke color texture"),
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
        label: Some("debug smoke encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("debug smoke pass"),
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
        let vertices = [
            super::debug::DebugVertex {
                position: [0.0, 0.0, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            super::debug::DebugVertex {
                position: [1.0, 0.0, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            super::debug::DebugVertex {
                position: [0.0, 0.0, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            super::debug::DebugVertex {
                position: [0.0, 1.0, 0.0],
                color: [0.0, 1.0, 0.0],
            },
        ];
        gizmos.upload(&device, &queue, &vertices);
        gizmos.draw(&mut pass, &camera_bind_group);
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
fn gpu_skybox_sampling_verifies_texture_content() {
    let (device, queue, float32_filterable) = require_headless_device();

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
        super::HDR_FORMAT,
        float32_filterable,
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
            BindGroupEntry {
                binding: 2,
                resource: resources.env_params_buffer.as_entire_binding(),
            },
        ],
    });
    let data = render_skybox_rgb(
        &device,
        &queue,
        &resources,
        &camera_bind_group,
        &known_bind_group,
    );
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
    let env =
        Environment::from_hdr_file(std::path::Path::new("test/test.hdr")).unwrap_or_else(|_| {
            Environment {
                width: 2,
                height: 1,
                rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }
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

/// GPU 路径：镜面预过滤 mip 链与 BRDF LUT 必须非黑。
///
/// 防回归：预过滤参数缓冲若在循环里复用同一块（`queue.write_buffer` 先于
/// `submit()` 执行），除顶层外的 mip 全部提前 return，预过滤图基本全黑。
#[test]
fn gpu_specular_ibl_outputs_nonblack() {
    let (device, queue, float32_filterable) = require_headless_device();

    let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("specular smoke camera layout"),
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
        label: Some("specular smoke camera buffer"),
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
    let resources = EnvironmentResources::new(
        &device,
        &queue,
        &camera_layout,
        super::HDR_FORMAT,
        float32_filterable,
    );

    // 红绿 2×1 环境图：预过滤 mip 0 与 BRDF LUT 都应有非零分量。
    let env = Environment {
        width: 2,
        height: 1,
        rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    };
    let gpu_env = resources.convert(&device, &queue, &env);

    let pre = read_texture_rgba32f(&device, &queue, &gpu_env.prefiltered_texture, 0, 0, 8, 8);
    let max_pre = pre
        .iter()
        .fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
    eprintln!("预过滤 mip0 读回最大分量：{max_pre}");
    assert!(max_pre > 0.0, "预过滤图 mip 0 全黑（参数缓冲复用 bug？）");

    let brdf = read_texture_rgba32f(&device, &queue, &gpu_env.brdf_lut_texture, 0, 0, 8, 8);
    let max_brdf = brdf.iter().fold(0.0f32, |m, p| m.max(p[0]).max(p[1]));
    eprintln!("BRDF LUT 读回最大分量：{max_brdf}");
    assert!(max_brdf > 0.0, "BRDF LUT 全黑");
}

/// 无窗口验证：HDR 中间目标清成白色（radiance=1）后经 blit 色调映射，
/// 写 Rgba8UnormSrgb 应仍是高亮——证明"HDR 值 → 可显示"的最后一环工作。
#[test]
fn gpu_blit_tone_maps_hdr_radiance() {
    let (device, queue, _) = require_headless_device();

    // 环境参数（AgX EV 窗口默认值）：blit 绑定组需要这块 uniform。
    let env_params = device.create_buffer(&BufferDescriptor {
        label: Some("blit test env params"),
        size: size_of::<EnvironmentParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &env_params,
        0,
        bytemuck::bytes_of(&EnvironmentParams::default()),
    );

    // HDR 目标清成白色：场景 pass 的占位（真实场景会画物体/天空盒）。
    let (_hdr_texture, hdr_view) = create_hdr_texture(&device, 4, 4);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blit test clear encoder"),
    });
    {
        // 只清屏不绘制：pass 存活到块尾以正确结束渲染通道。
        let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("blit test clear pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::WHITE),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        // 不画任何东西：只清屏。
    }
    queue.submit([encoder.finish()]);

    // blit → Rgba8UnormSrgb，读回。
    let blit = BlitResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let blit_bind_group = blit.create_bind_group(&device, &hdr_view, &env_params);
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("blit test color"),
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
    let color_view = color_texture.create_view(&TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("blit test encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("blit test pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        blit.draw(&mut pass, &blit_bind_group);
    }
    queue.submit([encoder.finish()]);

    let data = read_texture_rgba8(&device, &queue, &color_texture, 4, 4);
    let max = data
        .chunks_exact(4)
        .map(|p| p[0].max(p[1]).max(p[2]))
        .max()
        .unwrap_or(0);
    eprintln!("纯白 HDR blit 后最大分量：{max}");
    assert!(
        max > 180,
        "白色 HDR（radiance=1）经 blit 应映射为高亮，实际 {max}"
    );
}

/// GpuManager：`sync` 只上传 CPU 侧 Pinned 的条目；`unpin` 是软释放——
/// 已上传的条目不会在 sync 时被回收（回收归 `gc`）。
#[test]
fn gpu_manager_unpin_is_soft_release() {
    let (device, queue, _) = require_headless_device();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let mut assets = crate::engine::core::asset::AssetManager::new(
        crate::engine::core::resource::MergedResourceSpace::new(std::env::temp_dir()),
    );
    let mut gpu = crate::engine::render::asset::GpuManager::new(device, queue);
    let mesh = assets.register(crate::engine::Mesh::cube());

    // 未 pin：sync 后不应有显存资源。
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(mesh).is_none());

    // pin + sync：上传。
    assert!(assets.pin(mesh));
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(mesh).is_some());

    // unpin + sync：软释放，不立即回收。
    assert!(assets.unpin(mesh));
    gpu.sync(&mut assets);
    assert!(
        gpu.mesh_gpu_resident(mesh).is_some(),
        "unpin 后 sync 不应回收（软释放）"
    );
}

/// 自愈取用：有效句柄未 pin/未同步时，`mesh_gpu` 现场上传并置驻留；
/// 已移除的无效句柄返回 `None`（渲染器据此报错）。
#[test]
fn gpu_manager_mesh_gpu_uploads_on_demand() {
    let (device, queue, _) = require_headless_device();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let mut assets = crate::engine::core::asset::AssetManager::new(
        crate::engine::core::resource::MergedResourceSpace::new(std::env::temp_dir()),
    );
    let mut gpu = crate::engine::render::asset::GpuManager::new(device, queue);
    let mesh = assets.register(crate::engine::Mesh::cube());

    // 注册后既未 pin 也未 sync：ensure 应现场上传，且此后不回收。
    assert!(gpu.mesh_gpu(mesh, &mut assets).is_some(), "有效句柄应按需上传");
    assert!(gpu.mesh_gpu_resident(mesh).is_some());
    // 上传即刷新最近取用：sync 不会回收它。
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(mesh).is_some());

    // remove 后句柄无效：ensure 返回 None。
    assert!(assets.remove(mesh).is_some());
    assert!(gpu.mesh_gpu(mesh, &mut assets).is_none());
}

/// 预分配语义：`pin` 标记驻留（CPU 侧），`sync` 后上传；重复 pin 是引用计数，
/// 上传只发生一次。
#[test]
fn gpu_manager_pin_then_sync_preallocates() {
    let (device, queue, _) = require_headless_device();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let mut assets = crate::engine::core::asset::AssetManager::new(
        crate::engine::core::resource::MergedResourceSpace::new(std::env::temp_dir()),
    );
    let mut gpu = crate::engine::render::asset::GpuManager::new(device, queue);
    let mesh = assets.register(crate::engine::Mesh::cube());

    // pin 标记驻留，sync 上传。
    assert!(assets.pin(mesh));
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(mesh).is_some());

    // 重复 pin 幂等，不重复上传。
    assert!(assets.pin(mesh));
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(mesh).is_some());
}

/// 显存 GC（纯成员）：按最近使用窗口淘汰——窗口 0 只保留"当前时钟"最新取用的
/// 条目；被淘汰的条目 CPU 数据不受影响。
#[test]
fn gpu_manager_gc_evicts_cold_by_usage_window() {
    let (device, queue, _) = require_headless_device();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let mut assets = crate::engine::core::asset::AssetManager::new(
        crate::engine::core::resource::MergedResourceSpace::new(std::env::temp_dir()),
    );
    let mut gpu = crate::engine::render::asset::GpuManager::new(device, queue);
    let a = assets.register(crate::engine::Mesh::cube());
    let b = assets.register(crate::engine::Mesh::cube());

    // 分两次 pin + sync：a 先上传（时钟较早），b 后上传（时钟最新）。
    assert!(assets.pin(a));
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(a).is_some());
    assert!(assets.pin(b));
    gpu.sync(&mut assets);
    assert!(gpu.mesh_gpu_resident(b).is_some());

    // gc 窗口 0：a 超窗回收，b（当前时钟最新）保留。
    gpu.gc(&crate::engine::GcPolicy::default());
    assert!(gpu.mesh_gpu_resident(b).is_some(), "最新取用的保留");
    assert!(gpu.mesh_gpu_resident(a).is_none(), "超窗的回收");
    // GPU 释放不影响 CPU 数据。
    assert!(assets.get_cached(a).is_some());
    assert!(assets.get_cached(b).is_some());

    // 自愈：被回收的 a 下次取用自动重传。
    assert!(gpu.mesh_gpu(a, &mut assets).is_some());
    assert!(gpu.mesh_gpu_resident(a).is_some());
}

/// 把天空盒渲染到 HDR 目标，再过 blit 色调映射写 4×4 离屏 Rgba8UnormSrgb，
/// 读回像素字节。覆盖"场景 pass → HDR → blit → 交换链格式"的完整链路。
fn render_skybox_rgb(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &EnvironmentResources,
    camera_bind_group: &wgpu::BindGroup,
    skybox_bind_group: &wgpu::BindGroup,
) -> Vec<u8> {
    // 1) 场景 pass：天空盒 → HDR 中间目标（原始辐射值）。
    let (_hdr_texture, hdr_view) = create_hdr_texture(device, 4, 4);
    let depth = create_depth_texture(device, 4, 4);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("verify scene encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("verify scene pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &hdr_view,
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
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_bind_group(1, skybox_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit([encoder.finish()]);

    // 2) blit pass：HDR → AgX 色调映射 → Rgba8UnormSrgb。
    let blit = BlitResources::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let blit_bind_group = blit.create_bind_group(device, &hdr_view, &resources.env_params_buffer);
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
    let color_view = color_texture.create_view(&TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("verify blit encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("verify blit pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        blit.draw(&mut pass, &blit_bind_group);
    }
    queue.submit([encoder.finish()]);

    // 3) 读回 u8。
    read_texture_rgba8(device, queue, &color_texture, 4, 4)
}

/// 从 Rgba8UnormSrgb 纹理读回像素字节（逐行按 256 对齐）。
fn read_texture_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let aligned_row = 256u32; // 按 copy 要求每行对齐到 256
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("verify readback"),
        size: (aligned_row * height) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("verify readback encoder"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
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
    data[..(width * height * 4) as usize].to_vec()
}

/// 从 RGBA32F 纹理读回一小片区域（单层单 mip，逐行按 256 对齐）。
fn read_texture_rgba32f(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    layer: u32,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let aligned_row = 256u32;
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("texture readback"),
        size: (aligned_row * height) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("texture readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
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
    let mut out = Vec::with_capacity((width * height) as usize);
    for row in 0..height as usize {
        let base = row * aligned_row as usize;
        for col in 0..width as usize {
            let off = base + col * 16;
            out.push([
                f32::from_le_bytes(data[off..off + 4].try_into().unwrap()),
                f32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()),
                f32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()),
                f32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap()),
            ]);
        }
    }
    out
}
