//! wgpu 能力探针：强制 GL 后端，复现 BUG.md 记录的"数组存储纹理写入全零"。
//!
//! 背景：环境贴图（天空盒/IBL）最初用"GPU 计算着色器写数组存储纹理"实现，
//! 在 GL 后端上写入全零；同样代码在 Vulkan/Metal 正常（`examples/vulkan_probe.rs`）。
//! 本探针在真实 GL（桌面通常是 Mesa llvmpipe，GL 4.5 支持计算着色器）上逐项验证：
//!
//!   A    write_texture 上传 6 层数组 + **整块数组**拷贝读回（读回路径本身）
//!   A2   write_texture 上传 6 层 + **逐层**拷贝读回（上传路径本身）
//!   B    计算着色器写 6 层数组存储纹理 + 逐层读回（关键 bug）
//!   C    计算着色器写单层存储纹理 + 读回（对照组，已知 GL 上正常）
//!   D    最初方案的完整"等距矩形 → 立方体贴图"计算（端到端复现）
//!   E    逐层上传数组 → 采样 → 写单层存储 → 读回（真实回退路径）
//!   F    计算写数组存储 → 采样 → 写单层存储 → 读回（隔离"写入"与"读回"）
//!
//! 用逐层读回而不是整块读回，是为了区分"存储写入坏了"和"数组读回坏了"：
//! 两者在 GL 上都是 BUG.md 记录的不稳定点。
//!
//! 运行：`cargo run --example gl_probe`
//! 输出会打印实际选中的适配器；若本机没有可用 GL/EGL 驱动则提前退出。

use std::sync::mpsc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    size: u32,
    value: f32,
    _pad: [u32; 2],
}

const ARRAY_STORE_SHADER: &str = r#"
struct Params {
    size: u32,
    value: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var out_tex: texture_storage_2d_array<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size || gid.y >= params.size || gid.z >= 6u) {
        return;
    }
    textureStore(out_tex, vec2<i32>(gid.xy), i32(gid.z),
                 vec4<f32>(params.value, 0.0, 0.0, 1.0));
}
"#;

const SINGLE_STORE_SHADER: &str = r#"
struct Params {
    size: u32,
    value: f32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size || gid.y >= params.size) {
        return;
    }
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(params.value, 0.0, 0.0, 1.0));
}
"#;

/// 数组纹理 → 单层存储纹理（采样后直出，绕开数组 readback）。
/// 测试 E / F 的第二阶段用：验证数组数据经采样是否完好。
const ARRAY_TO_SINGLE_SHADER: &str = r#"
@group(0) @binding(0) var in_tex: texture_2d_array<f32>;
@group(0) @binding(1) var in_sampler: sampler;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 4u || gid.y >= 4u) {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / 4.0;
    let c = textureSampleLevel(in_tex, in_sampler, uv, i32(0), 0.0);
    textureStore(out_tex, vec2<i32>(gid.xy), c);
}
"#;

/// 最初的"等距矩形 → 立方体贴图"计算着色器（与当时实现一致的逻辑）。
/// 测试 D 用来回答：这个方案在 GL 上是不是端到端全零。
const EQUIRECT_TO_CUBEMAP_SHADER: &str = r#"
struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var equirect_tex: texture_2d<f32>;
@group(0) @binding(2) var equirect_sampler: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d_array<rgba32float, write>;

const PI: f32 = 3.141592653589793;

fn face_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let x = u * 2.0 - 1.0;
    let y = v * 2.0 - 1.0;
    switch face {
        case 0u: { return vec3<f32>(1.0, -y, -x); }
        case 1u: { return vec3<f32>(-1.0, -y, x); }
        case 2u: { return vec3<f32>(x, 1.0, y); }
        case 3u: { return vec3<f32>(x, -1.0, -y); }
        case 4u: { return vec3<f32>(x, -y, 1.0); }
        default: { return vec3<f32>(-x, -y, -1.0); }
    }
}

fn dir_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), theta / PI);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size || gid.y >= params.size || gid.z >= 6u) {
        return;
    }
    let u = (f32(gid.x) + 0.5) / f32(params.size);
    let v = (f32(gid.y) + 0.5) / f32(params.size);
    let dir = face_dir(gid.z, u, v);
    let color = textureSampleLevel(equirect_tex, equirect_sampler, dir_to_equirect(dir), 0.0);
    textureStore(out_tex, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(color.rgb, 1.0));
}
"#;

fn main() {
    // 1. 强制 GL（桌面通常走 EGL + Mesa）。
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::GL;
    let instance = wgpu::Instance::new(instance_desc);

    // 2. 枚举所有 GL 适配器。
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::GL));
    println!("== GL 适配器（{} 个）==", adapters.len());
    for adapter in &adapters {
        let info = adapter.get_info();
        println!(
            "  - {} | 后端 {:?} | 类型 {:?}",
            info.name, info.backend, info.device_type
        );
    }

    // 3. 请求 HighPerformance 适配器（GL 上通常就是 llvmpipe）。
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(e) => {
            println!("错误：请求 GL 适配器失败（{e}）——本机可能没有可用的 GL/EGL 驱动。");
            println!("结论：无法在 GL 上验证；可用 `cargo run --example vulkan_probe` 对照。");
            std::process::exit(1);
        }
    };
    let info = adapter.get_info();
    println!(
        "== 选中适配器：{} | {:?} | {:?} ==",
        info.name, info.backend, info.device_type
    );

    // 4. Rgba32Float 的格式能力（存储纹理 / 过滤）。
    let fmt = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Float);
    println!("== Rgba32Float 能力 ==");
    println!("  allowed_usages: {:#?}", fmt.allowed_usages);
    println!("  flags: {:#?}", fmt.flags);

    // 5. 设备。
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("probe device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("设备创建失败");

    // 测试 A：write_texture 上传 6 层数组 + 整块数组拷贝读回。
    // BUG.md 记录 GL 的"数组纹理 readback 拷贝"也不可靠，这里单独验证。
    println!("== 测试 A：write_texture 上传数组 + 整块数组拷贝读回 ==");
    let known: Vec<[f32; 4]> = vec![[0.8, 0.0, 0.0, 1.0]; (4 * 4 * 6) as usize];
    let known_tex = create_array_texture(&device, &queue, 4, &known, "known");
    let a_max = readback_max(&device, &queue, &known_tex, 4, 6, false);
    report("A", a_max);

    // 测试 A2：同一上传，但逐层拷贝读回（对照组：上传路径本身）。
    println!("== 测试 A2：write_texture 上传数组 + 逐层拷贝读回 ==");
    let a2_max = readback_max(&device, &queue, &known_tex, 4, 6, true);
    report("A2", a2_max);

    // 测试 B：计算着色器写 6 层数组存储纹理（关键问题所在）。
    println!("== 测试 B：compute textureStore 写数组存储纹理 ==");
    let (b_pipeline, b_layout, b_params) = compute_pipeline(&device, ARRAY_STORE_SHADER, true);
    let b_tex = create_storage_array_texture(&device, 4);
    run_storage_write(&device, &queue, &b_pipeline, &b_layout, &b_params, &b_tex, 4, true);
    let b_max = readback_max(&device, &queue, &b_tex, 4, 6, true);
    report("B", b_max);

    // 测试 C：计算着色器写单层存储纹理（对照组，已知 GL 上正常）。
    println!("== 测试 C：compute textureStore 写单层存储纹理 ==");
    let (c_pipeline, c_layout, c_params) = compute_pipeline(&device, SINGLE_STORE_SHADER, false);
    let c_tex = create_storage_single_texture(&device, 4);
    run_storage_write(&device, &queue, &c_pipeline, &c_layout, &c_params, &c_tex, 4, false);
    let c_max = readback_max(&device, &queue, &c_tex, 4, 1, true);
    report("C", c_max);

    // 测试 D：最初方案的完整"等距矩形 → 立方体贴图"计算逻辑。
    println!("== 测试 D：最初方案（等距矩形 → 立方体贴图）==");
    let src_pixels: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let src_tex = create_equirect_texture(&device, &queue, 2, 2, &src_pixels);
    let d_tex = create_storage_array_texture(&device, 4);
    run_equirect_convert(&device, &queue, &src_tex, &d_tex);
    let d_max = readback_max(&device, &queue, &d_tex, 4, 6, true);
    report("D", d_max);

    // 测试 E：逐层上传数组 → 采样 → 写单层存储 → 读回。
    // 对应真实回退路径（CPU 转换 + 逐层 write_texture + 采样），
    // 全程不碰数组 readback，用来确认这条路在 GL 上确实可信。
    println!("== 测试 E：上传数组 → 采样 → 单层输出 ==");
    let e_tex = create_array_texture(&device, &queue, 4, &known, "known-e");
    let e_out = create_storage_single_texture(&device, 4);
    run_array_to_single(&device, &queue, &e_tex, &e_out);
    let e_max = readback_max(&device, &queue, &e_out, 4, 1, true);
    report("E", e_max);

    // 测试 F：计算写数组存储 → 采样 → 写单层存储 → 读回。
    // 若 F 非零，说明数组存储"写入"本身没坏，B 的全零只是数组读回坏；
    // 若 F 全零，说明写入（或写入后的采样）确实不可靠。
    println!("== 测试 F：计算写数组存储 → 采样 → 单层输出 ==");
    let f_tex = create_storage_array_texture(&device, 4);
    run_storage_write(&device, &queue, &b_pipeline, &b_layout, &b_params, &f_tex, 4, true);
    let f_out = create_storage_single_texture(&device, 4);
    run_array_to_single(&device, &queue, &f_tex, &f_out);
    let f_max = readback_max(&device, &queue, &f_out, 4, 1, true);
    report("F", f_max);

    // 结论。
    println!("== 结论 ==");
    if c_max > 0.0 && e_max > 0.0 && f_max > 0.0 {
        println!("GL 上：上传、采样、数组存储写入全部正常，只有数组纹理 readback 拷贝全零。");
        println!("→ 修正 BUG.md 的表述：'写入全零'实为'读回全零'（A/A2/B/D 都卡在读回）；");
        println!("  黑天空盒另有原因（如参数缓冲分时复用），但 GL 上仍建议 CPU 回退，");
        println!("  避免任何数组 readback/存储依赖。");
    } else if c_max > 0.0 && e_max > 0.0 && f_max <= 0.0 {
        println!("GL 上：上传正常、单层存储写入正常，但数组存储纹理写入全零。");
        println!("→ 复现 BUG.md：GL 后端的数组存储纹理写入不可靠；");
        println!("  回退 CPU 转换 + 逐层 write_texture 上传（EnvConversionPath）的方案正确。");
    } else if a_max <= 0.0 && a2_max <= 0.0 && e_max > 0.0 {
        println!("GL 上：整块数组拷贝读回全零、逐层读回正常。");
        println!("→ 印证 BUG.md 的另一半：GL 的数组纹理 readback 拷贝不可靠。");
    } else if a2_max <= 0.0 {
        println!("GL 上：连逐层上传+读回都不正常，无法得出有效结论。");
    } else {
        println!("GL 上：单层存储写入也失败，属于更通用的计算/驱动问题。");
    }
}

fn create_array_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face: u32,
    rgba32f: &[[f32; 4]],
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face,
            height: face,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let layer_pixels = (face * face) as usize;
    for layer in 0..6u32 {
        let data = &rgba32f[(layer as usize * layer_pixels)..((layer as usize + 1) * layer_pixels)];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(face * 16),
                rows_per_image: Some(face),
            },
            wgpu::Extent3d {
                width: face,
                height: face,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

fn create_storage_array_texture(device: &wgpu::Device, face: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage array"),
        size: wgpu::Extent3d {
            width: face,
            height: face,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// 数组纹理 → 单层存储纹理（测试 E / F 第二阶段）。
fn run_array_to_single(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_tex: &wgpu::Texture,
    out_tex: &wgpu::Texture,
) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("array to single shader"),
        source: wgpu::ShaderSource::Wgsl(ARRAY_TO_SINGLE_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("array to single layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("array to single pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("array to single pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("array to single sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("array to single src view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("array to single bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&out_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("array to single encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
}

fn create_storage_single_texture(device: &wgpu::Device, face: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage single"),
        size: wgpu::Extent3d {
            width: face,
            height: face,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// 单层 RGBA32F 等距矩形源纹理（逐像素 RGBA 数据）。
fn create_equirect_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba: &[f32],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("equirect source"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(rgba),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// 跑最初的等距矩形 → 立方体贴图计算（测试 D）。
fn run_equirect_convert(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_tex: &wgpu::Texture,
    out_tex: &wgpu::Texture,
) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("equirect shader"),
        source: wgpu::ShaderSource::Wgsl(EQUIRECT_TO_CUBEMAP_SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("equirect layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("equirect pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("equirect pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("equirect params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &params,
        0,
        bytemuck::bytes_of(&Params {
            size: 4,
            value: 0.0,
            _pad: [0; 2],
        }),
    );
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("equirect sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    let src_view = src_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("equirect out view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("equirect bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&out_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("equirect encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 6); // 4×4 面，8×8 workgroup 覆盖
    }
    queue.submit([encoder.finish()]);
}

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &str,
    is_array: bool,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe shader"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: if is_array {
                        wgpu::TextureViewDimension::D2Array
                    } else {
                        wgpu::TextureViewDimension::D2
                    },
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (pipeline, layout, params)
}

fn run_storage_write(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    texture: &wgpu::Texture,
    face: u32,
    is_array: bool,
) {
    queue.write_buffer(
        params,
        0,
        bytemuck::bytes_of(&Params {
            size: face,
            value: 0.8,
            _pad: [0; 2],
        }),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("probe view"),
        dimension: Some(if is_array {
            wgpu::TextureViewDimension::D2Array
        } else {
            wgpu::TextureViewDimension::D2
        }),
        base_array_layer: 0,
        array_layer_count: Some(if is_array { 6 } else { 1 }),
        ..Default::default()
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        if is_array {
            pass.dispatch_workgroups(face.div_ceil(8), face.div_ceil(8), 6);
        } else {
            pass.dispatch_workgroups(face.div_ceil(8), face.div_ceil(8), 1);
        }
    }
    queue.submit([encoder.finish()]);
}

/// 读回最大像素值。
///
/// `per_layer` = true 时逐层拷贝（origin.z = 层号，depth = 1），绕开 GL 上
/// "整块数组拷贝"的不稳定路径；false 时整块拷贝（复现测试 A 的读回问题）。
fn readback_max(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    face: u32,
    layers: u32,
    per_layer: bool,
) -> f32 {
    let mut max = 0.0f32;
    for layer in 0..layers {
        let layer_count = if per_layer { 1 } else { layers };
        let origin_z = if per_layer { layer } else { 0 };
        let row_bytes = face * 16;
        let aligned = row_bytes.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (aligned * face * layer_count) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: origin_z,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(face),
                },
            },
            wgpu::Extent3d {
                width: face,
                height: face,
                depth_or_array_layers: layer_count,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll 应成功");
        rx.recv().expect("map 回调应触发").expect("map 应成功");
        let data = slice.get_mapped_range().expect("取范围应成功");
        max = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .fold(max, f32::max);
    }
    max
}

fn report(label: &str, max_value: f32) {
    let verdict = if max_value > 0.0 { "OK（数据非零）" } else { "FAIL（全零）" };
    println!("  测试 {label}：最大像素值 {max_value:.3} → {verdict}");
}
