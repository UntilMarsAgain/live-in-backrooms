//! Minimal repro: wgpu GL backend `copy_texture_to_buffer` from a 2D array
//! texture returns all zeros, while the same path from a single-layer texture
//! works.
//!
//! Run: `cargo run --example gl_array_readback_repro`
//!
//! Observed on: llvmpipe (Mesa 26.1.6, LLVM 22.1.8) via wgpu 30.0.0 GL backend.
//! Whole-array copy and per-layer copy (origin.z = layer) both return zeros;
//! upload (`write_texture`) into the array texture and sampling it are fine.

use std::sync::mpsc;

const SIZE: u32 = 4;
const LAYERS: u32 = 6;
const VALUE: f32 = 0.8;

fn main() {
    // Headless GL (EGL surfaceless / llvmpipe).
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::GL;
    let instance = wgpu::Instance::new(desc);

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no GL adapter");
    let info = adapter.get_info();
    println!(
        "Adapter: {} | {:?} | {:?}",
        info.name, info.backend, info.device_type
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("repro device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device");

    // 1) Control: single-layer texture, upload + readback.
    let single = create_texture(&device, 1);
    upload_layer(&queue, &single, 0);
    let single_max = readback_max(&device, &queue, &single, SIZE, 1, false);

    // 2) 2D array texture, per-layer upload, whole-array readback.
    let array = create_texture(&device, LAYERS);
    for layer in 0..LAYERS {
        upload_layer(&queue, &array, layer);
    }
    let whole_max = readback_max(&device, &queue, &array, SIZE, LAYERS, false);

    // 3) Same array texture, per-layer readback (origin.z = layer, depth = 1).
    let per_layer_max = readback_max(&device, &queue, &array, SIZE, LAYERS, true);

    println!(
        "single-layer readback:    max = {single_max:.3} -> {}",
        if single_max > 0.0 { "OK" } else { "ALL ZEROS" }
    );
    println!(
        "array whole readback:     max = {whole_max:.3} -> {}",
        if whole_max > 0.0 { "OK" } else { "ALL ZEROS" }
    );
    println!(
        "array per-layer readback: max = {per_layer_max:.3} -> {}",
        if per_layer_max > 0.0 {
            "OK"
        } else {
            "ALL ZEROS"
        }
    );
}

fn create_texture(device: &wgpu::Device, layers: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe texture"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn upload_layer(queue: &wgpu::Queue, texture: &wgpu::Texture, layer: u32) {
    let data: Vec<f32> = vec![VALUE, 0.0, 0.0, 1.0].repeat((SIZE * SIZE) as usize);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 16),
            rows_per_image: Some(SIZE),
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
}

/// Copy to buffer (whole array or per layer) and return the max texel value.
fn readback_max(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: u32,
    layers: u32,
    per_layer: bool,
) -> f32 {
    let mut max = 0.0f32;
    for layer in 0..layers {
        let (layer_count, origin_z) = if per_layer { (1, layer) } else { (layers, 0) };
        let row_bytes = size * 16;
        let aligned = row_bytes.div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (aligned * size * layer_count) as u64,
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
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layer_count,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        rx.recv().expect("map callback").expect("map");
        let data = slice.get_mapped_range().expect("range");
        max = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .fold(max, f32::max);
    }
    max
}
