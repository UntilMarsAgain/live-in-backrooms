//! 环境子系统：纹理创建与上传。
//!
//! 立方体贴图、mip 链立方体贴图、2D 纹理的创建与逐层上传（逐层避开 GL 后端
//! 整块写数组纹理的 bug），以及无环境时的默认黑色占位绑定组。

use wgpu::util::DeviceExt;
use wgpu::{BindGroupDescriptor, BindGroupEntry, TextureViewDescriptor};

use super::EnvironmentGpu;

/// 无环境贴图时的默认绑定组：1×1×6 黑色立方体贴图。
///
/// 保证 mesh 管线 @group(4) 与天空盒管线始终有可绑定的资源。
pub(crate) fn create_default_environment(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    environment_layout: &wgpu::BindGroupLayout,
    skybox_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    intensity_buffer: &wgpu::Buffer,
) -> EnvironmentGpu {
    let black = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("default black environment"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &[0u8; 6 * 16],
    );
    let view = black.create_view(&TextureViewDescriptor {
        label: Some("default black environment view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });
    // 默认 BRDF LUT：1×1 黑纹理（无镜面反射时贡献 0）。
    let black_2d = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("default black 2d texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &[0u8; 16],
    );
    let black_2d_view = black_2d.create_view(&TextureViewDescriptor::default());
    let mesh_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("default environment mesh bind group"),
        layout: environment_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 3,
                resource: intensity_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&black_2d_view),
            },
        ],
    });
    let skybox_bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("default skybox bind group"),
        layout: skybox_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: intensity_buffer.as_entire_binding(),
            },
        ],
    });
    EnvironmentGpu {
        environment_texture: black.clone(),
        environment_view: view.clone(),
        irradiance_texture: black.clone(),
        irradiance_view: view.clone(),
        prefiltered_texture: black,
        prefiltered_view: view,
        brdf_lut_texture: black_2d,
        brdf_lut_view: black_2d_view,
        sampler: sampler.clone(),
        mesh_bind_group,
        skybox_bind_group,
    }
}

/// 创建 6 层 RGBA32F 立方体贴图并逐层上传（层序 +X,-X,+Y,-Y,+Z,-Z）。
///
/// 逐层写而非整块写：wgpu 的 GL 后端对"一次 write_texture 上传整个
/// 2D 数组纹理"的实现不可靠（实测读出全零），逐层上传与单层纹理同样稳定。
pub(crate) fn create_cube_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face_size: u32,
    rgba32f: &[[f32; 4]],
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
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
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let layer_pixels = (face_size * face_size) as usize;
    for layer in 0..6u32 {
        let layer_data =
            &rgba32f[(layer as usize * layer_pixels)..((layer as usize + 1) * layer_pixels)];
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
            bytemuck::cast_slice(layer_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(face_size * 16),
                rows_per_image: Some(face_size),
            },
            wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// 创建带 mip 链的 6 层 RGBA32F 立方体贴图并逐层逐 mip 上传（CPU 路径用）。
///
/// `mips[mip]` 是第 mip 层 6 个面的数据（每面 `face_size>>mip` 见方）；
/// 逐层写避开 GL 后端整块写数组纹理的 bug，与 [`create_cube_texture`] 同策略。
pub(crate) fn create_mip_cube_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face_size: u32,
    mip_count: u32,
    mips: &[Vec<[f32; 4]>],
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for mip in 0..mip_count {
        let mip_size = face_size >> mip;
        let layer_pixels = (mip_size * mip_size) as usize;
        for layer in 0..6u32 {
            let layer_data =
                &mips[mip as usize][(layer as usize * layer_pixels)..((layer as usize + 1) * layer_pixels)];
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(layer_data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(mip_size * 16),
                    rows_per_image: Some(mip_size),
                },
                wgpu::Extent3d {
                    width: mip_size,
                    height: mip_size,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    texture
}

/// 创建单层 RGBA32F 2D 纹理并上传（CPU 路径的 BRDF LUT 用）。
pub(crate) fn create_2d_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgba32f: &[[f32; 4]],
    label: &str,
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(rgba32f),
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
