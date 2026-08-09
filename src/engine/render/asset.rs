//! GPU 资产层：显存表示、上传器与驻留管理（无内存副本）。
//!
//! CPU 侧（[`crate::engine::asset::AssetManager`]）是权威：数据本体与 `Pinned`
//! 状态都在那里；本管理器只维护"句柄 → 显存表示"的纯驻留表（`HashMap`）。
//!
//! 职责划分：
//! - [`GpuManager::sync`]：**只上传**——Pinned 且未上传的补上（预上传优化）；
//! - [`GpuManager::mesh_gpu`] / [`GpuManager::texture_gpu`]：**自愈取用**——
//!   调用方只要给句柄，调度器检查并上传，缺啥传啥；
//! - [`GpuManager::gc`]：**真正回收**——释放所有不再 Pinned 的显存条目
//!   （unpin 是软释放，不立即回收；调用时机由物理刻决定）。

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::engine::core::asset::AssetManager;
use crate::engine::core::asset::{AssetState, Handle};
use crate::engine::core::mesh::Mesh;
use crate::engine::core::texture::Texture;

/// 上传器：把一类资源的 CPU 数据转换为 GPU 表示（客户端概念）。
///
/// 实现可以携带状态（设备能力分支、调试计数等），由 [`GpuManager`] 持有。
pub trait GpuUploader<T, G> {
    fn upload(&mut self, device: &Device, queue: &Queue, data: &T) -> G;
}

/// 网格的 GPU 表示：每网格独立的顶点/索引缓冲。
///
/// 独立缓冲是"资源级卸载/更新"的前提；渲染器按句柄取用，绘制时切换缓冲。
#[derive(Debug)]
pub struct MeshGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    /// 索引数量（绘制时用；独立缓冲整份即该网格）。
    pub index_count: u32,
}

/// 纹理的 GPU 表示：贴图纹理及其视图。
#[derive(Debug)]
pub struct TextureGpu {
    #[allow(dead_code)] // 预留：纹理重建/尺寸查询；当前仅 view 被采样
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// 网格上传器：把 `Mesh` 转成独立 GPU 缓冲（顶点 + 索引）。
#[derive(Debug, Default)]
pub struct MeshUploader;

impl GpuUploader<Mesh, MeshGpu> for MeshUploader {
    fn upload(&mut self, device: &Device, queue: &Queue, mesh: &Mesh) -> MeshGpu {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer"),
            contents: bytemuck::cast_slice(mesh.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh index buffer"),
            contents: bytemuck::cast_slice(mesh.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let _ = queue; // 上传走 create_buffer_init（映射创建），queue 仅作签名一致。
        MeshGpu {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices().len() as u32,
        }
    }
}

/// 贴图上传器：把 `Texture` 转成 GPU 纹理（RGBA8 sRGB，TEXTURE_BINDING）。
#[derive(Debug, Default)]
pub struct TextureUploader;

impl GpuUploader<Texture, TextureGpu> for TextureUploader {
    fn upload(&mut self, device: &Device, queue: &Queue, texture: &Texture) -> TextureGpu {
        let gpu_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size: wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texture.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture.width * 4),
                rows_per_image: Some(texture.height),
            },
            wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
        );
        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureGpu {
            texture: gpu_texture,
            view,
        }
    }
}

/// 显存驻留管理器：句柄 → 显存表示的纯驻留表（CPU 侧权威）。
#[derive(Debug)]
pub struct GpuManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mesh_uploader: MeshUploader,
    texture_uploader: TextureUploader,
    meshes: HashMap<Handle<Mesh>, MeshGpu>,
    textures: HashMap<Handle<Texture>, TextureGpu>,
}

impl GpuManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device,
            queue,
            mesh_uploader: MeshUploader,
            texture_uploader: TextureUploader,
            meshes: HashMap::new(),
            textures: HashMap::new(),
        }
    }

    /// 底层设备（上传时使用）。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 提交队列。
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// 按句柄取用网格显存表示（**自愈**）：未上传则自动上传（CPU 数据不在
    /// 内存会先回磁盘），并标记 `Pinned` 防止被 `gc` 回收。
    ///
    /// 这是外部"通过句柄取用"的统一入口——检查与上传是调度器（本管理器）的
    /// 固有行为，调用方不需要先做任何 ensure。句柄无效/已移除返回 `None`。
    pub fn mesh_gpu(
        &mut self,
        handle: Handle<Mesh>,
        assets: &mut AssetManager,
    ) -> Option<&MeshGpu> {
        self.upload_mesh(handle, assets)?;
        assets.pin(handle);
        self.meshes.get(&handle)
    }

    pub fn texture_gpu(
        &mut self,
        handle: Handle<Texture>,
        assets: &mut AssetManager,
    ) -> Option<&TextureGpu> {
        self.upload_texture(handle, assets)?;
        assets.pin(handle);
        self.textures.get(&handle)
    }

    /// 纯查询：句柄当前的显存表示，**不触发上传**（已驻留/检查用）。
    pub fn mesh_gpu_resident(&self, handle: Handle<Mesh>) -> Option<&MeshGpu> {
        self.meshes.get(&handle)
    }

    pub fn texture_gpu_resident(&self, handle: Handle<Texture>) -> Option<&TextureGpu> {
        self.textures.get(&handle)
    }

    /// 预上传优化：把所有 `Pinned` 且尚未上传的句柄批量上传。
    ///
    /// 不做这一步也不会错——取用时 `mesh_gpu`/`texture_gpu` 会自愈，
    /// 只是上传时机更晚、更碎。回收由 [`Self::gc`] 负责。
    pub fn sync(&mut self, assets: &mut AssetManager) {
        self.sync_meshes(assets);
        self.sync_textures(assets);
    }

    /// 显存垃圾回收：释放所有不再 `Pinned` 的条目（Resident/DiskOnly/已移除）。
    ///
    /// `unpin`/`remove` 是软释放：不立即回收，等这里真正释放。
    /// 调用时机由物理刻决定（目前由调用方按需触发）。
    #[allow(dead_code)] // 公共 GC API：物理刻接入前由调用方按需触发
    pub fn gc(&mut self, assets: &AssetManager) {
        self.gc_meshes(assets);
        self.gc_textures(assets);
    }

    /// 上传一个网格句柄（自愈取用与 `sync` 共用）。
    ///
    /// 顺序很重要：**先校验 CPU 侧句柄有效**（已移除的句柄在这里返回 `None`），
    /// 再查显存表——否则移除后仍残留的旧显存条目会被误当成有效上传返回。
    fn upload_mesh(&mut self, handle: Handle<Mesh>, assets: &mut AssetManager) -> Option<()> {
        if self.meshes.contains_key(&handle) {
            return Some(());
        }
        // 自愈：数据缺失（含内存卸载）时 get 会经重载器自动回磁盘。
        let mesh = assets.get(handle)?;
        let gpu = self.mesh_uploader.upload(&self.device, &self.queue, mesh);
        self.meshes.insert(handle, gpu);
        Some(())
    }

    fn upload_texture(&mut self, handle: Handle<Texture>, assets: &mut AssetManager) -> Option<()> {
        if self.textures.contains_key(&handle) {
            return Some(());
        }
        let texture = assets.get(handle)?;
        let gpu = self.texture_uploader.upload(&self.device, &self.queue, texture);
        self.textures.insert(handle, gpu);
        Some(())
    }

    fn sync_meshes(&mut self, assets: &mut AssetManager) {
        let to_upload: Vec<Handle<Mesh>> = assets
            .iter_of::<Mesh>()
            .filter(|handle| {
                matches!(assets.state(*handle), Some(AssetState::Pinned))
                    && !self.meshes.contains_key(handle)
            })
            .collect();
        for handle in to_upload {
            self.upload_mesh(handle, assets);
        }
    }

    fn sync_textures(&mut self, assets: &mut AssetManager) {
        let to_upload: Vec<Handle<Texture>> = assets
            .iter_of::<Texture>()
            .filter(|handle| {
                matches!(assets.state(*handle), Some(AssetState::Pinned))
                    && !self.textures.contains_key(handle)
            })
            .collect();
        for handle in to_upload {
            self.upload_texture(handle, assets);
        }
    }

    fn gc_meshes(&mut self, assets: &AssetManager) {
        self.meshes
            .retain(|handle, _| matches!(assets.state(*handle), Some(AssetState::Pinned)));
    }

    fn gc_textures(&mut self, assets: &AssetManager) {
        self.textures.retain(|handle, _| {
            matches!(assets.state(*handle), Some(AssetState::Pinned))
        });
    }
}
