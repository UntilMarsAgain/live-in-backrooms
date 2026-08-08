//! 资产管理器的 GPU 实现层：GPU 表示类型、上传器与 [`AssetManager`] 本体。
//!
//! 按项目约定 core 层不包含 GPU 实际实现，因此这里的类型（`MeshGpu`、
//! 上传器等）从 core 独立出来，依赖方向 render → core：
//! core 只保留无 GPU 依赖的抽象（[`AssetRegistry`]、[`Handle`] 等），
//! GPU 上传 trait 与占位实现（[`GpuUploader`] / [`NoGpuUploader`]）也在这里。

use std::any::Any;
use std::collections::HashMap;
#[allow(unused_imports)] // Arc 由宏展开后的 AssetManager 使用，静态分析误报
use std::sync::Arc;

use paste::paste;
use wgpu::{Device, Queue};

use lbr_shared::asset::GlbLoader;
use lbr_shared::core::asset::{
    AssetLoader, AssetRegistry, AssetState, DataSource, Handle, MeshSource,
};
use lbr_shared::core::game_path::GamePath;
use lbr_shared::core::mesh::Mesh;
use lbr_shared::core::resource::MergedResourceSpace;
use lbr_shared::core::texture::Texture;

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

/// 上传器：把一类资源的 CPU 数据转换为 GPU 表示。
///
/// 客户端概念（CPU → GPU 转换），不进入共享层；实现可以携带状态（设备能力
/// 分支、调试计数等），由 [`AssetManager`] 持有并在同步/按需上传时调用。
/// 纯数据资源（GPU 类型 `()`）用 [`NoGpuUploader`] 空实现。
pub trait GpuUploader<T, G> {
    fn upload(&mut self, device: &Device, queue: &Queue, data: &T) -> G;
}

/// 纯 CPU 资源（GPU 类型 `()`）的占位上传器：`upload` 是空操作。
///
/// rustc 会把空函数调用内联掉，无性能损失。
#[derive(Debug, Default)]
pub struct NoGpuUploader;

impl<T> GpuUploader<T, ()> for NoGpuUploader {
    fn upload(&mut self, _device: &Device, _queue: &Queue, _data: &T) {}
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

/// 无文件加载的占位加载器（程序生成资产用）：`load` 必然失败。
///
/// mesh/texture 目前走程序注册（glTF 加载器尚未接 GamePath），下一步接入
/// GlbLoader（glTF 复合加载：一个文件产出网格 + 贴图 + 场景）后替换。
#[derive(Debug, Default)]
pub struct NoLoader;

impl<T> AssetLoader<T> for NoLoader {
    type Extra = ();
    type Parsed = ();

    fn load(&self, _space: &MergedResourceSpace, _path: &GamePath) -> anyhow::Result<()> {
        anyhow::bail!("该资源类型尚不支持从文件加载（加载器未接入）");
    }

    fn entries(&self, _parsed: &()) -> Vec<(T, ())> {
        Vec::new()
    }

    fn entry<'a>(&self, _parsed: &'a (), _extra: &()) -> Option<&'a T> {
        None
    }
}

/// 内存层：文件级解析结果（按 `GamePath` 缓存一份，供该文件所有条目共享）。
///
/// 值类型由各加载器的 `Parsed` 决定（`Box<dyn Any>` 擦除），取回时 downcast。
/// 内存卸载 = 从这里移除对应文件——"磁盘 → 内存 → 显存"三级驻留的内存层。
#[derive(Default)]
pub struct MemoryLayer {
    files: HashMap<GamePath, Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for MemoryLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryLayer {{ files: {} }}", self.files.len())
    }
}

impl MemoryLayer {
    fn insert(&mut self, source: GamePath, parsed: Box<dyn Any + Send + Sync>) {
        self.files.insert(source, parsed);
    }

    fn get(&self, source: &GamePath) -> Option<&(dyn Any + Send + Sync)> {
        self.files.get(source).map(|b| b.as_ref())
    }

    fn remove(&mut self, source: &GamePath) -> Option<Box<dyn Any + Send + Sync>> {
        self.files.remove(source)
    }
}

/// 声明资产管理器的资源类型集合。
///
/// 每个条目：`字段名: CPU类型 => GPU类型, 上传器类型, 加载器类型`。
/// 加载器实现 [`AssetLoader`]（从 `GamePath` 解析 CPU 数据）；程序生成资产
/// 用 [`NoLoader`] 占位。新增资源类型 = 加一行。
macro_rules! asset_types {
    ($($field:ident: $ty:ident => $gpu:ty, $uploader:ident, $loader:ident),* $(,)?) => {
        paste! {
            /// 统一资产管理器：持有设备/队列/合并资源空间/内存层与各类型
            /// 注册表、上传器、加载器。
            ///
            /// 三级驻留：磁盘（`load_*` 按 `GamePath` 读取）→ 内存（内存层
            /// 文件结果，`unload_*_memory` 卸载）→ 显存（`pin` / `sync_gpu`）。
            #[derive(Debug)]
            pub struct AssetManager {
                device: Option<Arc<Device>>,
                queue: Option<Arc<Queue>>,
                space: MergedResourceSpace,
                memory: MemoryLayer,
                $(
                    $field: AssetRegistry<$ty, $gpu>,
                    [<$field _uploader>]: $uploader,
                    [<$field _loader>]: $loader,
                )*
            }

            // 宏为每种资源全量生成 load/get/ensure/unload/pin 方法，
            // 部分在调用方未覆盖前属于公共 API 预留，暂时保留。
            #[allow(dead_code)]
            impl AssetManager {
                /// 新建管理器；`device` + `queue` 用于 GPU，`space` 是合并资源空间。
                pub fn new(
                    device: Arc<Device>,
                    queue: Arc<Queue>,
                    space: MergedResourceSpace,
                ) -> Self {
                    Self {
                        device: Some(device),
                        queue: Some(queue),
                        space,
                        memory: MemoryLayer::default(),
                        $( $field: AssetRegistry::new(), [<$field _uploader>]: $uploader::default(), [<$field _loader>]: $loader::default(), )*
                    }
                }

                /// 无 GPU 环境（纯数据/碰撞测试）用：`sync_gpu` 自动跳过。
                pub fn without_gpu(space: MergedResourceSpace) -> Self {
                    Self {
                        device: None,
                        queue: None,
                        space,
                        memory: MemoryLayer::default(),
                        $( $field: AssetRegistry::new(), [<$field _uploader>]: $uploader::default(), [<$field _loader>]: $loader::default(), )*
                    }
                }

                /// 底层设备（上传/同步时使用）；无 GPU 构造时为 `None`。
                pub fn device(&self) -> Option<&Device> {
                    self.device.as_deref()
                }

                /// 合并资源空间（`GamePath` → 文件流）。
                pub fn space(&self) -> &MergedResourceSpace {
                    &self.space
                }

                /// 同步所有注册表的 GPU 资源：pinned 且未上传的上传，
                /// 非 pinned 的回收。数据从内存层/内联来源取。
                pub fn sync_gpu(&mut self) {
                    let (Some(device), Some(queue)) = (&self.device, &self.queue) else {
                        return; // 无 GPU 构造：纯 CPU 使用，不上传
                    };
                    $(
                        let handles: Vec<_> = self.$field.iter().collect();
                        for handle in handles {
                            match self.$field.state(handle) {
                                Some(AssetState::Pinned) => {
                                    if self.$field.gpu(handle).is_none() {
                                        // 内联取数据（借 self 字段，可与 uploader 可变借用共存）。
                                        let data = match self.$field.data_source(handle) {
                                            Some(DataSource::Inline(data)) => Some(data),
                                            Some(DataSource::File { source, extra }) => {
                                                let parsed = self.memory.get(source)
                                                    .expect("File 条目的 source 应在内存层（先 ensure_loaded）");
                                                let parsed = parsed
                                                    .downcast_ref::<<$loader as AssetLoader<$ty>>::Parsed>()
                                                    .expect("内存层解析结果类型与加载器 Parsed 不匹配（解析器 bug）");
                                                let extra = extra
                                                    .downcast_ref::<<$loader as AssetLoader<$ty>>::Extra>()
                                                    .expect("extra 类型与加载器 Extra 不匹配（解析器 bug）");
                                                self.[<$field _loader>].entry(parsed, extra)
                                            }
                                            None => None,
                                        };
                                        if let Some(data) = data {
                                            let gpu = self.[<$field _uploader>]
                                                .upload(device, queue, data);
                                            *self.$field.gpu_mut(handle).expect("存活") = Some(gpu);
                                        }
                                    }
                                }
                                Some(AssetState::Resident) | Some(AssetState::DiskOnly) => {
                                    *self.$field.gpu_mut(handle).expect("存活") = None;
                                }
                                _ => {}
                            }
                        }
                    )*
                }

                $(
                    /// 对应类型的注册表访问器。
                    pub fn $field(&self) -> &AssetRegistry<$ty, $gpu> {
                        &self.$field
                    }

                    /// 对应类型的注册表可变访问器。
                    pub fn [<$field _mut>](&mut self) -> &mut AssetRegistry<$ty, $gpu> {
                        &mut self.$field
                    }

                    /// 取条目 CPU 数据：内联直接返回；文件条目从内存层按定位信息取。
                    ///
                    /// 文件条目数据不在内存层（被卸载）时返回 `None`，调用方应先
                    /// `ensure_*_loaded`；downcast 失败视为解析器 bug（panic 暴露）。
                    pub fn [<get_ $field>](&self, handle: Handle<$ty>) -> Option<&$ty> {
                        match self.$field.data_source(handle)? {
                            DataSource::Inline(data) => Some(data),
                            DataSource::File { source, extra } => {
                                let parsed = self.memory.get(source)?;
                                let parsed = parsed
                                    .downcast_ref::<<$loader as AssetLoader<$ty>>::Parsed>()
                                    .expect("内存层解析结果类型与加载器 Parsed 不匹配（解析器 bug）");
                                let extra = extra
                                    .downcast_ref::<<$loader as AssetLoader<$ty>>::Extra>()
                                    .expect("extra 类型与加载器 Extra 不匹配（解析器 bug）");
                                self.[<$field _loader>].entry(parsed, extra)
                            }
                        }
                    }

                    /// 从合并资源空间按 `GamePath` 加载：解析一次，文件结果进内存层，
                    /// 每个条目注册为 `File` 来源的句柄并返回。
                    pub fn [<load_ $field>](
                        &mut self,
                        path: &GamePath,
                    ) -> anyhow::Result<Vec<Handle<$ty>>> {
                        let parsed: <$loader as AssetLoader<$ty>>::Parsed =
                            <$loader as AssetLoader<$ty>>::load(
                                &self.[<$field _loader>],
                                self.space(),
                                path,
                            )?;
                        let entries: Vec<($ty, <$loader as AssetLoader<$ty>>::Extra)> =
                            self.[<$field _loader>].entries(&parsed);
                        self.memory.insert(path.clone(), Box::new(parsed));
                        Ok(entries
                            .into_iter()
                            // 数据已在内存层（文件结果），注册只记来源与定位。
                            .map(|(_data, extra)| {
                                self.$field
                                    .register_file(path.clone(), Box::new(extra))
                            })
                            .collect())
                    }

                    /// 确保条目 CPU 数据在内存：内联常驻；文件条目数据不在内存层时
                    /// 重新解析（`ensure_gpu` 会递归调用这里，形成"显存→内存→磁盘"）。
                    pub fn [<ensure_ $field _loaded>](
                        &mut self,
                        handle: Handle<$ty>,
                    ) -> Option<()> {
                        match self.$field.data_source(handle)? {
                            DataSource::Inline(_) => Some(()),
                            DataSource::File { source, .. } => {
                                if self.memory.get(source).is_none() {
                                    let parsed: <$loader as AssetLoader<$ty>>::Parsed =
                                        <$loader as AssetLoader<$ty>>::load(
                                            &self.[<$field _loader>],
                                            self.space(),
                                            source,
                                        )
                                        .ok()?;
                                    self.memory.insert(source.clone(), Box::new(parsed));
                                }
                                self.$field.set_state(handle, AssetState::Resident);
                                Some(())
                            }
                        }
                    }

                    /// 内存卸载（按文件）：释放内存层文件结果，该文件所有条目置
                    /// `DiskOnly`，GPU 一并回收（上传无源）。句柄仍有效，可重载。
                    pub fn [<unload_ $field _memory>](&mut self, source: &GamePath) {
                        self.memory.remove(source);
                        let handles: Vec<_> = self.$field.iter().collect();
                        for handle in handles {
                            if matches!(self.$field.data_source(handle),
                                Some(DataSource::File { source: s, .. }) if s == source)
                            {
                                *self.$field.gpu_mut(handle).expect("存活") = None;
                                self.$field.set_state(handle, AssetState::DiskOnly);
                            }
                        }
                    }

                    /// 按需确保该类型资源已上传（渲染器绘制前的兜底）：数据不在内存
                    /// 先递归 `ensure_*_loaded` 回磁盘，再上传并置 `Pinned`。
                    pub fn [<ensure_ $field _gpu>](
                        &mut self,
                        handle: Handle<$ty>,
                    ) -> Option<&$gpu> {
                        // 先确保内存数据（可能回磁盘重载），再取设备引用。
                        self.[<ensure_ $field _loaded>](handle)?;
                        let (device, queue) = (self.device.as_ref()?, self.queue.as_ref()?);
                        if self.$field.gpu(handle).is_none() {
                            let data = match self.$field.data_source(handle) {
                                Some(DataSource::Inline(data)) => Some(data),
                                Some(DataSource::File { source, extra }) => {
                                    let parsed = self.memory.get(source)
                                        .expect("File 条目的 source 应在内存层（先 ensure_loaded）");
                                    let parsed = parsed
                                        .downcast_ref::<<$loader as AssetLoader<$ty>>::Parsed>()
                                        .expect("内存层解析结果类型与加载器 Parsed 不匹配（解析器 bug）");
                                    let extra = extra
                                        .downcast_ref::<<$loader as AssetLoader<$ty>>::Extra>()
                                        .expect("extra 类型与加载器 Extra 不匹配（解析器 bug）");
                                    self.[<$field _loader>].entry(parsed, extra)
                                }
                                None => None,
                            };
                            let data = data?;
                            let gpu = self.[<$field _uploader>].upload(device, queue, data);
                            *self.$field.gpu_mut(handle).expect("存活") = Some(gpu);
                        }
                        self.$field.pin(handle);
                        self.$field.gpu(handle)
                    }

                    /// 预上传并驻留该类型资源（预分配语义）；无 GPU 构造只标记驻留状态。
                    pub fn [<pin_ $field>](&mut self, handle: Handle<$ty>) -> bool {
                        if self.$field.state(handle).is_none() {
                            return false;
                        }
                        if self.[<ensure_ $field _loaded>](handle).is_none() {
                            return false;
                        }
                        if let (Some(device), Some(queue)) = (&self.device, &self.queue) {
                            if self.$field.gpu(handle).is_none() {
                                let data = match self.$field.data_source(handle) {
                                    Some(DataSource::Inline(data)) => Some(data),
                                    Some(DataSource::File { source, extra }) => {
                                        let parsed = self.memory.get(source)
                                            .expect("File 条目的 source 应在内存层（先 ensure_loaded）");
                                        let parsed = parsed
                                            .downcast_ref::<<$loader as AssetLoader<$ty>>::Parsed>()
                                            .expect("内存层解析结果类型与加载器 Parsed 不匹配（解析器 bug）");
                                        let extra = extra
                                            .downcast_ref::<<$loader as AssetLoader<$ty>>::Extra>()
                                            .expect("extra 类型与加载器 Extra 不匹配（解析器 bug）");
                                        self.[<$field _loader>].entry(parsed, extra)
                                    }
                                    None => None,
                                };
                                if let Some(data) = data {
                                    let gpu = self.[<$field _uploader>]
                                        .upload(device, queue, data);
                                    *self.$field.gpu_mut(handle).expect("存活") = Some(gpu);
                                }
                            }
                        }
                        self.$field.pin(handle)
                    }
                )*
            }
        }
    };

}

asset_types! {
    meshes: Mesh => MeshGpu, MeshUploader, GlbLoader,
    textures: Texture => TextureGpu, TextureUploader, GlbLoader,
}

/// 资产管理器实现网格数据源：场景碰撞/调试按句柄取 CPU 网格。
impl MeshSource for AssetManager {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh> {
        self.get_meshes(handle)
    }
}

impl AssetManager {
    /// 把文件解析结果放入内存层（`GamePath` → 文件级数据，供 File 条目定位）。
    ///
    /// 加载器（如 [`GlbLoader`]）与 `load_scene` 共用：先注册 File 条目，
    /// 再把解析结果整体放入内存层。
    pub fn memory_insert(&mut self, source: GamePath, parsed: Box<dyn Any + Send + Sync>) {
        self.memory.insert(source, parsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 网格内联注册（程序生成资产路径）：data_source 与移除正常。
    #[test]
    fn inline_mesh_registers_and_removes() {
        let space = MergedResourceSpace::new(std::env::temp_dir());
        let mut assets = AssetManager::without_gpu(space);
        let handle = assets.meshes_mut().register(Mesh::cube());
        assert!(matches!(
            assets.meshes().data_source(handle),
            Some(DataSource::Inline(_))
        ));
        assert_eq!(assets.meshes().state(handle), Some(AssetState::Resident));
        assert!(assets.meshes_mut().remove(handle).is_some());
        assert!(assets.meshes().data_source(handle).is_none());
    }
}
