//! 统一资产管理（共享层）：唯一稳定句柄 + CPU 数据 + 磁盘/内存驻留状态机。
//!
//! 游戏逻辑侧只碰 [`Handle<T>`]：注册、查询、加载/卸载。GPU 表示类型与
//! 上传器（`MeshGpu`/`MeshUploader` 等）在客户端（lbr-client 的 render::asset）层；
//! 本模块只保留**无 GPU 依赖**的抽象：句柄、注册表、数据来源与驻留状态机。
//! 服务端用 `AssetRegistry<T, ()>`（无显存阶段）即可获得同样的磁盘/内存管理。
//!
//! 资源类型由渲染层的 [`asset_types!`] 宏注册：每类一个 [`AssetRegistry`]，
//! GPU 表示类型作为第二泛型参数（纯数据资源用 `()`，无显存阶段）。
//! 句柄带世代编号：卸载后旧句柄失效（不会误用已复用的槽位），且
//! `Handle<Mesh>` 与 `Handle<Texture>` 在编译期就不允许混用。
//!
//! 渲染器/游戏逻辑已接入；`AssetRegistry` 的部分查询方法在批量 pin、调试
//! 工具等场景使用，未全部覆盖前保留 allow。
#![allow(dead_code)]

use std::any::Any;
use std::marker::PhantomData;

use super::game_path::GamePath;
use super::mesh::Mesh;
use super::resource::MergedResourceSpace;

/// 资源句柄：世代编号 + 类型参数。
#[derive(Debug)]
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<T>,
}

// 句柄只按 (index, generation) 比较/哈希，类型参数 T 只是编译期标记，
// 因此这些 trait 的实现不依赖 T：Handle<Mesh> 与 Handle<Texture> 仍是
// 不同类型（T 在类型层面区分），但同一 T 的句柄可以自由拷贝比较。
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> Handle<T> {
    pub fn index(self) -> usize {
        self.index as usize
    }

    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// 资源驻留状态：内存（CPU 数据）与显存（GPU 表示）的生命周期。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetState {
    /// 已移除：句柄失效，数据与来源均不保留。
    Unloaded,
    /// 数据不在内存，但来源（`GamePath`）保留，可经 `ensure_loaded` 从磁盘重载。
    DiskOnly,
    /// CPU 数据驻留内存（内联数据或内存层文件结果）；GPU 表示可存在、可回收。
    Resident,
    /// 要求 GPU 驻留：上传后禁止回收。
    Pinned,
}

/// 条目的 CPU 数据来源。
///
/// - [`DataSource::Inline`]：程序生成/直接给的（`Mesh::cube`），数据随注册表持有；
/// - [`DataSource::File`]：来自文件解析结果（内存层），`extra` 是解析器自定义的
///   定位信息（如 glb 的 primitive 索引、音频的音轨名）——具体类型只有解析器
///   知道，取回时按 [`AssetLoader::Extra`] downcast。
pub enum DataSource<T> {
    Inline(T),
    File {
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
    },
}

// `extra` 是类型擦除的 `Box<dyn Any>`，无法 derive Debug；手动实现只打印来源。
impl<T: std::fmt::Debug> std::fmt::Debug for DataSource<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inline(data) => write!(f, "Inline({data:?})"),
            Self::File { source, .. } => write!(f, "File({source})"),
        }
    }
}

/// 资源加载器：把游戏路径解析为若干 CPU 数据条目。
///
/// 解析粒度为**文件**（一个文件可能产出多条，如 glb 的多个网格）；`Extra` 是
/// 条目在解析结果中的定位信息（类型由解析器自定，通过 `Box<dyn Any>` 存储），
/// `Parsed` 是文件级解析结果（内存层按 `GamePath` 缓存一份，供所有条目共享）。
pub trait AssetLoader<T> {
    /// 条目定位信息类型（如 glb 的 primitive 索引、音频的音轨名）。
    type Extra: Any + Send + Sync + PartialEq;
    /// 文件解析结果类型（内存层以 `Box<dyn Any>` 存储，重载时整份复用）。
    type Parsed: Any + Send + Sync;

    /// 解析文件（从合并资源空间读取），返回文件级解析结果。
    fn load(&self, space: &MergedResourceSpace, path: &GamePath)
        -> anyhow::Result<Self::Parsed>;

    /// 从解析结果产出条目（注册时调用：每条目一个 `(数据, 定位信息)`）。
    fn entries(&self, parsed: &Self::Parsed) -> Vec<(T, Self::Extra)>;

    /// 按定位信息从解析结果取回条目数据（重载/访问时调用）。
    ///
    /// 返回引用绑定 `parsed`（内存层数据），而非 `self`——这样调用方拿到的
    /// 数据引用只借内存层，可以与上传器等其他字段的可变借用共存。
    fn entry<'a>(&self, parsed: &'a Self::Parsed, extra: &Self::Extra) -> Option<&'a T>;
}

/// 网格数据源：场景碰撞/调试按句柄取 CPU 网格数据。
///
/// 由持有资源的资产管理器实现，隔离 core 与 GPU 实现层（scene 只依赖
/// core，不反向依赖 render）。
pub trait MeshSource {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh>;
}

/// 注册表槽位：世代 + 状态 + CPU 数据 + GPU 表示。
#[derive(Debug)]
struct Slot<T, G> {
    generation: u32,
    state: AssetState,
    /// `None` = 已移除（世代保留，槽位可复用）。
    data: Option<DataSource<T>>,
    gpu: Option<G>,
}

/// 单类资源注册表：世代句柄 + 槽位复用 + 版本号。
#[derive(Debug)]
pub struct AssetRegistry<T, G> {
    slots: Vec<Option<Slot<T, G>>>,
    /// 空闲槽索引栈（复用被卸载的槽位）。
    free: Vec<u32>,
    /// 变更版本：新增/卸载 +1，驱动 GPU 侧增量同步。
    version: u64,
}

impl<T, G> AssetRegistry<T, G> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            version: 0,
        }
    }

    /// 注册一份 CPU 数据，返回稳定句柄（世代防复用误用）。
    pub fn register(&mut self, data: T) -> Handle<T> {
        self.register_with_source(DataSource::Inline(data))
    }

    /// 注册一个来自文件解析结果的条目（`source` 为文件路径，`extra` 为
    /// 条目定位信息）。数据本体在内存层的文件结果中，注册表只记来源。
    pub fn register_file(
        &mut self,
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
    ) -> Handle<T> {
        self.register_with_source(DataSource::File { source, extra })
    }

    fn register_with_source(&mut self, data: DataSource<T>) -> Handle<T> {
        let handle = if let Some(index) = self.free.pop() {
            let slot = self.slots[index as usize]
                .as_mut()
                .expect("free 列表必然指向存活槽位");
            // remove 时世代已递增，复用直接用当前世代（旧句柄已失效）。
            slot.state = AssetState::Resident;
            slot.data = Some(data);
            slot.gpu = None;
            Handle {
                index,
                generation: slot.generation,
                _marker: PhantomData,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Some(Slot {
                generation: 1,
                state: AssetState::Resident,
                data: Some(data),
                gpu: None,
            }));
            Handle {
                index,
                generation: 1,
                _marker: PhantomData,
            }
        };
        self.version += 1;
        handle
    }

    /// 句柄是否仍指向存活资源（世代匹配）。
    pub fn is_valid(&self, handle: Handle<T>) -> bool {
        self.slot(handle).is_some()
    }

    /// 条目的数据来源（内联数据或文件引用）；句柄失效返回 `None`。
    pub fn data_source(&self, handle: Handle<T>) -> Option<&DataSource<T>> {
        self.slot(handle)?.data.as_ref()
    }

    /// GPU 表示访问（渲染器用）；未上传或句柄失效返回 `None`。
    pub fn gpu(&self, handle: Handle<T>) -> Option<&G> {
        self.slot(handle)?.gpu.as_ref()
    }

    /// 当前驻留状态。
    pub fn state(&self, handle: Handle<T>) -> Option<AssetState> {
        self.slot(handle).map(|s| s.state)
    }

    /// 设置驻留状态（AssetManager 在内存加载/卸载时调用）。
    pub fn set_state(&mut self, handle: Handle<T>, state: AssetState) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = state;
        true
    }

    /// 标记 GPU 驻留（不立即上传；上传由 `sync_gpu` 或 [`Self::pin_upload`]）。
    pub fn pin(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = AssetState::Pinned;
        true
    }

    /// 允许 GPU 回收：`sync_gpu` 时释放 GPU 表示（CPU 数据保留）。
    pub fn unpin(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = AssetState::Resident;
        true
    }

    /// 卸载：释放 CPU 数据（DataSource）与 GPU 表示，句柄从此失效。
    pub fn remove(&mut self, handle: Handle<T>) -> Option<DataSource<T>> {
        let slot = self.slot_mut(handle)?;
        let data = slot.data.take();
        slot.gpu = None; // drop GPU 资源，wgpu 延迟到队列空闲回收
        slot.state = AssetState::Unloaded;
        slot.generation += 1; // 旧句柄从此失效（世代不匹配）
        self.free.push(handle.index);
        self.version += 1;
        data
    }

    /// 变更版本（新增/卸载 +1）。
    pub fn version(&self) -> u64 {
        self.version
    }

    /// 存活资源数。
    pub fn len(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.as_ref().is_some_and(|slot| slot.data.is_some()))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 遍历所有存活资源句柄（供 AssetManager 同步 GPU / 收集清单用）。
    pub fn iter(&self) -> impl Iterator<Item = Handle<T>> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let slot = slot.as_ref()?;
            if slot.data.is_none() {
                return None;
            }
            Some(
                Handle {
                    index: index as u32,
                    generation: slot.generation,
                    _marker: PhantomData,
                },
            )
        })
    }

    /// GPU 表示的可变访问（AssetManager 上传/回收时写入）。
    pub fn gpu_mut(&mut self, handle: Handle<T>) -> Option<&mut Option<G>> {
        self.slot_mut(handle).map(|slot| &mut slot.gpu)
    }

    fn slot(&self, handle: Handle<T>) -> Option<&Slot<T, G>> {
        let slot = self.slots.get(handle.index as usize)?.as_ref()?;
        (slot.generation == handle.generation).then_some(slot)
    }

    fn slot_mut(&mut self, handle: Handle<T>) -> Option<&mut Slot<T, G>> {
        let slot = self.slots.get_mut(handle.index as usize)?.as_mut()?;
        (slot.generation == handle.generation).then_some(slot)
    }
}

impl<T, G> Default for AssetRegistry<T, G> {
    fn default() -> Self {
        Self::new()
    }
}

/// 声明资产管理器的资源类型集合。
///
/// 每个条目：`字段名: CPU类型 => GPU类型, 上传器类型`。纯数据资源
/// （关卡、AI 等）用 `()` 作 GPU 类型、上传器给一个空函数占位
/// （客户端的 NoGpuUploader，见 lbr-client 的 render::asset）；
/// 状态机没有显存阶段。新增资源类型 = 加一行。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::texture::Texture;

    #[test]
    fn register_and_get_roundtrip() {
        let mut meshes: AssetRegistry<Mesh, ()> = AssetRegistry::new();
        let handle = meshes.register(Mesh::triangle());
        assert!(meshes.is_valid(handle));
        assert!(matches!(meshes.data_source(handle), Some(DataSource::Inline(_))));
        assert_eq!(meshes.state(handle), Some(AssetState::Resident));
    }

    /// 世代句柄：卸载后旧句柄失效；复用槽位不会误用旧句柄。
    #[test]
    fn removed_handle_stays_invalid_across_slot_reuse() {
        let mut textures: AssetRegistry<Texture, ()> = AssetRegistry::new();
        let a = textures.register(Texture::white());
        let removed = textures.remove(a);
        assert!(removed.is_some());
        assert!(!textures.is_valid(a));
        assert!(textures.data_source(a).is_none());

        // 复用同一槽位注册新资源：旧句柄世代不匹配，仍无效。
        let b = textures.register(Texture::checkerboard(2, 1));
        assert_eq!(b.index(), a.index(), "应复用空闲槽位");
        assert_ne!(b.generation(), a.generation(), "世代应递增");
        assert!(!textures.is_valid(a));
        assert!(textures.is_valid(b));
        assert!(textures.data_source(a).is_none());
        assert!(textures.data_source(b).is_some());
    }

    /// pin/unpin 切换驻留状态；对失效句柄操作返回 false。
    #[test]
    fn pin_unpin_transitions_state() {
        let mut meshes: AssetRegistry<Mesh, ()> = AssetRegistry::new();
        let handle = meshes.register(Mesh::quad());
        assert!(meshes.pin(handle));
        assert_eq!(meshes.state(handle), Some(AssetState::Pinned));
        assert!(meshes.unpin(handle));
        assert_eq!(meshes.state(handle), Some(AssetState::Resident));

        let stale = Handle::<Mesh> {
            index: 999,
            generation: 1,
            _marker: PhantomData,
        };
        assert!(!meshes.pin(stale));
        assert!(!meshes.unpin(stale));
    }

    /// 迭代器遍历全部存活资源。
    #[test]
    fn iter_yields_all_live_assets() {
        let mut textures: AssetRegistry<Texture, ()> = AssetRegistry::new();
        let a = textures.register(Texture::white());
        let b = textures.register(Texture::checkerboard(2, 1));
        let keys: Vec<_> = textures.iter().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&a));
        assert!(keys.contains(&b));
    }

    /// 资源类型参数在编译期隔离：Handle<Mesh> 不能传给纹理注册表。
    #[test]
    fn handle_types_are_distinct() {
        let mesh: Handle<Mesh> = Handle {
            index: 0,
            generation: 1,
            _marker: PhantomData,
        };
        let texture: Handle<Texture> = Handle {
            index: 0,
            generation: 1,
            _marker: PhantomData,
        };
        // 不同 T 的 Handle 不是同一个类型（编译器保证），此处仅作存在性说明。
        let _ = (mesh, texture);
    }

}
