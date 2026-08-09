//! 统一资产管理（类型无关）：唯一稳定句柄 + 类型擦除存储 + 磁盘/内存驻留状态机。
//!
//! - [`Handle<T>`]：带世代的稳定句柄，`T` 是**编译期标记**（不同资源不同类型，
//!   不能混用）；槽位存储本身是类型擦除的；
//! - [`AssetManager`]：类型无关的存储与生命周期管理——注册/移除/驻留状态/
//!   内存层。数据以 `Box<dyn Any>` 存储，**解读留给外部**（资产层的 typed 助手
//!   负责把句柄解读成 `&Mesh`/`&Texture` 等）；
//! - [`MemoryLayer`]：按文件缓存解析结果（类型擦除，供该文件所有条目共享）。
//!
//! 职责边界：本模块只管理"内存"（槽位、内存层、状态机），不解读数据；
//! 文件解析/条目解读在 [`crate::engine::asset`]，显存驻留在渲染层 `GpuManager`。
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::HashMap;
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

/// 资源驻留状态：CPU 数据在不在内存、是否要求显存驻留。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetState {
    /// 已移除：句柄失效，数据与来源均不保留。
    Unloaded,
    /// 数据不在内存，但来源（`GamePath`）保留，可经 `ensure_loaded` 从磁盘重载。
    DiskOnly,
    /// CPU 数据驻留内存（内联数据或内存层文件结果）。
    Resident,
    /// 要求显存驻留：GPU 层上传后禁止回收。
    Pinned,
}

/// 条目数据（类型擦除）：内联数据或文件条目。
///
/// **单一存储点**：无论内联还是文件条目，数据本体都在槽位里（解析结果拷贝
/// 出来后原解析结果即丢弃）；文件条目额外记录来源与定位，供缺失时重读。
pub enum EntryData {
    /// 内存来源：数据本体在槽位里（注册时所有权移入）。
    Inline(Box<dyn Any + Send + Sync>),
    /// GamePath 来源：数据本体也在槽位里（解析结果拷贝出来），
    /// 额外记录来源 + 定位，`data = None` 表示已逐出（DiskOnly）。
    File {
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
        /// 条目自己的数据（重读时经重载器重新解析填入）。
        data: Option<Box<dyn Any + Send + Sync>>,
    },
}

impl std::fmt::Debug for EntryData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inline(_) => write!(f, "Inline(<erased>)"),
            Self::File { source, .. } => write!(f, "File({source})"),
        }
    }
}

/// 资源加载器：把游戏路径解析为若干 CPU 数据条目。
///
/// 解析粒度为**文件**（一个文件可能产出多条，如 glb 的多个网格）；`Extra` 是
/// 条目在解析结果中的定位信息（类型由解析器自定），`Parsed` 是文件级解析结果
/// （内存层按 `GamePath` 缓存一份，供所有条目共享）。
///
/// 本 trait 是**外部解读**的接口：资产层实现它，把类型无关的存储解读成 typed 数据。
pub trait AssetLoader<T> {
    /// 条目定位信息类型（如 glb 的 primitive 索引）。
    type Extra: Any + Send + Sync + PartialEq;
    /// 文件解析结果类型（内存层以 `Box<dyn Any>` 存储，重载时整份复用）。
    type Parsed: Any + Send + Sync;

    /// 解析文件（从合并资源空间读取），返回文件级解析结果。
    fn load(&self, space: &MergedResourceSpace, path: &GamePath) -> anyhow::Result<Self::Parsed>;

    /// 从解析结果产出条目（注册时调用：每条目一个 `(数据, 定位信息)`）。
    fn entries(&self, parsed: &Self::Parsed) -> Vec<(T, Self::Extra)>;

    /// 按定位信息从解析结果取回条目数据（重载/访问时调用）。
    fn entry<'a>(&self, parsed: &'a Self::Parsed, extra: &Self::Extra) -> Option<&'a T>;
}

/// 网格数据源：场景碰撞/调试按句柄取 CPU 网格数据。
///
/// 由资产层的解读视图（`MeshView`）实现——`scene` 只依赖 core，不反向依赖资产层。
pub trait MeshSource {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh>;
}

/// 注册表槽位：世代 + 状态 + 类型标记 + 数据。
#[derive(Debug)]
struct Slot {
    generation: u32,
    state: AssetState,
    /// 注册时的类型标记（`T`），迭代按类型过滤用。
    type_id: TypeId,
    /// `None` = 已移除（世代保留，槽位可复用）。
    data: Option<EntryData>,
}

/// 类型无关的资产管理器：内存（槽位 + 内存层）与驻留状态机。
///
/// 所有带句柄的方法都以 `Handle<T>` 为参数——`T` 只用于编译期区分与
/// `TypeId` 标记；存储本身不关心 `T` 是什么。
pub struct AssetManager {
    space: MergedResourceSpace,
    /// 每文件的"重载器"：条目数据缺失（DiskOnly）时**完整重新解析**该文件，
    /// 按 `extra` 取回对应条目数据。由资产层在加载时配置（解读外部）。
    reloaders: HashMap<
        GamePath,
        Box<
            dyn Fn(&MergedResourceSpace, &dyn Any) -> anyhow::Result<Box<dyn Any + Send + Sync>>
                + Send
                + Sync,
        >,
    >,
    /// 每文件的条目引用计数（**跨类型合计**）：计数归零（所有条目都被 `remove`）
    /// 时才释放重载器（关联计数）。
    file_refs: HashMap<GamePath, u32>,
    /// 反向索引：来源文件 → 该文件注册过的条目句柄（index, generation）。
    ///
    /// 用于"同一 GamePath 已加载则不重复注册"（去重），`remove` 时同步删除；
    /// 键是**规范化后**的 GamePath（`a//b` 与 `a/b` 是同一个键）。
    file_entries: HashMap<GamePath, Vec<(u32, u32)>>,
    slots: Vec<Option<Slot>>,
    /// 空闲槽索引栈（复用被卸载的槽位）。
    free: Vec<u32>,
    /// 变更版本：新增/卸载 +1。
    version: u64,
}

impl std::fmt::Debug for AssetManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AssetManager {{ slots: {}, reloaders: {}, file_refs: {}, file_entries: {} }}",
            self.slots.len(),
            self.reloaders.len(),
            self.file_refs.len(),
            self.file_entries.len(),
        )
    }
}

impl AssetManager {
    pub fn new(space: MergedResourceSpace) -> Self {
        Self {
            space,
            reloaders: HashMap::new(),
            file_refs: HashMap::new(),
            file_entries: HashMap::new(),
            slots: Vec::new(),
            free: Vec::new(),
            version: 0,
        }
    }

    /// 合并资源空间（`GamePath` → 文件流）。
    pub fn space(&self) -> &MergedResourceSpace {
        &self.space
    }

    /// 注册一份内联数据，返回带类型标记的句柄。
    pub fn register<T: Any + Send + Sync>(&mut self, data: T) -> Handle<T> {
        self.register_with_source(TypeId::of::<T>(), EntryData::Inline(Box::new(data)))
    }

    /// 注册一个文件条目：数据**拷贝进槽位**（单一存储点），来源 + 定位供重读。
    pub fn register_file<T: Any + Send + Sync>(
        &mut self,
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
        data: T,
    ) -> Handle<T> {
        *self.file_refs.entry(source.clone()).or_insert(0) += 1;
        let handle = self.register_with_source(
            TypeId::of::<T>(),
            EntryData::File {
                source: source.clone(),
                extra,
                data: Some(Box::new(data)),
            },
        );
        // 反向索引：路径 → 句柄（去重用）。
        self.file_entries
            .entry(source)
            .or_default()
            .push((handle.index, handle.generation));
        handle
    }

    /// 注册一个文件的"重载器"（每个来源一次）：内存层缺失时用它重新解析。
    pub fn set_file_reloader(
        &mut self,
        source: GamePath,
        reload: impl Fn(&MergedResourceSpace, &dyn Any) -> anyhow::Result<Box<dyn Any + Send + Sync>>
        + Send
        + Sync
        + 'static,
    ) {
        self.reloaders.insert(source, Box::new(reload));
    }

    fn register_with_source<T>(&mut self, type_id: TypeId, data: EntryData) -> Handle<T> {
        let handle = if let Some(index) = self.free.pop() {
            let slot = self.slots[index as usize]
                .as_mut()
                .expect("free 列表必然指向存活槽位");
            slot.state = AssetState::Resident;
            slot.type_id = type_id;
            slot.data = Some(data);
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
                type_id,
                data: Some(data),
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

    /// 取数据（downcast 到 `T`）：内联直接返回；文件条目直接取槽位里的数据，
    /// 数据已被逐出（DiskOnly）时经重载器**完整重解析**该文件并按 `extra` 取回。
    ///
    /// 句柄类型与注册类型一致（register 时由 `T` 决定），downcast 失败只可能是
    /// 程序 bug——debug 构建直接暴露，release 退化为 `None`。
    pub fn get<T: Any>(&mut self, handle: Handle<T>) -> Option<&T> {
        self.ensure_entry_data(handle)?;
        self.get_cached(handle)
    }

    /// 只读取用（不触发重载）：数据须已在内存（碰撞/调试等只读路径用）。
    pub fn get_cached<T: Any>(&self, handle: Handle<T>) -> Option<&T> {
        let slot = self.slot(handle)?;
        match slot.data.as_ref()? {
            EntryData::Inline(boxed) => {
                let out = boxed.downcast_ref::<T>();
                debug_assert!(out.is_some(), "句柄类型与槽位数据不符（程序 bug）");
                out
            }
            EntryData::File {
                data: Some(boxed), ..
            } => {
                let out = boxed.downcast_ref::<T>();
                debug_assert!(out.is_some(), "句柄类型与槽位数据不符（程序 bug）");
                out
            }
            EntryData::File { data: None, .. } => None,
        }
    }

    /// 条目的数据来源（外部据此解读文件条目）。
    pub fn data_source<T>(&self, handle: Handle<T>) -> Option<&EntryData> {
        self.slot(handle)?.data.as_ref()
    }

    /// 句柄的文件来源（File 条目返回 `GamePath`；内联条目返回 `None`）。
    pub fn source_of<T>(&self, handle: Handle<T>) -> Option<&GamePath> {
        match self.data_source(handle)? {
            EntryData::Inline(_) => None,
            EntryData::File { source, .. } => Some(source),
        }
    }

    /// 确保条目数据在槽位：DiskOnly 的文件条目经重载器**完整重解析**后取回。
    fn ensure_entry_data<T>(&mut self, handle: Handle<T>) -> Option<()> {
        // 1. 取来源（复制，结束槽位借用）。
        let source = match self.data_source(handle)? {
            EntryData::Inline(_) => return Some(()),
            EntryData::File { data: Some(_), .. } => return Some(()),
            EntryData::File { source, .. } => source.clone(),
        };
        // 2. 取重载器（move 出来）。
        let reload = self.reloaders.remove(&source)?;
        // 3. 读定位信息（不可变借用槽位），调用重载器（只需 &self.space）。
        let extra_any: &dyn Any = {
            let entry = self.data_source(handle)?;
            let EntryData::File { extra, .. } = entry else {
                return Some(());
            };
            &**extra
        };
        let parsed = (reload)(&self.space, extra_any).ok()?;
        // 4. 放回重载器 + 把数据写进槽位。
        self.reloaders.insert(source, reload);
        self.set_file_data(handle, parsed);
        Some(())
    }

    fn set_file_data<T>(&mut self, handle: Handle<T>, data: Box<dyn Any + Send + Sync>) {
        if let Some(slot) = self.slot_mut(handle) {
            if let Some(EntryData::File {
                data: slot_data, ..
            }) = &mut slot.data
            {
                *slot_data = Some(data);
                slot.state = AssetState::Resident;
            }
        }
    }

    /// 句柄是否仍指向存活资源（世代匹配）。
    pub fn is_valid<T>(&self, handle: Handle<T>) -> bool {
        self.slot(handle).is_some()
    }

    /// 当前驻留状态。
    pub fn state<T>(&self, handle: Handle<T>) -> Option<AssetState> {
        self.slot(handle).map(|s| s.state)
    }

    /// 设置驻留状态（资产层在内存加载/卸载时调用）。
    pub fn set_state<T>(&mut self, handle: Handle<T>, state: AssetState) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = state;
        true
    }

    /// 标记要求显存驻留（不立即上传；上传由渲染层 `GpuManager` 完成）。
    pub fn pin<T>(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = AssetState::Pinned;
        true
    }

    /// 允许显存回收（软释放）：`GpuManager::gc` 时释放对应显存表示。
    pub fn unpin<T>(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = AssetState::Resident;
        true
    }

    /// 卸载：释放条目数据（EntryData），句柄从此失效。
    pub fn remove<T>(&mut self, handle: Handle<T>) -> Option<EntryData> {
        // 先取出数据并结束槽位借用，再处理引用计数（需要 &mut 其他字段）。
        let data = {
            let slot = self.slot_mut(handle)?;
            slot.state = AssetState::Unloaded;
            slot.generation += 1;
            slot.data.take()
        };
        // 反向索引同步删除 + B1.2 引用计数。
        if let Some(EntryData::File { source, .. }) = &data {
            if let Some(handles) = self.file_entries.get_mut(source) {
                handles.retain(|&(index, generation)| {
                    !(index == handle.index && generation == handle.generation)
                });
                if handles.is_empty() {
                    self.file_entries.remove(source);
                }
            }
            if let Some(count) = self.file_refs.get_mut(source) {
                *count -= 1;
                if *count == 0 {
                    self.file_refs.remove(source);
                    self.reloaders.remove(source);
                }
            }
        }
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

    /// 按类型遍历存活句柄（TypeId 过滤；`T` 必须与注册时的类型一致）。
    pub fn iter_of<T: Any>(&self) -> impl Iterator<Item = Handle<T>> + '_ {
        let target = TypeId::of::<T>();
        self.slots
            .iter()
            .enumerate()
            .filter_map(move |(index, slot)| {
                let slot = slot.as_ref()?;
                if slot.data.is_none() || slot.type_id != target {
                    return None;
                }
                Some(Handle {
                    index: index as u32,
                    generation: slot.generation,
                    _marker: PhantomData,
                })
            })
    }

    /// 某来源文件已注册的 `T` 类型句柄（去重用；没有则返回空）。
    ///
    /// `load_entries`/`load_scene` 在注册前先查它：同路径同类型已加载就直接
    /// 复用句柄，不再解析、不再产生重复条目。
    pub fn loaded_handles_of<T: Any>(&self, source: &GamePath) -> Vec<Handle<T>> {
        let target = TypeId::of::<T>();
        self.file_entries
            .get(source)
            .into_iter()
            .flatten()
            .filter_map(|&(index, generation)| {
                let slot = self.slots.get(index as usize)?.as_ref()?;
                // 只看"是否已注册过该类型"：DiskOnly（数据已逐出）也算，避免
                // 逐出后重复注册；世代匹配保证不是已移除的旧句柄。
                (slot.generation == generation && slot.type_id == target).then(|| Handle {
                        index,
                        generation,
                        _marker: PhantomData,
                    })
            })
            .collect()
    }

    /// 内存卸载（按文件）：命中 File 条目的数据丢弃（置 `DiskOnly`），
    /// 来源 + 定位保留，下次 `get` 经重载器完整重解析。
    pub fn unload_memory(&mut self, source: &GamePath) {
        let indices: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, slot)| {
                let slot = slot.as_ref()?;
                matches!(slot.data.as_ref(), Some(EntryData::File { source: s, .. }) if s == source)
                    .then_some(index)
            })
            .collect();
        for index in indices {
            if let Some(slot) = self.slots[index].as_mut() {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                    }
                }
            }
        }
    }

    /// 内存垃圾回收：释放非 `Pinned` 文件条目的数据（→ `DiskOnly`，下次取用
    /// 自动重读），并清理失效来源的重载器/计数。
    ///
    /// 调用时机由物理刻决定（目前由调用方按需触发）。
    #[allow(dead_code)] // 公共 GC API：物理刻接入前由调用方按需触发
    pub fn gc(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            if slot.state != AssetState::Pinned {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                    }
                }
            }
        }
        let mut in_use = std::collections::HashSet::new();
        for slot in self.slots.iter().flatten() {
            if let Some(EntryData::File { source, .. }) = &slot.data {
                in_use.insert(source.clone());
            }
        }
        self.reloaders.retain(|source, _| in_use.contains(source));
        self.file_entries.retain(|source, _| in_use.contains(source));
        self.file_refs.retain(|source, _| in_use.contains(source));
    }

    fn slot<T>(&self, handle: Handle<T>) -> Option<&Slot> {
        let slot = self.slots.get(handle.index as usize)?.as_ref()?;
        (slot.generation == handle.generation).then_some(slot)
    }

    fn slot_mut<T>(&mut self, handle: Handle<T>) -> Option<&mut Slot> {
        let slot = self.slots.get_mut(handle.index as usize)?.as_mut()?;
        (slot.generation == handle.generation).then_some(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::core::resource::MergedResourceSpace;
    use crate::engine::core::texture::Texture;

    fn manager() -> AssetManager {
        AssetManager::new(MergedResourceSpace::new(std::env::temp_dir()))
    }

    /// 测试用文件重载器：模拟"完整重解析"，按 extra 返回对应数据。
    fn u32_reloader(
        _space: &MergedResourceSpace,
        extra: &dyn Any,
    ) -> anyhow::Result<Box<dyn Any + Send + Sync>> {
        let index = extra
            .downcast_ref::<u32>()
            .ok_or_else(|| anyhow::anyhow!("extra 类型不符"))?;
        Ok(Box::new(10 + *index))
    }

    #[test]
    fn register_and_get_roundtrip() {
        let mut assets = manager();
        let handle = assets.register(Mesh::triangle());
        assert!(assets.is_valid(handle));
        assert!(matches!(
            assets.data_source(handle),
            Some(EntryData::Inline(_))
        ));
        assert_eq!(assets.state(handle), Some(AssetState::Resident));
        assert!(
            assets.get::<Mesh>(handle).is_some(),
            "内联数据应能 downcast 取回"
        );
    }

    /// 世代句柄：卸载后旧句柄失效；复用槽位不会误用旧句柄。
    #[test]
    fn removed_handle_stays_invalid_across_slot_reuse() {
        let mut assets = manager();
        let a = assets.register(Texture::white());
        let removed = assets.remove(a);
        assert!(removed.is_some());
        assert!(!assets.is_valid(a));
        assert!(assets.data_source(a).is_none());

        // 复用同一槽位注册新资源：旧句柄世代不匹配，仍无效。
        let b = assets.register(Texture::checkerboard(2, 1));
        assert_eq!(b.index(), a.index(), "应复用空闲槽位");
        assert_ne!(b.generation(), a.generation(), "世代应递增");
        assert!(!assets.is_valid(a));
        assert!(assets.is_valid(b));
    }

    /// pin/unpin 切换驻留状态；对失效句柄操作返回 false。
    #[test]
    fn pin_unpin_transitions_state() {
        let mut assets = manager();
        let handle = assets.register(Mesh::quad());
        assert!(assets.pin(handle));
        assert_eq!(assets.state(handle), Some(AssetState::Pinned));
        assert!(assets.unpin(handle));
        assert_eq!(assets.state(handle), Some(AssetState::Resident));

        let stale = Handle::<Mesh> {
            index: 999,
            generation: 1,
            _marker: PhantomData,
        };
        assert!(!assets.pin(stale));
        assert!(!assets.unpin(stale));
    }

    /// 按类型遍历：只产出该类型的存活句柄。
    #[test]
    fn iter_of_filters_by_type() {
        let mut assets = manager();
        let tex_a = assets.register(Texture::white());
        let tex_b = assets.register(Texture::checkerboard(2, 1));
        let _mesh = assets.register(Mesh::cube());

        let tex_handles: Vec<_> = assets.iter_of::<Texture>().collect();
        assert_eq!(tex_handles.len(), 2);
        assert!(tex_handles.contains(&tex_a));
        assert!(tex_handles.contains(&tex_b));
        assert_eq!(assets.iter_of::<Mesh>().count(), 1);
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

    /// B1.2 关联计数：重载器只在最后一条存活条目被 remove 时释放。
    #[test]
    fn file_reloader_freed_when_last_entry_removed() {
        let mut assets = manager();
        let path: GamePath = "test:file.glb".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);
        let a = assets.register_file::<u32>(path.clone(), Box::new(0u32), 7u32);
        let b = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);
        assert_eq!(assets.file_refs.get(&path), Some(&2));

        // 只移除一条：引用计数仍 >0，重载器保留。
        assert!(assets.remove(a).is_some());
        assert!(!assets.reloaders.is_empty());

        // 最后一条被移除：重载器释放（无需 gc）。
        assert!(assets.remove(b).is_some());
        assert!(assets.file_refs.is_empty());
        assert!(assets.reloaders.is_empty());
    }

    /// 反向索引：路径 → 句柄；remove 时同步删除；规范化路径是同一个键。
    #[test]
    fn file_entries_index_tracks_and_sync_removes() {
        let mut assets = manager();
        let path: GamePath = "test:a//b.glb".parse().unwrap(); // 规范化 → a/b.glb
        assert_eq!(path.path(), "a/b.glb");
        let a = assets.register_file::<u32>(path.clone(), Box::new(0u32), 7u32);
        let b = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);

        let handles = assets.loaded_handles_of::<u32>(&path);
        assert_eq!(handles.len(), 2);
        assert!(handles.contains(&a));
        assert!(handles.contains(&b));

        // remove 同步删除索引条目。
        assets.remove(a);
        assert_eq!(assets.loaded_handles_of::<u32>(&path), vec![b]);
        assets.remove(b);
        assert!(assets.loaded_handles_of::<u32>(&path).is_empty());
        assert!(assets.file_entries.is_empty());
    }

    /// 文件条目：数据本体在槽位（单一存储点），来源可见。
    #[test]
    fn file_entry_holds_own_data_and_source_visible() {
        let mut assets = manager();
        let path: GamePath = "test:file.glb".parse().expect("合法路径");
        let handle = assets.register_file::<u32>(path.clone(), Box::new(1u32), 8u32);
        assert_eq!(assets.get(handle), Some(&8u32), "文件条目数据在槽位");
        assert_eq!(assets.source_of(handle), Some(&path));
        assert!(matches!(
            assets.data_source(handle),
            Some(EntryData::File { source, .. }) if source == &path
        ));
    }

    /// 文件条目封装：内存层缺失时 `get` 自动经重载器重读（无需外部 ensure）。
    #[test]
    fn file_entry_get_auto_reloads_from_disk() {
        let mut assets = manager();
        let path: GamePath = "test:data.bin".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);

        // 数据在槽位：get 直接返回。
        let handle = assets.register_file::<u32>(path.clone(), Box::new(2u32), 99u32);
        assert_eq!(assets.get(handle), Some(&99u32));

        // 卸载后数据丢弃 → get 经重载器完整重解析（按 extra 取回）。
        assets.unload_memory(&path);
        assert_eq!(assets.state(handle), Some(AssetState::DiskOnly));
        assert!(assets.get_cached(handle).is_none());
        assert_eq!(assets.get(handle), Some(&12u32), "重读后返回新数据");
        assert_eq!(assets.state(handle), Some(AssetState::Resident));
    }

    /// gc：释放非 Pinned 文件条目的数据（→ DiskOnly），Pinned 保留。
    #[test]
    fn gc_evicts_unpinned_file_data() {
        let mut assets = manager();
        let path: GamePath = "test:data.bin".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);
        let pinned = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
        let loose = assets.register_file::<u32>(path.clone(), Box::new(1u32), 2u32);
        assets.pin(pinned);

        assets.gc();
        // Pinned 保留数据；非 Pinned 逐出，但 get 能重读。
        assert!(assets.get_cached(pinned).is_some());
        assert!(assets.get_cached(loose).is_none());
        assert_eq!(assets.get(loose), Some(&11u32));
    }
}
