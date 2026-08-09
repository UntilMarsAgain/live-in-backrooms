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
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use slotmap::{DefaultKey, SlotMap};

use super::game_path::GamePath;
use super::mesh::Mesh;
use super::resource::MergedResourceSpace;

/// 资源句柄：slotmap 世代键 + 类型参数。
///
/// 键的世代由 [`SlotMap`] 管理（删除后旧键永不匹配复用的槽位），
/// `T` 只是编译期标记（不同资源不同类型）。
#[derive(Debug)]
pub struct Handle<T> {
    key: DefaultKey,
    _marker: PhantomData<T>,
}

// 句柄只按 key 比较/哈希，类型参数 T 只是编译期标记，
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
        self.key == other.key
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl<T> Handle<T> {
    /// 底层 slotmap 键（资产库内部使用；外部通常不需要）。
    pub fn key(self) -> DefaultKey {
        self.key
    }
}

/// 资源驻留状态：CPU 数据在不在内存、是否要求显存驻留。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetState {
    /// 已移除：句柄失效，数据与来源均不保留。
    Unloaded,
    /// 占位句柄：异步加载中，数据尚未就绪；`get` 会阻塞等待。
    Loading,
    /// 数据不在内存，但来源（`GamePath`）保留，可经 `ensure_loaded` 从磁盘重载。
    DiskOnly,
    /// CPU 数据驻留内存（内联数据或内存层文件结果）。
    Resident,
    /// 要求显存驻留：GPU 层上传后禁止回收。
    Pinned,
}

/// 单个句柄的取用状态（状态查询 API）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandleState {
    /// 异步加载中（占位句柄）：`get` 会阻塞到完成。
    Loading,
    /// 数据就绪（在槽位中，可直接取用）。
    Ready,
    /// 数据已逐出（DiskOnly）：取用时 `get` 会经重载器重读。
    DiskOnly,
    /// 句柄无效（已移除/不存在）。
    Invalid,
}

/// 整个资产库的状态（库状态查询 API）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetStatus {
    /// 存活条目总数。
    pub entries: usize,
    /// 加载中的条目数（占位句柄）。
    pub loading: usize,
    /// 数据就绪的条目数。
    pub ready: usize,
    /// 数据已逐出（DiskOnly）的条目数。
    pub disk_only: usize,
    /// 后台加载中的任务数（异步加载）。
    pub in_flight: usize,
}

/// 异步加载产出的单条条目（类型擦除的 data + extra）。
pub(crate) type LoadedEntry = (Box<dyn Any + Send + Sync>, Box<dyn Any + Send + Sync>);

/// 异步加载产出的**整个文件**的解析结果：每类型一批条目（按类型独立排序，
/// 与 [`FileLoader::scan`] 的注册顺序一一对应）。
pub(crate) type FileLoadResult = Vec<(TypeId, Vec<LoadedEntry>)>;

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

/// 文件级加载器：一次 scan + 一次 parse 产出**所有类型**的条目。
///
/// 同步（[`AssetManager::load_file`]）与异步（[`AssetManager::load_file_async`]）
/// 共用同一接口：一个文件 → 多种类型多条条目（如 glb 同时产出 Mesh 与 Texture），
/// 避免 per-type 双解析。
pub trait FileLoader: Send + Sync + 'static {
    /// 轻量结构扫描：只读文件结构（**不解析缓冲区**），返回每类型条目的定位信息
    /// （`extra` 列表；同类型内顺序与 [`Self::parse`] 产出的条目一一对应）。
    fn scan(
        &self,
        bytes: &[u8],
    ) -> anyhow::Result<Vec<(TypeId, Vec<Box<dyn Any + Send + Sync>>)>>;

    /// 完整解析：返回每类型条目（data + extra）。
    fn parse(&self, bytes: &[u8]) -> anyhow::Result<Vec<(TypeId, Vec<LoadedEntry>)>>;

    /// 两条定位信息是否指向同一条目（数据逐出后重载时，按它从重解析结果找回
    /// 对应条目；实现按自己 `extra` 的具体类型 downcast 比较即可）。
    fn extra_eq(&self, a: &dyn Any, b: &dyn Any) -> bool;
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
    state: AssetState,
    /// 注册时的类型标记（`T`），迭代按类型过滤用。
    type_id: TypeId,
    /// 最近使用序号（注册/get/pin 时更新；智能 GC 按它淘汰冷数据）。
    last_used: u64,
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
    /// 按 `(类型标记, extra)` 取回对应条目数据。由资产层在加载时配置
    /// （解读外部）；`TypeId` 参数让一个文件（glb）的所有类型共用一个重载器。
    reloaders: HashMap<
        GamePath,
        Box<
            dyn Fn(
                    &MergedResourceSpace,
                    TypeId,
                    &dyn Any,
                ) -> anyhow::Result<Box<dyn Any + Send + Sync>>
                + Send
                + Sync,
        >,
    >,
    /// 每文件的条目引用计数（**跨类型合计**）：计数归零（所有条目都被 `remove`）
    /// 时才释放重载器（关联计数）。
    file_refs: HashMap<GamePath, u32>,
    /// 反向索引：来源文件 → 该文件注册过的条目句柄（slotmap 键）。
    ///
    /// 用于"同一 GamePath 已加载则不重复注册"（去重），`remove` 时同步删除；
    /// 键是**规范化后**的 GamePath（`a//b` 与 `a/b` 是同一个键）。
    file_entries: HashMap<GamePath, Vec<DefaultKey>>,
    /// 槽位表（slotmap：世代键 + 槽位复用由库管理）。
    slots: SlotMap<DefaultKey, Slot>,
    /// 后台加载中的任务（按文件去重：同一文件最多一个解析任务）。
    in_flight: HashSet<GamePath>,
    load_tx: mpsc::Sender<(GamePath, anyhow::Result<FileLoadResult>)>,
    load_rx: mpsc::Receiver<(GamePath, anyhow::Result<FileLoadResult>)>,
    /// 加载完成的唤醒信号（后台线程发送结果后 notify）。
    load_cond: Arc<(Mutex<()>, Condvar)>,
    /// 最近使用时钟（单调递增；智能 GC 的"现在"）。
    use_clock: u64,
    /// 变更版本：新增/卸载 +1。
    version: u64,
}

impl std::fmt::Debug for AssetManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AssetManager {{ slots: {}, reloaders: {}, file_refs: {}, file_entries: {}, \
             in_flight: {} }}",
            self.slots.len(),
            self.reloaders.len(),
            self.file_refs.len(),
            self.file_entries.len(),
            self.in_flight.len(),
        )
    }
}

impl AssetManager {
    pub fn new(space: MergedResourceSpace) -> Self {
        let (load_tx, load_rx) = mpsc::channel();
        Self {
            space,
            reloaders: HashMap::new(),
            file_refs: HashMap::new(),
            file_entries: HashMap::new(),
            slots: SlotMap::with_key(),
            in_flight: HashSet::new(),
            load_tx,
            load_rx,
            load_cond: Arc::new((Mutex::new(()), Condvar::new())),
            use_clock: 0,
            version: 0,
        }
    }

    /// 合并资源空间（`GamePath` → 文件流）。
    pub fn space(&self) -> &MergedResourceSpace {
        &self.space
    }

    /// 注册一份内联数据，返回带类型标记的句柄。
    pub fn register<T: Any + Send + Sync>(&mut self, data: T) -> Handle<T> {
        self.register_with_source(
            AssetState::Resident,
            TypeId::of::<T>(),
            EntryData::Inline(Box::new(data)),
        )
    }

    /// 注册一个文件条目：数据**拷贝进槽位**（单一存储点），来源 + 定位供重读。
    pub fn register_file<T: Any + Send + Sync>(
        &mut self,
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
        data: T,
    ) -> Handle<T> {
        let key = self.register_file_erased(source, TypeId::of::<T>(), extra, Box::new(data));
        Handle {
            key,
            _marker: PhantomData,
        }
    }

    /// 注册一个**类型擦除**的文件条目：数据直接进槽位（`Resident`）。
    ///
    /// 同步加载（[`Self::load_file`]）与 `load_scene`（资产层，需要
    /// glTF document 的特殊入口）共用；调用方只有 `TypeId`，没有静态类型 `T`。
    fn register_file_erased(
        &mut self,
        source: GamePath,
        type_id: TypeId,
        extra: Box<dyn Any + Send + Sync>,
        data: Box<dyn Any + Send + Sync>,
    ) -> DefaultKey {
        *self.file_refs.entry(source.clone()).or_insert(0) += 1;
        let key = self.register_with_source_erased(
            AssetState::Resident,
            type_id,
            EntryData::File {
                source: source.clone(),
                extra,
                data: Some(data),
            },
        );
        // 反向索引：路径 → 句柄（去重用）。
        self.file_entries
            .entry(source)
            .or_default()
            .push(key);
        key
    }

    /// 注册一个**占位文件条目**（异步加载中）：数据 `None`、状态 `Loading`，
    /// 后台解析完成后由 `pump` 填充。
    pub fn register_file_pending<T: Any + Send + Sync>(
        &mut self,
        source: GamePath,
        extra: Box<dyn Any + Send + Sync>,
    ) -> Handle<T> {
        let key = self.register_pending_erased(source, TypeId::of::<T>(), extra);
        Handle {
            key,
            _marker: PhantomData,
        }
    }

    /// 注册一个**类型擦除**的占位文件条目（[`FileLoader`] 异步加载用：
    /// 调用方只有 `TypeId`，没有静态类型 `T`）。占位句柄由调用方经
    /// `loaded_handles_of::<T>` 取回。
    fn register_pending_erased(
        &mut self,
        source: GamePath,
        type_id: TypeId,
        extra: Box<dyn Any + Send + Sync>,
    ) -> DefaultKey {
        *self.file_refs.entry(source.clone()).or_insert(0) += 1;
        let key = self.register_with_source_erased(
            AssetState::Loading,
            type_id,
            EntryData::File {
                source: source.clone(),
                extra,
                data: None,
            },
        );
        self.file_entries
            .entry(source)
            .or_default()
            .push(key);
        key
    }

    /// 注册一个文件的"重载器"（每个来源一次）：内存层缺失时用它重新解析。
    pub fn set_file_reloader(
        &mut self,
        source: GamePath,
        reload: impl Fn(
                &MergedResourceSpace,
                TypeId,
                &dyn Any,
            ) -> anyhow::Result<Box<dyn Any + Send + Sync>>
            + Send
            + Sync
            + 'static,
    ) {
        self.reloaders.insert(source, Box::new(reload));
    }

    /// 文件级重载器：数据逐出（DiskOnly）后**完整重解析**文件，按（类型, extra）
    /// 找回对应条目。同步/异步入口共用，保证所有加载方式的重载行为一致。
    fn file_reloader_for<L>(
        loader: L,
        source: GamePath,
    ) -> Box<
        dyn Fn(
                &MergedResourceSpace,
                TypeId,
                &dyn Any,
            ) -> anyhow::Result<Box<dyn Any + Send + Sync>>
            + Send
            + Sync,
    >
    where
        L: FileLoader + Clone,
    {
        Box::new(move |space: &MergedResourceSpace, type_id: TypeId, extra: &dyn Any| {
            let bytes = space.read(&source)?;
            let parsed = loader.parse(&bytes)?;
            for (tid, entries) in parsed {
                if tid != type_id {
                    continue;
                }
                for (data, entry_extra) in entries {
                    if loader.extra_eq(extra, &*entry_extra) {
                        return Ok(data);
                    }
                }
            }
            anyhow::bail!("重载时找不到对应条目：{source}")
        })
    }

    fn register_with_source<T>(
        &mut self,
        state: AssetState,
        type_id: TypeId,
        data: EntryData,
    ) -> Handle<T> {
        let key = self.register_with_source_erased(state, type_id, data);
        Handle {
            key,
            _marker: PhantomData,
        }
    }

    fn register_with_source_erased(
        &mut self,
        state: AssetState,
        type_id: TypeId,
        data: EntryData,
    ) -> DefaultKey {
        // slotmap 内部管理槽位复用与世代：删除的键永不匹配复用后的槽位。
        self.use_clock += 1;
        let key = self.slots.insert(Slot {
            state,
            type_id,
            last_used: self.use_clock,
            data: Some(data),
        });
        self.version += 1;
        key
    }

    /// 取数据（downcast 到 `T`）：内联直接返回；文件条目直接取槽位里的数据，
    /// 数据已被逐出（DiskOnly）时经重载器**完整重解析**该文件并按 `extra` 取回。
    ///
    /// 句柄类型与注册类型一致（register 时由 `T` 决定），downcast 失败只可能是
    /// 程序 bug——debug 构建直接暴露，release 退化为 `None`。
    pub fn get<T: Any>(&mut self, handle: Handle<T>) -> Option<&T> {
        // Loading（异步加载中的占位句柄）：阻塞等待后台填充完成。
        loop {
            self.pump();
            if self.state(handle) != Some(AssetState::Loading) {
                break;
            }
            let (lock, cvar) = &*self.load_cond;
            let guard = lock.lock().unwrap();
            let _ = cvar.wait_timeout(guard, Duration::from_millis(50)).unwrap();
        }
        self.ensure_entry_data(handle)?;
        // 记录最近使用（智能 GC 按它淘汰冷数据）。
        self.use_clock += 1;
        let now = self.use_clock;
        if let Some(slot) = self.slot_mut(handle) {
            slot.last_used = now;
        }
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
        // 1. 取来源 + 类型标记（复制，结束槽位借用）。
        let (source, type_id) = {
            let slot = self.slot(handle)?;
            match &slot.data {
                Some(EntryData::Inline(_)) => return Some(()),
                Some(EntryData::File { data: Some(_), .. }) => return Some(()),
                Some(EntryData::File { source, .. }) => (source.clone(), slot.type_id),
                None => return None,
            }
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
        let parsed = (reload)(&self.space, type_id, extra_any).ok()?;
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
        self.use_clock += 1;
        let now = self.use_clock;
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.last_used = now;
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
        self.remove_key(handle.key)
    }

    /// 按原始键卸载（异步加载失败清理占位句柄也用）。
    fn remove_key(&mut self, key: DefaultKey) -> Option<EntryData> {
        let slot = self.slots.remove(key)?;
        let data = slot.data;
        // 反向索引同步删除 + B1.2 引用计数。
        if let Some(EntryData::File { source, .. }) = &data {
            if let Some(handles) = self.file_entries.get_mut(source) {
                handles.retain(|k| *k != key);
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
        self.version += 1;
        data
    }

    /// 变更版本（新增/卸载 +1）。
    pub fn version(&self) -> u64 {
        self.version
    }

    /// 存活资源数。
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 按类型遍历存活句柄（TypeId 过滤；`T` 必须与注册时的类型一致）。
    pub fn iter_of<T: Any>(&self) -> impl Iterator<Item = Handle<T>> + '_ {
        let target = TypeId::of::<T>();
        self.slots
            .iter()
            .filter_map(move |(key, slot)| {
                (slot.data.is_some() && slot.type_id == target).then(|| Handle {
                    key,
                    _marker: PhantomData,
                })
            })
    }

    /// 某来源文件已注册的 `T` 类型句柄（去重用；没有则返回空）。
    ///
    /// `load_file`/`load_file_async`/`load_scene` 注册前先查它：
    /// 同路径同类型已加载就直接复用句柄，不再解析、不再产生重复条目。
    pub fn loaded_handles_of<T: Any>(&self, source: &GamePath) -> Vec<Handle<T>> {
        let target = TypeId::of::<T>();
        self.file_entries
            .get(source)
            .into_iter()
            .flatten()
            .filter_map(|key| {
                let slot = self.slots.get(*key)?;
                // 只看"是否已注册过该类型"：DiskOnly（数据已逐出）也算，避免
                // 逐出后重复注册；slotmap 键世代保证不是已移除的旧句柄。
                (slot.type_id == target).then(|| Handle {
                    key: *key,
                    _marker: PhantomData,
                })
            })
            .collect()
    }

    // ---- 文件加载（同步 / 异步） ----

    /// **同步加载（文件级、多类型）**：主线程读文件并**完整解析一次**，立即把
    /// 该文件所有类型的条目注册进槽位（数据直接驻留内存，无占位阶段）。
    ///
    /// - 同文件已有条目（任意类型）→ 直接返回，不解析，复用已有句柄；
    /// - 与 [`Self::load_file_async`] 共用同一重载器逻辑（数据逐出后重放
    ///   [`FileLoader`] 完整重解析）；
    /// - 调用方用 `loaded_handles_of::<T>(&path)` 取各类型句柄。
    pub fn load_file<L>(&mut self, loader: L, path: GamePath) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        let bytes = self.space.read(&path)?;
        let parsed = loader.parse(&bytes)?;
        self.register_parsed_file(loader, path, parsed)
    }

    /// 注册一份**已解析**的文件条目（全部类型）并配置重载器；同文件已有条目
    /// 则跳过。同步/异步加载与 `load_scene`（资产层需要 glTF document 的
    /// 特殊入口）共用此逻辑，保证重载器一致。
    pub(crate) fn register_parsed_file<L>(
        &mut self,
        loader: L,
        path: GamePath,
        parsed: FileLoadResult,
    ) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        // 去重：同路径已有条目（任意类型）→ 复用，不再注册。
        if self.file_entries.contains_key(&path) {
            return Ok(());
        }
        if parsed.is_empty() {
            anyhow::bail!("文件没有可注册的条目：{path}");
        }
        // 条目数据直接进槽位（单一存储点，Resident）。
        for (type_id, entries) in parsed {
            for (data, extra) in entries {
                self.register_file_erased(path.clone(), type_id, extra, data);
            }
        }
        // 配置重载器：数据逐出后完整重解析，按（类型, extra）找回对应条目。
        self.set_file_reloader(path.clone(), Self::file_reloader_for(loader, path));
        Ok(())
    }

    /// **异步加载（文件级、多类型）**：立即注册每类型占位句柄（`Loading`），
    /// 后台线程**完整解析一次**产出该文件所有类型的条目后填充；`get` 遇到
    /// Loading 会阻塞等待。
    ///
    /// - 调用方先拿占位句柄（`loaded_handles_of::<T>(&path)`），随后 `get` 阻塞；
    /// - 同文件已有条目（任意类型，含加载中）→ 直接返回，不 scan、不 parse，
    ///   复用已有句柄；
    /// - 一个文件解析一次产出全部类型（如 glb 的 Mesh + Texture），没有
    ///   per-type 双解析；
    /// - 加载失败：占位句柄全部移除（调用方持有的句柄随之失效）；
    /// - 数据逐出（DiskOnly）后的重读经重载器完整重解析（重放 [`FileLoader`]）。
    pub fn load_file_async<L>(&mut self, loader: L, path: GamePath) -> anyhow::Result<()>
    where
        L: FileLoader + Clone,
    {
        // 去重：同路径已有条目（任意类型）→ 复用，不 scan、不 parse。
        if self.file_entries.contains_key(&path) {
            return Ok(());
        }
        // 1. 主线程读文件 + 轻量结构扫描（只读结构，不解析缓冲区）。
        let bytes = self.space.read(&path)?;
        let scanned = loader.scan(&bytes)?;
        if scanned.is_empty() {
            anyhow::bail!("文件没有可注册的条目：{path}");
        }
        // 2. 按类型注册占位句柄（各类型独立 extra 列表，顺序与 parse 对应）。
        for (type_id, extras) in scanned {
            for extra in extras {
                self.register_pending_erased(path.clone(), type_id, extra);
            }
        }
        // 3. 配置重载器（与同步入口共用同一份逻辑）。
        self.set_file_reloader(
            path.clone(),
            Self::file_reloader_for(loader.clone(), path.clone()),
        );
        // 4. 后台完整解析（一次产出所有类型）→ 回主线程 `pump` 填充占位。
        self.in_flight.insert(path.clone());
        let tx = self.load_tx.clone();
        let cond = self.load_cond.clone();
        std::thread::spawn(move || {
            let result = loader.parse(&bytes);
            let _ = tx.send((path, result));
            cond.1.notify_all();
        });
        Ok(())
    }

    /// 消费后台完成的结果：成功则填充占位句柄的数据，失败则移除占位句柄。
    fn pump(&mut self) {
        while let Ok((path, result)) = self.load_rx.try_recv() {
            self.in_flight.remove(&path);
            let keys: Vec<DefaultKey> = self
                .file_entries
                .get(&path)
                .map(|handles| handles.iter().copied().collect())
                .unwrap_or_default();
            match result {
                Ok(batches) => {
                    // 各类型条目与该类型占位句柄按注册顺序一一对应
                    // （scan 的注册顺序 = parse 的产出顺序）。
                    let mut filled = HashSet::new();
                    for (type_id, entries) in batches {
                        let typed_keys: Vec<DefaultKey> = keys
                            .iter()
                            .copied()
                            .filter(|key| {
                                self.slots
                                    .get(*key)
                                    .is_some_and(|slot| slot.type_id == type_id)
                            })
                            .collect();
                        for (key, (data, _extra)) in typed_keys.into_iter().zip(entries) {
                            if let Some(slot) = self.slots.get_mut(key) {
                                if let Some(EntryData::File { data: slot_data, .. }) =
                                    &mut slot.data
                                {
                                    *slot_data = Some(data);
                                }
                                slot.state = AssetState::Resident;
                            }
                            filled.insert(key);
                        }
                    }
                    // 保险：scan 注册了但 parse 没产出的占位（加载器 bug）按失败清理，
                    // 避免 `get` 永久阻塞。
                    for key in keys {
                        if !filled.contains(&key)
                            && self
                                .slots
                                .get(key)
                                .is_some_and(|slot| slot.state == AssetState::Loading)
                        {
                            self.remove_key(key);
                        }
                    }
                }
                Err(_) => {
                    // 加载失败：该文件的占位句柄全部移除（调用方句柄随之失效）。
                    for key in keys {
                        self.remove_key(key);
                    }
                }
            }
        }
    }

    // ---- 状态查询 ----

    /// 单个句柄的取用状态。
    pub fn handle_state<T>(&self, handle: Handle<T>) -> HandleState {
        match self.state(handle) {
            Some(AssetState::Loading) => HandleState::Loading,
            Some(AssetState::Resident) | Some(AssetState::Pinned) => HandleState::Ready,
            Some(AssetState::DiskOnly) => HandleState::DiskOnly,
            _ => HandleState::Invalid,
        }
    }

    /// 整个资产库的状态（先消费后台完成的结果）。
    pub fn status(&mut self) -> AssetStatus {
        self.pump();
        let mut loading = 0;
        let mut ready = 0;
        let mut disk_only = 0;
        for (_, slot) in self.slots.iter() {
            match slot.state {
                AssetState::Loading => loading += 1,
                AssetState::Resident | AssetState::Pinned => ready += 1,
                AssetState::DiskOnly => disk_only += 1,
                AssetState::Unloaded => {}
            }
        }
        AssetStatus {
            entries: self.slots.len(),
            loading,
            ready,
            disk_only,
            in_flight: self.in_flight.len(),
        }
    }

    /// 内存卸载（按文件）：命中 File 条目的数据丢弃（置 `DiskOnly`），
    /// 来源 + 定位保留，下次 `get` 经重载器完整重解析。
    pub fn unload_memory(&mut self, source: &GamePath) {
        let keys: Vec<DefaultKey> = self
            .slots
            .iter()
            .filter_map(|(key, slot)| {
                matches!(slot.data.as_ref(), Some(EntryData::File { source: s, .. }) if s == source)
                    .then_some(key)
            })
            .collect();
        for key in keys {
            if let Some(slot) = self.slots.get_mut(key) {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                    }
                }
            }
        }
    }

    /// 智能内存回收（基线：按最近使用窗口）：释放非 `Pinned` 且**最近
    /// `stale_window_uses` 次使用内未被取用**的文件条目数据（→ `DiskOnly`，
    /// 下次取用自动重读）；并清理失效来源的重载器/计数。
    ///
    /// 调用时机由物理刻决定（目前由调用方按需触发）。
    #[allow(dead_code)] // 公共 GC API：物理刻接入前由调用方按需触发
    pub fn gc(&mut self, stale_window_uses: u64) {
        let cutoff = self.use_clock.saturating_sub(stale_window_uses);
        for (_, slot) in self.slots.iter_mut() {
            if slot.state != AssetState::Pinned && slot.last_used < cutoff {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                    }
                }
            }
        }
        let mut in_use = std::collections::HashSet::new();
        for (_, slot) in self.slots.iter() {
            if let Some(EntryData::File { source, .. }) = &slot.data {
                in_use.insert(source.clone());
            }
        }
        self.reloaders.retain(|source, _| in_use.contains(source));
        self.file_entries.retain(|source, _| in_use.contains(source));
        self.file_refs.retain(|source, _| in_use.contains(source));
    }

    fn slot<T>(&self, handle: Handle<T>) -> Option<&Slot> {
        self.slots.get(handle.key)
    }

    fn slot_mut<T>(&mut self, handle: Handle<T>) -> Option<&mut Slot> {
        self.slots.get_mut(handle.key)
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
        _type_id: TypeId,
        extra: &dyn Any,
    ) -> anyhow::Result<Box<dyn Any + Send + Sync>> {
        let index = extra
            .downcast_ref::<u32>()
            .ok_or_else(|| anyhow::anyhow!("extra 类型不符"))?;
        Ok(Box::new(10 + *index))
    }

    /// 临时合并资源空间：在系统临时目录建 `test/{file}` 并写入内容。
    fn temp_manager_with(tag: &str, file: &str, content: &[u8]) -> (AssetManager, GamePath) {
        let dir = std::env::temp_dir().join(format!("asset-async-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        let ns = dir.join("test");
        std::fs::create_dir_all(&ns).expect("创建测试目录");
        std::fs::write(ns.join(file), content).expect("写测试文件");
        let space = MergedResourceSpace::new(dir);
        let path: GamePath = format!("test:{file}").parse().expect("合法路径");
        (AssetManager::new(space), path)
    }

    /// 测试用异步文件加载器：scan 读结构（忽略内容）、parse 产出一份数据。
    /// `fail_parse` 模拟"扫描成功但解析失败"；`multi_type` 额外产出 String 条目。
    #[derive(Clone)]
    struct FakeFileLoader {
        fail_parse: bool,
        multi_type: bool,
    }

    impl FileLoader for FakeFileLoader {
        fn scan(
            &self,
            _bytes: &[u8],
        ) -> anyhow::Result<Vec<(TypeId, Vec<Box<dyn Any + Send + Sync>>)>> {
            let mut out = vec![(
                TypeId::of::<u32>(),
                vec![Box::new(0u32) as Box<dyn Any + Send + Sync>],
            )];
            if self.multi_type {
                out.push((
                    TypeId::of::<String>(),
                    vec![Box::new(0u32) as Box<dyn Any + Send + Sync>],
                ));
            }
            Ok(out)
        }

        fn parse(&self, _bytes: &[u8]) -> anyhow::Result<Vec<(TypeId, Vec<LoadedEntry>)>> {
            // 慢一点，让"Loading"状态可观察。
            std::thread::sleep(Duration::from_millis(50));
            if self.fail_parse {
                anyhow::bail!("模拟解析失败");
            }
            let mut out = vec![(
                TypeId::of::<u32>(),
                vec![(
                    Box::new(7u32) as Box<dyn Any + Send + Sync>,
                    Box::new(0u32) as Box<dyn Any + Send + Sync>,
                )],
            )];
            if self.multi_type {
                out.push((
                    TypeId::of::<String>(),
                    vec![(
                        Box::new("hi".to_string()) as Box<dyn Any + Send + Sync>,
                        Box::new(0u32) as Box<dyn Any + Send + Sync>,
                    )],
                ));
            }
            Ok(out)
        }

        fn extra_eq(&self, a: &dyn Any, b: &dyn Any) -> bool {
            match (a.downcast_ref::<u32>(), b.downcast_ref::<u32>()) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            }
        }
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

        // 槽位复用由 slotmap 管理：旧键世代不匹配，永远失效。
        let b = assets.register(Texture::checkerboard(2, 1));
        assert_ne!(b.key(), a.key(), "新句柄应是不同键（世代不同）");
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
            key: DefaultKey::default(),
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
            key: DefaultKey::default(),
            _marker: PhantomData,
        };
        let texture: Handle<Texture> = Handle {
            key: DefaultKey::default(),
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

    /// 智能 gc（按最近使用窗口）：释放非 Pinned 且超窗未使用的文件数据，
    /// Pinned 与最近使用过的保留。
    #[test]
    fn gc_evicts_unpinned_file_data() {
        let mut assets = manager();
        let path: GamePath = "test:data.bin".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);
        let pinned = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
        let loose = assets.register_file::<u32>(path.clone(), Box::new(1u32), 2u32);
        assets.pin(pinned);

        assets.gc(0); // 窗口 0：只保留 Pinned 与"此刻"使用的。
        // Pinned 保留数据；非 Pinned 逐出，但 get 能重读。
        assert!(assets.get_cached(pinned).is_some());
        assert!(assets.get_cached(loose).is_none());
        assert_eq!(assets.get(loose), Some(&11u32));
    }

    /// 最近使用保护：get 过的条目在窗口内不被 gc 逐出。
    #[test]
    fn gc_keeps_recently_used_entries() {
        let mut assets = manager();
        let path: GamePath = "test:data.bin".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);
        let old = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
        let fresh = assets.register_file::<u32>(path.clone(), Box::new(1u32), 2u32);
        assets.get(fresh); // 使用 fresh，把它标记为最近使用。

        assets.gc(0);
        assert!(assets.get_cached(fresh).is_some(), "最近使用的应保留");
        assert!(assets.get_cached(old).is_none(), "未使用的应逐出");
    }

    /// 异步加载（文件级）：立即注册占位句柄（Loading），get 阻塞等待填充完成。
    #[test]
    fn load_file_async_parses_off_thread_and_get_waits() {
        let (mut assets, path) = temp_manager_with("wait", "async.bin", b"x");
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: false,
                },
                path.clone(),
            )
            .expect("scan 应成功");
        let handles = assets.loaded_handles_of::<u32>(&path);
        assert_eq!(handles.len(), 1);
        assert_eq!(assets.handle_state(handles[0]), HandleState::Loading);
        assert!(assets.status().in_flight >= 1);

        // get 强制等待：阻塞到后台填充完成。
        assert_eq!(*assets.get(handles[0]).unwrap(), 7);
        assert_eq!(assets.handle_state(handles[0]), HandleState::Ready);
        assert_eq!(assets.status().in_flight, 0);
    }

    /// FileLoader 一次 parse 产出多种类型（1:N）：各类型占位句柄都被填充。
    #[test]
    fn load_file_async_produces_multiple_types() {
        let (mut assets, path) = temp_manager_with("multi", "multi.bin", b"x");
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: true,
                },
                path.clone(),
            )
            .expect("scan 应成功");
        let numbers = assets.loaded_handles_of::<u32>(&path);
        let strings = assets.loaded_handles_of::<String>(&path);
        assert_eq!(numbers.len(), 1);
        assert_eq!(strings.len(), 1);
        assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
        assert_eq!(assets.get(strings[0]).unwrap().as_str(), "hi");
    }

    /// 同步加载（文件级、多类型）：一次 parse 立即注册全部类型条目（无占位阶段）。
    #[test]
    fn load_file_registers_all_types_immediately() {
        let (mut assets, path) = temp_manager_with("sync", "sync.bin", b"x");
        assets
            .load_file(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: true,
                },
                path.clone(),
            )
            .expect("同步加载应成功");
        let numbers = assets.loaded_handles_of::<u32>(&path);
        let strings = assets.loaded_handles_of::<String>(&path);
        assert_eq!(numbers.len(), 1);
        assert_eq!(strings.len(), 1);
        // 同步路径没有占位阶段：注册完就是 Ready。
        assert_eq!(assets.handle_state(numbers[0]), HandleState::Ready);
        assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
        assert_eq!(assets.get(strings[0]).unwrap().as_str(), "hi");
        assert_eq!(assets.status().in_flight, 0);

        // 逐出后 get 经重载器自动重读（与异步入口共用同一重载器逻辑）。
        assets.unload_memory(&path);
        assert_eq!(assets.handle_state(numbers[0]), HandleState::DiskOnly);
        assert_eq!(*assets.get(numbers[0]).unwrap(), 7);
    }

    /// 同文件二次 `load_file_async`：不 scan、不 parse，复用已有句柄。
    #[test]
    fn load_file_async_dedupes_same_path() {
        let (mut assets, path) = temp_manager_with("dedup", "dedup.bin", b"x");
        for _ in 0..2 {
            assets
                .load_file_async(
                    FakeFileLoader {
                        fail_parse: false,
                        multi_type: false,
                    },
                    path.clone(),
                )
                .expect("scan 应成功");
        }
        // 等第一次完成，然后再次调用：仍复用同一个句柄，不新增条目。
        let handles = assets.loaded_handles_of::<u32>(&path);
        assert_eq!(handles.len(), 1);
        assert_eq!(*assets.get(handles[0]).unwrap(), 7);
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: false,
                },
                path.clone(),
            )
            .expect("scan 应成功");
        assert_eq!(assets.loaded_handles_of::<u32>(&path).len(), 1);
        assert_eq!(assets.iter_of::<u32>().count(), 1);
    }

    /// 解析失败：占位句柄全部移除（句柄失效），引用计数与反向索引清理干净。
    #[test]
    fn load_file_async_failure_removes_placeholders() {
        let (mut assets, path) = temp_manager_with("fail", "fail.bin", b"x");
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: true,
                    multi_type: false,
                },
                path.clone(),
            )
            .expect("scan 应成功");
        let handles = assets.loaded_handles_of::<u32>(&path);
        assert_eq!(handles.len(), 1);
        assert!(assets.status().in_flight >= 1);

        // get 阻塞到失败清理完成（占位移除 → 返回 None），比轮询确定。
        assert!(assets.get(handles[0]).is_none(), "失败后 get 应返回 None");
        assert_eq!(assets.status().in_flight, 0);
        assert!(assets.loaded_handles_of::<u32>(&path).is_empty());
        assert_eq!(assets.handle_state(handles[0]), HandleState::Invalid);
        assert!(assets.file_refs.is_empty(), "引用计数应随占位移除清零");
    }

    /// 异步加载的数据逐出（DiskOnly）后，get 经重载器完整重解析并自动取回。
    #[test]
    fn load_file_async_data_reloads_after_eviction() {
        let (mut assets, path) = temp_manager_with("reload", "reload.bin", b"x");
        assets
            .load_file_async(
                FakeFileLoader {
                    fail_parse: false,
                    multi_type: false,
                },
                path.clone(),
            )
            .expect("scan 应成功");
        let handle = assets.loaded_handles_of::<u32>(&path)[0];
        assert_eq!(*assets.get(handle).unwrap(), 7);

        assets.unload_memory(&path);
        assert_eq!(assets.handle_state(handle), HandleState::DiskOnly);
        assert_eq!(*assets.get(handle).unwrap(), 7, "重读后取回新数据");
        assert_eq!(assets.handle_state(handle), HandleState::Ready);
    }

    /// 状态查询：句柄状态（Ready/DiskOnly/Invalid）与库状态。
    #[test]
    fn handle_state_and_status_queries() {
        let mut assets = manager();
        let h = assets.register(5u32);
        assert_eq!(assets.handle_state(h), HandleState::Ready);

        let path: GamePath = "test:data.bin".parse().expect("合法路径");
        assets.set_file_reloader(path.clone(), u32_reloader);
        let f = assets.register_file::<u32>(path.clone(), Box::new(0u32), 1u32);
        assets.unload_memory(&path);
        assert_eq!(assets.handle_state(f), HandleState::DiskOnly);

        let status = assets.status();
        assert_eq!(status.entries, 2);
        assert_eq!(status.ready, 1);
        assert_eq!(status.disk_only, 1);
        assert_eq!(status.in_flight, 0);

        assets.remove(h);
        assert_eq!(assets.handle_state(h), HandleState::Invalid);
    }
}
