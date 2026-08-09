//! 统一资产管理（类型无关）：唯一稳定句柄 + 类型擦除存储 + 磁盘/内存驻留状态机。
//!
//! - [`Handle<T>`]：带世代的稳定句柄，`T` 是**编译期标记**（不同资源不同类型，
//!   不能混用）；槽位存储本身是类型擦除的；
//! - [`AssetManager`]：类型无关的存储与生命周期管理——槽位表（slotmap）、
//!   注册/移除/驻留状态机、文件来源重载器与反向索引。数据以 `Box<dyn Any>`
//!   存储在**槽位单一存储点**，**解读留给外部**（资产层的 typed 助手负责把
//!   句柄解读成 `&Mesh`/`&Texture` 等）。
//!
//! 职责边界：本模块只管理"内存"（槽位、重载器、状态机），不解读数据；
//! 文件解析/条目解读在 [`crate::engine::asset`]，显存驻留在渲染层 `GpuManager`。
//!
//! 按职责拆分为子模块：
//! - [`load`](self::load)：同步/异步文件加载与重载器；
//! - [`gc`](self::gc)：显式逐出与统一回收；
//! - `tests`：核心测试。
#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use slotmap::{DefaultKey, SlotMap};

use super::data::mesh::Mesh;
use super::gc::GcInfo;
use super::resource::game_path::GamePath;
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

/// 作用域式 pin 守卫（RAII）：由 [`AssetManager::pin_guard`] 构造，
/// 离开作用域时自动 `unpin` 一次（与构造时的 `pin` 配对）。
pub struct PinGuard<'a, T> {
    assets: &'a mut AssetManager,
    handle: Handle<T>,
}

impl<T> PinGuard<'_, T> {
    /// 守卫持有的句柄。
    pub fn handle(&self) -> Handle<T> {
        self.handle
    }
}

impl<T> Drop for PinGuard<'_, T> {
    fn drop(&mut self) {
        self.assets.unpin(self.handle);
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
    /// 要求驻留：至少一个 pin 持有者（引用计数见 [`Slot::pins`]）；GC 不淘汰、
    /// GPU 层上传后不回收。
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
    fn scan(&self, bytes: &[u8]) -> anyhow::Result<Vec<(TypeId, Vec<Box<dyn Any + Send + Sync>>)>>;

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
    /// GC 记录（最近取用、pin 计数；见 [`super::gc::GcInfo`]）。
    gc: GcInfo,
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
        self.file_entries.entry(source).or_default().push(key);
        key
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
            gc: GcInfo {
                last_used: self.use_clock,
                pins: 0,
            },
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
            slot.gc.last_used = now;
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

    /// 标记要求驻留（**引用计数**：每次调用 +1，需与 [`Self::unpin`] 配对）。
    /// 不立即上传；上传由渲染层 `GpuManager` 完成（GPU 驻留按最近使用窗口
    /// 判定，pin 保护的是 CPU 数据不被 `gc` 逐出）。
    pub fn pin<T>(&mut self, handle: Handle<T>) -> bool {
        self.use_clock += 1;
        let now = self.use_clock;
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.gc.pins += 1;
        slot.gc.last_used = now;
        slot.state = AssetState::Pinned;
        true
    }

    /// 解除一次驻留要求（**引用计数**：计数减一，归零才允许回收）。
    /// 软释放：不立即逐出，等 `GpuManager::gc` / [`Self::gc`] 真正执行。
    pub fn unpin<T>(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.gc.pins = slot.gc.pins.saturating_sub(1);
        if slot.gc.pins == 0 {
            slot.state = AssetState::Resident;
        }
        true
    }

    /// 句柄当前是否仍有驻留要求（pin 计数 > 0）。
    pub fn pinned<T>(&self, handle: Handle<T>) -> bool {
        self.slot(handle).is_some_and(|s| s.gc.pins > 0)
    }

    /// 作用域式 pin（RAII）：构造时 `pin`，守卫离开作用域自动 `unpin`。
    ///
    /// 守卫持有 `&mut AssetManager`，存活期间不能再被其他可变借用；需要边
    /// 持有驻留边取用其他资源时，改用显式 `pin`/`unpin` 配对。
    pub fn pin_guard<T>(&mut self, handle: Handle<T>) -> Option<PinGuard<'_, T>> {
        if self.pin(handle) {
            Some(PinGuard {
                assets: self,
                handle,
            })
        } else {
            None
        }
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
        self.slots.iter().filter_map(move |(key, slot)| {
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
    fn slot<T>(&self, handle: Handle<T>) -> Option<&Slot> {
        self.slots.get(handle.key)
    }

    fn slot_mut<T>(&mut self, handle: Handle<T>) -> Option<&mut Slot> {
        self.slots.get_mut(handle.key)
    }
}

mod gc;
mod load;
#[cfg(test)]
mod tests;
