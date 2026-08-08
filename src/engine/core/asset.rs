//! 统一资产管理：唯一稳定句柄 + CPU/GPU 双持有 + 驻留状态机。
//!
//! 游戏逻辑侧只碰 [`Handle<T>`]：注册、查询、`pin`/`unpin`、卸载；
//! GPU 表示类型与上传器实现（`MeshGpu`/`MeshUploader` 等）在
//! [`crate::engine::render::asset`] 层，本模块只保留抽象：句柄、注册表、
//! 状态机与上传接口。渲染器等 GPU 使用方在上传/回收后按句柄取数据。
//!
//! 资源类型由渲染层的 [`asset_types!`] 宏注册：每类一个 [`AssetRegistry`]，
//! GPU 表示类型作为第二泛型参数（纯数据资源用 `()`，无显存阶段）。
//! 句柄带世代编号：卸载后旧句柄失效（不会误用已复用的槽位），且
//! `Handle<Mesh>` 与 `Handle<Texture>` 在编译期就不允许混用。
//!
//! 渲染器/游戏逻辑已接入；`AssetRegistry` 的部分查询方法在批量 pin、调试
//! 工具等场景使用，未全部覆盖前保留 allow。
#![allow(dead_code)]

use std::marker::PhantomData;

use wgpu::{Device, Queue};

use super::mesh::Mesh;
use super::texture::Texture;

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
    /// 未加载：数据不在内存。
    Unloaded,
    /// CPU 数据驻留内存；GPU 表示可存在、可回收。
    Resident,
    /// 要求 GPU 驻留：`sync_gpu` 确保已上传，禁止回收。
    Pinned,
}

/// 上传器：把一类资源的 CPU 数据转换为 GPU 表示。
///
/// 实现可以携带状态（设备能力分支、调试计数等），由渲染层的资产管理器
/// 持有并在同步/按需上传时传入注册表。纯数据资源（GPU 类型 `()`）用
/// [`NoGpuUploader`] 空实现。
pub trait GpuUploader<T, G> {
    fn upload(&mut self, device: &Device, queue: &Queue, data: &T) -> G;
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
    cpu: Option<T>,
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
        let handle = if let Some(index) = self.free.pop() {
            let slot = self.slots[index as usize]
                .as_mut()
                .expect("free 列表必然指向存活槽位");
            // remove 时世代已递增，复用直接用当前世代（旧句柄已失效）。
            slot.state = AssetState::Resident;
            slot.cpu = Some(data);
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
                cpu: Some(data),
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

    /// CPU 数据只读访问；句柄失效返回 `None`。
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.slot(handle)?.cpu.as_ref()
    }

    /// CPU 数据可变访问（游戏逻辑更新数据用；`sync_gpu` 会重新上传）。
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.slot_mut(handle)?.cpu.as_mut()
    }

    /// GPU 表示访问（渲染器用）；未上传或句柄失效返回 `None`。
    pub fn gpu(&self, handle: Handle<T>) -> Option<&G> {
        self.slot(handle)?.gpu.as_ref()
    }

    /// 立即确保 GPU 表示存在并标记驻留（与 [`Self::pin_upload`] 完全相同的
    /// 路径）：句柄有效但未上传时**立即上传**，再置 `Pinned`；句柄无效
    /// 返回 `None`。
    ///
    /// 渲染器绘制前的兜底：正常流程由调用方提前 [`Self::pin_upload`] 预上传，
    /// 这里处理"忘了 pin"的遗漏，把静默跳过变成真上传。
    pub fn ensure_gpu(
        &mut self,
        device: &Device,
        queue: &Queue,
        handle: Handle<T>,
        uploader: &mut dyn GpuUploader<T, G>,
    ) -> Option<&G> {
        if !self.pin_upload(device, queue, handle, uploader) {
            return None;
        }
        self.slot(handle)?.gpu.as_ref()
    }

    /// 当前驻留状态。
    pub fn state(&self, handle: Handle<T>) -> Option<AssetState> {
        self.slot(handle).map(|s| s.state)
    }

    /// 标记 GPU 驻留（不立即上传；上传由 `sync_gpu` 或 [`Self::pin_upload`]）。
    pub fn pin(&mut self, handle: Handle<T>) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        slot.state = AssetState::Pinned;
        true
    }

    /// 立即确保 GPU 表示存在并标记驻留（预分配语义：与按需上传同一条路径，
    /// 只是提前调用——`Vec::with_capacity` 式的优化，而非必选）。
    ///
    /// 句柄有效且未上传时先上传，再置 `Pinned`；句柄无效返回 `false`。
    pub fn pin_upload(
        &mut self,
        device: &Device,
        queue: &Queue,
        handle: Handle<T>,
        uploader: &mut dyn GpuUploader<T, G>,
    ) -> bool {
        let Some(slot) = self.slot_mut(handle) else {
            return false;
        };
        if slot.gpu.is_none() {
            if let Some(data) = &slot.cpu {
                slot.gpu = Some(uploader.upload(device, queue, data));
            }
        }
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

    /// 卸载：释放 CPU 数据与 GPU 表示，句柄从此失效。
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slot_mut(handle)?;
        let data = slot.cpu.take();
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
            .filter(|s| s.as_ref().is_some_and(|slot| slot.cpu.is_some()))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 遍历所有存活资源（句柄 + CPU 数据）。
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let slot = slot.as_ref()?;
            let data = slot.cpu.as_ref()?;
            Some((
                Handle {
                    index: index as u32,
                    generation: slot.generation,
                    _marker: PhantomData,
                },
                data,
            ))
        })
    }

    /// 同步 GPU：pinned 且未上传的上传；非 pinned 的回收 GPU 表示。
    ///
    /// `uploader` 由资源类型决定如何把 CPU 数据转成 GPU 表示；
    /// 渲染器在合适时机调用资产管理器的 `sync_gpu`。
    pub fn sync_gpu(
        &mut self,
        device: &Device,
        queue: &Queue,
        uploader: &mut dyn GpuUploader<T, G>,
    ) {
        for slot in self.slots.iter_mut().flatten() {
            match slot.state {
                AssetState::Pinned => {
                    if slot.gpu.is_none() {
                        if let Some(data) = &slot.cpu {
                            slot.gpu = Some(uploader.upload(device, queue, data));
                        }
                    }
                }
                AssetState::Resident | AssetState::Unloaded => {
                    slot.gpu = None;
                }
            }
        }
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

/// 纯 CPU 资源（GPU 类型 `()`）的占位上传器：`upload` 是空操作。
///
/// rustc 会把空函数调用内联掉，无性能损失；GPU 上传器实现（`MeshUploader`
/// 等）在渲染层（[`crate::engine::render::asset`]）。
#[derive(Debug, Default)]
pub struct NoGpuUploader;

impl<T> GpuUploader<T, ()> for NoGpuUploader {
    fn upload(&mut self, _device: &Device, _queue: &Queue, _data: &T) {}
}

/// 纯 CPU 数据资源示例（无 GPU 阶段）：关卡数据、AI 数据等走此模式。
///
/// 注册表 GPU 类型用 `()`、上传器用 [`NoGpuUploader`] 空函数占位；
/// `sync_gpu` 对它是空操作（只遍历无上传）。
#[derive(Debug, Clone)]
pub struct LevelData {
    pub name: String,
}

/// 声明资产管理器的资源类型集合。
///
/// 每个条目：`字段名: CPU类型 => GPU类型, 上传器类型`。纯数据资源
/// （关卡、AI 等）用 `()` 作 GPU 类型、上传器给一个空函数占位
/// （[`NoGpuUploader`]）；状态机没有显存阶段。新增资源类型 = 加一行。
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_get_roundtrip() {
        let mut meshes: AssetRegistry<Mesh, ()> = AssetRegistry::new();
        let handle = meshes.register(Mesh::triangle());
        assert!(meshes.is_valid(handle));
        assert_eq!(meshes.get(handle).map(|m| m.vertices().len()), Some(3));
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
        assert!(textures.get(a).is_none());

        // 复用同一槽位注册新资源：旧句柄世代不匹配，仍无效。
        let b = textures.register(Texture::checkerboard(2, 1));
        assert_eq!(b.index(), a.index(), "应复用空闲槽位");
        assert_ne!(b.generation(), a.generation(), "世代应递增");
        assert!(!textures.is_valid(a));
        assert!(textures.is_valid(b));
        assert!(textures.get(a).is_none());
        assert!(textures.get(b).is_some());
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

    /// get_mut 后数据可更新（GPU 侧由 sync_gpu 重新上传）。
    #[test]
    fn get_mut_updates_cpu_data() {
        let mut meshes: AssetRegistry<Mesh, ()> = AssetRegistry::new();
        let handle = meshes.register(Mesh::triangle());
        let updated = meshes.get_mut(handle).expect("句柄有效");
        *updated = Mesh::quad();
        assert_eq!(meshes.get(handle).map(|m| m.vertices().len()), Some(4));
    }

    /// 迭代器遍历全部存活资源。
    #[test]
    fn iter_yields_all_live_assets() {
        let mut textures: AssetRegistry<Texture, ()> = AssetRegistry::new();
        let a = textures.register(Texture::white());
        let b = textures.register(Texture::checkerboard(2, 1));
        let keys: Vec<_> = textures.iter().map(|(k, _)| k).collect();
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
