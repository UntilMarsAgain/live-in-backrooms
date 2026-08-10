//! 内存回收：显式逐出（`unload_memory`）与统一 GC（`gc`）。
#![allow(dead_code)]

use super::*;
use crate::engine::core::gc::GcPolicy;

impl AssetManager {
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
        let count = keys.len();
        for key in keys {
            if let Some(slot) = self.slots.get_mut(key) {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                    }
                }
            }
        }
        tracing::debug!("资产内存卸载：{source}（{count} 个条目）");
    }

    /// 智能内存回收（与 GPU 侧共用 [`GcPolicy::should_keep`] 判定）：
    /// 被判定淘汰的**文件条目**数据释放（→ `DiskOnly`，下次取用自动重读）；
    /// 并清理失效来源的重载器/计数。
    ///
    /// 调用时机由物理刻决定（目前由调用方按需触发）。
    #[allow(dead_code)] // 公共 GC API：物理刻接入前由调用方按需触发
    pub fn gc(&mut self, policy: &GcPolicy) {
        let now = Instant::now();
        let total = self.slots.len();
        let mut evicted = 0usize;
        for (_, slot) in self.slots.iter_mut() {
            if !policy.should_keep(&slot.gc, now) {
                if let Some(EntryData::File { data, .. }) = &mut slot.data {
                    if data.take().is_some() {
                        slot.state = AssetState::DiskOnly;
                        evicted += 1;
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
        self.file_entries
            .retain(|source, _| in_use.contains(source));
        self.file_refs.retain(|source, _| in_use.contains(source));
        tracing::debug!(
            "资产库 GC：扫描 {total} 个条目，内存逐出 {evicted} 个文件条目"
        );
    }
}
