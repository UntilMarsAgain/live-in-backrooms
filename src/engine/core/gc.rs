//! 统一 GC：CPU（[`crate::engine::core::asset::AssetManager`]）与 GPU
//! （渲染层 `GpuManager`）共用同一种淘汰算法与同一种记录结构。
//!
//! - [`GcInfo`]：**每条目**的记录（最近取用、pin 计数；后续加内存占用、
//!   优先级等只扩这个结构体）；
//! - [`GcPolicy`]：**每次 GC 调用**的控制参数 + 通用判定
//!   （[`GcPolicy::should_keep`]）。两侧的 `gc()` 都只是拿自己的时钟和条目
//!   记录喂给这个函数，算法本身只在这里改。

/// 条目级 GC 记录（CPU 槽位与 GPU 驻留表**共用**）。
///
/// 后续要加"内存占用/优先级/预算水位"等更智能的淘汰信息时，**只改这个
/// 结构体**（连同 [`GcPolicy::should_keep`] 的判定），两个管理器的 API 不动。
#[derive(Debug, Clone, Copy)]
pub struct GcInfo {
    /// 最近一次"取用"的时钟序号（注册/取用/pin 时刷新）。
    pub last_used: u64,
    /// pin 引用计数：谁需要驻留谁 +1；归零即允许按窗口淘汰。
    /// （GPU 侧条目目前恒为 0——GPU 驻留按最近使用窗口判定。）
    pub pins: u32,
}

/// GC 控制参数：CPU 与 GPU 的 `gc()` 共用**同一算法**，只是参数不同
/// （CPU 逐出会触发磁盘重读，GPU 会触发重新上传，代价不同 → 窗口不同）。
#[derive(Debug, Clone, Copy)]
pub struct GcPolicy {
    /// 最近使用窗口（时钟序号）：`last_used < now - stale_window` 的条目淘汰。
    pub stale_window: u64,
    /// 是否允许淘汰仍被 pin 的条目（默认否；内存压力场景可开）。
    pub evict_pinned: bool,
    /// 为 `true` 时**忽略 `stale_window`**：不区分最近使用，释放全部未 pin 的
    /// 条目（全量清扫；配合 `evict_pinned` 可连 pinned 一起清）。
    pub ignore_stale_window: bool,
}

impl Default for GcPolicy {
    fn default() -> Self {
        Self {
            stale_window: 0,
            evict_pinned: false,
            ignore_stale_window: false,
        }
    }
}

impl GcPolicy {
    /// **通用淘汰判定**：给定一条目的记录与"现在"（所属管理器的时钟），
    /// 返回是否保留。两侧 `gc()` 只调用这一个函数决定去留——调算法就改这里。
    ///
    /// 当前规则：被 pin 的条目保留（除非 [`Self::evict_pinned`]）；否则
    /// 窗口内（`last_used >= now - stale_window`）保留，超窗淘汰。
    pub fn should_keep(&self, info: &GcInfo, now: u64) -> bool {
        let pinned_kept = info.pins > 0 && !self.evict_pinned;
        if self.ignore_stale_window {
            // 全量清扫：不管最近使用，只有 pin 保护生效。
            return pinned_kept;
        }
        let fresh = info.last_used >= now.saturating_sub(self.stale_window);
        pinned_kept || fresh
    }
}
