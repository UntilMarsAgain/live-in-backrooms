//! 游戏逻辑的包：游戏内容与玩法逻辑的入口。
//!
//! 内容以"包"为单位组织在 `game-data/` 下，加载与运行时模型封装在
//! [`package`] 模块中；引擎能力通过 [`crate::engine`] 使用。
//! 包加载器只提供单包加载与发现（见 [`package::loader`]），依赖解析、
//! 加载顺序与存储由调用方处理。

pub mod package;

#[allow(unused_imports)] // 预留：接入 App 后使用
pub use package::Package;
