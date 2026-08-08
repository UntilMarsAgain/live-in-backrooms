//! 磁盘的定义：与 `game-data/` 下 TOML 文件一一对应的数据结构（serde），
//! 只用于读写文件；运行时模型见 [`crate::game::package::Package`]，两者不混用。
//! 后续关卡、物品、实体、语言等内容格式也放这里，一个磁盘文件一个模块。

pub mod package;

pub use package::PackageData;
