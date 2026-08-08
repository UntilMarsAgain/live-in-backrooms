//! 磁盘上 package 的定义：对应包目录下的 `package.toml`。
//!
//! 字段与文件内容一一对应，仅用于读写；加载后的运行时模型见
//! [`crate::game::package::Package`]。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// 包清单的磁盘格式（`package.toml`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageData {
    /// 包名，全局唯一（如 `vanilla`）。
    pub name: String,
    /// 人类可读的描述。
    pub description: String,
    /// 包自身版本；对 vanilla 与游戏版本相同。
    pub version: String,
    /// 依赖：包名 → 版本要求（解析方式同 Cargo）。
    ///
    /// 清单中没有 `[dependencies]` 时默认为空，便于旧文件向后兼容。
    #[serde(default)]
    pub dependencies: BTreeMap<String, String>,
}
