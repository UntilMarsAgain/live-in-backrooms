//! 游戏运行时视角下的包：加载完成后游戏逻辑直接消费的包模型。
//!
//! 磁盘定义见 [`loader::data`]（serde，与文件一一对应），加载器见 [`loader`]；
//! 磁盘格式与运行时模型严格分离，不混用。

pub mod loader;

use std::collections::BTreeMap;
use std::path::PathBuf;

use loader::data::PackageData;

/// 加载后的内容包（运行时模型，不派生 serde）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    /// 包名，全局唯一（如 `vanilla`）。
    pub name: String,
    /// 人类可读的描述。
    pub description: String,
    /// 包自身版本；对 vanilla 与游戏版本相同。
    pub version: String,
    /// 依赖：包名 → 版本要求。
    pub dependencies: BTreeMap<String, String>,
    /// 包目录在磁盘上的位置（后续读取关卡/资产用，磁盘格式里没有此字段）。
    pub root: PathBuf,
}

impl Package {
    /// 从磁盘定义转换，附加包目录位置等运行时信息。
    pub fn from_data(data: PackageData, root: PathBuf) -> Self {
        Self {
            name: data.name,
            description: data.description,
            version: data.version,
            dependencies: data.dependencies,
            root,
        }
    }
}
