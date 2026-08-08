//! package 的加载器：发现 `game-data/` 下的内容包，
//! 把磁盘定义（[`data`]）解析并转换为运行时模型（[`crate::game::package::Package`]）。
//!
//! 本模块只负责"单包加载"与"包发现"；依赖解析、加载顺序与包的存储
//! 由调用方（关卡加载流程 / App）处理。

pub mod data;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use super::Package;

/// 辅助函数：在 `game_data_root` 下搜索所有一级子目录（不递归），
/// 对存在 `package.toml` 的目录读取并解析，转换为运行时包后返回。
#[allow(dead_code)] // 预留：接入关卡加载流程后由调用方使用
pub fn discover_packages(game_data_root: &Path) -> Result<Vec<Package>> {
    let mut packages = Vec::new();
    for entry in fs::read_dir(game_data_root)
        .with_context(|| format!("无法读取游戏数据目录 {}", game_data_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest = path.join("package.toml");
        if !manifest.is_file() {
            continue;
        }
        packages.push(load_package(&path)?);
    }
    Ok(packages)
}

/// 读取单个包目录的 `package.toml`，转换为运行时包。
///
/// 只负责这一个包的加载；依赖解析、加载顺序与存储由调用方处理。
#[allow(dead_code)] // 预留：接入关卡加载流程后由调用方使用
pub fn load_package(package_root: &Path) -> Result<Package> {
    let manifest = package_root.join("package.toml");
    let text = fs::read_to_string(&manifest)
        .with_context(|| format!("读取 {} 失败", manifest.display()))?;
    let data: data::PackageData =
        toml::from_str(&text).with_context(|| format!("解析 {} 失败", manifest.display()))?;
    Ok(Package::from_data(data, package_root.to_path_buf()))
}
