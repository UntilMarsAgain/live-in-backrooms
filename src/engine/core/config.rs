//! 配置文件子系统：启动时读取/生成的配置。
//!
//! 配置分两类，存放位置不同：
//! - **内容/数据配置**（资源包顺序、关卡定义）随数据走，放 `game-data/`；
//! - **用户/机器配置**（窗口、键位、图形）放 `config/`（后续实现）。
//!
//! 本模块实现资源包清单 [`PackConfig`]（`game-data/packs.toml`）：
//! - 每次启动扫描 `game-data/` 下的有效包（见 [`super::pack`]）；
//! - 更新顺序：失效包移除、新包按依赖插入、清单不存在则创建；
//! - 生成阶段依赖图有环 → Err（报错退出）；
//! - 加载阶段顺序不满足依赖或存在冲突 → Err（报错退出）。

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::pack::{self, version_satisfies, Package};

/// 资源包清单：声明加载顺序（`game-data/packs.toml`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PackConfig {
    /// 加载顺序：pack ID = `game-data/` 下的文件夹名；数组靠后的包覆盖靠前。
    pub order: Vec<String>,
}

impl Default for PackConfig {
    fn default() -> Self {
        Self {
            order: vec!["vanilla".to_string()],
        }
    }
}

impl PackConfig {
    /// 当前加载顺序。
    pub fn order(&self) -> &[String] {
        &self.order
    }

    /// 读清单；文件缺失/解析失败时回退到默认（打印原因，不阻断）。
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        match std::fs::read_to_string(path.as_ref()) {
            Ok(text) => match toml::from_str(&text) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!(
                        "解析资源包清单失败（{}）：{e}；使用默认顺序",
                        path.as_ref().display()
                    );
                    Self::default()
                }
            },
            Err(e) => {
                eprintln!(
                    "读取资源包清单失败（{}）：{e}；使用默认顺序",
                    path.as_ref().display()
                );
                Self::default()
            }
        }
    }

    /// 启动主流程：扫描有效包 → 按依赖更新顺序 → 写回 `packs.toml`。
    ///
    /// 返回（最新清单, 包表）。生成阶段：
    /// - 没有有效包 → Err；
    /// - 依赖图存在环 → Err（生成时报错退出）。
    pub fn discover_and_update(
        data_root: impl AsRef<Path>,
    ) -> Result<(Self, HashMap<String, Package>)> {
        let data_root = data_root.as_ref();
        let packages = pack::scan_packs(data_root)?;
        if packages.is_empty() {
            bail!("{} 下没有任何有效资源包", data_root.display());
        }
        pack::detect_cycle(&packages)?;

        let manifest_path = data_root.join("packs.toml");
        let mut config = Self::load_or_default(&manifest_path);
        config.reconcile(&packages)?;
        config.save_if_changed(&manifest_path)?;
        Ok((config, packages))
    }

    /// 加载校验：order 中的包必须有效、依赖顺序满足、无冲突；违反 → Err（报错退出）。
    pub fn validate(&self, packages: &HashMap<String, Package>) -> Result<()> {
        for id in &self.order {
            if !packages.contains_key(id) {
                bail!("packs.toml 顺序中的包 {id} 不存在或不是有效包");
            }
        }
        // 依赖：必须启用、版本满足、在自身之前。
        for (index, id) in self.order.iter().enumerate() {
            let pkg = &packages[id];
            for (dep, req) in &pkg.dependencies {
                let dep_pkg = packages
                    .get(dep)
                    .ok_or_else(|| anyhow::anyhow!("包 {id} 的依赖 {dep} 不存在或不是有效包"))?;
                if !version_satisfies(req, &dep_pkg.version)? {
                    bail!(
                        "包 {id} 依赖 {dep}{req}，但已启用版本是 {}",
                        dep_pkg.version
                    );
                }
                match self.order.iter().position(|x| x == dep) {
                    Some(p) if p < index => {}
                    Some(_) => bail!("包 {id} 的依赖 {dep} 必须在它之前加载（当前顺序不满足）"),
                    None => bail!("包 {id} 的依赖 {dep} 未启用"),
                }
            }
        }
        // 冲突：启用且版本满足时不能共存。
        for id in &self.order {
            for (other, req) in &packages[id].conflicts {
                if let Some(other_pkg) = packages.get(other) {
                    if self.order.iter().any(|x| x == other)
                        && version_satisfies(req, &other_pkg.version)?
                    {
                        bail!("包 {id} 与 {other}{req} 冲突，不能同时启用");
                    }
                }
            }
        }
        Ok(())
    }

    /// 按顺序映射为包根目录列表（`game-data/<id>`），用于构建合并资源空间。
    pub fn pack_roots(&self, data_root: impl AsRef<Path>) -> Vec<PathBuf> {
        self.order
            .iter()
            .map(|id| data_root.as_ref().join(id))
            .collect()
    }

    /// 与扫描结果对齐：移除失效包、按依赖插入新包。
    fn reconcile(&mut self, packages: &HashMap<String, Package>) -> Result<()> {
        // 删除：已不存在或不再有效的包从顺序里移除（警告，不报错——
        // "包被删除后自动清理"是合法生命周期；报错只留给依赖/冲突不满足）。
        let removed: Vec<String> = self
            .order
            .iter()
            .filter(|id| !packages.contains_key(*id))
            .cloned()
            .collect();
        self.order.retain(|id| packages.contains_key(id));
        for id in removed {
            eprintln!("警告：packs.toml 中的包 {id} 不存在或不是有效包，已从加载顺序中移除");
        }
        // 新增：按依赖插入（依赖也是新包时先递归插入依赖）。
        let mut inserted: HashSet<String> = self.order.iter().cloned().collect();
        let mut ids: Vec<String> = packages.keys().cloned().collect();
        ids.sort();
        for id in ids {
            if !inserted.contains(&id) {
                insert_pack(&id, &mut self.order, &mut inserted, packages, &mut Vec::new())?;
            }
        }
        Ok(())
    }

    /// 写回 `packs.toml`；内容未变化时不写（避免无谓的 mtime 抖动）。
    fn save_if_changed(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("序列化资源包清单失败")?;
        match std::fs::read_to_string(path) {
            Ok(existing) if existing == text => return Ok(()),
            _ => {}
        }
        std::fs::write(path, text)
            .with_context(|| format!("写资源包清单失败：{}", path.display()))?;
        Ok(())
    }
}

/// 把新包插入顺序：先递归插入依赖（有效包），再放到最后一个依赖之后。
fn insert_pack(
    id: &str,
    order: &mut Vec<String>,
    inserted: &mut HashSet<String>,
    packages: &HashMap<String, Package>,
    stack: &mut Vec<String>,
) -> Result<()> {
    if inserted.contains(id) {
        return Ok(());
    }
    if stack.iter().any(|s| s == id) {
        bail!("资源包依赖存在环：{} → {id}", stack.join(" → "));
    }
    let pkg = packages.get(id).expect("insert_pack 只处理有效包");
    stack.push(id.to_string());
    for dep in pkg.dependencies.keys() {
        if !packages.contains_key(dep) {
            bail!("包 {id} 依赖的 {dep} 不存在或不是有效包");
        }
        insert_pack(dep, order, inserted, packages, stack)?;
    }
    stack.pop();
    // 插入位置：最后一个已存在依赖之后（无依赖则追加到末尾）。
    let pos = pkg
        .dependencies
        .keys()
        .filter_map(|dep| order.iter().position(|x| x == dep))
        .max()
        .map_or(order.len(), |p| p + 1);
    order.insert(pos, id.to_string());
    inserted.insert(id.to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 在临时目录搭一个包布局：`dir/<id>/package.toml`。
    /// `deps` / `conflicts` 为 `(包 ID, 版本要求)`。
    fn make_pack(
        dir: &Path,
        id: &str,
        deps: &[(&str, &str)],
        conflicts: &[(&str, &str)],
    ) {
        let pack_dir = dir.join(id);
        std::fs::create_dir_all(&pack_dir).expect("创建包目录");
        let mut text = format!("id = \"{id}\"\nversion = \"0.1.0\"\n");
        if !deps.is_empty() {
            text.push_str("\n[dependencies]\n");
            for (dep, req) in deps {
                text.push_str(&format!("{dep} = \"{req}\"\n"));
            }
        }
        if !conflicts.is_empty() {
            text.push_str("\n[conflicts]\n");
            for (other, req) in conflicts {
                text.push_str(&format!("{other} = \"{req}\"\n"));
            }
        }
        std::fs::write(pack_dir.join("package.toml"), text).expect("写 package.toml");
    }

    #[test]
    fn parses_order_and_maps_roots() {
        let dir = std::env::temp_dir().join("pack-config-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建目录");
        std::fs::write(dir.join("packs.toml"), "order = [\"vanilla\", \"mod_a\"]\n")
            .expect("写文件");

        let config = PackConfig::load_or_default(dir.join("packs.toml"));
        assert_eq!(config.order(), ["vanilla", "mod_a"]);
        assert_eq!(
            config.pack_roots(&dir),
            vec![dir.join("vanilla"), dir.join("mod_a")]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn defaults_when_file_missing() {
        let dir = std::env::temp_dir().join("pack-config-missing");
        let _ = std::fs::remove_dir_all(&dir);
        let config = PackConfig::load_or_default(dir.join("packs.toml"));
        assert_eq!(config.order(), ["vanilla"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 扫描 + 更新：新包按依赖插入，清单文件被创建。
    #[test]
    fn discover_creates_and_updates_order() {
        let dir = std::env::temp_dir().join("pack-discover-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建目录");
        make_pack(&dir, "vanilla", &[], &[]);
        make_pack(&dir, "mod_a", &[("vanilla", "=0.1.0")], &[]);
        make_pack(&dir, "mod_b", &[("mod_a", "*")], &[]);

        let (config, packages) = PackConfig::discover_and_update(&dir).expect("应成功");
        assert_eq!(config.order(), ["vanilla", "mod_a", "mod_b"]);
        assert!(dir.join("packs.toml").is_file(), "清单文件应被创建");
        config.validate(&packages).expect("生成后的顺序应满足依赖");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 删除：已消失的包从顺序中移除。
    #[test]
    fn deleted_pack_removed_from_order() {
        let dir = std::env::temp_dir().join("pack-delete-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建目录");
        make_pack(&dir, "vanilla", &[], &[]);
        std::fs::write(dir.join("packs.toml"), "order = [\"vanilla\", \"gone\"]\n")
            .expect("写清单");

        let (config, _) = PackConfig::discover_and_update(&dir).expect("应成功");
        assert_eq!(config.order(), ["vanilla"]);
        // 清单文件也被重写，不再包含已删除的包。
        let rewritten = std::fs::read_to_string(dir.join("packs.toml")).expect("读回清单");
        assert!(!rewritten.contains("gone"), "重写后的清单不应包含 gone：{rewritten}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 生成阶段：依赖环 → Err（报错退出）。
    #[test]
    fn cycle_rejected_at_generation() {
        let dir = std::env::temp_dir().join("pack-cycle-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建目录");
        make_pack(&dir, "a", &[("b", "*")], &[]);
        make_pack(&dir, "b", &[("a", "*")], &[]);

        let err = PackConfig::discover_and_update(&dir).expect_err("依赖环应报错");
        assert!(err.to_string().contains("环"), "错误信息应指明环：{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 生成阶段：依赖不存在 → Err。
    #[test]
    fn missing_dependency_rejected_at_generation() {
        let dir = std::env::temp_dir().join("pack-missing-dep-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建目录");
        make_pack(&dir, "a", &[("vanilla", "*")], &[]);

        let err = PackConfig::discover_and_update(&dir).expect_err("缺失依赖应报错");
        assert!(err.to_string().contains("vanilla"), "错误信息应提到缺失依赖：{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 加载阶段：手改顺序不满足依赖 → Err。
    #[test]
    fn load_validation_rejects_wrong_dep_order() {
        let mut packages = HashMap::new();
        packages.insert(
            "vanilla".to_string(),
            Package {
                id: "vanilla".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: HashMap::new(),
                conflicts: HashMap::new(),
            },
        );
        packages.insert(
            "mod_a".to_string(),
            Package {
                id: "mod_a".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: [("vanilla".to_string(), "=0.1.0".to_string())]
                    .into_iter()
                    .collect(),
                conflicts: HashMap::new(),
            },
        );
        let config = PackConfig {
            order: vec!["mod_a".into(), "vanilla".into()],
        };
        assert!(config.validate(&packages).is_err(), "依赖顺序错误应报错");
    }

    /// 加载阶段：冲突包同时启用 → Err。
    #[test]
    fn load_validation_rejects_conflict() {
        let mut packages = HashMap::new();
        packages.insert(
            "a".to_string(),
            Package {
                id: "a".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: HashMap::new(),
                conflicts: [("b".to_string(), "*".to_string())].into_iter().collect(),
            },
        );
        packages.insert(
            "b".to_string(),
            Package {
                id: "b".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: HashMap::new(),
                conflicts: HashMap::new(),
            },
        );
        let config = PackConfig {
            order: vec!["a".into(), "b".into()],
        };
        assert!(config.validate(&packages).is_err(), "冲突应报错");
    }

    /// 加载阶段：依赖的已启用版本不满足版本要求 → Err。
    #[test]
    fn load_validation_rejects_version_mismatch() {
        let mut packages = HashMap::new();
        packages.insert(
            "vanilla".to_string(),
            Package {
                id: "vanilla".into(),
                name: None,
                version: "0.2.0".into(),
                dependencies: HashMap::new(),
                conflicts: HashMap::new(),
            },
        );
        packages.insert(
            "mod_a".to_string(),
            Package {
                id: "mod_a".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: [("vanilla".to_string(), "=0.1.0".to_string())]
                    .into_iter()
                    .collect(),
                conflicts: HashMap::new(),
            },
        );
        let config = PackConfig {
            order: vec!["vanilla".into(), "mod_a".into()],
        };
        let err = config.validate(&packages).expect_err("版本不满足应报错");
        assert!(err.to_string().contains("0.2.0"), "错误应提到实际版本：{err}");
    }

}
