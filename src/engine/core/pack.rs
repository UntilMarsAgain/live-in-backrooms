//! 资源包：`package.toml` 的定义、扫描与依赖图。
//!
//! 包 ID = `game-data/` 下的文件夹名；有效包必须存在 `package.toml` 且能正确
//! 解析，并且其中的 `id` 与文件夹名一致。依赖/冲突字段按包 ID 引用。
//! 版本约束采用 **Cargo 语义**（`semver::VersionReq`）：如 `=0.1.0`（精确）、
//! `^0.1`、`~0.1.2`、`>=1.2`、`*` 等。
//! 暂不引入压缩包：只扫描文件夹（`game-data/<id>/package.toml`）。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};

/// 包的元数据（`game-data/<id>/package.toml`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Package {
    /// 包 ID，必须与包文件夹名一致（扫描时校验）。
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// 包版本（semver，如 `0.1.0`）：版本检测与依赖约束都基于它。
    pub version: String,
    /// 依赖：包 ID → 版本要求（Cargo 语义）；被依赖包必须在自身之前加载。
    #[serde(default)]
    pub dependencies: HashMap<String, String>,
    /// 冲突：包 ID → 版本要求；启用且版本满足时不允许共存。
    #[serde(default)]
    pub conflicts: HashMap<String, String>,
}

/// 解析一份 package.toml 文本。
pub fn parse_package(text: &str) -> Result<Package> {
    let pkg: Package = toml::from_str(text).context("package.toml 解析失败")?;
    // 版本必须是合法 semver；依赖/冲突里的版本要求也必须是合法 Cargo 语法，
    // 在扫描阶段就报错，而不是拖到加载阶段。
    Version::parse(&pkg.version)
        .with_context(|| format!("version=\"{}\" 不是合法 semver", pkg.version))?;
    for (dep, req) in &pkg.dependencies {
        VersionReq::parse(req)
            .with_context(|| format!("依赖 {dep} 的版本要求 {req:?} 非法"))?;
    }
    for (other, req) in &pkg.conflicts {
        VersionReq::parse(req)
            .with_context(|| format!("冲突 {other} 的版本要求 {req:?} 非法"))?;
    }
    Ok(pkg)
}

/// 版本要求是否被满足（Cargo 语义：`VersionReq::matches`）。
pub fn version_satisfies(requirement: &str, version: &str) -> Result<bool> {
    let req = VersionReq::parse(requirement)
        .with_context(|| format!("非法版本要求：{requirement:?}"))?;
    let ver = Version::parse(version).with_context(|| format!("非法包版本：{version:?}"))?;
    Ok(req.matches(&ver))
}

/// 扫描 `data_root` 下所有包目录，返回 包 ID → 包元数据。
///
/// - 没有 `package.toml` 的目录：不是包，跳过；
/// - `package.toml` 读取/解析失败，或 `id` 与文件夹名不一致：警告并跳过
///   （不阻断启动；这样的目录不是有效包）。
pub fn scan_packs(data_root: &Path) -> Result<HashMap<String, Package>> {
    let mut packs = HashMap::new();
    let entries = std::fs::read_dir(data_root)
        .with_context(|| format!("扫描资源包目录失败：{}", data_root.display()))?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let folder = entry.file_name().to_string_lossy().into_owned();
        let manifest = path.join("package.toml");
        if !manifest.is_file() {
            continue; // 没有 package.toml：不是有效包
        }
        let text = match std::fs::read_to_string(&manifest) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("警告：{folder} 的 package.toml 读取失败，忽略该包：{e}");
                continue;
            }
        };
        match parse_package(&text) {
            Ok(pkg) if pkg.id == folder => {
                packs.insert(folder, pkg);
            }
            Ok(pkg) => {
                eprintln!(
                    "警告：{folder} 的 package.toml 中 id=\"{}\" 与文件夹名不一致，忽略该包",
                    pkg.id
                );
            }
            Err(e) => {
                eprintln!("警告：{folder} 的 package.toml 解析失败，忽略该包：{e:#}");
            }
        }
    }
    Ok(packs)
}

/// 依赖图环检测：存在环时返回 Err（生成顺序阶段报错退出）。
pub(crate) fn detect_cycle(packages: &HashMap<String, Package>) -> Result<()> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mark {
        Visiting,
        Done,
    }

    fn visit(
        id: &str,
        packages: &HashMap<String, Package>,
        marks: &mut HashMap<String, Mark>,
        path: &mut Vec<String>,
    ) -> Result<()> {
        match marks.get(id) {
            Some(Mark::Done) => return Ok(()),
            Some(Mark::Visiting) => {
                bail!("资源包依赖存在环：{} → {id}", path.join(" → "));
            }
            None => {}
        }
        marks.insert(id.to_string(), Mark::Visiting);
        path.push(id.to_string());
        if let Some(pkg) = packages.get(id) {
            for dep in pkg.dependencies.keys() {
                if packages.contains_key(dep) {
                    visit(dep, packages, marks, path)?;
                }
            }
        }
        path.pop();
        marks.insert(id.to_string(), Mark::Done);
        Ok(())
    }

    let mut marks = HashMap::new();
    let ids: Vec<&String> = packages.keys().collect();
    for id in ids {
        visit(id, packages, &mut marks, &mut Vec::new())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_package_with_deps_and_conflicts() {
        let pkg = parse_package(
            r#"
id = "mod_a"
name = "模组 A"
version = "0.1.0"

[dependencies]
vanilla = "=0.1.0"
base_lib = "^0.1"

[conflicts]
old_mod = ">=1.0"
"#,
        )
        .expect("应能解析");
        assert_eq!(pkg.id, "mod_a");
        assert_eq!(pkg.version, "0.1.0");
        assert_eq!(pkg.dependencies.get("vanilla").map(String::as_str), Some("=0.1.0"));
        assert_eq!(pkg.conflicts.get("old_mod").map(String::as_str), Some(">=1.0"));
    }

    #[test]
    fn scan_finds_valid_packs_and_skips_others() {
        let dir = std::env::temp_dir().join("pack-scan-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("vanilla")).expect("创建目录");
        std::fs::write(
            dir.join("vanilla/package.toml"),
            "id = \"vanilla\"\nversion = \"0.1.0\"\n",
        )
        .expect("写文件");
        // 没有 package.toml 的目录：不是包。
        std::fs::create_dir_all(dir.join("not-a-pack")).expect("创建目录");
        // package.toml 存在但 id 不匹配：忽略。
        std::fs::create_dir_all(dir.join("bad")).expect("创建目录");
        std::fs::write(
            dir.join("bad/package.toml"),
            "id = \"wrong\"\nversion = \"0.1.0\"\n",
        )
        .expect("写文件");

        let packs = scan_packs(&dir).expect("扫描成功");
        assert_eq!(packs.len(), 1);
        assert!(packs.contains_key("vanilla"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn invalid_version_requirement_rejected() {
        let err = parse_package(
            r#"
id = "mod_a"
version = "0.1.0"

[dependencies]
vanilla = "不是版本要求"
"#,
        )
        .expect_err("非法版本要求应报错");
        assert!(err.to_string().contains("非法"), "错误信息应指明：{err}");
    }

    #[test]
    fn version_satisfies_cargo_semantics() {
        assert!(version_satisfies("=0.1.0", "0.1.0").expect("可解析"));
        assert!(!version_satisfies("=0.1.0", "0.1.1").expect("可解析"));
        assert!(version_satisfies("^0.1", "0.1.9").expect("可解析"));
        assert!(!version_satisfies("^0.1", "0.2.0").expect("可解析"));
        assert!(version_satisfies(">=1.2", "2.0.0").expect("可解析"));
        assert!(version_satisfies("*", "9.9.9").expect("可解析"));
    }

    #[test]
    fn cycle_detection_reports_cycle() {
        let mut packages = HashMap::new();
        packages.insert(
            "a".to_string(),
            Package {
                id: "a".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: [("b".to_string(), "*".to_string())].into_iter().collect(),
                conflicts: HashMap::new(),
            },
        );
        packages.insert(
            "b".to_string(),
            Package {
                id: "b".into(),
                name: None,
                version: "0.1.0".into(),
                dependencies: [("a".to_string(), "*".to_string())].into_iter().collect(),
                conflicts: HashMap::new(),
            },
        );
        assert!(detect_cycle(&packages).is_err());
    }
}
