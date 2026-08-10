//! 游戏资源路径：`namespace:xxx/xxx/xx.xx`。
//!
//! 纯标识符类型，不触碰文件系统；"路径 → 实际文件"的映射由
//! [`GamePath::resolve`] 提供（当前简单映射 `root/{namespace}/{path}`）。
//! 未来 namespace 支持类似 Minecraft 数据包的合并/覆盖机制时，只需
//! 扩展这里的解析逻辑（多个包按优先级叠加、返回首个命中的文件），
//! 本类型与校验规则不变。

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Result, bail};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 游戏资源路径：`namespace:path`（path 用 `/` 分隔，可带扩展名）。
///
/// 例如 `vanilla:models/tiles/corridor.glb` 对应包根下的
/// `vanilla/models/tiles/corridor.glb`。namespace 与 path 的解析保持
/// 纯字符串，namespace 当前与包目录一一对应。
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GamePath {
    namespace: String,
    path: String,
}

impl GamePath {
    pub fn new(namespace: impl Into<String>, path: impl Into<String>) -> Result<Self> {
        let namespace = namespace.into();
        let path = normalize_path(&path.into());
        validate_namespace(&namespace)?;
        validate_path(&path)?;
        Ok(Self { namespace, path })
    }

    /// 资源命名空间（`namespace:` 前缀）。
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// 命名空间内的相对路径（`:` 后的部分）。
    pub fn path(&self) -> &str {
        &self.path
    }

    /// 映射到包根目录下的实际文件路径：`root/{namespace}/{path}`。
    ///
    /// 当前 namespace 与磁盘目录一一对应；未来数据包合并/覆盖机制
    /// （多个包按优先级叠加）会扩展这里的解析——例如返回"首个命中的
    /// 文件"或按优先级合并目录，`GamePath` 本身不变。
    pub fn resolve(&self, root: &Path) -> PathBuf {
        root.join(&self.namespace).join(&self.path)
    }
}

/// 路径规范化：把"不同写法但指向同一文件"的地址折叠成唯一形式。
///
/// 当前规则（path 段已不允许 `.`/`..`、不允许反斜杠，因此只剩这两类）：
/// - 折叠连续 `/`：`a//b` → `a/b`；
/// - 去掉尾部 `/`：`a/b/` → `a/b`。
///
/// 大小写**不**折叠（Linux 文件系统区分大小写；折叠会把不同文件合并）。
/// 规范化在构造时完成，因此 `==`/`Hash` 会把等价写法视为同一路径——
/// 资产去重索引（GamePath → 句柄）天然把它们当成同一个文件。
fn normalize_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_slash = false;
    for c in s.chars() {
        if c == '/' {
            if !prev_slash {
                out.push(c);
            }
            prev_slash = true;
        } else {
            out.push(c);
            prev_slash = false;
        }
    }
    while out.ends_with('/') {
        out.pop();
    }
    out
}

impl FromStr for GamePath {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let (namespace, path) = s.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("游戏路径缺少 ':' 分隔符（应为 namespace:path）：{s:?}")
        })?;
        Self::new(namespace, path)
    }
}

impl fmt::Display for GamePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.path)
    }
}

// serde：以字符串形式序列化（关卡数据/模组配置里直接写路径字符串）。
impl Serialize for GamePath {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for GamePath {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

fn validate_namespace(s: &str) -> Result<()> {
    if s.is_empty() {
        bail!("namespace 不能为空");
    }
    // "." / ".." 会被 `resolve` 拼进文件系统路径造成穿越；以 '.' 开头
    // 会命中隐藏目录，一并拒绝。
    if s == "." || s == ".." || s.starts_with('.') {
        bail!("namespace 不能是 '.' / '..' 或以 '.' 开头：{s:?}");
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        bail!("namespace 含非法字符：{s:?}（允许字母数字 _ - .）");
    }
    Ok(())
}

fn validate_path(s: &str) -> Result<()> {
    if s.is_empty() {
        bail!("path 不能为空");
    }
    if s.starts_with('/') {
        bail!("path 不能以 / 开头：{s:?}");
    }
    // 段级检查：任一段为 "." 或 ".." 都拒绝（防路径穿越与当前目录混入）。
    // 与 `contains("..")` 相比不误伤合法文件名（如 `a..b`）。
    if s.split('/').any(|seg| seg == "." || seg == "..") {
        bail!("path 不能含 '.' / '..' 段（禁止路径穿越）：{s:?}");
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/')
    {
        bail!("path 含非法字符：{s:?}（允许字母数字 _ - . /）");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 解析、访问器与 Display roundtrip。
    #[test]
    fn parse_and_roundtrip() {
        let p: GamePath = "vanilla:models/tiles/corridor.glb".parse().unwrap();
        assert_eq!(p.namespace(), "vanilla");
        assert_eq!(p.path(), "models/tiles/corridor.glb");
        assert_eq!(p.to_string(), "vanilla:models/tiles/corridor.glb");
    }

    /// 规范化：连续斜杠与尾部斜杠折叠成同一路径（等价写法 = 同一个 GamePath）。
    #[test]
    fn normalize_equivalent_addresses() {
        let a: GamePath = "vanilla:a//b/c.glb".parse().unwrap();
        let b: GamePath = "vanilla:a/b/c.glb".parse().unwrap();
        assert_eq!(a, b, "a//b 与 a/b 应等价");
        assert_eq!(a.path(), "a/b/c.glb");

        let c: GamePath = "vanilla:a/b/c.glb/".parse().unwrap();
        assert_eq!(c, b, "尾部斜杠应折叠");
        assert_eq!(c.path(), "a/b/c.glb");

        // 规范化后仍拒绝空路径 / 绝对路径。
        assert!("vanilla://".parse::<GamePath>().is_err());
        assert!("vanilla://a".parse::<GamePath>().is_err());
    }

    /// resolve：包根 + namespace + path。
    #[test]
    fn resolve_to_package_root() {
        let p: GamePath = "vanilla:data/levels/level-0.toml".parse().unwrap();
        assert_eq!(
            p.resolve(Path::new("game-data")),
            PathBuf::from("game-data/vanilla/data/levels/level-0.toml")
        );
    }

    /// 非法输入拒绝：缺冒号、空段、路径穿越、绝对路径、非法字符。
    #[test]
    fn rejects_invalid_input() {
        assert!("vanilla/models.glb".parse::<GamePath>().is_err()); // 缺冒号
        assert!(":models.glb".parse::<GamePath>().is_err()); // 空 namespace
        assert!("vanilla:".parse::<GamePath>().is_err()); // 空 path
        assert!("..:data/x".parse::<GamePath>().is_err()); // namespace 穿越
        assert!(".:data/x".parse::<GamePath>().is_err()); // namespace 当前目录
        assert!(".hidden:data/x".parse::<GamePath>().is_err()); // namespace 隐藏目录
        assert!("vanilla:../secret".parse::<GamePath>().is_err()); // 路径穿越
        assert!("vanilla:a/./b".parse::<GamePath>().is_err()); // 当前目录段
        assert!("vanilla:/etc/passwd".parse::<GamePath>().is_err()); // 绝对路径
        assert!("vanilla:models\\x.glb".parse::<GamePath>().is_err()); // 反斜杠
        assert!("vanilla:models/x.glb".parse::<GamePath>().is_ok()); // 正常
        assert!("vanilla:models/a..b.glb".parse::<GamePath>().is_ok()); // 合法点段
    }

    /// serde：字符串与 GamePath 互转（经 toml 表验证）。
    #[test]
    fn serde_roundtrip() {
        let p: GamePath = "vanilla:data/levels/level-0.toml".parse().unwrap();
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            path: GamePath,
        }
        let text = toml::to_string(&Holder { path: p.clone() }).unwrap();
        assert_eq!(text, "path = \"vanilla:data/levels/level-0.toml\"\n");
        let back: Holder = toml::from_str(&text).unwrap();
        assert_eq!(back.path, p);
    }
}
