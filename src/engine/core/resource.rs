//! 合并资源空间：把 [`GamePath`] 解析到实际文件系统、集中文件访问的调度器。
//!
//! 游戏逻辑/加载器**不直接拼文件路径**，而是把游戏路径交给这里，由它
//! 定位到合并资源空间下的实际文件并返回系统文件句柄。
//!
//! 合并语义：
//! - 包 ID = `game-data/` 下的文件夹名（见 [`crate::engine::core::config::PackConfig`]）；
//! - 包根列表按加载顺序传入（基础包在前，覆盖包在后），`order = ["vanilla", "mod_a"]`
//!   表示 `mod_a` 覆盖 `vanilla` 的同名文件；
//! - [`Self::resolve`] 从最高优先级包开始遍历，返回**首个命中**的文件。

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::game_path::GamePath;

/// 合并资源空间：持有按优先级排序的包根列表，提供"游戏路径 → 文件句柄"的映射。
#[derive(Debug, Clone)]
pub struct MergedResourceSpace {
    /// 包根列表，按加载顺序（基础 → 覆盖）。
    roots: Vec<PathBuf>,
}

impl MergedResourceSpace {
    /// 单包（便捷/测试）：等价于 [`Self::from_pack_roots`] 传入一个根。
    pub fn new(root: PathBuf) -> Self {
        Self::from_pack_roots(vec![root])
    }

    /// 多包合并：`roots` 按加载顺序传入（基础包在前，覆盖包在后）。
    pub fn from_pack_roots(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// 合并资源空间包根列表（加载顺序）。
    #[allow(dead_code)] // 公共 API：包根查询，暂无调用方
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// 游戏路径对应的实际文件：从最高优先级包开始遍历，返回首个命中；
    /// 任何包中都不存在时返回 `None`。
    pub fn resolve(&self, path: &GamePath) -> Option<PathBuf> {
        self.roots
            .iter()
            .rev() // 靠后的包优先级更高，先查
            .map(|root| path.resolve(root))
            .find(|p| p.is_file())
    }

    /// 文件是否存在（合并空间内，任意包命中）。
    pub fn exists(&self, path: &GamePath) -> bool {
        self.resolve(path).is_some()
    }

    /// 打开文件，返回只读流句柄。
    ///
    /// 故意只暴露 [`Read`] 而不暴露 `Seek`：调用方不需要猜测资源是系统文件、
    /// zip 压缩流还是内存映射——统一按顺序流读取。后端内部（如 zip 读中央
    /// 目录定位）可以自由使用 `Seek`，不泄露给调用方；将来若某类资源确实
    /// 需要随机访问（大文件内部按偏移分块），单独设计流式接口，不污染这里。
    /// `Send` 让句柄可跨线程（异步加载工作线程）。
    pub fn open(&self, path: &GamePath) -> Result<Box<dyn Read + Send>> {
        let real = self.resolve(path).with_context(|| {
            format!(
                "资源文件不存在于任何资源包：{path}（已查 {} 个包）",
                self.roots.len()
            )
        })?;
        let file = std::fs::File::open(&real)
            .with_context(|| format!("打开资源文件失败：{path}（{}）", real.display()))?;
        Ok(Box::new(file))
    }

    /// 读取文件全部字节（便捷接口；处理大文件仍应直接用 [`Self::open`]）。
    pub fn read(&self, path: &GamePath) -> Result<Vec<u8>> {
        let mut file = self.open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .with_context(|| format!("读取资源文件失败：{path}"))?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 在临时目录模拟合并资源空间：打开/读取/存在性检查，越界路径被 GamePath
    /// 拦在解析之前。
    #[test]
    fn opens_files_within_merged_space() {
        let dir = std::env::temp_dir().join("merged-space-test");
        let _ = std::fs::remove_dir_all(&dir);
        let ns_dir = dir.join("vanilla/data");
        std::fs::create_dir_all(&ns_dir).expect("创建测试目录");
        std::fs::write(ns_dir.join("x.toml"), b"hello").expect("写测试文件");

        let space = MergedResourceSpace::new(dir.clone());
        let path: GamePath = "vanilla:data/x.toml".parse().expect("合法路径");
        assert!(space.exists(&path));
        assert_eq!(space.read(&path).expect("读取成功"), b"hello");

        // 不存在：exists false、open 报错。
        let missing: GamePath = "vanilla:data/nope.toml".parse().expect("合法路径");
        assert!(!space.exists(&missing));
        assert!(space.open(&missing).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 多包合并：靠后的包覆盖靠前的同名文件；只在基础包的文件仍可命中。
    #[test]
    fn later_pack_overrides_earlier() {
        let dir = std::env::temp_dir().join("merged-space-override");
        let _ = std::fs::remove_dir_all(&dir);

        let base = dir.join("base");
        let override_pack = dir.join("override");
        for (root, content) in [
            (&base, b"base".as_slice()),
            (&override_pack, b"override".as_slice()),
        ] {
            let ns = root.join("test");
            std::fs::create_dir_all(&ns).expect("创建目录");
            std::fs::write(ns.join("x.txt"), content).expect("写文件");
        }
        // 只在基础包里存在的文件。
        let base_only = base.join("test");
        std::fs::create_dir_all(&base_only).expect("创建目录");
        std::fs::write(base_only.join("only-base.txt"), b"only").expect("写文件");

        let space =
            MergedResourceSpace::from_pack_roots(vec![base.clone(), override_pack.clone()]);
        let overridden: GamePath = "test:x.txt".parse().expect("合法路径");
        assert_eq!(space.read(&overridden).expect("读取成功"), b"override");

        let base_only_path: GamePath = "test:only-base.txt".parse().expect("合法路径");
        assert_eq!(space.read(&base_only_path).expect("读取成功"), b"only");

        let missing: GamePath = "test:nope.txt".parse().expect("合法路径");
        assert!(!space.exists(&missing));
        assert!(space.open(&missing).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
