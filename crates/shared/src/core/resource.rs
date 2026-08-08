//! 合并资源空间：把 [`GamePath`] 解析到实际文件系统、集中文件访问的调度器。
//!
//! 游戏逻辑/加载器**不直接拼文件路径**，而是把游戏路径交给这里，由它
//! 定位到合并资源空间下的实际文件并返回系统文件句柄。当前空间等价于
//! 一个包根目录（`game-data/vanilla/`）；未来多个数据包（模组）合并/
//! 覆盖时，扩展为按优先级遍历包根列表、返回首个命中的文件，本类型接口
//! 不变（这正是"合并"的落点）。
//!
//! TODO：当前仍是**单包根简化实现**（假设合并资源空间 = 一个包根目录，
//! namespace 目录在其下），不扫描、不合并。真正的合并资源空间需要：
//! 1. 扫描 `game-data/` 下所有含 `package.toml` 的包目录（包发现）；
//! 2. 按依赖/加载顺序排序，形成优先级列表；
//! 3. `resolve` 改为遍历包根列表、返回首个命中的文件（模组覆盖原版）。
//! 届时把 `root: PathBuf` 换成包根列表即可，`GamePath` 与接口不变。

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::game_path::GamePath;

/// 合并资源空间：持有包根，提供"游戏路径 → 文件句柄"的映射。
#[derive(Debug, Clone)]
pub struct MergedResourceSpace {
    root: PathBuf,
}

impl MergedResourceSpace {
    /// 以包根目录创建合并资源空间（当前：`game-data/vanilla/`）。
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// 合并资源空间根目录（包根）。
    #[allow(dead_code)] // 公共 API：合并根查询，暂无调用方
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// 游戏路径对应的实际文件系统路径：`root/{namespace}/{path}`。
    ///
    /// [`GamePath`] 构造时已做穿越校验，这里只做拼接；未来多包合并时
    /// 改为"按优先级遍历包根、返回首个命中"。
    pub fn resolve(&self, path: &GamePath) -> PathBuf {
        path.resolve(&self.root)
    }

    /// 文件是否存在（合并空间内）。
    pub fn exists(&self, path: &GamePath) -> bool {
        self.resolve(path).is_file()
    }

    /// 打开文件，返回只读流句柄。
    ///
    /// 故意只暴露 [`Read`] 而不暴露 `Seek`：调用方不需要猜测资源是系统文件、
    /// zip 压缩流还是内存映射——统一按顺序流读取。后端内部（如 zip 读中央
    /// 目录定位）可以自由使用 `Seek`，不泄露给调用方；将来若某类资源确实
    /// 需要随机访问（大文件内部按偏移分块），单独设计流式接口，不污染这里。
    /// `Send` 让句柄可跨线程（异步加载工作线程）。
    pub fn open(&self, path: &GamePath) -> Result<Box<dyn Read + Send>> {
        let real = self.resolve(path);
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
}
