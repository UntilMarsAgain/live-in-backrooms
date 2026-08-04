//! 纹理模块：CPU 侧的贴图数据与全局纹理资产库。
//!
//! 与 [`MeshLibrary`](crate::mesh::MeshLibrary) 同一套模式：只追加、永久持有、
//! 版本号驱动 GPU 侧增量上传。句柄是稠密编号（不删除因此稳定）。

/// 纹理：RGBA8 未压缩像素（sRGB 语义，上传时按 `Rgba8UnormSrgb` 处理）。
#[derive(Debug, Clone)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

impl Texture {
    /// 1×1 白色纹理：无贴图材质的默认值。
    pub fn white() -> Self {
        Self {
            width: 1,
            height: 1,
            rgba8: vec![255, 255, 255, 255],
        }
    }

    /// 1×1 中性法线贴图（RGB(128,128,255) → 切线空间法线 (0,0,1)）。
    pub fn neutral_normal() -> Self {
        Self {
            width: 1,
            height: 1,
            rgba8: vec![128, 128, 255, 255],
        }
    }

    /// 棋盘格纹理：用于验证贴图采样（后室风的暗黄两色）。
    pub fn checkerboard(size: u32, tile: u32) -> Self {
        let mut rgba8 = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let on = ((x / tile) + (y / tile)) % 2 == 0;
                let color: [u8; 4] = if on {
                    [214, 184, 112, 255]
                } else {
                    [66, 56, 36, 255]
                };
                rgba8.extend_from_slice(&color);
            }
        }
        Self {
            width: size,
            height: size,
            rgba8,
        }
    }
}

/// 纹理句柄：在 [`TextureLibrary`] 中的稠密编号。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureKey(usize);

impl TextureKey {
    pub fn index(self) -> usize {
        self.0
    }
}

/// 全局纹理资产库：只追加、永久持有。
#[derive(Debug, Default)]
pub struct TextureLibrary {
    textures: Vec<Texture>,
    version: u64,
}

impl TextureLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// 批量注册：一次调用追加多张贴图并返回各自的句柄。
    pub fn register_many(&mut self, textures: impl IntoIterator<Item = Texture>) -> Vec<TextureKey> {
        let start = self.textures.len();
        self.textures.extend(textures);
        let keys: Vec<_> = (start..self.textures.len()).map(TextureKey).collect();
        if !keys.is_empty() {
            self.version += 1;
        }
        keys
    }

    #[allow(dead_code)] // 预留：单张贴图注册 API
    pub fn register(&mut self, texture: Texture) -> TextureKey {
        self.register_many([texture])[0]
    }

    pub fn texture(&self, key: TextureKey) -> Option<&Texture> {
        self.textures.get(key.0)
    }

    pub fn len(&self) -> usize {
        self.textures.len()
    }

    #[allow(dead_code)] // 预留
    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }

    pub fn textures(&self) -> &[Texture] {
        &self.textures
    }

    pub fn version(&self) -> u64 {
        self.version
    }
}
