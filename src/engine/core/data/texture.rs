//! 纹理模块：CPU 侧的贴图数据。
//!
//! 注册与生命周期管理统一走 [`crate::engine::core::asset::AssetManager`]。

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
