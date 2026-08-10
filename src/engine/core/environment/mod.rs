//! 环境模块：CPU 侧的环境光数据（等距矩形环境贴图）。
//!
//! 每个关卡一份环境资产（机制上与天空盒同源），启动时一次性解码，
//! 由渲染器转换成 GPU 上的环境立方体贴图与辐照度图。
//!
//! 输入格式：
//! - **HDR**（Radiance .hdr）：高动态范围，物理正确的 PBR 环境光；
//! - **LDR**（PNG/JPEG 等普通图片）：sRGB → 线性 + 曝光补偿，
//!   给无法制作 HDR 的模组作者当简易天空盒/环境光用；
//! - [`Environment::from_file`] 按文件内容自动识别两种格式。

use std::path::Path;

use anyhow::{Context, Result};

/// 环境贴图：Radiance HDR 解码后的线性 RGB 浮点像素。
#[derive(Debug, Clone)]
pub struct Environment {
    pub width: u32,
    pub height: u32,
    /// 行优先（从上到下、从左到右）的线性 RGB 像素。
    pub rgb: Vec<[f32; 3]>,
}

impl Environment {
    /// LDR 环境贴图的默认曝光（自动识别路径使用；显式 LDR 接口可自定）。
    ///
    /// LDR 值域 0..1 且对比度低，乘曝光能部分补偿动态范围损失；
    /// 保守取 1.0，模组作者可按需调高。
    pub const DEFAULT_LDR_EXPOSURE: f32 = 1.0;

    /// 从 Radiance HDR（.hdr）字节解码。
    pub fn from_hdr_bytes(bytes: &[u8]) -> Result<Self> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .context("环境贴图解码失败")?;
        let rgb32f = image.to_rgb32f();
        let (width, height) = rgb32f.dimensions();
        let rgb = rgb32f.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, rgb })
    }

    /// 从磁盘读取并解码 Radiance HDR 文件。
    #[allow(dead_code)] // 显式入口：调用方明确指定 HDR 时用；App 默认走自动识别
    pub fn from_hdr_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).context("无法读取环境贴图文件")?;
        Self::from_hdr_bytes(&bytes)
    }

    /// 从文件按后缀加载：`.hdr`（不区分大小写）走 Radiance 解码，
    /// 其余（PNG/JPEG 等）按 LDR 处理（默认曝光）。
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).context("无法读取环境贴图文件")?;
        let is_hdr = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("hdr"));
        if is_hdr {
            Self::from_hdr_bytes(&bytes)
        } else {
            Self::from_ldr_bytes(&bytes, Self::DEFAULT_LDR_EXPOSURE)
        }
    }

    /// 从字节按内容（magic bytes）识别格式：HDR 走 Radiance，其余按 LDR（默认曝光）。
    #[allow(dead_code)] // 显式入口：无文件名的字节流用；App 默认走 from_file 后缀识别
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        match image::guess_format(bytes) {
            Ok(image::ImageFormat::Hdr) => Self::from_hdr_bytes(bytes),
            _ => Self::from_ldr_bytes(bytes, Self::DEFAULT_LDR_EXPOSURE),
        }
    }

    /// 从 LDR 图片字节（PNG/JPEG 等）构造环境：sRGB → 线性 × 曝光。
    pub fn from_ldr_bytes(bytes: &[u8], exposure: f32) -> Result<Self> {
        let image = image::load_from_memory(bytes).context("环境贴图解码失败")?;
        Self::from_ldr_image(&image, exposure)
    }

    /// 从磁盘读取 LDR 图片并构造环境。
    #[allow(dead_code)] // 显式入口：需要自定义曝光时用；App 默认走自动识别
    pub fn from_ldr_file(path: &Path, exposure: f32) -> Result<Self> {
        let bytes = std::fs::read(path).context("无法读取环境贴图文件")?;
        Self::from_ldr_bytes(&bytes, exposure)
    }

    /// 从已解码的图片构造环境：8-bit sRGB 像素 → 线性 RGB × 曝光。
    ///
    /// LDR 图片本身是 sRGB 语义，直接当线性用会让画面偏暗偏灰；
    /// 线性化后再乘曝光，才能和 HDR 路径共用同一套天空盒/IBL 流程。
    pub fn from_ldr_image(image: &image::DynamicImage, exposure: f32) -> Result<Self> {
        let rgb8 = image.to_rgb8();
        let (width, height) = rgb8.dimensions();
        let rgb = rgb8
            .pixels()
            .map(|p| {
                [
                    srgb_to_linear(p.0[0] as f32 / 255.0) * exposure,
                    srgb_to_linear(p.0[1] as f32 / 255.0) * exposure,
                    srgb_to_linear(p.0[2] as f32 / 255.0) * exposure,
                ]
            })
            .collect();
        Ok(Self { width, height, rgb })
    }
}

/// sRGB 编码值 → 线性亮度（精确曲线）。
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests;
