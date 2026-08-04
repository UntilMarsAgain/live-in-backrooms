//! 材质模块：物体的表面属性。
//!
//! 目前只实现基础色（因子 + 可选贴图）；金属度/粗糙度/法线/自发光等
//! PBR 通道按同样模式扩展（对应 glTF 的 metallic-roughness/normal/emissive）。

use crate::texture::TextureKey;

/// 材质：基础颜色因子 + 可选的基础色贴图。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// 基础颜色因子（RGBA，glTF `baseColorFactor`）。
    pub base_color: [f32; 4],
    /// 基础色贴图（glTF `baseColorTexture`）。
    pub base_color_texture: Option<TextureKey>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_color_texture: None,
        }
    }
}
