//! 材质模块：物体的表面属性。
//!
//! 目前只实现基础色（因子 + 可选贴图）；金属度/粗糙度/法线/自发光等
//! PBR 通道按同样模式扩展（对应 glTF 的 metallic-roughness/normal/emissive）。

use super::asset::Handle;
use super::texture::Texture;

/// 材质：基础色 + 金属度/粗糙度 + 法线贴图（glTF PBR 子集）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// 基础颜色因子（RGBA，glTF `baseColorFactor`）。
    pub base_color: [f32; 4],
    /// 基础色贴图（glTF `baseColorTexture`）。
    pub base_color_texture: Option<Handle<Texture>>,
    /// 金属度因子（glTF `metallicFactor`，默认 0）。
    pub metallic_factor: f32,
    /// 粗糙度因子（glTF `roughnessFactor`，默认 1）。
    pub roughness_factor: f32,
    /// 金属度/粗糙度贴图（glTF `metallicRoughnessTexture`，B=金属度、G=粗糙度）。
    pub metallic_roughness_texture: Option<Handle<Texture>>,
    /// 法线贴图（glTF `normalTexture`）。
    pub normal_texture: Option<Handle<Texture>>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_color_texture: None,
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
        }
    }
}

impl Material {
    /// 材质引用的全部贴图句柄（基础色/金属度粗糙度/法线）。
    ///
    /// 关卡加载时用来收集"场景需要的贴图清单"统一 pin。
    pub fn texture_handles(&self) -> impl Iterator<Item = Handle<Texture>> + '_ {
        [
            self.base_color_texture,
            self.metallic_roughness_texture,
            self.normal_texture,
        ]
        .into_iter()
        .flatten()
    }
}
