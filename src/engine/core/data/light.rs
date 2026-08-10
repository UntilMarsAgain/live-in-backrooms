//! 灯光模块：CPU 侧的光源数据。
//!
//! 三种光源类型：方向光 / 点光 / 面光（矩形面板）。
//! 灯光是场景对象的一种（`SceneObjectKind::Light`）：
//! - 方向光：方向由物体旋转决定（局部 -Z 指向**光行进方向**，位置无意义；
//!   与面光的发射方向同义，着色器对方向光取反得到"表面 → 光源"向量）；
//! - 点光：位置由物体位置决定，平方反比衰减；
//! - 面光：位置 + 朝向（局部 -Z 是发射方向）+ 面板尺寸，当前按朗伯发射面板
//!   近似（真实矩形面光需要 LTC，见优化账本）。

use glam::Vec3;

/// 光源类型。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightKind {
    /// 平行光：方向由旋转决定。
    Directional,
    /// 点光：位置 + 平方反比衰减。
    Point,
    /// 矩形面光：位置 + 朝向 + 尺寸（朗伯发射面板近似）。
    Area { width: f32, height: f32 },
}

/// 光源：类型 + 颜色 + 强度。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    pub color: Vec3,
    pub intensity: f32,
}

impl Light {
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self {
            kind: LightKind::Directional,
            color,
            intensity,
        }
    }

    pub fn point(color: Vec3, intensity: f32) -> Self {
        Self {
            kind: LightKind::Point,
            color,
            intensity,
        }
    }

    pub fn area(width: f32, height: f32, color: Vec3, intensity: f32) -> Self {
        Self {
            kind: LightKind::Area { width, height },
            color,
            intensity,
        }
    }
}
