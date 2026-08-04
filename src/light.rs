//! 灯光模块：CPU 侧的光源数据。
//!
//! 目前只实现方向光；点光、聚光灯后续以新类型加入。
//! 灯光是场景对象的一种（`SceneObjectKind::Light`），方向由物体的旋转决定
//! （局部 -Z 指向场景的方向就是光照方向），位置字段对方向光无意义。

use glam::Vec3;

/// 方向光：颜色 + 强度（方向由场景对象的旋转决定）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub color: Vec3,
    pub intensity: f32,
}

impl Light {
    /// 白光、单位强度。
    pub const WHITE: Self = Self {
        color: Vec3::ONE,
        intensity: 1.0,
    };
}
