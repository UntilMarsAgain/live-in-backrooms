//! 变换组件：位置 + 旋转 + 缩放（TRS）。
//!
//! 场景对象、灯光、相机等一切需要“摆放在世界坐标系里”的东西都可以复用这个
//! 结构；世界矩阵 = 沿父链累乘各节点的 [`Transform::to_mat4`]。

use glam::{Mat4, Quat, Vec3};

/// 局部变换：位置 + 旋转 + 缩放。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    /// 单位变换：原点、无旋转、单位缩放。
    pub const IDENTITY: Self = Self {
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// 转成 4×4 矩阵（模型矩阵）。
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}
