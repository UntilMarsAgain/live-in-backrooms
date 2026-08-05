//! 相机模块。
//!
//! 分层设计：
//! - 本文件：`Camera`，纯数学核心（位置、朝向、投影），不依赖 wgpu；
//! - [`uniform`]：`CameraUniform`，GPU 内存布局与数据填充；
//!
//! 输入控制（自由视角）已拆到 [`super::controller`]，相机本身不依赖任何输入。

pub mod uniform;

pub use uniform::CameraUniform;

use std::f32::consts::FRAC_PI_2;

use glam::camera::rh::proj::directx;
use glam::camera::rh::view;
use glam::{Mat4, Vec3};

/// 自由相机：位置 + 欧拉角朝向 + 透视投影参数。
///
/// 本阶段世界坐标与物体坐标等同，相机只负责"看"，不做物体变换；
/// 作为场景节点时（[`SceneObjectKind::Camera`]），位置/朝向由本结构自身决定，
/// 节点的局部 Transform 暂不参与合成。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    position: Vec3,
    /// 绕 Y 轴的偏航角（弧度）。
    yaw: f32,
    /// 俯仰角（弧度），正值向上。
    pitch: f32,
    /// 垂直视野角（弧度）。
    fov_y: f32,
    /// 宽高比（宽 / 高）。
    aspect: f32,
    near: f32,
    far: f32,
}

impl Camera {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        position: Vec3,
        yaw: f32,
        pitch: f32,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            yaw,
            pitch,
            fov_y,
            aspect,
            near,
            far,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// 视线方向（含俯仰）。
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    /// 水平视线方向（移动用，忽略俯仰）。
    pub fn forward_horizontal(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    /// 水平右方向。
    pub fn right(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos())
    }

    /// 旋转相机（偏航 / 俯仰，弧度），俯仰限制在 ±90° 内。
    pub fn rotate(&mut self, yaw_delta: f32, pitch_delta: f32) {
        self.yaw += yaw_delta;
        self.pitch = (self.pitch + pitch_delta).clamp(-FRAC_PI_2 + 0.01, FRAC_PI_2 - 0.01);
    }

    /// 平移相机。
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    /// 沿视线方向前后移动。
    #[allow(dead_code)] // 预留：滚轮改调速度后暂无调用方
    pub fn move_forward(&mut self, distance: f32) {
        self.position += self.forward() * distance;
    }

    /// 视图 × 投影矩阵。
    pub fn view_proj(&self) -> Mat4 {
        let view_mat = view::look_to_mat4(self.position, self.forward(), Vec3::Y);
        // directx 系列投影的深度范围为 0..1，与 wgpu 的 NDC 一致。
        let proj_mat = directx::perspective(self.fov_y, self.aspect, self.near, self.far);
        proj_mat * view_mat
    }
}
