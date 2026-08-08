//! 相机的 GPU uniform：只负责内存布局与数据填充，不创建任何 GPU 对象。
//!
//! 缓冲区与绑定组的创建在 `render` 模块完成，这里保持纯数据。

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use super::Camera;

/// 与 WGSL 中 `CameraUniform` 对应的 CPU 侧布局：
/// `mat4x4<f32>`（64 字节）+ `vec3<f32>` + 4 字节填充 = 80 字节，
/// 再 + `mat4x4<f32>`（64 字节）= 144 字节。
/// 逆视图×投影矩阵供天空盒从 NDC 反推视线方向，也方便日后重建世界坐标。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: Mat4,
    pub position: Vec3,
    pub _padding: u32,
    pub inverse_view_proj: Mat4,
}

impl CameraUniform {
    /// 从相机生成一帧的 uniform 数据。
    pub fn from_camera(camera: &Camera) -> Self {
        let view_proj = camera.view_proj();
        Self {
            view_proj,
            position: camera.position(),
            _padding: 0,
            inverse_view_proj: view_proj.inverse(),
        }
    }
}
