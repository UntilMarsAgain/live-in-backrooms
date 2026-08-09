//! 基础数据：场景 / 渲染 / 资产共用的纯数据与数学类型。
//!
//! 每个类型一个文件，集中在一个模块下，避免 `core` 顶层过于零散：
//! [`Transform`](transform::Transform) / [`Aabb`](aabb::Aabb) /
//! [`Light`](light::Light) / [`Material`](material::Material) /
//! [`Texture`](texture::Texture) / [`Mesh`](mesh::Mesh)。

pub mod aabb;
pub mod light;
pub mod material;
pub mod mesh;
pub mod texture;
pub mod transform;
