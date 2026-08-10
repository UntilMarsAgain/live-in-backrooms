//! 引擎模块：按子系统分层，依赖方向单向。
//!
//! ```text
//! core（共享数据/数学） ← scene ← render / asset
//!                       ← ecs（场景/渲染/输入统一在 ECS 上）
//! ```
//!
//! - [`core`]：无 GPU、无场景依赖的共享数据层（Transform/Mesh/Texture/Material/Light/Camera）；
//! - [`scene`]：层级场景图；
//! - [`render`]：渲染；
//! - [`asset`]：资产加载；
//! - [`ecs`]：实体组件系统（场景生成、固定步长物理刻、自由相机）。
//!
//! 未来的 physics / audio 等子系统同样消费 core/scene。游戏内容不属于这里，
//! 见 [`crate::game`]。

pub mod asset;
pub mod core;
pub mod ecs;
pub mod render;
pub mod scene;

// 引擎对外 API（应用层/游戏层从这里取用；部分类型当前尚未用到，属预留）。
#[allow(unused_imports)]
pub use asset::MeshView;
#[allow(unused_imports)]
pub use core::asset::{
    AssetManager, AssetState, AssetStatus, FileLoader, Handle, HandleState, MeshSource, PinGuard,
};
#[allow(unused_imports)]
pub use core::camera::Camera;
#[allow(unused_imports)]
pub use core::camera::CameraAction;
#[allow(unused_imports)]
pub use core::data::aabb::Aabb;
#[allow(unused_imports)]
pub use core::data::light::Light;
#[allow(unused_imports)]
pub use core::data::material::Material;
#[allow(unused_imports)]
pub use core::data::mesh::{Mesh, Vertex};
#[allow(unused_imports)]
pub use core::data::texture::Texture;
#[allow(unused_imports)]
pub use core::data::transform::Transform;
#[allow(unused_imports)]
pub use core::environment::Environment;
#[allow(unused_imports)]
pub use core::gc::GcPolicy;
#[allow(unused_imports)]
pub use core::resource::config::PackConfig;
#[allow(unused_imports)]
pub use core::resource::game_path::GamePath;
#[allow(unused_imports)]
pub use core::resource::pack::Package;
#[allow(unused_imports)]
pub use core::resource::MergedResourceSpace;
#[allow(unused_imports)]
pub use render::{DisplayHandle, GpuManager, MeshGpu, Renderer, TextureGpu};
#[allow(unused_imports)]
pub use scene::{ObjectKey, SceneObject, SceneObjectKind, Scene};
