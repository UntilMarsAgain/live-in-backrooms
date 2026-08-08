//! 引擎模块：按子系统分层，依赖方向单向。
//!
//! ```text
//! core（共享数据/数学） ← scene ← render / asset
//!                       ← input（只依赖 core）
//! ```
//!
//! - [`core`]：无 GPU、无场景依赖的共享数据层（Transform/Mesh/Texture/Material/Light/Camera）；
//! - [`scene`]：层级场景图；
//! - [`render`]：渲染；
//! - [`asset`]：资产加载；
//! - [`input`]：输入控制。
//!
//! 未来的 physics / audio 等子系统同样消费 core/scene。游戏内容不属于这里，
//! 见 [`crate::game`]。

pub mod asset;
pub mod core;
pub mod input;
pub mod render;
pub mod scene;

// 引擎对外 API（应用层/游戏层从这里取用；部分类型当前尚未用到，属预留）。
#[allow(unused_imports)]
pub use core::aabb::Aabb;
#[allow(unused_imports)]
pub use core::asset::{
    AssetRegistry, AssetState, GpuUploader, Handle, MeshSource, NoGpuUploader,
};
#[allow(unused_imports)]
pub use core::camera::Camera;
#[allow(unused_imports)]
pub use core::camera::CameraAction;
#[allow(unused_imports)]
pub use core::config::PackConfig;
#[allow(unused_imports)]
pub use core::environment::Environment;
#[allow(unused_imports)]
pub use core::game_path::GamePath;
#[allow(unused_imports)]
pub use core::light::Light;
#[allow(unused_imports)]
pub use core::material::Material;
#[allow(unused_imports)]
pub use core::mesh::{Mesh, Vertex};
#[allow(unused_imports)]
pub use core::pack::Package;
#[allow(unused_imports)]
pub use core::resource::MergedResourceSpace;
#[allow(unused_imports)]
pub use core::texture::Texture;
#[allow(unused_imports)]
pub use core::transform::Transform;
#[allow(unused_imports)]
pub use input::{FreeCameraController, InputController};
#[allow(unused_imports)]
pub use render::{AssetManager, DisplayHandle, MeshGpu, Renderer, TextureGpu};
#[allow(unused_imports)]
pub use scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};
