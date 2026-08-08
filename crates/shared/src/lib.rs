//! 双方共有层（shared）：客户端与服务端都依赖的数据、场景与解析逻辑。
//!
//! ```text
//! core（纯数据/数学） ← scene（服务端视图的场景树） ← asset（glTF 解析）
//! ```
//!
//! - [`core`]：无 GPU、无场景依赖的共享数据层（Transform/Mesh/Material/
//!   Light/Camera/Environment/GamePath/资源句柄等）；
//! - [`scene`]：**服务端视图**的场景树——只含节点、局部变换与引用（网格/灯光），
//!   不含渲染信息（材质/相机/环境）。客户端在客户端视图场景（`ClientScene`，
//!   lbr-client crate）里叠加渲染信息；
//! - [`asset`]：glTF 等文件的纯解析（客户端用于上传 GPU，服务端用于碰撞数据）。
//!
//! 未来 physics / 协议等子系统同样消费这一层。游戏内容不属于这里，
//! 见游戏内容层（lbr-game crate）。

pub mod asset;
pub mod core;
pub mod protocol;
pub mod scene;

// 共享层对外 API（应用层/游戏层从这里取用；部分类型当前尚未用到，属预留）。
#[allow(unused_imports)]
pub use core::aabb::Aabb;
#[allow(unused_imports)]
pub use core::asset::{AssetLoader, AssetRegistry, AssetState, DataSource, Handle, MeshSource};
#[allow(unused_imports)]
pub use core::camera::{Camera, CameraAction};
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
pub use core::resource::MergedResourceSpace;
#[allow(unused_imports)]
pub use core::texture::Texture;
#[allow(unused_imports)]
pub use core::transform::Transform;
#[allow(unused_imports)]
pub use protocol::{EntityKind, EntityState, WorldSnapshot};
#[allow(unused_imports)]
pub use scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};
