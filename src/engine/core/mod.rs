//! 共享数据/数学层：无 GPU、无场景依赖，所有子系统从这里取数据。
//!
//! 物理（碰撞需要 Transform/Mesh）、音频等未来子系统也消费这一层。

pub mod aabb;
pub mod asset;
pub mod camera;
pub mod config;
pub mod environment;
pub mod game_path;
pub mod light;
pub mod material;
pub mod mesh;
pub mod pack;
pub mod resource;
pub mod texture;
pub mod transform;
