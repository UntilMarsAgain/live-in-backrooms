//! 共享数据/数学层：无 GPU、无场景依赖，所有子系统从这里取数据。
//!
//! 物理（碰撞需要 Transform/Mesh）、音频等未来子系统也消费这一层。

pub mod asset;
pub mod camera;
pub mod data;
pub mod environment;
pub mod gc;
pub mod resource;
