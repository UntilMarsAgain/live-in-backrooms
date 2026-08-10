//! 组件（C）：场景物体的数据，全部由 bevy_ecs 存储。
//!
//! 对应旧 `SceneObjectKind` 的拆解：变换 / 网格句柄 / 材质 / 灯光 / 相机
//! 各自成为独立组件，层级用 bevy 内置的 `ChildOf` / `Children` 关系
//! （自动维护 + 级联 despawn，见 `bevy_ecs::hierarchy`），碰撞盒从网格 AABB
//! 派生为 `Collider`（系统只读它，不碰资产库）。

use bevy_ecs::prelude::*;
use glam::Mat4;

use crate::engine::core::asset::Handle;
use crate::engine::core::camera::Camera;
use crate::engine::core::data::aabb::Aabb;
use crate::engine::core::data::light::Light;
use crate::engine::core::data::material::Material;
use crate::engine::core::data::mesh::Mesh;
use crate::engine::core::data::transform::Transform;

/// 局部变换（相对父实体）。
#[derive(Component, Debug, Clone, Copy)]
pub struct LocalTransform(pub Transform);

/// 世界矩阵（由 [`super::hierarchy::propagate_world_transforms`] 维护）。
#[derive(Component, Debug, Clone, Copy)]
pub struct WorldMatrix(pub Mat4);

/// 引用统一资产库里的网格。
#[derive(Component, Debug, Clone, Copy)]
pub struct MeshHandle(pub Handle<Mesh>);

/// 表面材质。
#[derive(Component, Debug, Clone)]
pub struct MaterialC(pub Material);

/// 灯光数据（位置/朝向由世界矩阵决定）。
#[derive(Component, Debug, Clone, Copy)]
pub struct LightC(pub Light);

/// 相机数据（位置/朝向/投影自持，与旧 `SceneObjectKind::Camera` 一致）。
#[derive(Component, Debug, Clone)]
pub struct CameraC(pub Camera);

/// 主相机标记（场景生成时打在主相机实体上）。
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct MainCamera;

/// 局部空间碰撞盒（由网格 AABB 派生；渲染/碰撞系统只读这个，不碰资产库）。
#[derive(Component, Debug, Clone, Copy)]
pub struct Collider(pub Aabb);
