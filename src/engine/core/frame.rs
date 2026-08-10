//! 渲染指令（拍扁的场景）：ECS 与渲染器之间的数据契约。
//!
//! 只含语义数据：资源库句柄（`Handle<Mesh>` / `Handle<Texture>`）、世界矩阵、
//! 灯光的类型 + 位置/朝向、碰撞盒的 AABB + 世界矩阵。**不含任何 GPU 布局**
//! （uniform 打包、线框顶点生成都是渲染侧的工作）。
//!
//! 本模块属于 core：既不依赖 ECS 也不依赖 wgpu，服务端（无头）与客户端共用。
//! ECS 的 `prepare_frame` 系统负责填充它，渲染器消费它。

use glam::{Mat4, Quat, Vec3};

use super::asset::Handle;
use super::camera::Camera;
use super::data::aabb::Aabb;
use super::data::light::LightKind;
use super::data::material::Material;
use super::data::mesh::Mesh;

/// 一帧可绘制的物体（只含网格实体；实例下标 = ObjectData 数组下标）。
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub world_matrix: Mat4,
    pub material: Material,
    pub mesh: Handle<Mesh>,
}

/// 语义灯光：渲染指令里的灯光描述（类型 + 世界位置/朝向 + 光参数）。
///
/// 只描述"场景里有什么光"，不含任何 GPU 布局；打包成 uniform 是渲染侧的工作。
#[derive(Debug, Clone, Copy)]
pub struct LightData {
    pub kind: LightKind,
    /// 世界位置（方向光忽略）。
    pub position: Vec3,
    /// 世界朝向（方向光/面光：局部 -Z = 光行进方向；点光忽略）。
    pub rotation: Quat,
    pub color: Vec3,
    pub intensity: f32,
}

/// 语义碰撞箱：渲染指令里的调试碰撞箱描述（局部 AABB + 世界变换）。
///
/// 只描述"哪里有什么样的碰撞箱"；生成线框顶点是渲染侧的工作。
#[derive(Debug, Clone, Copy)]
pub struct ColliderData {
    pub aabb: Aabb,
    pub world: Mat4,
}

/// 渲染指令：一帧的绘制描述（相机 + 物体 + 灯光 + 碰撞箱）——拍扁的场景。
///
/// ECS 的 `prepare_frame` 系统每帧填充它，渲染器消费它。
#[derive(Debug, Clone, Default)]
pub struct RenderCommand {
    pub camera: Option<Camera>,
    pub objects: Vec<RenderObject>,
    pub lights: Vec<LightData>,
    pub colliders: Vec<ColliderData>,
    pub show_light_debug: bool,
    pub show_collision_debug: bool,
}
