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

/// 一组可实例化绘制：同一个网格 + 同一材质，多个世界变换一次 draw 画完。
///
/// 组内实例在 `ObjectData` 数组中连续排列（渲染侧按组顺序编号），
/// 着色器按 `instance_index` 取矩阵——这是 multi-draw 参数表的雏形。
#[derive(Debug, Clone)]
pub struct RenderMeshGroup {
    pub mesh: Handle<Mesh>,
    pub material: Material,
    /// 组内实例的世界矩阵（顺序 = 该组在 object_data 数组中的连续段）。
    pub instances: Vec<Mat4>,
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
    /// 网格绘制组：每组对应一次实例化 `draw_indexed`。
    pub meshes: Vec<RenderMeshGroup>,
    pub lights: Vec<LightData>,
    pub colliders: Vec<ColliderData>,
    pub show_light_debug: bool,
    pub show_collision_debug: bool,
}

impl RenderCommand {
    /// 追加一个网格实例：并入同网格同材质的组（没有则新建）。
    ///
    /// 线性查找即可——场景物体数小；以后量大可换成按
    /// `(Handle<Mesh>, Material)` 哈希索引或排序归并。
    pub fn push_mesh_instance(&mut self, mesh: Handle<Mesh>, material: Material, world: Mat4) {
        for group in &mut self.meshes {
            if group.mesh == mesh && group.material == material {
                group.instances.push(world);
                return;
            }
        }
        self.meshes.push(RenderMeshGroup {
            mesh,
            material,
            instances: vec![world],
        });
    }
}

#[cfg(test)]
mod tests {
    use glam::Mat4;
    use slotmap::SlotMap;

    use super::*;

    /// 从临时 SlotMap 生成互不相同的句柄（测试不需要真实资产）。
    fn handles() -> [Handle<Mesh>; 2] {
        let mut map = SlotMap::new();
        let a = Handle::from_key(map.insert(()));
        let b = Handle::from_key(map.insert(()));
        [a, b]
    }

    /// 同网格同材质 → 合并成一个组；材质或网格不同 → 分开。
    #[test]
    fn push_mesh_instance_groups_by_mesh_and_material() {
        let [mesh_a, mesh_b] = handles();
        let mut material_a = Material::default();
        material_a.base_color = [1.0, 0.0, 0.0, 1.0];
        let mut material_b = material_a;
        material_b.base_color = [0.0, 1.0, 0.0, 1.0];

        let mut command = RenderCommand::default();
        command.push_mesh_instance(mesh_a, material_a, Mat4::IDENTITY);
        command.push_mesh_instance(mesh_a, material_a, Mat4::from_translation(glam::Vec3::X));
        command.push_mesh_instance(mesh_a, material_b, Mat4::IDENTITY);
        command.push_mesh_instance(mesh_b, material_a, Mat4::IDENTITY);

        assert_eq!(command.meshes.len(), 3);
        assert_eq!(command.meshes[0].instances.len(), 2);
        assert_eq!(command.meshes[1].instances.len(), 1);
        assert_eq!(command.meshes[2].instances.len(), 1);
    }
}
