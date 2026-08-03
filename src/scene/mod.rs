//! 场景模块：网格调色盘 + 物体列表。
//!
//! 场景是"关卡"的静态描述：网格在调色盘中去重，物体通过索引引用网格，
//! 并携带自己的位置与旋转。切换场景由 App 层 API 触发（见 `App::load_scene`）。

use glam::{Quat, Vec3};

use crate::mesh::Mesh;

/// 场景物体：位置 + 旋转 + 调色盘网格索引。
#[derive(Debug, Clone, Copy)]
pub struct SceneObject {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub mesh_index: u32,
}

/// 场景：网格调色盘与物体列表。
#[derive(Debug, Clone, Default)]
pub struct Scene {
    meshes: Vec<Mesh>,
    objects: Vec<SceneObject>,
}

impl Scene {
    pub fn new(meshes: Vec<Mesh>, objects: Vec<SceneObject>) -> Self {
        Self { meshes, objects }
    }

    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn objects(&self) -> &[SceneObject] {
        &self.objects
    }

    pub fn add_object(&mut self, object: SceneObject) {
        self.objects.push(object);
    }

    /// 演示场景：调色盘里有三角形、四边形和立方体三种网格（顶点/索引数量各不相同），
    /// 物体以不同位置/旋转摆放，用来验证物体变换与多网格区间绘制。
    pub fn demo() -> Self {
        let palette = vec![Mesh::triangle(), Mesh::quad(), Mesh::cube()];
        let mut scene = Self::new(palette, Vec::new());
        scene.add_object(SceneObject {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            mesh_index: 0,
        });
        scene.add_object(SceneObject {
            position: Vec3::new(1.8, 0.0, 0.6),
            rotation: Quat::from_rotation_y(0.9),
            scale: Vec3::ONE,
            mesh_index: 0,
        });
        // 四边形：X 轴拉长，演示非等比缩放。
        scene.add_object(SceneObject {
            position: Vec3::new(-1.8, 0.4, 0.8),
            rotation: Quat::from_rotation_z(0.7),
            scale: Vec3::new(1.6, 1.0, 1.0),
            mesh_index: 1, // 四边形
        });
        scene.add_object(SceneObject {
            position: Vec3::new(0.6, 1.6, -0.8),
            rotation: Quat::from_rotation_x(1.1),
            scale: Vec3::ONE,
            mesh_index: 0,
        });
        // 立方体：放在视野正上方偏后，绕 Y 和 X 各转一点，让多个面可见。
        scene.add_object(SceneObject {
            position: Vec3::new(0.0, 1.5, -1.6),
            rotation: Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.6),
            scale: Vec3::splat(1.3),
            mesh_index: 2,
        });
        scene
    }
}
