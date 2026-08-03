//! 场景模块：物体列表。
//!
//! 场景是"关卡"的实例状态：物体携带自己的位置/旋转/缩放，通过 [`MeshKey`]
//! 引用全局资产库里的网格。网格资产由 `MeshLibrary` 永久持有，不属于某个场景。
//! 切换场景由 App 层 API 触发（见 `App::load_scene`）。
//!
//! 物体存放在带代际标签的稀疏集（[`SlotMap`]）中：添加/删除/访问都是 O(1)
//! （Vec 扩容除外），遍历 O(n) 且只包含存活物体、没有空洞；句柄带代际，
//! 删除后不会复用同一数值，避免"悬空句柄撞上新物体"。

use glam::{Quat, Vec3};
use slotmap::{new_key_type, SlotMap};

use crate::mesh::MeshKey;

new_key_type! {
    pub struct ObjectKey;
}

/// 场景物体：位置 + 旋转 + 缩放 + 全局网格句柄。
#[derive(Debug, Clone, Copy)]
pub struct SceneObject {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// 引用的全局网格资产。
    pub mesh: MeshKey,
}

/// 场景：物体列表（网格资产在全局 `MeshLibrary` 中）。
#[derive(Debug, Clone, Default)]
pub struct Scene {
    objects: SlotMap<ObjectKey, SceneObject>,
}

impl Scene {
    pub fn new(objects: SlotMap<ObjectKey, SceneObject>) -> Self {
        Self { objects }
    }

    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// 遍历所有存活物体（O(n)，无空洞）。
    pub fn objects(&self) -> impl Iterator<Item = (ObjectKey, &SceneObject)> + '_ {
        self.objects.iter()
    }

    /// 添加物体，返回句柄（O(1)，扩容除外）。
    pub fn add_object(&mut self, object: SceneObject) -> ObjectKey {
        self.objects.insert(object)
    }

    /// 删除物体（O(1)），返回被删除的物体；句柄已失效时返回 `None`。
    #[allow(dead_code)] // 预留：等场景编辑/卸载逻辑接入后使用
    pub fn remove_object(&mut self, key: ObjectKey) -> Option<SceneObject> {
        self.objects.remove(key)
    }

    /// 按句柄访问（O(1)）。
    #[allow(dead_code)] // 预留
    pub fn object(&self, key: ObjectKey) -> Option<&SceneObject> {
        self.objects.get(key)
    }

    /// 按句柄可变访问（O(1)）。
    #[allow(dead_code)] // 预留
    pub fn object_mut(&mut self, key: ObjectKey) -> Option<&mut SceneObject> {
        self.objects.get_mut(key)
    }

    /// 演示场景：三角形、四边形、立方体三种资产，物体以不同位置/旋转/缩放摆放。
    pub fn demo(triangle: MeshKey, quad: MeshKey, cube: MeshKey) -> Self {
        let mut scene = Self::new(SlotMap::with_key());
        scene.add_object(SceneObject {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            mesh: triangle,
        });
        scene.add_object(SceneObject {
            position: Vec3::new(1.8, 0.0, 0.6),
            rotation: Quat::from_rotation_y(0.9),
            scale: Vec3::ONE,
            mesh: triangle,
        });
        // 四边形：X 轴拉长，演示非等比缩放。
        scene.add_object(SceneObject {
            position: Vec3::new(-1.8, 0.4, 0.8),
            rotation: Quat::from_rotation_z(0.7),
            scale: Vec3::new(1.6, 1.0, 1.0),
            mesh: quad,
        });
        scene.add_object(SceneObject {
            position: Vec3::new(0.6, 1.6, -0.8),
            rotation: Quat::from_rotation_x(1.1),
            scale: Vec3::ONE,
            mesh: triangle,
        });
        // 立方体：放在视野正上方偏后，绕 Y 和 X 各转一点，让多个面可见。
        scene.add_object(SceneObject {
            position: Vec3::new(0.0, 1.5, -1.6),
            rotation: Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.6),
            scale: Vec3::splat(1.3),
            mesh: cube,
        });
        scene
    }
}
