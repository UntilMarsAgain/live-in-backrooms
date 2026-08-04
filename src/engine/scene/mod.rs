//! 场景模块：层级化的场景图。
//!
//! 树结构由 [`indextree`] 提供（arena 树 + 带代际标签的节点句柄），本模块只保留
//! 场景语义层：物体（节点）携带**局部**变换与可选网格，以及世界矩阵计算、
//! 增删/重挂载 API。
//!
//! 语义约定：
//! - 物体组织成一棵树，每个节点有唯一父节点；`remove_object` 删除**整棵子树**，
//!   父节点没了，子节点不允许残存为孤儿；
//! - 变换是相对父节点的局部值，世界变换 = 沿祖先链向上累乘
//!   （见 [`Scene::world_transform`]）；
//! - 句柄带代际，删除后不会复用同一数值，失效句柄被安全识别（返回 `None`/`false`）；
//! - 成环在库层就被禁止（`append`/`checked_append` 拒绝自挂与挂祖先），
//!   [`Scene::reparent`] 也会预检查后代关系，因此遍历不需要环保护。
//!
//! 网格资产由 `MeshLibrary` 永久持有，不属于某个场景；场景对象用
//! [`SceneObjectKind`] 区分类型（网格 / 空分组节点，灯光、相机等后续加入）。
//! 切换场景由 App 层 API 触发。

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use indextree::{Arena, NodeId};

use super::core::light::Light;
use super::core::material::Material;
use super::core::mesh::MeshKey;
use super::core::texture::TextureKey;
use super::core::transform::Transform;

/// 场景节点句柄：indextree 的节点 ID（带代际，删除后不失效复用）。
pub type ObjectKey = NodeId;

/// 场景对象的类型。
///
/// 灯光、相机等系统落地后，作为新的变体挂到这里；枚举让所有处理分支
/// （渲染、剔除、灯光收集等）都能被编译器强制检查。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneObjectKind {
    /// 纯分组节点：只承载子节点，本身不可见。
    Empty,
    /// 引用全局资产库里的网格。
    Mesh(MeshKey),
    /// 方向光：方向由物体旋转决定（局部 -Z 指向场景）。
    Light(Light),
}

/// 场景节点数据：局部变换 + 类型。
///
/// 父子关系由 `indextree` 的 arena 树维护，本结构不存任何句柄/指针。
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 相对父节点的局部变换；没有父节点时即世界变换。
    pub transform: Transform,
    /// 场景对象的类型。
    pub kind: SceneObjectKind,
    /// 表面材质（仅 `Mesh` 类型生效）。
    pub material: Material,
}

impl SceneObject {
    /// 新建一个节点。要挂到树上用 [`Scene::attach`] 或 [`Scene::reparent`]。
    pub fn new(kind: SceneObjectKind, transform: Transform) -> Self {
        Self {
            transform,
            kind,
            material: Material::default(),
        }
    }

    /// 设置材质（构建时链式调用）。
    pub fn with_material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// 若该对象是网格，返回其 `MeshKey`；否则返回 `None`。
    pub fn mesh_key(&self) -> Option<MeshKey> {
        match self.kind {
            SceneObjectKind::Mesh(key) => Some(key),
            SceneObjectKind::Empty | SceneObjectKind::Light(_) => None,
        }
    }
}

/// 场景：层级化的物体树（网格资产在全局 `MeshLibrary` 中）。
#[derive(Debug, Clone, Default)]
pub struct Scene {
    tree: Arena<SceneObject>,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// 存活节点总数（含纯分组节点）。
    pub fn object_count(&self) -> usize {
        self.tree.iter_node_ids().count()
    }

    /// 遍历所有存活节点（O(n)，无空洞）。
    pub fn objects(&self) -> impl Iterator<Item = (ObjectKey, &SceneObject)> + '_ {
        self.tree.iter_node_ids().filter_map(|id| {
            let node = self.tree.get(id)?;
            (!node.is_removed()).then(|| (id, node.get()))
        })
    }

    /// 所有根节点（无父节点的节点）。
    pub fn roots(&self) -> impl Iterator<Item = (ObjectKey, &SceneObject)> + '_ {
        self.objects().filter(|(id, _)| id.parent(&self.tree).is_none())
    }

    /// 直接子节点（O(n) 扫描，场景规模小所以足够）。
    pub fn children_of(&self, key: ObjectKey) -> impl Iterator<Item = ObjectKey> + '_ {
        self.tree
            .iter_node_ids()
            .filter(move |id| id.parent(&self.tree) == Some(key))
    }

    /// 添加一个根节点（O(1)）。要作为子节点挂载请用 [`Scene::attach`]。
    pub fn add_object(&mut self, object: SceneObject) -> ObjectKey {
        self.tree.new_node(object)
    }

    /// 把新节点挂到 `parent` 下并返回句柄；父节点已失效时返回 `None`（O(1)）。
    pub fn attach(&mut self, parent: ObjectKey, object: SceneObject) -> Option<ObjectKey> {
        if parent.is_removed(&self.tree) {
            return None;
        }
        let child = self.tree.new_node(object);
        // 新节点无任何关系，append 不可能失败（不会自挂/挂祖先/已删除）。
        parent.append(child, &mut self.tree);
        Some(child)
    }

    /// 把已有节点移到 `new_parent` 下（`None` 表示变为根节点）。
    ///
    /// 新父节点是自身或自身后代（会成环）时拒绝操作并返回 `false`，
    /// 且**不产生任何副作用**；节点已失效同样返回 `false`。
    pub fn reparent(&mut self, key: ObjectKey, new_parent: Option<ObjectKey>) -> bool {
        if key.is_removed(&self.tree) {
            return false;
        }
        match new_parent {
            None => {
                key.detach(&mut self.tree);
                true
            }
            Some(parent) => {
                if parent == key || parent.is_removed(&self.tree) {
                    return false;
                }
                // ancestors 包含节点自身；若 parent 的祖先链里有 key，则 key 是 parent 的祖先。
                if parent.ancestors(&self.tree).any(|id| id == key) {
                    return false;
                }
                key.detach(&mut self.tree);
                // 已排除自挂/挂祖先/已删除，append 不会失败。
                parent.append(key, &mut self.tree);
                true
            }
        }
    }

    /// 删除节点及其**整棵子树**（O(子树大小)），返回被删除的根节点数据。
    ///
    /// 父节点被删除后，子节点不会残存为孤儿；句柄已失效时返回 `None`。
    pub fn remove_object(&mut self, key: ObjectKey) -> Option<SceneObject> {
        if key.is_removed(&self.tree) {
            return None;
        }
        let removed = self.tree[key].get().clone();
        key.remove_subtree(&mut self.tree);
        Some(removed)
    }

    /// 把 `other` 中的所有物体复制进本场景（保持层级关系），返回新复制的根节点句柄。
    ///
    /// 用于把 glTF 等外部场景并入现有场景（例如并进演示场景）。
    pub fn merge(&mut self, other: &Scene) -> Vec<ObjectKey> {
        // 第一遍：复制所有节点，记录旧句柄 → 新句柄。
        let mut remap: HashMap<ObjectKey, ObjectKey> = HashMap::new();
        for (old_key, object) in other.objects() {
            // 整份复制（含材质）；之前只重建 kind+transform 会把材质丢掉。
            let new_key = self.add_object(object.clone());
            remap.insert(old_key, new_key);
        }
        // 第二遍：按旧场景的父子关系重建层级。
        for (old_key, _) in other.objects() {
            if let Some(old_parent) = old_key.parent(&other.tree) {
                let new_child = remap[&old_key];
                let new_parent = remap[&old_parent];
                let _ = self.reparent(new_child, Some(new_parent));
            }
        }
        other.roots().map(|(key, _)| remap[&key]).collect()
    }

    /// 沿祖先链向上累乘得到世界矩阵（O(深度)）。
    ///
    /// 树结构保证祖先链无环、必然终止；节点已失效时返回 `None`。
    pub fn world_transform(&self, key: ObjectKey) -> Option<Mat4> {
        if key.is_removed(&self.tree) {
            return None;
        }
        let chain: Vec<Transform> = key
            .ancestors(&self.tree)
            .map(|id| self.tree[id].get().transform)
            .collect();
        Some(
            chain
                .iter()
                .rev()
                .fold(Mat4::IDENTITY, |world, transform| world * transform.to_mat4()),
        )
    }

    /// 按句柄访问（O(1)）；句柄失效时返回 `None`。
    pub fn object(&self, key: ObjectKey) -> Option<&SceneObject> {
        if key.is_removed(&self.tree) {
            return None;
        }
        self.tree.get(key).map(|node| node.get())
    }

    /// 按句柄可变访问（O(1)）；句柄失效时返回 `None`。
    pub fn object_mut(&mut self, key: ObjectKey) -> Option<&mut SceneObject> {
        if key.is_removed(&self.tree) {
            return None;
        }
        self.tree.get_mut(key).map(|node| node.get_mut())
    }

    /// 演示场景：三角形、四边形、立方体三种资产，物体以不同位置/旋转/缩放摆放。
    ///
    /// 最后一个物体故意挂在立方体下：小三角形会跟随立方体一起旋转，
    /// 用来验证层级变换（world_transform）工作正常。
    pub fn demo(
        triangle: MeshKey,
        quad: MeshKey,
        cube: MeshKey,
        cube_texture: Option<TextureKey>,
    ) -> Self {
        let mut scene = Self::new();
        // 方向光：从右上前方照向场景。
        let light_direction = Vec3::new(0.5, 0.6, 0.6).normalize();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::WHITE),
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_arc(Vec3::NEG_Z, light_direction),
                Vec3::ONE,
            ),
        ));
        // 点光（辅助灯）：暖色，放在演示物体右上方。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::new(1.0, 0.85, 0.6), 100.0)),
            Transform::new(Vec3::new(2.2, 1.8, 0.8), Quat::IDENTITY, Vec3::ONE),
        ));
        // 面光（荧光灯面板）：朝下照亮扳手区域。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::area(1.5, 0.6, Vec3::new(0.9, 0.95, 1.0), 45.0)),
            Transform::new(
                Vec3::new(1.8, 2.8, -0.8),
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                Vec3::ONE,
            ),
        ));
        scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(triangle),
            Transform::IDENTITY,
        ));
        scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(triangle),
            Transform::new(
                Vec3::new(1.8, 0.0, 0.6),
                Quat::from_rotation_y(0.9),
                Vec3::ONE,
            ),
        ));
        // 四边形：X 轴拉长，演示非等比缩放。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(quad),
            Transform::new(
                Vec3::new(-1.8, 0.4, 0.8),
                Quat::from_rotation_z(0.7),
                Vec3::new(1.6, 1.0, 1.0),
            ),
        ));
        scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(triangle),
            Transform::new(
                Vec3::new(0.6, 1.6, -0.8),
                Quat::from_rotation_x(1.1),
                Vec3::ONE,
            ),
        ));
        // 立方体：放在视野正上方偏后，绕 Y 和 X 各转一点，让多个面可见。
        let cube = scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(cube),
            Transform::new(
                Vec3::new(0.0, 1.5, -1.6),
                Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.6),
                Vec3::splat(1.3),
            ),
        )
        .with_material(Material {
            base_color: [1.0; 4],
            base_color_texture: cube_texture,
            ..Material::default()
        }));
        // 小三角形挂在立方体正上方，跟随立方体一起旋转（验证层级）。
        let _ = scene.attach(
            cube,
            SceneObject::new(
                SceneObjectKind::Mesh(triangle),
                Transform::new(Vec3::new(0.0, 1.2, 0.0), Quat::IDENTITY, Vec3::splat(0.4)),
            ),
        );
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 世界变换 = 根 × 父 × 自身（平移链沿祖先累乘）。
    #[test]
    fn world_transform_multiplies_parent_chain() {
        let mut scene = Scene::new();
        let root = scene.add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
        ));
        let child = scene
            .attach(
                root,
                SceneObject::new(
                    SceneObjectKind::Empty,
                    Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                ),
            )
            .expect("父节点存活");

        let world = scene.world_transform(child).unwrap();
        // 子节点原点的世界位置 = (1, 2, 0)
        assert_eq!(world.transform_point3(Vec3::ZERO), Vec3::new(1.0, 2.0, 0.0));
    }

    /// 子节点跟随父节点旋转：父绕 Y 转 90°，子局部 +X 方向点应落到 ±Z 轴上。
    #[test]
    fn world_transform_follows_parent_rotation() {
        let mut scene = Scene::new();
        let root = scene.add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::new(Vec3::ZERO, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2), Vec3::ONE),
        ));
        let child = scene
            .attach(
                root,
                SceneObject::new(
                    SceneObjectKind::Empty,
                    Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                ),
            )
            .expect("父节点存活");

        let world = scene.world_transform(child).unwrap();
        let p = world.transform_point3(Vec3::ZERO);
        // 局部 (1,0,0) 旋转 90° 后落在 (0,0,±1)
        assert!(p.z.abs() > 0.99, "p = {p:?}");
        assert!(p.x.abs() < 0.01, "p = {p:?}");
        assert!(p.y.abs() < 0.01, "p = {p:?}");
    }

    /// merge 必须保留材质（glTF 场景合并进演示场景时材质不能丢）。
    #[test]
    fn merge_preserves_material() {
        let mut a = Scene::new();
        a.add_object(
            SceneObject::new(SceneObjectKind::Empty, Transform::IDENTITY).with_material(Material {
                base_color: [0.2, 0.3, 0.4, 1.0],
                ..Material::default()
            }),
        );

        let mut b = Scene::new();
        let merged = b.merge(&a);
        assert_eq!(merged.len(), 1);
        let obj = b.object(merged[0]).expect("合并后的节点应存活");
        assert_eq!(obj.material.base_color, [0.2, 0.3, 0.4, 1.0]);
    }
}
