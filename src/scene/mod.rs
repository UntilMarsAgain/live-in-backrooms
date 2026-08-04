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
//! 网格资产由 `MeshLibrary` 永久持有，不属于某个场景；`SceneObject::mesh` 为
//! `None` 表示纯分组节点（只承载子节点）。切换场景由 App 层 API 触发。

use glam::{Mat4, Quat, Vec3};
use indextree::{Arena, NodeId};

use crate::mesh::MeshKey;

/// 场景节点句柄：indextree 的节点 ID（带代际，删除后不失效复用）。
pub type ObjectKey = NodeId;

/// 场景节点数据：局部变换 + 可选网格。
///
/// 父子关系由 `indextree` 的 arena 树维护，本结构不存任何句柄/指针。
#[derive(Debug, Clone)]
pub struct SceneObject {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// 引用的全局网格资产；`None` 表示纯分组节点（只承载子节点）。
    pub mesh: Option<MeshKey>,
}

impl SceneObject {
    /// 新建一个节点。要挂到树上用 [`Scene::attach`] 或 [`Scene::reparent`]。
    pub fn new(mesh: Option<MeshKey>, position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
            mesh,
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

    /// 沿祖先链向上累乘得到世界矩阵（O(深度)）。
    ///
    /// 树结构保证祖先链无环、必然终止；节点已失效时返回 `None`。
    pub fn world_transform(&self, key: ObjectKey) -> Option<Mat4> {
        if key.is_removed(&self.tree) {
            return None;
        }
        let mut chain = Vec::new();
        for id in key.ancestors(&self.tree) {
            let obj = &self.tree[id].get();
            chain.push((obj.scale, obj.rotation, obj.position));
        }
        let mut world = Mat4::IDENTITY;
        for (scale, rotation, position) in chain.iter().rev() {
            world *= Mat4::from_scale_rotation_translation(*scale, *rotation, *position);
        }
        Some(world)
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
    pub fn demo(triangle: MeshKey, quad: MeshKey, cube: MeshKey) -> Self {
        let mut scene = Self::new();
        scene.add_object(SceneObject::new(
            Some(triangle),
            Vec3::new(0.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        scene.add_object(SceneObject::new(
            Some(triangle),
            Vec3::new(1.8, 0.0, 0.6),
            Quat::from_rotation_y(0.9),
            Vec3::ONE,
        ));
        // 四边形：X 轴拉长，演示非等比缩放。
        scene.add_object(SceneObject::new(
            Some(quad),
            Vec3::new(-1.8, 0.4, 0.8),
            Quat::from_rotation_z(0.7),
            Vec3::new(1.6, 1.0, 1.0),
        ));
        scene.add_object(SceneObject::new(
            Some(triangle),
            Vec3::new(0.6, 1.6, -0.8),
            Quat::from_rotation_x(1.1),
            Vec3::ONE,
        ));
        // 立方体：放在视野正上方偏后，绕 Y 和 X 各转一点，让多个面可见。
        let cube = scene.add_object(SceneObject::new(
            Some(cube),
            Vec3::new(0.0, 1.5, -1.6),
            Quat::from_rotation_x(0.35) * Quat::from_rotation_y(0.6),
            Vec3::splat(1.3),
        ));
        // 小三角形挂在立方体正上方，跟随立方体一起旋转（验证层级）。
        let _ = scene.attach(
            cube,
            SceneObject::new(
                Some(triangle),
                Vec3::new(0.0, 1.2, 0.0),
                Quat::IDENTITY,
                Vec3::splat(0.4),
            ),
        );
        scene
    }
}
