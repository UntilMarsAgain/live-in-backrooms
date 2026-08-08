//! 场景模块（服务端视图）：层级化的场景树。
//!
//! 树结构由 [`indextree`] 提供（arena 树 + 带代际标签的节点句柄），本模块只保留
//! 场景语义层：物体（节点）携带**局部**变换与类型（网格/灯光/分组），以及
//! 世界矩阵计算、增删/重挂载 API。
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
//! 网格资产由统一 `AssetManager` 持有，不属于某个场景；场景对象用
//! [`SceneObjectKind`] 区分类型（网格 / 空分组节点 / 灯光）。
//!
//! 本模块是**服务端视图**：不携带任何渲染信息（材质、相机、环境、色调映射参数
//! 都在客户端视图场景 ClientScene 里叠加）。服务端与客户端共用这棵纯树，
//! 客户端在本地解析实体 ID 后重建它。

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use indextree::{Arena, NodeId};

use super::core::aabb::Aabb;
use super::core::asset::{Handle, MeshSource};
use super::core::light::Light;
use super::core::mesh::Mesh;
use super::core::transform::Transform;

/// 场景节点句柄：indextree 的节点 ID（带代际，删除后不失效复用）。
pub type ObjectKey = NodeId;

/// 场景对象的类型。
///
/// 枚举让所有处理分支（渲染、剔除、灯光收集等）都能被编译器强制检查；
/// 新类型作为变体挂到这里。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneObjectKind {
    /// 纯分组节点：只承载子节点，本身不可见。
    Empty,
    /// 引用统一资产管理器里的网格（[`Handle<Mesh>`]）。
    Mesh(Handle<Mesh>),
    /// 灯光：位置/朝向由节点变换决定（见 [`Light`] 的约定）。
    Light(Light),
}

/// 场景节点数据：局部变换 + 类型（服务端视图不携带渲染信息）。
///
/// 父子关系由 `indextree` 的 arena 树维护，本结构不存任何句柄/指针。
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// 相对父节点的局部变换；没有父节点时即世界变换。
    pub transform: Transform,
    /// 场景对象的类型。
    pub kind: SceneObjectKind,
}

impl SceneObject {
    /// 新建一个节点。要挂到树上用 [`Scene::attach`] 或 [`Scene::reparent`]。
    pub fn new(kind: SceneObjectKind, transform: Transform) -> Self {
        Self { transform, kind }
    }

    /// 若该对象是网格，返回其句柄；否则返回 `None`。
    pub fn mesh_handle(&self) -> Option<Handle<Mesh>> {
        match self.kind {
            SceneObjectKind::Mesh(handle) => Some(handle),
            SceneObjectKind::Empty | SceneObjectKind::Light(_) => None,
        }
    }
}

/// 场景：层级化的物体树（网格资产在统一 `AssetManager` 中）。
#[derive(Debug, Clone)]
pub struct Scene {
    tree: Arena<SceneObject>,
    /// 灯光节点缓存（增删时维护，渲染每帧按距离收集）。
    light_nodes: Vec<ObjectKey>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            tree: Arena::default(),
            light_nodes: Vec::new(),
        }
    }
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
        self.objects()
            .filter(|(id, _)| id.parent(&self.tree).is_none())
    }

    /// 直接子节点（O(n) 扫描，场景规模小所以足够）。
    pub fn children_of(&self, key: ObjectKey) -> impl Iterator<Item = ObjectKey> + '_ {
        self.tree
            .iter_node_ids()
            .filter(move |id| id.parent(&self.tree) == Some(key))
    }

    /// 添加一个根节点（O(1)）。要作为子节点挂载请用 [`Scene::attach`]。
    ///
    /// 灯光节点会登记进灯光缓存（[`Scene::lights`]），供渲染每帧按距离收集。
    pub fn add_object(&mut self, object: SceneObject) -> ObjectKey {
        let key = self.tree.new_node(object);
        if matches!(self.tree[key].get().kind, SceneObjectKind::Light(_)) {
            self.light_nodes.push(key);
        }
        key
    }

    /// 把新节点挂到 `parent` 下并返回句柄；父节点已失效时返回 `None`（O(1)）。
    ///
    /// 灯光节点同样登记进灯光缓存。
    pub fn attach(&mut self, parent: ObjectKey, object: SceneObject) -> Option<ObjectKey> {
        if parent.is_removed(&self.tree) {
            return None;
        }
        let child = self.tree.new_node(object);
        // 新节点无任何关系，append 不可能失败（不会自挂/挂祖先/已删除）。
        parent.append(child, &mut self.tree);
        if matches!(self.tree[child].get().kind, SceneObjectKind::Light(_)) {
            self.light_nodes.push(child);
        }
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
        // 缓存维护：整棵子树里的灯光节点一并移出（灯光可能挂在被删节点的子树上）。
        let subtree: Vec<ObjectKey> = key.descendants(&self.tree).collect();
        self.light_nodes.retain(|k| !subtree.contains(k));
        key.remove_subtree(&mut self.tree);
        Some(removed)
    }

    /// 把 `other` 中的所有物体复制进本场景（保持层级关系），返回**完整重映射**
    /// （旧句柄 → 新句柄），调用方据此复制节点级附加数据（如客户端的材质）。
    ///
    /// 用于把 glTF 等外部场景并入现有场景（例如并进演示场景）。
    pub fn merge(&mut self, other: &Scene) -> HashMap<ObjectKey, ObjectKey> {
        // 第一遍：复制所有节点，记录旧句柄 → 新句柄。
        let mut remap: HashMap<ObjectKey, ObjectKey> = HashMap::new();
        for (old_key, object) in other.objects() {
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
        remap
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
        Some(chain.iter().rev().fold(Mat4::IDENTITY, |world, transform| {
            world * transform.to_mat4()
        }))
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

    /// 场景中所有灯光节点（灯光缓存，增删时维护）。
    ///
    /// 读取时再做一次存活/类型过滤兜底，防止 `object_mut` 把节点改成其他类型后
    /// 留下脏缓存；缓存规模很小（灯光数），过滤代价可忽略。
    pub fn lights(&self) -> impl Iterator<Item = ObjectKey> + '_ {
        self.light_nodes.iter().copied().filter(|key| {
            self.tree.get(*key).is_some_and(|node| {
                !node.is_removed() && matches!(node.get().kind, SceneObjectKind::Light(_))
            })
        })
    }

    // ---- AABB 碰撞查询 ----
    //
    // 网格数据由资产管理器持有、场景不持有，因此查询需要调用方传入
    // [`MeshSource`]（core 接口，scene 不反向依赖 render）。世界 AABB = 网格局部 bounds
    // 经该节点世界矩阵变换后的包围盒（旋转会使其变大，属 AABB 的正常行为）。
    //
    // 现阶段查询为 O(物体数)：场景规模小足够；Level 0 区块落地时再上空间分区。

    /// 节点在世界空间中的 AABB；句柄失效、非网格节点或空包围盒时返回 `None`。
    pub fn object_aabb_world(&self, meshes: &dyn MeshSource, key: ObjectKey) -> Option<Aabb> {
        let handle = self.object(key)?.mesh_handle()?;
        let local = meshes.mesh(handle)?.bounds();
        if local.is_empty() {
            return None;
        }
        let world = self.world_transform(key)?;
        Some(local.transformed_by(&world))
    }

    /// 世界点是否落在 `key` 所指物体的世界 AABB 内（含边界）。
    pub fn point_inside(&self, meshes: &dyn MeshSource, key: ObjectKey, point: Vec3) -> bool {
        self.object_aabb_world(meshes, key)
            .is_some_and(|aabb| aabb.contains(point))
    }

    /// 两个已存在物体是否碰撞（世界 AABB 相交）。
    pub fn objects_collide(&self, meshes: &dyn MeshSource, a: ObjectKey, b: ObjectKey) -> bool {
        let (Some(aa), Some(bb)) = (
            self.object_aabb_world(meshes, a),
            self.object_aabb_world(meshes, b),
        ) else {
            return false;
        };
        aa.intersects(&bb)
    }

    /// 外部物体（尚未加入场景，如玩家）与场景的碰撞测试：给定物体在**局部
    /// 空间**的 AABB（相对自身原点，全尺寸 min/max）与摆放 [`Transform`]，
    /// 返回第一个碰撞到的场景物体句柄。
    ///
    /// `exclude` 用于跳过不需要参与测试的节点（如玩家脚下的地板、自身的
    /// 手持物）；传入空切片表示测试全部网格节点。
    pub fn collides_with(
        &self,
        meshes: &dyn MeshSource,
        transform: &Transform,
        local: Aabb,
        exclude: &[ObjectKey],
    ) -> Option<ObjectKey> {
        let probe = local.transformed(transform);
        self.objects()
            .filter(|(key, _)| !exclude.contains(key))
            .filter_map(|(key, _)| self.object_aabb_world(meshes, key).map(|aabb| (key, aabb)))
            .find(|(_, aabb)| aabb.intersects(&probe))
            .map(|(key, _)| key)
    }
}

#[cfg(test)]
mod tests;
