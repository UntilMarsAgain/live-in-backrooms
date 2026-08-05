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
//! [`SceneObjectKind`] 区分类型（网格 / 空分组节点 / 灯光 / 相机）。
//! 关卡环境（天空盒 + IBL）跟随场景：加载场景时由 App 层一并上传，
//! 模组作者按"一个关卡 = 场景 + 环境"来组织资产。
//! 切换场景由 App 层 API 触发。

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use indextree::{Arena, NodeId};

use super::core::camera::Camera;
use super::core::environment::Environment;
use super::core::light::Light;
use super::core::material::Material;
use super::core::mesh::MeshKey;
use super::core::texture::TextureKey;
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
    /// 引用全局资产库里的网格。
    Mesh(MeshKey),
    /// 灯光：位置/朝向由节点变换决定（见 [`Light`] 的约定）。
    Light(Light),
    /// 相机：位置与朝向由 `Camera` 内部状态决定，节点 Transform 暂不参与合成
    /// （保留给将来"相机挂在角色/载具下"的层级用法）。
    Camera(Camera),
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
            SceneObjectKind::Empty | SceneObjectKind::Light(_) | SceneObjectKind::Camera(_) => None,
        }
    }
}

/// 场景：层级化的物体树（网格资产在全局 `MeshLibrary` 中）。
#[derive(Debug, Clone)]
pub struct Scene {
    tree: Arena<SceneObject>,
    /// 灯光节点缓存（增删时维护，渲染每帧按距离收集）。
    light_nodes: Vec<ObjectKey>,
    /// 主相机节点（场景的"出生点视角"）。`None` = 未指定，App 层加载时会补默认相机。
    main_camera: Option<ObjectKey>,
    /// 关卡环境（天空盒 + IBL）。`None` = 纯手动布光 / 保持默认黑环境。
    environment: Option<Arc<Environment>>,
    /// 环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    environment_intensity: f32,
    /// AgX 色调映射的 EV 窗口（相对中间灰 0.18 的 EV 档位），默认与 Blender 一致。
    agx_min_ev: f32,
    agx_max_ev: f32,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            tree: Arena::default(),
            light_nodes: Vec::new(),
            main_camera: None,
            environment: None,
            // 默认满环境光：不显式设置强度时保持"环境图参与光照"的既有行为。
            environment_intensity: 1.0,
            // 默认 EV 窗口 = Blender AgX（中间灰上下 -10 ~ +6.5 EV）。
            agx_min_ev: crate::engine::render::uniform::AGX_DEFAULT_EV_MIN,
            agx_max_ev: crate::engine::render::uniform::AGX_DEFAULT_EV_MAX,
        }
    }
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// 给场景绑定环境（天空盒 + IBL）；`load_scene` 时自动上传。
    ///
    /// `Arc` 避免环境像素（MB 级）在场景间复制；多个关卡可共享同一环境。
    pub fn with_environment(mut self, environment: Arc<Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// 场景绑定的环境（无则 `None`）。
    pub fn environment(&self) -> Option<&Environment> {
        self.environment.as_deref()
    }

    /// 设置环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    #[allow(dead_code)] // 公共配置 API：关卡数据构建场景时使用
    pub fn with_environment_intensity(mut self, intensity: f32) -> Self {
        self.environment_intensity = intensity;
        self
    }

    /// 场景的环境强度（默认 1.0）。
    pub fn environment_intensity(&self) -> f32 {
        self.environment_intensity
    }

    /// 覆盖 AgX 色调映射的 EV 窗口（场景级风格配置）。
    ///
    /// 参数是**相对中间灰 0.18 的 EV 档位**：默认 -10 ~ +6.5 EV（Blender 一致），
    /// 一般无需修改；想让某个层级更亮/更暗或动态范围更宽/更窄时再调。
    /// 窗口越宽曲线越平缓（整体偏灰），越窄对比越强；要求 `min_ev < max_ev`。
    #[allow(dead_code)] // 公共配置 API：关卡数据构建场景时使用
    pub fn with_environment_agx_ev(mut self, min_ev: f32, max_ev: f32) -> Self {
        debug_assert!(min_ev < max_ev, "AgX EV 窗口要求 min_ev < max_ev");
        self.agx_min_ev = min_ev;
        self.agx_max_ev = max_ev;
        self
    }

    /// 场景的 AgX EV 窗口下界（默认 -10，相对中间灰的 EV）。
    pub fn agx_min_ev(&self) -> f32 {
        self.agx_min_ev
    }

    /// 场景的 AgX EV 窗口上界（默认 +6.5，相对中间灰的 EV）。
    pub fn agx_max_ev(&self) -> f32 {
        self.agx_max_ev
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

    /// 添加一个相机节点（**不会**自动设为主相机），返回节点句柄。
    ///
    /// 要把它设为主相机请再调用 [`Scene::set_main_camera`]；"添加"与"设为主相机"
    /// 是两个独立操作，避免添加相机时意外抢走主相机身份。
    pub fn add_camera(&mut self, camera: Camera) -> ObjectKey {
        self.add_object(SceneObject::new(
            SceneObjectKind::Camera(camera),
            Transform::IDENTITY,
        ))
    }

    /// 把指定节点设为主相机；必须是存活的 `Camera` 节点，否则返回 `false` 且不改变现状。
    #[allow(dead_code)] // 公共配置 API：场景构建/游戏逻辑切换主相机时使用
    pub fn set_main_camera(&mut self, key: ObjectKey) -> bool {
        if !self
            .object(key)
            .is_some_and(|obj| matches!(obj.kind, SceneObjectKind::Camera(_)))
        {
            return false;
        }
        self.main_camera = Some(key);
        true
    }

    /// 主相机节点句柄；未指定时返回 `None`。
    ///
    /// 主相机引用是**不变量**：一旦是 `Some`，必须指向一个存活的相机节点。
    /// 指向已删除的节点或非相机节点属于运行违例，这里直接 panic，
    /// 而不是静默返回 `None` 让上层无感知地跳过渲染。
    pub fn main_camera(&self) -> Option<ObjectKey> {
        let key = self.main_camera?;
        let obj = self.object(key).unwrap_or_else(|| {
            panic!("场景主相机运行违例：main_camera 指向的节点已删除（key = {key:?}）")
        });
        assert!(
            matches!(obj.kind, SceneObjectKind::Camera(_)),
            "场景主相机运行违例：main_camera 指向的节点不是相机（key = {key:?}）"
        );
        Some(key)
    }

    /// 主相机的只读引用。
    pub fn main_camera_ref(&self) -> Option<&Camera> {
        let key = self.main_camera()?;
        match &self.object(key)?.kind {
            SceneObjectKind::Camera(camera) => Some(camera),
            _ => None,
        }
    }

    /// 主相机的可变引用（输入控制、窗口尺寸变化等）。
    pub fn main_camera_mut(&mut self) -> Option<&mut Camera> {
        let key = self.main_camera()?;
        match &mut self.object_mut(key)?.kind {
            SceneObjectKind::Camera(camera) => Some(camera),
            _ => None,
        }
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
        // 主相机随子树一起失效：被删子树里含主相机节点时清空引用。
        if let Some(main) = self.main_camera {
            if subtree.contains(&main) {
                self.main_camera = None;
            }
        }
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
        // 主相机随场景一起并入（外部场景自带主相机时保留，句柄重映射到本场景）。
        if let Some(old_main) = other.main_camera() {
            if let Some(new_key) = remap.get(&old_main) {
                self.main_camera = Some(*new_key);
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
        // 约定：物体局部 -Z = 光行进方向（光源 → 场景，与面光一致）；
        // 光从右上前方照来，所以行进方向 = 光的来向取反。
        let light_arrival = Vec3::new(0.5, 0.6, 0.6).normalize();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::directional(Vec3::ONE, 0.7)),
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_arc(Vec3::NEG_Z, -light_arrival),
                Vec3::ONE,
            ),
        ));
        // 点光（辅助灯）：暖色，放在演示物体右上方。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::new(1.0, 0.85, 0.6), 18.0)),
            Transform::new(Vec3::new(2.2, 1.8, 0.8), Quat::IDENTITY, Vec3::ONE),
        ));
        // 面光（荧光灯面板）：朝下照亮扳手区域。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::area(1.5, 0.6, Vec3::new(0.9, 0.95, 1.0), 20.0)),
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
        let cube = scene.add_object(
            SceneObject::new(
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
            }),
        );
        // 小三角形挂在立方体正上方，跟随立方体一起旋转（验证层级）。
        let _ = scene.attach(
            cube,
            SceneObject::new(
                SceneObjectKind::Mesh(triangle),
                Transform::new(Vec3::new(0.0, 1.2, 0.0), Quat::IDENTITY, Vec3::splat(0.4)),
            ),
        );
        // 主相机（出生点视角）：与 App 缺省相机一致的位置/朝向，宽高比按默认窗口 16:9。
        let camera = scene.add_camera(Camera::new(
            Vec3::new(0.0, 1.0, 3.0),
            -std::f32::consts::FRAC_PI_2,
            -0.15,
            std::f32::consts::FRAC_PI_4,
            16.0 / 9.0,
            0.1,
            100.0,
        ));
        assert!(
            scene.set_main_camera(camera),
            "新添加的相机节点必然能设为主相机"
        );
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 环境强度默认应为 1.0（满环境光），避免手误改成 0 后物体失去环境光。
    #[test]
    fn default_environment_intensity_is_full() {
        assert_eq!(Scene::new().environment_intensity(), 1.0);
    }

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
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                Vec3::ONE,
            ),
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

    /// 灯光缓存：add/attach 登记，删除整棵子树时一并移出。
    #[test]
    fn light_cache_tracks_add_and_remove() {
        let mut scene = Scene::new();
        let point = scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::ONE, 1.0)),
            Transform::IDENTITY,
        ));
        let parent = scene.add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::IDENTITY,
        ));
        let child_light = scene
            .attach(
                parent,
                SceneObject::new(
                    SceneObjectKind::Light(Light::directional(Vec3::ONE, 1.0)),
                    Transform::IDENTITY,
                ),
            )
            .expect("父节点存活");

        let keys: Vec<_> = scene.lights().collect();
        assert_eq!(keys, vec![point, child_light]);

        // 删除父节点：子灯光随子树一起移出缓存。
        scene.remove_object(parent);
        let keys: Vec<_> = scene.lights().collect();
        assert_eq!(keys, vec![point]);

        scene.remove_object(point);
        assert_eq!(scene.lights().count(), 0);
    }

    /// 主相机：add_camera 只添加不设为主相机，set_main_camera 显式切换并校验类型。
    #[test]
    fn main_camera_lifecycle() {
        let mut scene = Scene::new();
        assert!(scene.main_camera().is_none());

        let cam = scene.add_camera(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0));
        // 添加相机 ≠ 设为主相机。
        assert!(scene.main_camera().is_none());
        assert!(scene.set_main_camera(cam));
        assert_eq!(scene.main_camera(), Some(cam));
        assert!(scene.main_camera_ref().is_some());

        // 非相机节点不能设为主相机。
        let empty = scene.add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::IDENTITY,
        ));
        assert!(!scene.set_main_camera(empty));
        assert_eq!(scene.main_camera(), Some(cam));

        // 删除相机节点后引用清空。
        scene.remove_object(cam);
        assert!(scene.main_camera().is_none());
    }

    /// merge 保留外部场景的主相机（句柄重映射到本场景）。
    #[test]
    fn merge_preserves_main_camera() {
        let mut a = Scene::new();
        let cam = a.add_camera(Camera::new(
            Vec3::new(1.0, 2.0, 3.0),
            0.0,
            0.0,
            1.0,
            1.0,
            0.1,
            100.0,
        ));
        assert!(a.set_main_camera(cam));
        assert_eq!(a.main_camera(), Some(cam));

        let mut b = Scene::new();
        let merged = b.merge(&a);
        assert_eq!(merged.len(), 1);
        assert_eq!(b.main_camera(), Some(merged[0]));
        assert_eq!(
            b.main_camera_ref().unwrap().position(),
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    /// 主相机指向非相机节点是运行违例：访问时直接 panic，而不是静默返回 `None`。
    #[test]
    #[should_panic(expected = "不是相机")]
    fn main_camera_invalid_kind_panics() {
        let mut scene = Scene::new();
        let cam = scene.add_camera(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0));
        assert!(scene.set_main_camera(cam));
        // 绕过 set_main_camera 的校验把相机节点改成其他类型（模拟脏状态）。
        scene.object_mut(cam).unwrap().kind = SceneObjectKind::Empty;
        let _ = scene.main_camera();
    }
}
