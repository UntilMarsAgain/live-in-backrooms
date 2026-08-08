//! 客户端同步层：消费服务端快照，组装/更新客户端视图场景。
//!
//! - 实体**结构不变**时（集合/类型/模型来源相同）只更新变换，零重建；
//! - 结构变化时整树重建，并返回 `true` 让调用方重新 `Renderer::load_scene`
//!   （材质绑定组等 GPU 静态部分跟着重建）；
//! - 模型模板（GamePath → 视图场景）按路径缓存，解析一次、材质/层级复用。

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use lbr_shared::core::transform::Transform;
use lbr_shared::protocol::{EntityKind, EntityState, WorldSnapshot};
use lbr_shared::scene::{ObjectKey, SceneObject, SceneObjectKind};
use lbr_shared::GamePath;

use crate::asset;
use crate::render::AssetManager;
use crate::scene::ClientScene;

/// 快照播放器：把服务端权威状态应用到本地视图场景。
pub struct SnapshotPlayer {
    /// 已解析的模型模板（GamePath → 视图场景；材质/层级一次建好）。
    templates: HashMap<GamePath, ClientScene>,
    /// 实体 ID → 客户端节点及其模板根局部变换（合成世界变换用）。
    nodes: HashMap<u32, Vec<(ObjectKey, Transform)>>,
    /// 实体集合签名（检测结构变化；不含 tick/变换）。
    signature: Option<u64>,
}

impl SnapshotPlayer {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            nodes: HashMap::new(),
            signature: None,
        }
    }

    /// 应用一帧快照到 `target`（保留相机/环境等渲染信息）。
    ///
    /// 返回 `true` 表示实体结构发生变化，调用方需要重新
    /// `Renderer::load_scene`（并 pin 资产）。
    pub fn apply(
        &mut self,
        assets: &mut AssetManager,
        snap: &WorldSnapshot,
        target: &mut ClientScene,
    ) -> anyhow::Result<bool> {
        let signature = snapshot_signature(snap);
        if self.signature != Some(signature) {
            // 结构变化：整树重建，重新建立实体 → 节点映射。
            target.clear_entities();
            self.nodes.clear();
            for entity in &snap.entities {
                self.spawn_entity(assets, target, entity)?;
            }
            self.signature = Some(signature);
            return Ok(true);
        }

        // 结构不变：只更新变换（实体 ID → 节点句柄）。
        for (id, nodes) in &self.nodes {
            if let Some(entity) = snap.entities.iter().find(|e| e.id == *id) {
                for (key, base) in nodes {
                    if let Some(object) = target.scene_mut().object_mut(*key) {
                        object.transform = compose(entity.transform, *base);
                    }
                }
            }
        }
        Ok(false)
    }

    /// 生成一个实体：网格从模板合并（解析一次并缓存），灯光直接建节点。
    fn spawn_entity(
        &mut self,
        assets: &mut AssetManager,
        target: &mut ClientScene,
        entity: &EntityState,
    ) -> anyhow::Result<()> {
        match &entity.kind {
            EntityKind::Mesh(path) => {
                if !self.templates.contains_key(path) {
                    let template = asset::load_scene(path, assets)?;
                    self.templates.insert(path.clone(), template);
                }
                let template = self.templates.get(path).expect("刚插入");
                // 模板根节点的局部变换记为 base：世界变换 = 实体变换 × base。
                let roots = target.merge(template);
                let mut nodes = Vec::with_capacity(roots.len());
                for root in roots {
                    let base = target
                        .scene()
                        .object(root)
                        .map(|o| o.transform)
                        .unwrap_or(Transform::IDENTITY);
                    if let Some(object) = target.scene_mut().object_mut(root) {
                        object.transform = compose(entity.transform, base);
                    }
                    nodes.push((root, base));
                }
                self.nodes.insert(entity.id, nodes);
            }
            EntityKind::Light(light) => {
                let key = target.scene_mut().add_object(SceneObject::new(
                    SceneObjectKind::Light(*light),
                    entity.transform,
                ));
                self.nodes.insert(entity.id, vec![(key, Transform::IDENTITY)]);
            }
        }
        Ok(())
    }
}

impl Default for SnapshotPlayer {
    fn default() -> Self {
        Self::new()
    }
}

/// 实体集合签名：只看"有哪些实体、什么类型、模型来源"，不看 tick/变换。
fn snapshot_signature(snap: &WorldSnapshot) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for entity in &snap.entities {
        entity.id.hash(&mut hasher);
        match &entity.kind {
            EntityKind::Mesh(path) => {
                1u8.hash(&mut hasher);
                path.hash(&mut hasher);
            }
            EntityKind::Light(_) => {
                2u8.hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// 世界变换合成：`outer`（实体摆放）× `inner`（模板根局部变换）。
fn compose(outer: Transform, inner: Transform) -> Transform {
    let (scale, rotation, translation) = (outer.to_mat4() * inner.to_mat4())
        .to_scale_rotation_translation();
    Transform::new(translation, rotation, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lbr_shared::core::resource::MergedResourceSpace;

    /// 真实 test.glb 走一遍"快照 → 视图场景"：首次应用重建（true），
    /// 结构不变的第二次应用只更新变换（false）且旋转生效。
    #[test]
    fn apply_builds_then_updates_transforms() {
        let space = MergedResourceSpace::new("game-data/vanilla/".into());
        let path: GamePath = "test:test.glb".parse().expect("合法路径");
        if !space.exists(&path) {
            eprintln!("跳过：test/test.glb 未准备（测试数据不入库）");
            return;
        }

        let mut assets = AssetManager::without_gpu(space);
        let mut player = SnapshotPlayer::new();
        let mut scene = ClientScene::new();

        let mut snap = WorldSnapshot {
            tick: 1,
            entities: vec![EntityState {
                id: 7,
                kind: EntityKind::Mesh(path),
                transform: Transform::new(
                    glam::Vec3::new(1.8, 0.0, -1.2),
                    glam::Quat::IDENTITY,
                    glam::Vec3::splat(5.0),
                ),
            }],
        };

        let rebuilt = player.apply(&mut assets, &snap, &mut scene).expect("首次应用");
        assert!(rebuilt, "首次快照应重建结构");
        let yaw0 = scene
            .scene()
            .roots()
            .next()
            .map(|(k, _)| scene.scene().object(k).unwrap().transform.rotation)
            .expect("应有根节点");

        // 同一个实体集合（tick/旋转变化）→ 不重建，只更新变换。
        snap.tick = 2;
        if let EntityKind::Mesh(_) = &mut snap.entities[0].kind {
            snap.entities[0].transform.rotation *=
                glam::Quat::from_rotation_y(0.1);
        }
        let rebuilt = player.apply(&mut assets, &snap, &mut scene).expect("再次应用");
        assert!(!rebuilt, "结构不变不应重建");
        let yaw1 = scene
            .scene()
            .roots()
            .next()
            .map(|(k, _)| scene.scene().object(k).unwrap().transform.rotation)
            .expect("应有根节点");
        assert_ne!(yaw0, yaw1, "更新路径应应用新的旋转");
    }
}
