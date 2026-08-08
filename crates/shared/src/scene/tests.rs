//! 场景模块测试（服务端视图）：层级变换、合并、灯光缓存与碰撞查询。

use super::*;
use crate::Mesh;
use crate::core::asset::{AssetRegistry, DataSource, Handle, MeshSource};
use glam::{Quat, Vec3};

/// 本地网格来源：用共享层的注册表替代客户端 AssetManager，
/// 保持共享层测试不依赖渲染（服务端同样只需要这类 CPU 网格数据）。
#[derive(Default)]
struct MeshSourceRegistry(AssetRegistry<Mesh, ()>);

impl MeshSource for MeshSourceRegistry {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh> {
        match self.0.data_source(handle)? {
            DataSource::Inline(mesh) => Some(mesh),
            DataSource::File { .. } => None,
        }
    }
}

/// 场景碰撞查询的公共脚手架：一个边长 1 的立方体资产 + 摆在 `position` 的实例。
fn cube_world(assets: &mut MeshSourceRegistry, scene: &mut Scene, position: Vec3) -> ObjectKey {
    let key = assets.0.register(Mesh::cube());
    scene.add_object(SceneObject::new(
        SceneObjectKind::Mesh(key),
        Transform::new(position, Quat::IDENTITY, Vec3::ONE),
    ))
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

/// merge 返回完整重映射（旧句柄 → 新句柄），层级与世界变换保留。
/// 客户端用重映射复制节点级附加数据（如材质），服务端视图本身不携带。
#[test]
fn merge_reparents_and_returns_full_remap() {
    let mut a = Scene::new();
    let root = a.add_object(SceneObject::new(
        SceneObjectKind::Empty,
        Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
    ));
    let child = a
        .attach(
            root,
            SceneObject::new(
                SceneObjectKind::Empty,
                Transform::new(Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            ),
        )
        .expect("父节点存活");

    let mut b = Scene::new();
    let remap = b.merge(&a);
    assert_eq!(remap.len(), 2);
    let new_root = remap[&root];
    let new_child = remap[&child];
    assert_eq!(b.children_of(new_root).collect::<Vec<_>>(), vec![new_child]);
    assert_eq!(
        b.world_transform(new_child).unwrap().transform_point3(Vec3::ZERO),
        Vec3::new(1.0, 2.0, 0.0)
    );
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

/// 点包含：立方体中心在盒内，远处点在外，边界算在内。
#[test]
fn point_inside_uses_world_aabb() {
    let mut assets = MeshSourceRegistry::default();
    let mut scene = Scene::new();
    let cube = cube_world(&mut assets, &mut scene, Vec3::new(2.0, 0.0, 0.0));

    assert!(scene.point_inside(&assets, cube, Vec3::new(2.0, 0.0, 0.0)));
    assert!(scene.point_inside(&assets, cube, Vec3::new(2.5, 0.5, 0.5))); // 边界上
    assert!(!scene.point_inside(&assets, cube, Vec3::new(3.0, 0.0, 0.0)));
    // 非网格节点没有包围盒，任何点都不在内。
    let empty = scene.add_object(SceneObject::new(
        SceneObjectKind::Empty,
        Transform::IDENTITY,
    ));
    assert!(!scene.point_inside(&assets, empty, Vec3::ZERO));
}

/// 两物体碰撞：中心距 < 边长和的一半时相交，> 时不相交。
#[test]
fn objects_collide_uses_world_aabb() {
    let mut assets = MeshSourceRegistry::default();
    let mut scene = Scene::new();
    let a = cube_world(&mut assets, &mut scene, Vec3::ZERO);
    let b = cube_world(&mut assets, &mut scene, Vec3::new(0.5, 0.0, 0.0));
    let c = cube_world(&mut assets, &mut scene, Vec3::new(2.0, 0.0, 0.0));

    assert!(
        scene.objects_collide(&assets, a, b),
        "中心距 0.5 < 1 应相交"
    );
    assert!(!scene.objects_collide(&assets, a, c), "中心距 2 > 1 应分离");
    // 已删除的句柄不参与碰撞。
    scene.remove_object(c);
    assert!(!scene.objects_collide(&assets, a, c));
}

/// 外部物体：给定 transform + half_extents 与世界碰撞，支持旋转与排除。
#[test]
fn collides_with_external_probe() {
    let mut assets = MeshSourceRegistry::default();
    let mut scene = Scene::new();
    let cube = cube_world(&mut assets, &mut scene, Vec3::ZERO);

    // 玩家盒子局部 AABB（以自身原点为中心，半尺寸 (0.3, 0.9, 0.3)）：
    // 中心距 0.6 时与边长 1 的立方体相交。
    let player = Transform::new(Vec3::new(0.6, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
    let player_box = Aabb::from_half_extents(Vec3::ZERO, Vec3::new(0.3, 0.9, 0.3));
    assert_eq!(
        scene.collides_with(&assets, &player, player_box, &[]),
        Some(cube)
    );

    // 远离时无碰撞。
    let away = Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
    assert_eq!(scene.collides_with(&assets, &away, player_box, &[]), None);

    // 旋转 45° 后包围盒变大：probe 的 x 半宽从 0.3 增到 0.3√2 ≈ 0.424，
    // 中心距 0.9 < 0.5 + 0.424 所以相交；不旋转时 0.9 会分离（0.6 > 0.5）。
    let rotated = Transform::new(
        Vec3::new(0.9, 0.0, 0.0),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        Vec3::ONE,
    );
    assert_eq!(
        scene.collides_with(&assets, &rotated, player_box, &[]),
        Some(cube)
    );

    // 排除后跳过该物体。
    assert_eq!(
        scene.collides_with(&assets, &player, player_box, &[cube]),
        None
    );
}
