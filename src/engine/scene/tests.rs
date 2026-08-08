//! 场景模块测试：层级变换、合并、灯光缓存与主相机生命周期。

use super::*;
use crate::engine::Mesh;
use crate::engine::AssetManager;
use crate::engine::MergedResourceSpace;
use glam::{Quat, Vec3};

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

/// 主相机操作：平移与旋转按操作应用；无主相机时返回 `false`。
#[test]
fn apply_main_camera_action_moves_and_rotates() {
    let mut scene = Scene::new();
    let cam = scene.add_camera(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0));
    assert!(scene.set_main_camera(cam));

    let action = CameraAction {
        translate: Vec3::new(1.0, 2.0, 3.0),
        yaw_delta: 0.5,
        pitch_delta: -0.25,
    };
    assert!(scene.apply_main_camera_action(action));

    let camera = scene.main_camera_ref().expect("主相机应存在");
    assert_eq!(camera.position(), Vec3::new(1.0, 2.0, 3.0));
    // forward = (cos yaw·cos pitch, sin pitch, sin yaw·cos pitch)。
    let f = camera.forward();
    assert!((f.x - 0.5_f32.cos() * 0.25_f32.cos()).abs() < 1e-6);
    assert!((f.y + 0.25_f32.sin()).abs() < 1e-6, "俯仰向下：{f:?}");
    assert!((f.z - 0.5_f32.sin() * 0.25_f32.cos()).abs() < 1e-6);

    // 没有主相机时应用失败，返回 false。
    let mut empty = Scene::new();
    assert!(!empty.apply_main_camera_action(CameraAction::default()));
}

/// 场景碰撞查询的公共脚手架：一个边长 1 的立方体资产 + 放在原点的实例。
fn cube_world(assets: &mut AssetManager, scene: &mut Scene, position: Vec3) -> ObjectKey {
    let key = assets.meshes_mut().register(Mesh::cube());
    scene.add_object(SceneObject::new(
        SceneObjectKind::Mesh(key),
        Transform::new(position, Quat::IDENTITY, Vec3::ONE),
    ))
}

/// 点包含：立方体中心在盒内，远处点在外，边界算在内。
#[test]
fn point_inside_uses_world_aabb() {
    let mut assets = AssetManager::without_gpu(MergedResourceSpace::new(std::env::temp_dir()));
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
    let mut assets = AssetManager::without_gpu(MergedResourceSpace::new(std::env::temp_dir()));
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
    let mut assets = AssetManager::without_gpu(MergedResourceSpace::new(std::env::temp_dir()));
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
