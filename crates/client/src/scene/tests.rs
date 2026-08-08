//! 客户端视图场景测试：渲染信息（相机/环境/材质）的生命周期。

use super::*;
use lbr_shared::core::material::Material;
use lbr_shared::scene::{SceneObject, SceneObjectKind};
use lbr_shared::core::transform::Transform;
use glam::{Quat, Vec3};

/// 新场景自带默认主相机（出生点视角）。
#[test]
fn new_scene_has_default_camera() {
    let scene = ClientScene::new();
    assert_eq!(scene.camera().position(), Vec3::new(0.0, 1.0, 3.0));
}

/// 相机操作：平移与旋转按操作应用（先旋转再平移）。
#[test]
fn apply_camera_action_moves_and_rotates() {
    let mut scene = ClientScene::new();
    // 摆到已知姿态：原点、水平向前（避免依赖默认出生点相机）。
    scene.set_camera(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0));
    let action = CameraAction {
        translate: Vec3::new(1.0, 2.0, 3.0),
        yaw_delta: 0.5,
        pitch_delta: -0.25,
    };
    scene.apply_camera_action(action);

    assert_eq!(scene.camera().position(), Vec3::new(1.0, 2.0, 3.0));
    // forward = (cos yaw·cos pitch, sin pitch, sin yaw·cos pitch)。
    let f = scene.camera().forward();
    assert!((f.x - 0.5_f32.cos() * 0.25_f32.cos()).abs() < 1e-6);
    assert!((f.y + 0.25_f32.sin()).abs() < 1e-6, "俯仰向下：{f:?}");
    assert!((f.z - 0.5_f32.sin() * 0.25_f32.cos()).abs() < 1e-6);
}

/// 环境强度默认应为 1.0（满环境光），避免手误改成 0 后物体失去环境光。
#[test]
fn default_environment_intensity_is_full() {
    assert_eq!(ClientScene::new().environment_intensity(), 1.0);
}

/// 材质按节点句柄存取（共享树不携带，客户端视图场景持有）。
#[test]
fn material_is_stored_per_node() {
    let mut scene = ClientScene::new();
    let key = scene.scene_mut().add_object(SceneObject::new(
        SceneObjectKind::Empty,
        Transform::IDENTITY,
    ));
    assert!(scene.material(key).is_none());

    scene.set_material(
        key,
        Material {
            base_color: [0.2, 0.3, 0.4, 1.0],
            ..Material::default()
        },
    );
    assert_eq!(scene.material(key).unwrap().base_color, [0.2, 0.3, 0.4, 1.0]);
}

/// merge 保留外部视图场景的材质（句柄重映射后复制）。
#[test]
fn merge_preserves_materials() {
    let mut a = ClientScene::new();
    let key = a.scene_mut().add_object(SceneObject::new(
        SceneObjectKind::Empty,
        Transform::IDENTITY,
    ));
    a.set_material(
        key,
        Material {
            base_color: [0.2, 0.3, 0.4, 1.0],
            ..Material::default()
        },
    );

    let mut b = ClientScene::new();
    let merged_roots = b.merge(&a);
    assert_eq!(merged_roots.len(), 1);
    let merged = merged_roots[0];
    assert_eq!(b.material(merged).unwrap().base_color, [0.2, 0.3, 0.4, 1.0]);

    // 未设置材质的节点合并后保持无材质。
    let plain = b
        .scene_mut()
        .add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
        ));
    assert!(b.material(plain).is_none());
}
