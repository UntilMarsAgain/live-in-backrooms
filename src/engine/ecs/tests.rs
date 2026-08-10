//! ECS 实验测试：固定步长、层级传播、相机碰撞、场景生成。

use std::collections::HashSet;
use std::time::Duration;

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use winit::keyboard::KeyCode;

use super::camera::free_camera;
use super::components::{
    CameraC, Children, Collider, LocalTransform, MainCamera, MeshHandle, Parent, WorldMatrix,
};
use super::hierarchy::propagate_world_transforms;
use super::scene::spawn_scene;
use super::{FixedStep, FixedTimestep, InputSnapshot};
use crate::engine::core::camera::Camera;
use crate::engine::core::data::aabb::Aabb;
use crate::engine::core::data::transform::Transform;
use crate::engine::scene::{Scene, SceneObject, SceneObjectKind};
use crate::engine::AssetManager;
use crate::engine::MergedResourceSpace;
use crate::engine::MeshView;

#[test]
fn fixed_timestep_accumulates_ticks() {
    let step = Duration::from_secs_f64(1.0 / 60.0);
    let mut timestep = FixedTimestep::new(step);
    // 2.5 步 → 2 个 tick，alpha ≈ 0.5
    let (ticks, alpha) = timestep.advance(step.mul_f64(2.5));
    assert_eq!(ticks, 2);
    assert!((alpha - 0.5).abs() < 1e-3);
    // 不足一步 → 0 tick
    let (ticks, _) = timestep.advance(step.mul_f64(0.2));
    assert_eq!(ticks, 0);
}

/// 层级传播：子实体世界矩阵 = 父世界矩阵 × 局部矩阵。
#[test]
fn hierarchy_propagates_parent_chain() {
    let mut world = World::new();
    let parent = world
        .spawn((
            LocalTransform(Transform::new(
                Vec3::new(1.0, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            )),
            WorldMatrix(Transform::IDENTITY.to_mat4()),
        ))
        .id();
    let child = world
        .spawn((
            LocalTransform(Transform::new(
                Vec3::new(0.0, 1.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            )),
            WorldMatrix(Transform::IDENTITY.to_mat4()),
        ))
        .id();
    world.entity_mut(parent).insert(Children(vec![child]));
    world.entity_mut(child).insert(Parent(parent));

    let mut schedule = Schedule::default();
    schedule.add_systems(propagate_world_transforms);
    schedule.run(&mut world);

    let child_world = world.get::<WorldMatrix>(child).unwrap().0;
    let (_, _, translation) = child_world.to_scale_rotation_translation();
    assert_eq!(translation, Vec3::new(1.0, 1.0, 0.0));
}

/// 相机：输入 + 固定步长 → 移动；撞墙分轴滑动被挡。
#[test]
fn camera_moves_and_slides_along_walls() {
    let mut world = World::new();
    // 大步长（0.1s × speed 5 = 0.5/步），保证一步就越过"墙距"（0.2）。
    world.insert_resource(FixedStep(Duration::from_secs_f64(0.1)));
    world.insert_resource(InputSnapshot {
        keys: HashSet::from([KeyCode::KeyW]),
        look_delta: (0.0, 0.0),
        scroll_delta: 0.0,
    });
    let camera = world
        .spawn((
            CameraC(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0)),
            LocalTransform(Transform::IDENTITY),
            WorldMatrix(Transform::IDENTITY.to_mat4()),
            MainCamera,
        ))
        .id();
    // 障碍：中心 (1,0,0)、半尺寸 0.5 的立方体。
    world.spawn((
        WorldMatrix(Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE).to_mat4()),
        Collider(Aabb::from_half_extents(Vec3::ZERO, Vec3::splat(0.5))),
    ));

    let mut schedule = Schedule::default();
    schedule.add_systems(free_camera);
    schedule.run(&mut world);

    let cam = world.get::<CameraC>(camera).unwrap().0;
    // W = +X，speed 5 × dt 1/60 ≈ 0.083；墙距 1 < 0.5 + 0.3，应被挡在原地。
    assert_eq!(cam.position(), Vec3::ZERO, "撞墙应被挡在原地");
}

/// 无障碍时正常移动：W 走 speed × dt。
#[test]
fn camera_moves_free_when_no_obstacle() {
    let mut world = World::new();
    world.insert_resource(FixedStep(Duration::from_secs_f64(1.0 / 60.0)));
    world.insert_resource(InputSnapshot {
        keys: HashSet::from([KeyCode::KeyW]),
        look_delta: (0.0, 0.0),
        scroll_delta: 0.0,
    });
    let camera = world
        .spawn((
            CameraC(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0)),
            LocalTransform(Transform::IDENTITY),
            WorldMatrix(Transform::IDENTITY.to_mat4()),
            MainCamera,
        ))
        .id();

    let mut schedule = Schedule::default();
    schedule.add_systems(free_camera);
    schedule.run(&mut world);

    let cam = world.get::<CameraC>(camera).unwrap().0;
    assert!((cam.position().x - 5.0 / 60.0).abs() < 1e-6);
}

/// 场景模板生成：层级链接、网格碰撞盒、主相机标记。
#[test]
fn spawn_scene_creates_hierarchy_and_main_camera() {
    let mut assets = AssetManager::new(MergedResourceSpace::new(std::env::temp_dir()));
    let cube = assets.register(crate::engine::Mesh::cube());
    let mut template = Scene::new();
    let root = template.add_object(SceneObject::new(
        SceneObjectKind::Empty,
        Transform::IDENTITY,
    ));
    let mesh = template.attach(
        root,
        SceneObject::new(
            SceneObjectKind::Mesh(cube),
            Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
        ),
    );
    let cam = template.add_camera(Camera::new(
        Vec3::new(0.0, 1.0, 3.0),
        0.0,
        0.0,
        1.0,
        1.0,
        0.1,
        100.0,
    ));
    template.set_main_camera(cam);

    let mut world = World::new();
    let main = spawn_scene(&template, &mut world, &MeshView::new(&assets));
    assert!(main.is_some(), "模板指定了主相机");
    assert!(world.get::<MainCamera>(main.unwrap()).is_some());
    assert_eq!(world.query::<&MeshHandle>().iter(&world).count(), 1);
    assert_eq!(world.query::<&Collider>().iter(&world).count(), 1);
    // 层级：mesh 是 root 的子实体。
    let root_entity = world
        .query::<(&Children, &LocalTransform)>()
        .iter(&world)
        .find(|(_, t)| t.0 == Transform::IDENTITY)
        .map(|(children, _)| children.0[0])
        .expect("根节点有子实体");
    assert!(world.get::<Parent>(root_entity).is_some());
    let _ = mesh;
}
