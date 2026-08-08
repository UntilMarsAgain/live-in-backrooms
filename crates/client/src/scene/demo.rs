//! 演示视图场景构建：内置三角形/四边形/立方体示例，验证层级变换与材质。

use glam::{Quat, Vec3};

use super::ClientScene;
use lbr_shared::core::asset::Handle;
use lbr_shared::core::light::Light;
use lbr_shared::core::material::Material;
use lbr_shared::core::mesh::Mesh;
use lbr_shared::core::texture::Texture;
use lbr_shared::core::transform::Transform;
use lbr_shared::scene::{SceneObject, SceneObjectKind};

impl ClientScene {
    pub fn demo(
        triangle: Handle<Mesh>,
        quad: Handle<Mesh>,
        cube: Handle<Mesh>,
        cube_texture: Option<Handle<Texture>>,
    ) -> Self {
        let mut out = Self::new();
        let scene = out.scene_mut();
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
            ),
        );
        // 小三角形挂在立方体正上方，跟随立方体一起旋转（验证层级）。
        let _ = scene.attach(
            cube,
            SceneObject::new(
                SceneObjectKind::Mesh(triangle),
                Transform::new(Vec3::new(0.0, 1.2, 0.0), Quat::IDENTITY, Vec3::splat(0.4)),
            ),
        );
        // 材质属于渲染信息，由客户端视图场景持有（共享树不携带）。
        out.set_material(
            cube,
            Material {
                base_color: [1.0; 4],
                base_color_texture: cube_texture,
                ..Material::default()
            },
        );
        out
    }
}
