//! 渲染器使用的 GPU uniform 布局与灯光收集。
//!
//! 与 WGSL（mesh.wgsl / environment.wgsl）的内存布局一一对应，Rust 侧只负责
//! 布局与数据填充，缓冲/绑定组的创建在 `super::Renderer` 与 `super::environment`。
//!
//! 灯光是**动态数据**：每帧从场景灯光缓存收集"所有方向光 + 离相机最近的
//! [`MAX_NEARBY_LIGHTS`] 盏局部光"，写进只读 storage 数组（数量走独立 uniform）。
//! 玩家手电筒等动态光以后并入同一数组即可，无需改管线。

use glam::{Mat4, Vec3};

use crate::engine::core::light::LightKind;
use crate::engine::scene::{Scene, SceneObjectKind};

/// 每帧参与着色的局部光（点光/面光）数量上限：离相机最近的 X 盏。
/// 方向光不占此额度，总是全部参与。
pub(super) const MAX_NEARBY_LIGHTS: usize = 8;

/// 灯光 storage 缓冲容量（盏）。必须 ≥ 方向光数 + [`MAX_NEARBY_LIGHTS`]；
/// 预分配固定容量，每帧只写实际数量，避免运行时重建缓冲。
pub(super) const LIGHT_CAPACITY: usize = 64;

/// 从场景灯光缓存收集每帧灯光：
/// 所有方向光（按场景顺序）+ 离 `camera_position` 最近的
/// [`MAX_NEARBY_LIGHTS`] 盏局部光（欧氏距离，近 → 远）。
///
/// 世界方向由物体的世界旋转决定（局部 -Z = 光行进方向）。
pub(super) fn collect_lights(scene: &Scene, camera_position: Vec3) -> Vec<LightUniform> {
    let mut directional = Vec::new();
    let mut nearby: Vec<(f32, LightUniform)> = Vec::new();

    for key in scene.lights() {
        let object = scene
            .object(key)
            .expect("lights() 只产出存活灯光节点");
        let SceneObjectKind::Light(light) = object.kind else {
            continue;
        };
        let world = scene
            .world_transform(key)
            .expect("objects() 只产出存活节点");
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        if matches!(light.kind, LightKind::Directional) {
            directional.push(LightUniform {
                kind: 0,
                _pad: [0; 3],
                direction: (rotation * Vec3::NEG_Z).to_array(),
                _pad_direction: 0.0,
                position: [0.0; 3],
                _pad_position: 0.0,
                color: light.color.to_array(),
                intensity: light.intensity,
                size: [0.0; 2],
                _pad_size: [0.0; 2],
            });
            continue;
        }
        // 局部光（点光/面光）：记下到相机的欧氏距离，用于每帧就近筛选。
        let entry = match light.kind {
            LightKind::Point => LightUniform {
                kind: 1,
                _pad: [0; 3],
                direction: [0.0; 3],
                _pad_direction: 0.0,
                position: translation.to_array(),
                _pad_position: 0.0,
                color: light.color.to_array(),
                intensity: light.intensity,
                size: [0.0; 2],
                _pad_size: [0.0; 2],
            },
            LightKind::Area { width, height } => LightUniform {
                kind: 2,
                _pad: [0; 3],
                direction: (rotation * Vec3::NEG_Z).to_array(),
                _pad_direction: 0.0,
                position: translation.to_array(),
                _pad_position: 0.0,
                color: light.color.to_array(),
                intensity: light.intensity,
                size: [width, height],
                _pad_size: [0.0; 2],
            },
            LightKind::Directional => unreachable!("方向光已在上方处理"),
        };
        nearby.push((translation.distance(camera_position), entry));
    }

    // 最近的 X 盏局部光（欧氏距离），方向光始终优先且不占额度。
    nearby.sort_by(|a, b| a.0.total_cmp(&b.0));
    nearby.truncate(MAX_NEARBY_LIGHTS);

    directional.extend(nearby.into_iter().map(|(_, light)| light));
    // 兜底：超出 storage 容量时截断（正常场景不会触发）。
    directional.truncate(LIGHT_CAPACITY);
    directional
}


/// 每物体 uniform：模型矩阵 + 法线矩阵（逆转置，正确处理非等比缩放）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ObjectData {
    pub(super) model: Mat4,
    /// 法线矩阵（WGSL `mat3x3<f32>` 布局：每列 16 字节，含填充）。
    pub(super) normal_matrix: [[f32; 4]; 3],
    /// 材质基础色因子（RGBA）。
    pub(super) base_color: [f32; 4],
    pub(super) metallic: f32,
    pub(super) roughness: f32,
    pub(super) _pad: [f32; 2],
}

/// 单个光源在 GPU 缓冲里的布局（80 字节，std140 兼容）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightUniform {
    /// 0=方向光 1=点光 2=面光。
    pub(super) kind: u32,
    pub(super) _pad: [u32; 3],
    /// 方向光/面光：世界空间行进方向（局部 -Z 经旋转）。
    pub(super) direction: [f32; 3],
    pub(super) _pad_direction: f32,
    /// 点光/面光：世界位置。
    pub(super) position: [f32; 3],
    pub(super) _pad_position: f32,
    pub(super) color: [f32; 3],
    pub(super) intensity: f32,
    /// 面光：面板尺寸（当前近似未直接使用，为 LTC 预留）。
    pub(super) size: [f32; 2],
    pub(super) _pad_size: [f32; 2],
}

/// 灯光数量 uniform：每帧写入实际参与着色的灯光数（16 字节，含填充）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightCountUniform {
    pub(super) count: u32,
    pub(super) _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<LightUniform>() == 80);
const _: () = assert!(std::mem::size_of::<LightCountUniform>() == 16);

/// 环境计算着色器参数 uniform：`size` = 每面尺寸，`sample_count` = 辐照度采样数。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EnvParams {
    pub(super) size: u32,
    pub(super) sample_count: u32,
    pub(super) _pad: [u32; 2],
}

/// 镜面预过滤计算着色器参数 uniform：当前 mip 的尺寸、mip 序号、总 mip 数、采样数。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PrefilterParams {
    pub(super) size: u32,
    pub(super) mip: u32,
    pub(super) mip_count: u32,
    pub(super) sample_count: u32,
}

const _: () = assert!(std::mem::size_of::<PrefilterParams>() == 16);

/// 中间灰 0.18 的 log2 值：`log2(0.18)`。
/// 把"相对中间灰的 EV"换算成 shader 里的绝对 log2 锚点：`绝对 = EV + 该值`
/// （如 -10 EV → -12.47393，+6.5 EV → 4.026069，即 Blender/Filament 的默认窗口）。
pub(crate) const AGX_MIDDLE_GRAY_LOG2: f32 = -2.47393;

/// AgX 默认 EV 窗口（相对中间灰的 EV 档位）：-10 ~ +6.5 EV，与 Blender 一致。
pub(crate) const AGX_DEFAULT_EV_MIN: f32 = -10.0;
pub(crate) const AGX_DEFAULT_EV_MAX: f32 = 6.5;

/// 上述默认窗口换算后的绝对 log2 锚点（uniform 初值）。
pub(super) const AGX_DEFAULT_MIN_EV: f32 = AGX_DEFAULT_EV_MIN + AGX_MIDDLE_GRAY_LOG2;
pub(super) const AGX_DEFAULT_MAX_EV: f32 = AGX_DEFAULT_EV_MAX + AGX_MIDDLE_GRAY_LOG2;

/// 环境参数 uniform（mesh 管线 @group(4) binding 3，天空盒管线 @group(1) binding 2）。
/// `intensity` = IBL 环境光强度（天空盒侧兼作曝光），0 = 纯手动布光，1 = 满环境光；
/// `agx_min_ev` / `agx_max_ev` = AgX 色调映射 EV 窗口，场景可按层级覆盖（默认 Blender 值）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EnvironmentParams {
    pub(super) intensity: f32,
    pub(super) agx_min_ev: f32,
    pub(super) agx_max_ev: f32,
    pub(super) _pad: u32,
}

impl Default for EnvironmentParams {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            agx_min_ev: AGX_DEFAULT_MIN_EV,
            agx_max_ev: AGX_DEFAULT_MAX_EV,
            _pad: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::core::light::Light;
    use crate::engine::core::transform::Transform;
    use crate::engine::scene::{Scene, SceneObject, SceneObjectKind};
    use glam::Quat;

    /// 方向光的 uniform direction 应为行进方向（光源 → 场景），
    /// 即物体局部 -Z 经旋转后的方向，与面光"发射方向"语义一致。
    #[test]
    fn directional_light_direction_is_travel_direction() {
        let mut scene = Scene::new();
        // 光从右上前方照向场景（来向 = arrival），行进方向 = -arrival。
        let arrival = Vec3::new(0.5, 0.6, 0.6).normalize();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::directional(Vec3::ONE, 1.0)),
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_arc(Vec3::NEG_Z, -arrival),
                Vec3::ONE,
            ),
        ));

        let lights = collect_lights(&scene, Vec3::ZERO);
        assert_eq!(lights.len(), 1);
        let dir = Vec3::from(lights[0].direction);
        assert!(
            dir.normalize().dot(-arrival).abs() > 0.99,
            "uniform direction 应为行进方向（-arrival），实际 {dir:?}"
        );
    }

    /// 收集规则：方向光总是包含且在前，局部光按离相机距离取最近 X 盏。
    #[test]
    fn collect_lights_keeps_directionals_and_nearest_local() {
        let mut scene = Scene::new();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::directional(Vec3::ONE, 1.0)),
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_arc(
                    Vec3::NEG_Z,
                    -Vec3::new(0.5, 0.6, 0.6).normalize(),
                ),
                Vec3::ONE,
            ),
        ));
        // 远处点光（距离 50）与近处点光（距离 2）。
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::ONE, 10.0)),
            Transform::new(Vec3::new(50.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
        ));
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::ONE, 10.0)),
            Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
        ));

        let lights = collect_lights(&scene, Vec3::ZERO);
        assert_eq!(lights.len(), 3);
        // 顺序：方向光在前，局部光按距离近 → 远。
        assert_eq!(lights[0].kind, 0);
        assert!(Vec3::from(lights[1].position).distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-5);
        assert!(Vec3::from(lights[2].position).distance(Vec3::new(50.0, 0.0, 0.0)) < 1e-5);
    }

    /// 局部光超过 X 盏时只保留最近的 X 盏。
    #[test]
    fn collect_lights_truncates_to_nearest_max() {
        let mut scene = Scene::new();
        // 9 盏点光，距离 1..=9，X = MAX_NEARBY_LIGHTS = 8。
        for i in 0..9 {
            scene.add_object(SceneObject::new(
                SceneObjectKind::Light(Light::point(Vec3::ONE, 1.0)),
                Transform::new(
                    Vec3::new((i + 1) as f32, 0.0, 0.0),
                    Quat::IDENTITY,
                    Vec3::ONE,
                ),
            ));
        }

        let lights = collect_lights(&scene, Vec3::ZERO);
        assert_eq!(lights.len(), MAX_NEARBY_LIGHTS);
        // 第 9 盏（距离 9）被截掉，其余都在额度内。
        for light in &lights {
            assert!(Vec3::from(light.position).x < 9.0);
        }
    }
}
