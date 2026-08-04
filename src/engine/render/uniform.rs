//! 渲染器使用的 GPU uniform 布局与灯光收集。
//!
//! 与 WGSL（mesh.wgsl / environment.wgsl）的内存布局一一对应，Rust 侧只负责
//! 布局与数据填充，缓冲/绑定组的创建在 `super::Renderer` 与 `super::environment`。

use glam::{Mat4, Vec3};

use crate::engine::core::light::LightKind;
use crate::engine::scene::{Scene, SceneObjectKind};

/// 从场景收集方向光（世界方向由物体的世界旋转决定）。
///
/// 在 `load_scene` 时调用一次：静态光源不需要每帧重新推导。
/// 将来出现会动的光源（移动/闪烁/玩家手电）时，再从这里加刷新入口。
pub(super) fn collect_lights(scene: &Scene) -> LightsUniform {
    let mut light_uniform = LightsUniform {
        count: 0,
        _pad: [0; 3],
        lights: [LightUniform {
            kind: 0,
            _pad: [0; 3],
            direction: [0.0; 3],
            _pad_direction: 0.0,
            position: [0.0; 3],
            _pad_position: 0.0,
            color: [0.0; 3],
            intensity: 0.0,
            size: [0.0; 2],
            _pad_size: [0.0; 2],
        }; MAX_LIGHTS],
    };
    for (key, object) in scene.objects() {
        if light_uniform.count as usize >= MAX_LIGHTS {
            break;
        }
        let SceneObjectKind::Light(light) = object.kind else {
            continue;
        };
        let world = scene
            .world_transform(key)
            .expect("objects() 只产出存活节点");
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        let entry = &mut light_uniform.lights[light_uniform.count as usize];
        match light.kind {
            LightKind::Directional => {
                entry.kind = 0;
                entry.direction = (rotation * Vec3::NEG_Z).to_array();
            }
            LightKind::Point => {
                entry.kind = 1;
                entry.position = translation.to_array();
            }
            LightKind::Area { width, height } => {
                entry.kind = 2;
                entry.direction = (rotation * Vec3::NEG_Z).to_array();
                entry.position = translation.to_array();
                entry.size = [width, height];
            }
        }
        entry.intensity = light.intensity;
        entry.color = light.color.to_array();
        light_uniform.count += 1;
    }
    light_uniform
}


/// 最多同时支持的方向光数量（与 WGSL 中 `MAX_LIGHTS` 一致）。
pub(super) const MAX_LIGHTS: usize = 8;


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

/// 单个光源在 uniform 缓冲里的布局（80 字节，std140 兼容）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    /// 0=方向光 1=点光 2=面光。
    kind: u32,
    _pad: [u32; 3],
    /// 方向光/面光：世界空间光照方向（局部 -Z 经旋转）。
    direction: [f32; 3],
    _pad_direction: f32,
    /// 点光/面光：世界位置。
    position: [f32; 3],
    _pad_position: f32,
    color: [f32; 3],
    intensity: f32,
    /// 面光：面板尺寸（当前近似未直接使用，为 LTC 预留）。
    size: [f32; 2],
    _pad_size: [f32; 2],
}

/// 灯光 uniform：数量 + 固定大小数组。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LightsUniform {
    count: u32,
    _pad: [u32; 3],
    lights: [LightUniform; MAX_LIGHTS],
}

const _: () = assert!(std::mem::size_of::<LightsUniform>() == 16 + 80 * MAX_LIGHTS);

/// 环境计算着色器参数 uniform：`size` = 每面尺寸，`sample_count` = 辐照度采样数。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EnvParams {
    pub(super) size: u32,
    pub(super) sample_count: u32,
    pub(super) _pad: [u32; 2],
}

/// 环境参数 uniform（mesh 管线 @group(4) binding 3）：
/// `intensity` = IBL 环境光强度，0 = 纯手动布光，1 = 满环境光。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EnvironmentIntensity {
    pub(super) intensity: f32,
    pub(super) _pad: [u32; 3],
}
