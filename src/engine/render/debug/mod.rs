//! 调试可视化：把场景里的光源与碰撞箱画成线框 Gizmo。
//!
//! - 灯光：每种光源一个"灯泡"标记，外加指向性的调试射线
//!   （方向光圆盘、点光八面体、面光矩形面板，见 [`build_light_gizmos`]）；
//! - 碰撞箱：每个网格物体的世界 AABB 画成 12 条边的线框盒子
//!   （见 [`build_collision_gizmos`]）。
//!
//! 线段在 `load_scene` 时生成并上传一次（光源与静态物体目前都是静态场景数据，
//! 物体支持动画后再改成每帧重建），走专用线框管线绘制：
//! 深度比较 Always + 不写深度，保证灯泡被物体挡住时依然可见，方便调试。

use glam::Vec3;
use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CompareFunction, DepthStencilState, FragmentState, PipelineLayoutDescriptor, PrimitiveState,
    PrimitiveTopology, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, VertexState,
};

use crate::engine::core::aabb::Aabb;
use crate::engine::core::asset::MeshSource;
use crate::engine::core::light::LightKind;
use crate::engine::scene::{Scene, SceneObjectKind};

/// 调试线条顶点：位置 + 颜色（与 debug.wgsl 顶点输入一一对应）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DebugVertex {
    pub(super) position: [f32; 3],
    pub(super) color: [f32; 3],
}

impl DebugVertex {
    fn new(position: Vec3, color: Vec3) -> Self {
        Self {
            position: position.to_array(),
            color: color.to_array(),
        }
    }

    /// 顶点缓冲布局（位置 float32×3 + 颜色 float32×3）。
    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DebugVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        }
    }
}

/// 灯泡 gizmo 的尺寸常量。
const BULB_RADIUS: f32 = 0.3;
const RAY_LENGTH: f32 = 1.4;
const DISC_SEGMENTS: usize = 16;

/// 从场景收集所有光源，生成调试线框的顶点列表（2 顶点 = 1 条线段）。
///
/// 灯光是静态场景数据，在 `load_scene` 时调用一次；光源将来支持动画时
/// 再改成每帧重建。
pub(super) fn build_light_gizmos(scene: &Scene) -> Vec<DebugVertex> {
    let mut vertices = Vec::new();
    for (key, object) in scene.objects() {
        let SceneObjectKind::Light(light) = object.kind else {
            continue;
        };
        let world = scene
            .world_transform(key)
            .expect("objects() 只产出存活节点，world_transform 必然有值");
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        // 太暗的光源用下限兜底，保证线框始终可见（保持光源自身的色相便于区分）。
        let color = light.color.max(Vec3::splat(0.35));
        match light.kind {
            LightKind::Directional => {
                // 局部 -Z（经世界旋转）= 光行进方向（光源 → 场景），
                // 与面光一致；调试箭头直接画行进方向。
                push_directional(&mut vertices, translation, rotation * Vec3::NEG_Z, color);
            }
            LightKind::Point => push_point(&mut vertices, &world, color),
            LightKind::Area { width, height } => {
                push_area(&mut vertices, &world, rotation, width, height, color);
            }
        }
    }
    vertices
}

/// 从场景收集所有网格物体的世界 AABB，生成线框盒子的顶点列表
/// （12 条边 = 24 个顶点）。
///
/// 固定橙色线框；空包围盒（无网格数据）的节点自动跳过。
pub(super) fn build_collision_gizmos(scene: &Scene, meshes: &dyn MeshSource) -> Vec<DebugVertex> {
    let color = Vec3::new(1.0, 0.55, 0.1);
    let mut vertices = Vec::new();
    for (key, _) in scene.objects() {
        if let Some(aabb) = scene.object_aabb_world(meshes, key) {
            push_aabb(&mut vertices, &aabb, color);
        }
    }
    vertices
}

/// 追加一个 AABB 的 12 条边（4 底 + 4 顶 + 4 竖柱）。
fn push_aabb(out: &mut Vec<DebugVertex>, aabb: &Aabb, color: Vec3) {
    let min = aabb.min;
    let max = aabb.max;
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    for &(i, j) in &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        push_segment(out, corners[i], corners[j], color);
    }
}

/// 方向光：垂直于光行进方向的圆盘 + 沿行进方向的长射线（带箭头）。
fn push_directional(out: &mut Vec<DebugVertex>, center: Vec3, dir: Vec3, color: Vec3) {
    let dir = dir.normalize();
    let u = dir.any_orthonormal_vector();
    let v = dir.cross(u);

    // 圆盘：16 段折线。
    for i in 0..DISC_SEGMENTS {
        let a = (i as f32 / DISC_SEGMENTS as f32) * std::f32::consts::TAU;
        let b = ((i + 1) as f32 / DISC_SEGMENTS as f32) * std::f32::consts::TAU;
        let p0 = center + (u * a.cos() + v * a.sin()) * BULB_RADIUS;
        let p1 = center + (u * b.cos() + v * b.sin()) * BULB_RADIUS;
        push_segment(out, p0, p1, color);
    }

    // 朝向场景的射线 + 箭头。
    push_arrow(out, center, dir, RAY_LENGTH, color);
}

/// 点光：八面体线框（灯泡）+ 三个轴向的短放射线。
fn push_point(out: &mut Vec<DebugVertex>, world: &glam::Mat4, color: Vec3) {
    // 八面体 6 个顶点（±X / ±Y / ±Z）。
    let axes = [
        Vec3::X * BULB_RADIUS,
        Vec3::NEG_X * BULB_RADIUS,
        Vec3::Y * BULB_RADIUS,
        Vec3::NEG_Y * BULB_RADIUS,
        Vec3::Z * BULB_RADIUS,
        Vec3::NEG_Z * BULB_RADIUS,
    ];
    let verts: Vec<Vec3> = axes
        .iter()
        .map(|local| world.transform_point3(*local))
        .collect();
    // 相邻顶点连线：两个顶点所在轴不同即为八面体的一条边（12 条）。
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            if axes[i].dot(axes[j]) == 0.0 {
                push_segment(out, verts[i], verts[j], color);
            }
        }
    }

    // 六个方向的放射线：从灯泡表面伸出去一小段，表示点光向四周辐射。
    let center = world.transform_point3(Vec3::ZERO);
    for axis in [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ] {
        let tip = world.transform_point3(axis * (BULB_RADIUS + 0.35));
        push_segment(out, center, tip, color);
    }
}

/// 面光：矩形面板线框（宽度/高度跟随世界变换）+ 中心发射射线（带箭头）。
fn push_area(
    out: &mut Vec<DebugVertex>,
    world: &glam::Mat4,
    rotation: glam::Quat,
    width: f32,
    height: f32,
    color: Vec3,
) {
    let half_w = width * 0.5;
    let half_h = height * 0.5;
    // 面板局部坐标：XY 平面内，发射方向为局部 -Z。
    let corners = [
        glam::Vec3::new(-half_w, -half_h, 0.0),
        glam::Vec3::new(half_w, -half_h, 0.0),
        glam::Vec3::new(half_w, half_h, 0.0),
        glam::Vec3::new(-half_w, half_h, 0.0),
    ];
    let corners: Vec<Vec3> = corners.iter().map(|c| world.transform_point3(*c)).collect();
    for i in 0..4 {
        push_segment(out, corners[i], corners[(i + 1) % 4], color);
    }

    // 面板中心十字（局部 X/Y 方向），标出中心与面板朝向。
    let center = world.transform_point3(Vec3::ZERO);
    let u = rotation * Vec3::X;
    let v = rotation * Vec3::Y;
    push_segment(out, center - u * half_w, center + u * half_w, color);
    push_segment(out, center - v * half_h, center + v * half_h, color);

    // 发射方向射线（局部 -Z 经世界旋转）+ 箭头。
    push_arrow(out, center, rotation * Vec3::NEG_Z, RAY_LENGTH, color);
}

/// 从 `origin` 沿 `dir` 画一条长度 `length` 的线段，末端带小箭头。
fn push_arrow(out: &mut Vec<DebugVertex>, origin: Vec3, dir: Vec3, length: f32, color: Vec3) {
    let dir = dir.normalize();
    let tip = origin + dir * length;
    let u = dir.any_orthonormal_vector();
    push_segment(out, origin, tip, color);
    let base = tip - dir * 0.18;
    push_segment(out, tip, base + u * 0.09, color);
    push_segment(out, tip, base - u * 0.09, color);
}

/// 追加一条线段（2 个顶点）。
fn push_segment(out: &mut Vec<DebugVertex>, a: Vec3, b: Vec3, color: Vec3) {
    out.push(DebugVertex::new(a, color));
    out.push(DebugVertex::new(b, color));
}

/// 线框调试的 GPU 资源：线框管线 + 可增长的顶点缓冲。
///
/// 灯光与碰撞箱各持一个实例（管线相同但缓冲独立，可同时显示）。
pub(super) struct LineGizmos {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    capacity_vertices: u32,
    /// 当前已上传的顶点数（load_scene 时刷新，渲染时直接按此绘制）。
    vertex_count: u32,
}

impl LineGizmos {
    /// 创建线框管线（复用相机绑定组布局，@group(0)）与初始顶点缓冲。
    pub(super) fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("debug line shader"),
            source: ShaderSource::Wgsl(include_str!("debug.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("debug line pipeline layout"),
            bind_group_layouts: &[Some(camera_bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("debug line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(DebugVertex::layout())],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                // 线框无正面背面之分。
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                // 不写深度 + Always：灯泡即使被墙挡住也保持可见，便于调试。
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        // 初始 1 顶点占位；绘制时顶点更多再由 ensure_capacity 重建。
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug line vertex buffer"),
            size: std::mem::size_of::<DebugVertex>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            vertex_buffer,
            capacity_vertices: 1,
            vertex_count: 0,
        }
    }

    /// 确保顶点缓冲能装下至少 `needed` 个顶点；不足时重建（旧缓冲自动释放）。
    pub(super) fn ensure_capacity(&mut self, device: &wgpu::Device, needed: u32) {
        if needed <= self.capacity_vertices {
            return;
        }
        self.vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("debug line vertex buffer"),
            size: needed as u64 * std::mem::size_of::<DebugVertex>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity_vertices = needed;
    }

    /// 上传顶点数据（容量不足时先重建缓冲）。灯光是静态数据，
    /// 在 `load_scene` 时调用一次即可，渲染循环不再上传。
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[DebugVertex],
    ) {
        self.ensure_capacity(device, vertices.len() as u32);
        if !vertices.is_empty() {
            queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        }
        self.vertex_count = vertices.len() as u32;
    }

    /// 绑定管线 + 相机 + 顶点缓冲，绘制已上传的线段。
    pub(super) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera_bind_group: &wgpu::BindGroup,
    ) {
        if self.vertex_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::AssetManager;
    use crate::engine::Camera;
    use crate::engine::core::light::Light;
    use crate::engine::core::transform::Transform;
    use crate::engine::scene::SceneObject;
    use glam::Quat;

    /// 方向光调试射线应指向光线行进方向（光源 → 场景），
    /// 即物体局部 -Z（统一约定的行进方向）经旋转后的方向。
    #[test]
    fn directional_gizmo_ray_points_along_travel_direction() {
        let mut scene = Scene::new();
        // 与演示场景相同的布光：光从右上前方照向场景（来向 = light_dir），
        // 物体局部 -Z 按约定对齐行进方向（-light_dir）。
        let light_dir = Vec3::new(0.5, 0.6, 0.6).normalize();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::directional(Vec3::ONE, 1.0)),
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_arc(Vec3::NEG_Z, -light_dir),
                Vec3::ONE,
            ),
        ));

        let gizmos = build_light_gizmos(&scene);
        // 找从圆盘中心出发、长度等于 RAY_LENGTH 的射线（push_arrow 第一条线段）。
        let ray = gizmos
            .windows(2)
            .find(|pair| {
                pair[0].position == [0.0, 0.0, 0.0]
                    && (Vec3::from(pair[1].position) - Vec3::from(pair[0].position)).length()
                        > RAY_LENGTH * 0.99
            })
            .expect("方向光应有一条从中心出发的行进方向射线");
        let dir = Vec3::from(ray[1].position) - Vec3::from(ray[0].position);
        assert!(
            dir.normalize().dot(-light_dir).abs() > 0.99,
            "射线应沿 -light_dir（行进方向），实际 {dir:?}"
        );
    }

    /// 一个边长 1 的立方体：碰撞箱生成 12 条边（24 顶点），
    /// 且线段的端点落在立方体 8 个角点上。
    #[test]
    fn collision_gizmos_wire_a_single_cube() {
        let mut scene = Scene::new();
        let mut assets = AssetManager::without_gpu();
        let key = assets.meshes_mut().register(crate::engine::Mesh::cube());
        scene.add_object(SceneObject::new(
            SceneObjectKind::Mesh(key),
            Transform::new(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
        ));

        let vertices = build_collision_gizmos(&scene, &assets);
        assert_eq!(vertices.len(), 24, "12 条边 × 2 顶点");

        // 所有端点都应落在立方体角点集合（±0.5）上。
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 != 0 { 0.5 } else { -0.5 },
                    if i & 2 != 0 { 0.5 } else { -0.5 },
                    if i & 4 != 0 { 0.5 } else { -0.5 },
                )
            })
            .collect();
        for vertex in &vertices {
            let p = Vec3::from(vertex.position);
            assert!(
                corners.iter().any(|c| (*c - p).length() < 1e-6),
                "端点应落在角点上：{p:?}"
            );
        }
    }

    /// 空场景：无碰撞箱可画；非网格节点（分组/灯光）不会产出线段。
    #[test]
    fn collision_gizmos_skip_non_mesh_nodes() {
        let mut scene = Scene::new();
        let assets = AssetManager::without_gpu();
        scene.add_object(SceneObject::new(
            SceneObjectKind::Empty,
            Transform::IDENTITY,
        ));
        scene.add_object(SceneObject::new(
            SceneObjectKind::Light(Light::point(Vec3::ZERO, 1.0)),
            Transform::IDENTITY,
        ));
        let cam = scene.add_camera(Camera::new(Vec3::ZERO, 0.0, 0.0, 1.0, 1.0, 0.1, 100.0));
        assert!(scene.set_main_camera(cam));

        assert!(build_collision_gizmos(&scene, &assets).is_empty());
    }
}
