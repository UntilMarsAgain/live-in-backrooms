//! 场景数据上传：网格合并、纹理增量上传、环境设置与关卡加载。

use wgpu::util::DeviceExt;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages};

use crate::engine::core::environment::Environment;
use crate::engine::core::mesh::{MeshLibrary, Vertex};
use crate::engine::core::texture::TextureLibrary;
use crate::engine::render::debug;
use crate::engine::render::init::create_texture_view;
use crate::engine::render::uniform::{
    AGX_DEFAULT_EV_MAX, AGX_DEFAULT_EV_MIN, AGX_MIDDLE_GRAY_LOG2, ObjectDataUniform,
};
use crate::engine::render::{MeshGpu, MeshRange, Renderer};
use crate::engine::scene::Scene;

impl Renderer {
    /// 把网格库中的全部资产合并成一份顶点/索引缓冲，永久驻留。
    ///
    /// 版本没变则跳过；新增资产后整体重传（前面的数据保持不变）。
    pub fn upload_meshes(&mut self, library: &MeshLibrary) {
        if library.version() == self.mesh_uploaded_version {
            return;
        }

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut mesh_ranges = Vec::with_capacity(library.len());
        for mesh in library.meshes() {
            let vertex_offset = vertices.len() as u32;
            mesh_ranges.push(MeshRange {
                index_offset: indices.len() as u32,
                index_count: mesh.indices().len() as u32,
            });
            vertices.extend_from_slice(mesh.vertices());
            // 合并时索引已按该网格的顶点起始偏移平移，因此绘制时 base_vertex 必须为 0。
            indices.extend(mesh.indices().iter().map(|i| i + vertex_offset));
        }

        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh library vertex buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh library index buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::INDEX,
            });

        self.mesh_buffer = Some(MeshGpu {
            vertex_buffer,
            index_buffer,
            mesh_ranges,
        });
        self.mesh_uploaded_version = library.version();
    }

    /// 把纹理库中新增的贴图上传为 GPU 纹理并建好绑定组（只追加，增量上传）。
    pub fn upload_textures(&mut self, library: &TextureLibrary) {
        if library.version() == self.uploaded_texture_version {
            return;
        }
        for texture in library.textures().iter().skip(self.texture_views.len()) {
            let view = create_texture_view(&self.device, &self.queue, texture);
            self.texture_views.push(view);
        }
        self.uploaded_texture_version = library.version();
    }

    /// 上传环境贴图（HDRI 等距矩形图）并转换成环境立方体贴图 + 辐照度图。
    ///
    /// 转换由两个计算着色器在启动时一次性完成，之后每帧只采样；
    /// 关卡切换换环境时重建纹理与绑定组，旧资源随替换自动释放。
    pub fn set_environment(&mut self, environment: &Environment) {
        self.environment =
            self.environment_resources
                .convert(&self.device, &self.queue, environment);
    }

    /// 设置环境强度（IBL 系数）：0 = 纯手动布光，1 = 满环境光。
    /// 只写 uniform，不重建环境资源。
    pub fn set_environment_intensity(&self, intensity: f32) {
        self.environment_resources
            .set_intensity(&self.queue, intensity);
    }

    /// 覆盖 AgX 色调映射的 EV 窗口（场景级风格配置，默认与 Blender 一致）。
    ///
    /// 参数是**相对中间灰 0.18 的 EV 档位**（如 -10 ~ +6.5），内部换算成
    /// shader 需要的绝对 log2 锚点；只写 uniform，不重建任何资源。
    pub fn set_environment_agx_ev(&self, ev_min: f32, ev_max: f32) {
        self.environment_resources.set_agx_range(
            &self.queue,
            ev_min + AGX_MIDDLE_GRAY_LOG2,
            ev_max + AGX_MIDDLE_GRAY_LOG2,
        );
    }

    /// 清除环境：切回默认的 1×1 黑环境（无天空盒、无 IBL），并把环境强度与
    /// AgX 窗口恢复默认。用于加载不带环境的场景时，避免残留上一关卡的天空盒。
    pub fn reset_environment(&mut self) {
        self.environment = self.environment_resources.default_environment.clone();
        self.set_environment_intensity(1.0);
        self.set_environment_agx_ev(AGX_DEFAULT_EV_MIN, AGX_DEFAULT_EV_MAX);
    }

    /// 加载场景：按物体数量重建动态 uniform 缓冲（网格资产已在 `upload_meshes` 中常驻）。
    pub fn load_scene(&mut self, scene: &Scene, meshes: &MeshLibrary) {
        // 按物体数量重建动态 uniform 缓冲与绑定组。
        let stride = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(std::mem::size_of::<ObjectDataUniform>() as u32);
        let object_data_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("object data buffer"),
            size: (scene.object_count() as u64).max(1) * stride as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("object data bind group"),
            layout: &self.object_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &object_data_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ObjectDataUniform>() as u64),
                }),
            }],
        });

        self.object_data_buffer = object_data_buffer;
        self.object_bind_group = object_bind_group;
        self.object_stride = stride as u32;

        // 灯光改为每帧收集（所有方向光 + 离相机最近的 X 盏局部光），见 render()；
        // 这里只保留静态数据（调试线框）的加载。

        // 调试线框同样是静态数据：加载时生成并上传一次，
        // 渲染时只按开关决定是否绘制，避免每帧重建/上传。
        let light_gizmos = debug::build_light_gizmos(scene);
        self.light_gizmos.upload(&self.device, &self.queue, &light_gizmos);
        let collision_gizmos = debug::build_collision_gizmos(scene, meshes);
        self.collision_gizmos
            .upload(&self.device, &self.queue, &collision_gizmos);

        // 每个物体的材质绑定组（与 objects() 迭代顺序一致，渲染时按同一下标取用）。
        let mut material_bind_groups = Vec::with_capacity(scene.object_count());
        for (_, object) in scene.objects() {
            if object.mesh_key().is_none() {
                material_bind_groups.push(self.default_material_bind_group.clone());
                continue;
            }
            let mat = &object.material;
            let base_view = mat
                .base_color_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_white_view);
            let mr_view = mat
                .metallic_roughness_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_white_view);
            let normal_view = mat
                .normal_texture
                .map(|k| &self.texture_views[k.index()])
                .unwrap_or(&self.default_normal_view);
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("material bind group"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(base_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(mr_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                ],
            });
            material_bind_groups.push(bind_group);
        }
        self.material_bind_groups = material_bind_groups;
    }
}
