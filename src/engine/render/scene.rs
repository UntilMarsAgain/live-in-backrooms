//! 场景数据上传：物体数据缓冲、材质绑定组、环境设置与关卡加载。

use std::mem::size_of;

use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages};

use crate::engine::asset::MeshView;
use crate::engine::core::asset::AssetManager;
use crate::engine::core::environment::Environment;
use crate::engine::render::asset::GpuManager;
use crate::engine::render::debug;
use crate::engine::render::uniform::{
    ObjectData, AGX_DEFAULT_EV_MAX, AGX_DEFAULT_EV_MIN, AGX_MIDDLE_GRAY_LOG2,
};
use crate::engine::render::Renderer;
use crate::engine::scene::Scene;

impl Renderer {
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

    /// 加载场景：按物体数量重建物体数据 storage 缓冲与材质绑定组。
    ///
    /// 网格/贴图的显存表示由 [`GpuManager`] 持有；这里只取视图构建绑定组，
    /// 不拥有资源。取用贴图视图走 `GpuManager::texture_gpu`——调度器会
    /// 检查并上传（自愈），调用方只需给句柄。
    pub fn load_scene(&mut self, scene: &Scene, gpu: &mut GpuManager, assets: &mut AssetManager) {
        // 按物体数量重建 storage 缓冲与绑定组（紧凑数组，无对齐步长浪费）。
        let object_data_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("object data buffer"),
            size: (scene.object_count() as u64).max(1) * size_of::<ObjectData>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let object_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("object data bind group"),
            layout: &self.object_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                // 绑定整个缓冲：着色器按实例索引访问，无动态偏移。
                resource: object_data_buffer.as_entire_binding(),
            }],
        });

        self.object_data_buffer = object_data_buffer;
        self.object_bind_group = object_bind_group;

        // 灯光改为每帧收集（所有方向光 + 离相机最近的 X 盏局部光），见 render()；
        // 这里只保留静态数据（调试线框）的加载。

        // 调试线框同样是静态数据：加载时生成并上传一次，
        // 渲染时只按开关决定是否绘制，避免每帧重建/上传。
        let light_gizmos = debug::build_light_gizmos(scene);
        self.light_gizmos
            .upload(&self.device, &self.queue, &light_gizmos);
        let collision_gizmos = debug::build_collision_gizmos(scene, &MeshView::new(assets));
        self.collision_gizmos
            .upload(&self.device, &self.queue, &collision_gizmos);

        // 每个物体的材质绑定组（与 objects() 迭代顺序一致，渲染时按同一下标取用）。
        let mut material_bind_groups = Vec::with_capacity(scene.object_count());
        for (_, object) in scene.objects() {
            if object.mesh_handle().is_none() {
                material_bind_groups.push(self.default_material_bind_group.clone());
                continue;
            }
            let mat = &object.material;
            // 自愈取用：贴图视图经 GpuManager 检查并上传；wgpu 视图是引用计数
            // 句柄，clone 便宜，避免持有返回引用的借用冲突。
            let base_view = mat
                .base_color_texture
                .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                .unwrap_or_else(|| self.default_white_view.clone());
            let mr_view = mat
                .metallic_roughness_texture
                .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                .unwrap_or_else(|| self.default_white_view.clone());
            let normal_view = mat
                .normal_texture
                .and_then(|handle| gpu.texture_gpu(handle, assets).map(|g| g.view.clone()))
                .unwrap_or_else(|| self.default_normal_view.clone());
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("material bind group"),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&base_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&mr_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&normal_view),
                    },
                ],
            });
            material_bind_groups.push(bind_group);
        }
        self.material_bind_groups = material_bind_groups;
    }
}
