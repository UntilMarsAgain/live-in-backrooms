//! 资产管理器的 GPU 实现层：GPU 表示类型、上传器与 [`AssetManager`] 本体。
//!
//! 按项目约定 core 层不包含 GPU 实际实现，因此这里的类型（`MeshGpu`、
//! 上传器等）从 core 独立出来，依赖方向 render → core：
//! core 只保留抽象（[`AssetRegistry`]、[`Handle`]、[`GpuUploader`] trait）。

#[allow(unused_imports)] // Arc 由宏展开后的 AssetManager 使用，静态分析误报
use std::sync::Arc;

use paste::paste;
use wgpu::{Device, Queue};

use crate::engine::core::asset::{
    AssetRegistry, GpuUploader, Handle, LevelData, MeshSource, NoGpuUploader,
};
use crate::engine::core::mesh::Mesh;
use crate::engine::core::texture::Texture;

/// 网格的 GPU 表示：每网格独立的顶点/索引缓冲。
///
/// 独立缓冲是"资源级卸载/更新"的前提；渲染器按句柄取用，绘制时切换缓冲。
#[derive(Debug)]
pub struct MeshGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    /// 索引数量（绘制时用；独立缓冲整份即该网格）。
    pub index_count: u32,
}

/// 纹理的 GPU 表示：贴图纹理及其视图。
#[derive(Debug)]
pub struct TextureGpu {
    #[allow(dead_code)] // 预留：纹理重建/尺寸查询；当前仅 view 被采样
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

/// 网格上传器：把 `Mesh` 转成独立 GPU 缓冲（顶点 + 索引）。
#[derive(Debug, Default)]
pub struct MeshUploader;

impl GpuUploader<Mesh, MeshGpu> for MeshUploader {
    fn upload(&mut self, device: &Device, queue: &Queue, mesh: &Mesh) -> MeshGpu {
        use wgpu::util::DeviceExt;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer"),
            contents: bytemuck::cast_slice(mesh.vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh index buffer"),
            contents: bytemuck::cast_slice(mesh.indices()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let _ = queue; // 上传走 create_buffer_init（映射创建），queue 仅作签名一致。
        MeshGpu {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices().len() as u32,
        }
    }
}

/// 贴图上传器：把 `Texture` 转成 GPU 纹理（RGBA8 sRGB，TEXTURE_BINDING）。
#[derive(Debug, Default)]
pub struct TextureUploader;

impl GpuUploader<Texture, TextureGpu> for TextureUploader {
    fn upload(&mut self, device: &Device, queue: &Queue, texture: &Texture) -> TextureGpu {
        let gpu_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("texture"),
            size: wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texture.rgba8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture.width * 4),
                rows_per_image: Some(texture.height),
            },
            wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: 1,
            },
        );
        let view = gpu_texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureGpu {
            texture: gpu_texture,
            view,
        }
    }
}

/// 声明资产管理器的资源类型集合。
///
/// 每个条目：`字段名: CPU类型 => GPU类型, 上传器类型`。纯数据资源
/// （关卡、AI 等）用 `()` 作 GPU 类型、上传器给一个空函数占位
/// （[`NoGpuUploader`]）；状态机没有显存阶段。新增资源类型 = 加一行。
macro_rules! asset_types {
    ($($field:ident: $ty:ident => $gpu:ty, $uploader:ident),* $(,)?) => {
        paste! {
            /// 统一资产管理器：持有设备/队列与各类型注册表、上传器。
            ///
            /// 游戏逻辑侧：注册/查询/pin/unpin（见各注册表方法）；
            /// 渲染器侧：`sync_gpu` / `ensure_*_gpu` 后按句柄取 GPU 数据。
            #[derive(Debug)]
            pub struct AssetManager {
                device: Option<Arc<Device>>,
                queue: Option<Arc<Queue>>,
                $(
                    $field: AssetRegistry<$ty, $gpu>,
                    [<$field _uploader>]: $uploader,
                )*
            }

            // 宏为每种资源全量生成 ensure/pin 便捷方法，部分在调用方
            // 未覆盖前属于公共 API 预留，暂时保留。
            #[allow(dead_code)]
            impl AssetManager {
                /// 新建管理器；`device` + `queue` 用于创建/管理 GPU 资源
                /// （wgpu 的 Device/Queue 都是内部引用计数，clone 廉价）。
                pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
                    Self {
                        device: Some(device),
                        queue: Some(queue),
                        $( $field: AssetRegistry::new(), [<$field _uploader>]: $uploader::default(), )*
                    }
                }

                /// 无 GPU 环境（纯数据/碰撞测试）用：`sync_gpu` 自动跳过。
                pub fn without_gpu() -> Self {
                    Self {
                        device: None,
                        queue: None,
                        $( $field: AssetRegistry::new(), [<$field _uploader>]: $uploader::default(), )*
                    }
                }

                /// 底层设备（上传/同步时使用）；无 GPU 构造时为 `None`。
                pub fn device(&self) -> Option<&Device> {
                    self.device.as_deref()
                }

                /// 同步所有注册表的 GPU 资源：pinned 且未上传的上传，
                /// 非 pinned 的回收。渲染器每帧或脏时调用。
                pub fn sync_gpu(&mut self) {
                    let (Some(device), Some(queue)) = (&self.device, &self.queue) else {
                        return; // 无 GPU 构造：纯 CPU 使用，不上传
                    };
                    $(
                        self.$field
                            .sync_gpu(device, queue, &mut self.[<$field _uploader>]);
                    )*
                }

                $(
                    /// 对应类型的注册表访问器。
                    pub fn $field(&self) -> &AssetRegistry<$ty, $gpu> {
                        &self.$field
                    }

                    /// 对应类型的注册表可变访问器。
                    pub fn [<$field _mut>](&mut self) -> &mut AssetRegistry<$ty, $gpu> {
                        &mut self.$field
                    }

                    /// 按需确保该类型资源已上传（渲染器绘制前的兜底）；
                    /// 句柄无效返回 `None`。
                    pub fn [<ensure_ $field _gpu>](
                        &mut self,
                        handle: Handle<$ty>,
                    ) -> Option<&$gpu> {
                        let (device, queue) = (self.device.as_ref()?, self.queue.as_ref()?);
                        self.$field.ensure_gpu(
                            device,
                            queue,
                            handle,
                            &mut self.[<$field _uploader>],
                        )
                    }

                    /// 预上传并驻留该类型资源（预分配语义）；无 GPU 构造
                    /// 只标记驻留状态。
                    pub fn [<pin_ $field>](&mut self, handle: Handle<$ty>) -> bool {
                        match (&self.device, &self.queue) {
                            (Some(device), Some(queue)) => self.$field.pin_upload(
                                device,
                                queue,
                                handle,
                                &mut self.[<$field _uploader>],
                            ),
                            _ => self.$field.pin(handle),
                        }
                    }
                )*
            }
        }
    };
}

asset_types! {
    meshes: Mesh => MeshGpu, MeshUploader,
    textures: Texture => TextureGpu, TextureUploader,
    levels: LevelData => (), NoGpuUploader,
}

/// 资产管理器实现网格数据源：场景碰撞/调试按句柄取 CPU 网格。
impl MeshSource for AssetManager {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh> {
        self.meshes().get(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::core::asset::AssetState;

    /// 纯 CPU 资源（NoGpuUploader）：注册/查询/移除正常，不涉及 GPU。
    #[test]
    fn pure_cpu_asset_registers_and_queries() {
        let mut assets = AssetManager::without_gpu();
        let handle = assets
            .levels_mut()
            .register(LevelData { name: "level-0".into() });
        assert_eq!(
            assets.levels().get(handle).map(|l| l.name.as_str()),
            Some("level-0")
        );
        assert_eq!(assets.levels().state(handle), Some(AssetState::Resident));
        assert!(assets.levels_mut().remove(handle).is_some());
        assert!(assets.levels().get(handle).is_none());
    }
}
