//! 资产加载与解读模块：把 glTF 2.0 文件转换成运行时的网格资产与场景。
//!
//! 目前只处理“模型”部分：
//! - 节点层级与局部变换 → [`SceneTemplate`]（每个 glTF 节点一个容器物体，网格 primitive
//!   作为其子物体，保证“父节点动、子节点跟着动”的层级关系）；
//! - 每个 primitive → [`Mesh`] 注册进类型无关的
//!   [`AssetManager`](crate::engine::core::asset::AssetManager)
//!   （按网格索引去重，多节点共享同一网格）；
//! - 顶点属性按 glTF 2.0 语义读取：POSITION（必需）、NORMAL、TEXCOORD_0、
//!   COLOR_0，各种存储格式（f32 / 归一化整数）统一转换成运行时的 f32 布局。
//! - 材质基础色（`baseColorFactor` + `baseColorTexture`）读取，贴图注册进
//!   [`AssetManager`]；金属度/粗糙度、
//!   法线、自发光等 PBR 通道暂不读取。
//!
//! 加载一律走**泛型入口**：`AssetManager::load_file` / `load_file_async`
//! （实现 [`FileLoader`] 的 [`GlbFileLoader`]），句柄取用走泛型的
//! `loaded_handles_of::<T>` / `get` / `get_cached`，不提供 per-type 便捷函数。
//! `MeshView` 把管理器包装成 [`MeshSource`] 供碰撞/调试使用。`load_scene_template`
//! 是特殊入口：需要 glTF `Document` 构建场景树（节点层级 + 材质绑定），
//! 条目注册与重载仍与文件加载共用同一逻辑。
//!
//! 相机、动画、蒙皮暂不读取。

use std::any::{Any, TypeId};
use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use glam::{Mat4, Quat, Vec3};
use gltf::mesh::Mode;
use gltf::scene::Transform as GltfTransform;

use super::core::data::material::Material;
use super::core::data::mesh::{Mesh, Vertex};
use super::core::data::texture::Texture;
use super::core::data::transform::Transform;
use super::scene::{ObjectKey, SceneObject, SceneObjectKind, SceneTemplate};
use crate::engine::core::asset::{
    AssetManager, FileLoadResult, FileLoader, Handle, LoadedEntry, MeshSource,
};
use crate::engine::core::resource::game_path::GamePath;

/// 碰撞数据源视图：把类型无关的 [`AssetManager`] 包装成 [`MeshSource`]
/// （解读需要加载器，故在资产层实现）。
pub struct MeshView<'a> {
    manager: &'a AssetManager,
}

impl<'a> MeshView<'a> {
    pub fn new(manager: &'a AssetManager) -> Self {
        Self { manager }
    }
}

impl MeshSource for MeshView<'_> {
    fn mesh(&self, handle: Handle<Mesh>) -> Option<&Mesh> {
        self.manager.get_cached(handle)
    }
}

impl AssetManager {
    /// 从 glTF 文件加载场景（按游戏路径从合并资源空间读取）：
    /// 网格/贴图资产注册进管理器自身，返回带层级的 [`SceneTemplate`]。
    ///
    /// 需要 glTF `Document`（节点层级、变换、材质→贴图绑定），所以保留自己的
    /// 解析入口；条目注册与重载器和 [`Self::load_file`] 共用同一逻辑。
    pub fn load_scene_template(&mut self, path: &GamePath) -> Result<SceneTemplate> {
        // 先从合并资源空间读字节（借用 self.space() 在 read 后结束），
        // 之后可安全地 &mut self 注册资产。
        let bytes = self.space().read(path)?;
        let (document, _buffers, _images, glb) =
            parse_glb(&bytes).with_context(|| format!("解析 glTF 失败：{path}"))?;
        let scene = document.default_scene().context("glTF 文件没有默认场景")?;

        // 1. 注册该文件全部条目（mesh + texture）并配置重载器——与
        //    `load_file`/`load_file_async` 共用同一逻辑（`register_parsed_file`），
        //    保证重载行为一致；已注册过则直接复用句柄。
        let parsed: FileLoadResult = vec![
            (
                TypeId::of::<Mesh>(),
                glb.meshes
                    .iter()
                    .enumerate()
                    .map(|(i, mesh)| {
                        (
                            Box::new(mesh.clone()) as Box<dyn Any + Send + Sync>,
                            Box::new(i as u32) as Box<dyn Any + Send + Sync>,
                        )
                    })
                    .collect(),
            ),
            (
                TypeId::of::<Texture>(),
                glb.textures
                    .iter()
                    .enumerate()
                    .map(|(i, texture)| {
                        (
                            Box::new(texture.clone()) as Box<dyn Any + Send + Sync>,
                            Box::new(i as u32) as Box<dyn Any + Send + Sync>,
                        )
                    })
                    .collect(),
            ),
        ];
        self.register_parsed_file(GlbFileLoader, path.clone(), parsed)?;
        let mesh_handles = self.loaded_handles_of::<Mesh>(path);
        let texture_handles = self.loaded_handles_of::<Texture>(path);

        // 2. document 网格 → primitive 句柄列表（mesh_handles 是全局 primitive 顺序）。
        let mut mesh_keys: HashMap<usize, Vec<Handle<Mesh>>> = HashMap::new();
        let mut offset = 0;
        for mesh in document.meshes() {
            let count = mesh.primitives().count();
            mesh_keys.insert(mesh.index(), mesh_handles[offset..offset + count].to_vec());
            offset += count;
        }

        // 3. 场景树构建：句柄已注册，Loader 只负责层级与材质引用。
        let mut loader = Loader {
            mesh_keys,
            texture_keys: texture_handles,
        };

        let mut out = SceneTemplate::new();
        for node in scene.nodes() {
            loader.load_node(&mut out, node, None)?;
        }
        Ok(out)
    }
}

/// 加载过程中的临时状态。
struct Loader {
    /// glTF 网格索引 → 已注册的句柄列表（每个 primitive 一个）。
    mesh_keys: HashMap<usize, Vec<Handle<Mesh>>>,
    /// glTF 图片索引 → 已注册的贴图句柄（File 条目，索引即 image 下标）。
    texture_keys: Vec<Handle<Texture>>,
}

impl Loader {
    /// 递归加载 glTF 节点：每个节点一个容器物体，网格与子节点挂在它下面。
    fn load_node(
        &mut self,
        scene: &mut SceneTemplate,
        node: gltf::scene::Node<'_>,
        parent: Option<ObjectKey>,
    ) -> Result<()> {
        let (translation, rotation, scale) = match node.transform() {
            GltfTransform::Decomposed {
                translation,
                rotation,
                scale,
            } => (
                Vec3::from(translation),
                Quat::from_array(rotation),
                Vec3::from(scale),
            ),
            GltfTransform::Matrix { matrix } => {
                let (scale, rotation, translation) =
                    Mat4::from_cols_array_2d(&matrix).to_scale_rotation_translation();
                (translation, rotation, scale)
            }
        };
        let children: Vec<_> = node.children().collect();
        let mesh_materials = match node.mesh() {
            Some(mesh) => Some(self.register_mesh(&mesh)?),
            None => None,
        };
        // 既没有网格也没有子节点的空节点：跳过。
        // TODO: 拼图接口需要此类节点，届时需关闭剪枝
        if mesh_materials.is_none() && children.is_empty() {
            return Ok(());
        }
        // 容器物体：承载局部变换，无网格；primitive 与子节点都挂在它下面。
        let container = match parent {
            Some(p) => scene
                .attach(
                    p,
                    SceneObject::new(
                        SceneObjectKind::Empty,
                        Transform::new(translation, rotation, scale),
                    ),
                )
                .expect("父节点必然存活"),
            None => scene.add_object(SceneObject::new(
                SceneObjectKind::Empty,
                Transform::new(translation, rotation, scale),
            )),
        };
        // 每个 primitive 一个可绘制物体（单位局部变换，位置已由容器决定）。
        if let Some(meshes) = mesh_materials {
            for (key, material) in meshes {
                let _ = scene.attach(
                    container,
                    SceneObject::new(SceneObjectKind::Mesh(key), Transform::IDENTITY)
                        .with_material(material),
                );
            }
        }
        for child in children {
            self.load_node(scene, child, Some(container))?;
        }
        Ok(())
    }

    /// 取一个 glTF 网格各 primitive 的 (句柄, 材质) 列表（句柄在 load_scene_template 预注册）。
    fn register_mesh(&self, mesh: &gltf::Mesh<'_>) -> Result<Vec<(Handle<Mesh>, Material)>> {
        let keys = self
            .mesh_keys
            .get(&mesh.index())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(keys.len());
        for (key, primitive) in keys.iter().zip(mesh.primitives()) {
            out.push((*key, self.material_from_primitive(&primitive)?));
        }
        Ok(out)
    }

    /// 读取 primitive 的材质：基础色因子 + 基础色贴图。
    fn material_from_primitive(&self, primitive: &gltf::Primitive<'_>) -> Result<Material> {
        // gltf::Material 总是存在（未指定时是默认材质，因子为 1）。
        let gltf_material = primitive.material();
        let pbr = gltf_material.pbr_metallic_roughness();
        // 贴图已在 load_scene_template 预注册为 File 条目，这里按 image 索引直接取句柄。
        // 基础色/金属度粗糙度是 `Info`，法线是 `NormalTexture`，分开处理。
        let tex_of_info = |info: Option<gltf::texture::Info>| {
            info.map(|i| self.texture_keys[i.texture().source().index()])
        };
        let tex_of_normal = |info: Option<gltf::material::NormalTexture>| {
            info.map(|i| self.texture_keys[i.texture().source().index()])
        };
        Ok(Material {
            base_color: pbr.base_color_factor(),
            base_color_texture: tex_of_info(pbr.base_color_texture()),
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture: tex_of_info(pbr.metallic_roughness_texture()),
            normal_texture: tex_of_normal(gltf_material.normal_texture()),
        })
    }
}

/// 把一个 glTF primitive 转换成运行时 `Mesh`（顶点属性统一转 f32，索引转 u32）。
///
/// 从 [`Loader::mesh_from_primitive`] 提取，供场景加载与 [`GlbFileLoader`] 共用。
fn mesh_from_primitive(buffers: &[&[u8]], primitive: &gltf::Primitive<'_>) -> Result<Mesh> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).copied());
    let Some(positions) = reader.read_positions() else {
        bail!("primitive 缺少 POSITION 属性");
    };
    let positions: Vec<[f32; 3]> = positions.collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_default();
    let tangents: Vec<[f32; 4]> = reader
        .read_tangents()
        .map(|iter| iter.collect())
        .unwrap_or_default();
    let tex_coords: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect())
        .unwrap_or_default();
    let colors: Vec<[f32; 3]> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgb_f32().collect())
        .unwrap_or_default();
    let indices: Vec<u32> = match reader.read_indices() {
        Some(iter) => iter.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    // 只支持三角形拓扑；条带/扇形在这里转换成三角形列表。
    let indices = match primitive.mode() {
        Mode::Triangles => indices,
        Mode::TriangleStrip => strip_to_triangles(&indices),
        Mode::TriangleFan => fan_to_triangles(&indices),
        other => bail!("不支持的 primitive 模式：{other:?}"),
    };

    // 切线：文件自带 TANGENT 则直接用；否则按 MikkTSpace 计算（Blender 同款算法），
    // 无需在 Blender 手动导出切线。
    let tangents = if !tangents.is_empty() {
        tangents
    } else if !tex_coords.is_empty() {
        compute_tangents(&positions, &normals, &tex_coords, &indices)
    } else {
        vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]
    };

    let mut vertices = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        vertices.push(Vertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
            tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
            tex_coord: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
            color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
        });
    }

    Ok(Mesh::new(vertices, indices))
}

/// glTF 文件解析结果：该文件的所有网格与贴图（不含场景树）。
///
/// 内存层按 `GamePath` 缓存一份，所有条目共享；mesh / texture 的
/// `Extra` 分别是各自数组里的索引。
#[derive(Debug)]
pub struct GlbAssets {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<Texture>,
}

/// 解析 glb 字节：返回文档（场景树）与解析出的网格/贴图资产。
///
/// [`GlbFileLoader`]（文件加载）与 [`load_scene_template`]（场景加载）共用同一解析，
/// 保证两类入口产出的条目顺序一致（`GlbAssets` 数组索引即 File 条目的 Extra）。
fn parse_glb(
    bytes: &[u8],
) -> Result<(
    gltf::Document,
    Vec<gltf::buffer::Data>,
    Vec<gltf::image::Data>,
    GlbAssets,
)> {
    let (document, buffers, images) = gltf::import_slice(bytes)?;
    let buffer_slices: Vec<&[u8]> = buffers.iter().map(|b| b.0.as_slice()).collect();
    let meshes = document
        .meshes()
        .flat_map(|mesh| mesh.primitives())
        .map(|primitive| mesh_from_primitive(&buffer_slices, &primitive))
        .collect::<Result<Vec<_>>>()?;
    let textures = images
        .iter()
        .map(gltf_image_to_texture)
        .collect::<Result<Vec<_>>>()?;
    Ok((document, buffers, images, GlbAssets { meshes, textures }))
}

/// glTF 文件级加载器：一次 scan + 一次 parse 产出 glb 文件的
/// **全部类型**条目——Mesh 按 primitive 全局序、Texture 按 image 序。
///
/// - `scan` 只解析 glTF 结构（JSON 头，不读缓冲区），用于立即注册占位句柄；
/// - `parse` 完整解析（同步/异步入口共用 [`parse_glb`]）；
/// - 重载时 `extra_eq` 按 primitive/image 索引找回对应条目。
#[derive(Debug, Default, Clone)]
pub struct GlbFileLoader;

impl FileLoader for GlbFileLoader {
    fn scan(&self, bytes: &[u8]) -> Result<Vec<(TypeId, Vec<Box<dyn Any + Send + Sync>>)>> {
        // 轻量结构扫描：只解析 JSON 结构（缓冲区数据不读不解码）。
        let gltf =
            gltf::Gltf::from_slice(bytes).with_context(|| "扫描 glTF 结构失败".to_string())?;
        let document = &gltf.document;
        let mesh_extras: Vec<_> = document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .enumerate()
            .map(|(i, _)| Box::new(i as u32) as Box<dyn Any + Send + Sync>)
            .collect();
        let texture_extras: Vec<_> = document
            .images()
            .enumerate()
            .map(|(i, _)| Box::new(i as u32) as Box<dyn Any + Send + Sync>)
            .collect();
        Ok(vec![
            (TypeId::of::<Mesh>(), mesh_extras),
            (TypeId::of::<Texture>(), texture_extras),
        ])
    }

    fn parse(&self, bytes: &[u8]) -> Result<Vec<(TypeId, Vec<LoadedEntry>)>> {
        let (_, _, _, assets) = parse_glb(bytes)?;
        let meshes: Vec<LoadedEntry> = assets
            .meshes
            .into_iter()
            .enumerate()
            .map(|(i, mesh)| {
                (
                    Box::new(mesh) as Box<dyn Any + Send + Sync>,
                    Box::new(i as u32) as Box<dyn Any + Send + Sync>,
                )
            })
            .collect();
        let textures: Vec<LoadedEntry> = assets
            .textures
            .into_iter()
            .enumerate()
            .map(|(i, texture)| {
                (
                    Box::new(texture) as Box<dyn Any + Send + Sync>,
                    Box::new(i as u32) as Box<dyn Any + Send + Sync>,
                )
            })
            .collect();
        Ok(vec![
            (TypeId::of::<Mesh>(), meshes),
            (TypeId::of::<Texture>(), textures),
        ])
    }

    fn extra_eq(&self, a: &dyn Any, b: &dyn Any) -> bool {
        match (a.downcast_ref::<u32>(), b.downcast_ref::<u32>()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }
}

/// 用 MikkTSpace（Blender 同款算法）计算逐顶点切线。
fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tex_coords: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut geometry = TangentGeometry {
        positions,
        normals,
        tex_coords,
        indices,
        tangents: vec![[0.0; 4]; indices.len()],
    };
    if !mikktspace::generate_tangents(&mut geometry) {
        return vec![[1.0, 0.0, 0.0, 1.0]; positions.len()];
    }

    // 角点切线 → 逐顶点平均（xyz 累加归一化；w 按多数符号）。
    let mut sum = vec![[0.0f32; 3]; positions.len()];
    let mut w_sum = vec![0.0f32; positions.len()];
    for (corner, &index) in indices.iter().enumerate() {
        let tangent = geometry.tangents[corner];
        let i = index as usize;
        for k in 0..3 {
            sum[i][k] += tangent[k];
        }
        w_sum[i] += tangent[3];
    }

    (0..positions.len())
        .map(|i| {
            let t = sum[i];
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if len < 1e-8 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [
                    t[0] / len,
                    t[1] / len,
                    t[2] / len,
                    if w_sum[i] < 0.0 { -1.0 } else { 1.0 },
                ]
            }
        })
        .collect()
}

/// mikktspace 的几何适配器：把我们的顶点/索引喂给算法，切线输出到 `tangents`。
struct TangentGeometry<'a> {
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    tex_coords: &'a [[f32; 2]],
    indices: &'a [u32],
    /// 每个角点（face*3+vert）的切线。
    tangents: Vec<[f32; 4]>,
}

impl mikktspace::Geometry for TangentGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals
            .get(self.indices[face * 3 + vert] as usize)
            .copied()
            .unwrap_or([0.0, 0.0, 1.0])
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[face * 3 + vert] = tangent;
    }
}

/// 把 glTF 解码出的图片数据转成运行时 RGBA8 纹理。
fn gltf_image_to_texture(image: &gltf::image::Data) -> Result<Texture> {
    use gltf::image::Format;

    let pixel_count = image.width as usize * image.height as usize;
    let mut rgba8 = Vec::with_capacity(pixel_count * 4);

    fn u16_to_u8(pair: &[u8]) -> u8 {
        (u16::from_le_bytes([pair[0], pair[1]]) >> 8) as u8
    }
    fn f32_to_u8(bytes: &[u8]) -> u8 {
        let v = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        (v.clamp(0.0, 1.0) * 255.0) as u8
    }

    let pixels = &image.pixels;
    match image.format {
        Format::R8 => {
            for &v in pixels {
                rgba8.extend_from_slice(&[v, v, v, 255]);
            }
        }
        Format::R8G8 => {
            for c in pixels.chunks(2) {
                rgba8.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        Format::R8G8B8 => {
            for c in pixels.chunks(3) {
                rgba8.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Format::R8G8B8A8 => rgba8.extend_from_slice(pixels),
        Format::R16 => {
            for c in pixels.chunks(2) {
                let v = u16_to_u8(c);
                rgba8.extend_from_slice(&[v, v, v, 255]);
            }
        }
        Format::R16G16 => {
            for c in pixels.chunks(4) {
                let r = u16_to_u8(&c[0..2]);
                rgba8.extend_from_slice(&[r, r, r, u16_to_u8(&c[2..4])]);
            }
        }
        Format::R16G16B16 => {
            for c in pixels.chunks(6) {
                rgba8.extend_from_slice(&[
                    u16_to_u8(&c[0..2]),
                    u16_to_u8(&c[2..4]),
                    u16_to_u8(&c[4..6]),
                    255,
                ]);
            }
        }
        Format::R16G16B16A16 => {
            for c in pixels.chunks(8) {
                rgba8.extend_from_slice(&[
                    u16_to_u8(&c[0..2]),
                    u16_to_u8(&c[2..4]),
                    u16_to_u8(&c[4..6]),
                    u16_to_u8(&c[6..8]),
                ]);
            }
        }
        Format::R32G32B32FLOAT => {
            for c in pixels.chunks(12) {
                rgba8.extend_from_slice(&[
                    f32_to_u8(&c[0..4]),
                    f32_to_u8(&c[4..8]),
                    f32_to_u8(&c[8..12]),
                    255,
                ]);
            }
        }
        Format::R32G32B32A32FLOAT => {
            for c in pixels.chunks(16) {
                rgba8.extend_from_slice(&[
                    f32_to_u8(&c[0..4]),
                    f32_to_u8(&c[4..8]),
                    f32_to_u8(&c[8..12]),
                    f32_to_u8(&c[12..16]),
                ]);
            }
        }
    }
    if rgba8.len() != pixel_count * 4 {
        bail!("图片像素数据长度与尺寸不匹配");
    }

    Ok(Texture {
        width: image.width,
        height: image.height,
        rgba8,
    })
}

/// 三角带 → 三角形列表（交替绕序对应反向）。
fn strip_to_triangles(indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len().saturating_sub(2) * 3);
    for i in 0..indices.len().saturating_sub(2) {
        if i % 2 == 0 {
            out.extend_from_slice(&[indices[i], indices[i + 1], indices[i + 2]]);
        } else {
            out.extend_from_slice(&[indices[i + 1], indices[i], indices[i + 2]]);
        }
    }
    out
}

/// 扇形 → 三角形列表。
fn fan_to_triangles(indices: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(indices.len().saturating_sub(2) * 3);
    for i in 1..indices.len().saturating_sub(1) {
        out.extend_from_slice(&[indices[0], indices[i], indices[i + 1]]);
    }
    out
}

#[cfg(test)]
mod tests;
