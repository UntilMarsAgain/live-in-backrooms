//! 资产加载与解读模块：把 glTF 2.0 文件转换成运行时的网格资产与场景。
//!
//! 目前只处理“模型”部分：
//! - 节点层级与局部变换 → [`Scene`]（每个 glTF 节点一个容器物体，网格 primitive
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
//! 本模块还提供 **typed 解读助手**：`AssetManager` 只存不解释，这里把句柄
//! 解读成 `&Mesh`/`&Texture`（`get_mesh`/`get_texture`），并负责磁盘加载与
//! 重载（`load_meshes` + 注册重载器，`get<T>` 缺失时自动回磁盘）。`MeshView` 把管理器包装成
//! [`MeshSource`] 供碰撞/调试使用。
//!
//! 相机、动画、蒙皮暂不读取。

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec3};
use gltf::mesh::Mode;
use gltf::scene::Transform as GltfTransform;

use crate::engine::core::asset::{AssetLoader, AssetManager, Handle, MeshSource};
use crate::engine::core::game_path::GamePath;
use crate::engine::core::resource::MergedResourceSpace;
use std::any::Any;
use super::core::material::Material;
use super::core::mesh::{Mesh, Vertex};
use super::core::texture::Texture;
use super::core::transform::Transform;
use super::scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};

/// 取网格 CPU 数据（只读，不触发重载；文件条目经注册的解读器取回）。
pub fn get_mesh<'a>(manager: &'a AssetManager, handle: Handle<Mesh>) -> Option<&'a Mesh> {
    manager.get_cached(handle)
}

/// 取贴图 CPU 数据（同上）。
pub fn get_texture<'a>(manager: &'a AssetManager, handle: Handle<Texture>) -> Option<&'a Texture> {
    manager.get_cached(handle)
}

/// 文件重载器（泛型）：条目数据缺失时**完整重解析**文件，按 `extra` 取回对应条目。
fn file_reloader<T, L>(
    loader: L,
    path: GamePath,
) -> Box<
    dyn Fn(&MergedResourceSpace, &dyn Any) -> anyhow::Result<Box<dyn Any + Send + Sync>>
        + Send
        + Sync,
>
where
    T: Any + Send + Sync + 'static,
    L: AssetLoader<T> + Clone + Send + Sync + 'static,
{
    Box::new(move |space: &MergedResourceSpace, extra: &dyn Any| {
        let parsed = <L as AssetLoader<T>>::load(&loader, space, &path)?;
        let entries = <L as AssetLoader<T>>::entries(&loader, &parsed);
        let extra = extra
            .downcast_ref::<L::Extra>()
            .ok_or_else(|| anyhow::anyhow!("重载定位信息类型不符"))?;
        entries
            .into_iter()
            .find(|(_data, e)| e == extra)
            .map(|(data, _)| Box::new(data) as Box<dyn Any + Send + Sync>)
            .ok_or_else(|| anyhow::anyhow!("重载时找不到对应条目"))
    })
}

/// **一次加载、多次注册（B1.1）**：按 `GamePath` 加载，解析一次并缓存进
/// 共享存储，然后把该文件的全部条目注册为 `T` 类型句柄。
///
/// - 同文件已被加载过（mesh/texture 各自调用本函数）时复用缓存，不重复解析；
/// - 自动配置解读器与重载器，`get<T>` 取用/重载无需额外设置；
/// - 文件解析结果的生命周期由引用计数管理（B1.2）：最后一条条目被
///   `remove` 时整份释放。
pub fn load_entries<T, L>(
    manager: &mut AssetManager,
    loader: &L,
    path: &GamePath,
) -> Result<Vec<Handle<T>>>
where
    T: Any + Send + Sync + 'static,
    L: AssetLoader<T> + Clone + Send + Sync + 'static,
{
    // 去重：同路径同类型已加载 → 直接复用句柄，不再解析。
    let existing = manager.loaded_handles_of::<T>(path);
    if !existing.is_empty() {
        return Ok(existing);
    }
    // 1. 完整解析一次，条目数据拷贝进槽位（解析结果随即丢弃——单一存储点）。
    let parsed = loader.load(manager.space(), path)?;
    let entries = loader.entries(&parsed);
    // 2. 配置文件重载器（数据逐出后重读用）。
    manager.set_file_reloader(
        path.clone(),
        file_reloader::<T, L>(loader.clone(), path.clone()),
    );
    // 3. 注册全部条目：数据移入槽位。
    Ok(entries
        .into_iter()
        .map(|(data, extra)| manager.register_file::<T>(path.clone(), Box::new(extra), data))
        .collect())
}

/// 从磁盘加载网格（`load_entries` 的便捷封装）。
pub fn load_meshes(manager: &mut AssetManager, path: &GamePath) -> Result<Vec<Handle<Mesh>>> {
    load_entries::<Mesh, GlbLoader>(manager, &GlbLoader, path)
}

/// 从磁盘加载贴图（同上）。
pub fn load_textures(manager: &mut AssetManager, path: &GamePath) -> Result<Vec<Handle<Texture>>> {
    load_entries::<Texture, GlbLoader>(manager, &GlbLoader, path)
}

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
        get_mesh(self.manager, handle)
    }
}

/// 从 glTF 文件加载场景（按游戏路径从合并资源空间读取）：
/// 网格/贴图资产注册进 `assets`，返回带层级的 `Scene`。
pub fn load_scene(
    path: &GamePath,
    assets: &mut AssetManager,
) -> Result<Scene> {
    // 先从合并资源空间读字节（借用 assets.space() 在 read 后结束），
    // 之后可安全地 &mut assets 注册资产。
    let bytes = assets.space().read(path)?;
    let (document, _buffers, _images, glb) =
        parse_glb(&bytes).with_context(|| format!("解析 glTF 失败：{path}"))?;
    let scene = document.default_scene().context("glTF 文件没有默认场景")?;

    // 1. 解析结果进内存层 + 注册 File 条目（mesh / texture 各按数组索引，
    //    与 [`GlbLoader`] 的 entries 顺序一致，可重载）。
    assets.set_file_reloader(
        path.clone(),
        file_reloader::<Mesh, GlbLoader>(GlbLoader, path.clone()),
    );
    let mesh_handles: Vec<Handle<Mesh>> = {
        let existing = assets.loaded_handles_of::<Mesh>(path);
        if !existing.is_empty() {
            existing
        } else {
            glb.meshes
                .iter()
                .enumerate()
                .map(|(i, mesh)| {
                    // 单一存储点：数据拷贝进槽位，解析结果 glb 随后丢弃。
                    assets.register_file::<Mesh>(path.clone(), Box::new(i as u32), mesh.clone())
                })
                .collect()
        }
    };
    let texture_handles: Vec<Handle<Texture>> = {
        let existing = assets.loaded_handles_of::<Texture>(path);
        if !existing.is_empty() {
            existing
        } else {
            glb.textures
                .iter()
                .enumerate()
                .map(|(i, texture)| {
                    assets
                        .register_file::<Texture>(path.clone(), Box::new(i as u32), texture.clone())
                })
                .collect()
        }
    };

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

    let mut out = Scene::new();
    for node in scene.nodes() {
        loader.load_node(&mut out, node, None)?;
    }
    Ok(out)
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
        scene: &mut Scene,
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

    /// 取一个 glTF 网格各 primitive 的 (句柄, 材质) 列表（句柄在 load_scene 预注册）。
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
        // 贴图已在 load_scene 预注册为 File 条目，这里按 image 索引直接取句柄。
        // 基础色/金属度粗糙度是 `Info`，法线是 `NormalTexture`，分开处理。
        let tex_of_info =
            |info: Option<gltf::texture::Info>| info.map(|i| self.texture_keys[i.texture().source().index()]);
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
/// 从 [`Loader::mesh_from_primitive`] 提取，供场景加载与 [`GlbLoader`] 共用。
fn mesh_from_primitive(
    buffers: &[&[u8]],
    primitive: &gltf::Primitive<'_>,
) -> Result<Mesh> {
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
/// [`GlbLoader`]（按文件加载）与 [`load_scene`]（场景加载）共用同一解析，
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

/// glTF 加载器：一个 `.glb` 文件 → 多个 Mesh + 多个 Texture。
///
/// 实现 [`AssetLoader`] 的两个实例（mesh 与 texture）：文件解析一次、
/// 结果进内存层，两类条目分别按各自数组索引定位。
#[derive(Debug, Default, Clone)]
pub struct GlbLoader;

impl AssetLoader<Mesh> for GlbLoader {
    type Extra = u32;
    type Parsed = GlbAssets;

    fn load(&self, space: &MergedResourceSpace, path: &GamePath) -> Result<GlbAssets> {
        let bytes = space.read(path)?;
        let (_, _, _, assets) = parse_glb(&bytes)
            .with_context(|| format!("解析 glTF 失败：{path}"))?;
        Ok(assets)
    }

    fn entries(&self, parsed: &GlbAssets) -> Vec<(Mesh, u32)> {
        parsed
            .meshes
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i as u32))
            .collect()
    }

    fn entry<'a>(&self, parsed: &'a GlbAssets, extra: &u32) -> Option<&'a Mesh> {
        parsed.meshes.get(*extra as usize)
    }
}

impl AssetLoader<Texture> for GlbLoader {
    type Extra = u32;
    type Parsed = GlbAssets;

    fn load(&self, space: &MergedResourceSpace, path: &GamePath) -> Result<GlbAssets> {
        <Self as AssetLoader<Mesh>>::load(self, space, path)
    }

    fn entries(&self, parsed: &GlbAssets) -> Vec<(Texture, u32)> {
        parsed
            .textures
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect()
    }

    fn entry<'a>(&self, parsed: &'a GlbAssets, extra: &u32) -> Option<&'a Texture> {
        parsed.textures.get(*extra as usize)
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
