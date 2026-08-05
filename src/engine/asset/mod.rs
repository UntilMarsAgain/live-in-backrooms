//! 资产加载模块：把 glTF 2.0 文件转换成运行时的网格资产与场景。
//!
//! 目前只处理“模型”部分：
//! - 节点层级与局部变换 → [`Scene`]（每个 glTF 节点一个容器物体，网格 primitive
//!   作为其子物体，保证“父节点动、子节点跟着动”的层级关系）；
//! - 每个 primitive → [`Mesh`] 注册进 [`MeshLibrary`]（按网格索引去重，多节点
//!   共享同一网格时不会重复上传）；
//! - 顶点属性按 glTF 2.0 语义读取：POSITION（必需）、NORMAL、TEXCOORD_0、
//!   COLOR_0，各种存储格式（f32 / 归一化整数）统一转换成运行时的 f32 布局。
//! - 材质基础色（`baseColorFactor` + `baseColorTexture`）读取，贴图注册进
//!   [`TextureLibrary`]；金属度/粗糙度、法线、自发光等 PBR 通道暂不读取。
//!
//! 相机、动画、蒙皮暂不读取。

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

use glam::{Mat4, Quat, Vec3};
use gltf::mesh::Mode;
use gltf::scene::Transform as GltfTransform;

use super::core::material::Material;
use super::core::mesh::{Mesh, MeshKey, MeshLibrary, Vertex};
use super::core::texture::{Texture, TextureKey, TextureLibrary};
use super::core::transform::Transform;
use super::scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};

/// 从 glTF 文件加载场景：网格资产注册进 `mesh_library`，返回带层级的 `Scene`。
pub fn load_scene(
    path: &Path,
    mesh_library: &mut MeshLibrary,
    texture_library: &mut TextureLibrary,
) -> Result<Scene, LoaderError> {
    let (document, buffers, images) = gltf::import(path)?;
    let scene = document
        .default_scene()
        .ok_or(LoaderError::NoDefaultScene)?;

    let buffer_slices: Vec<&[u8]> = buffers.iter().map(|b| b.0.as_slice()).collect();
    let mut loader = Loader {
        mesh_library,
        texture_library,
        buffers: &buffer_slices,
        images: &images,
        mesh_keys: HashMap::new(),
        texture_keys: HashMap::new(),
    };

    let mut out = Scene::new();
    for node in scene.nodes() {
        loader.load_node(&mut out, node, None)?;
    }
    Ok(out)
}

/// 加载过程中的临时状态。
struct Loader<'a> {
    mesh_library: &'a mut MeshLibrary,
    texture_library: &'a mut TextureLibrary,
    buffers: &'a [&'a [u8]],
    images: &'a [gltf::image::Data],
    /// glTF 网格索引 → 已注册的 MeshKey 列表（每个 primitive 一个）。
    mesh_keys: HashMap<usize, Vec<MeshKey>>,
    /// glTF 图片索引 → 已注册的 TextureKey（按图去重）。
    texture_keys: HashMap<usize, TextureKey>,
}

impl Loader<'_> {
    /// 递归加载 glTF 节点：每个节点一个容器物体，网格与子节点挂在它下面。
    fn load_node(
        &mut self,
        scene: &mut Scene,
        node: gltf::scene::Node<'_>,
        parent: Option<ObjectKey>,
    ) -> Result<(), LoaderError> {
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

    /// 注册一个 glTF 网格（按网格索引去重），返回每个 primitive 对应的
    /// (MeshKey, Material) 列表。
    fn register_mesh(
        &mut self,
        mesh: &gltf::Mesh<'_>,
    ) -> Result<Vec<(MeshKey, Material)>, LoaderError> {
        if self.mesh_keys.contains_key(&mesh.index()) {
            // 材质属于 primitive，不随网格去重缓存，需要重新读取。
            return self.materials_for(mesh);
        }
        let mut meshes = Vec::new();
        let mut materials = Vec::new();
        for primitive in mesh.primitives() {
            meshes.push(self.mesh_from_primitive(&primitive)?);
            materials.push(self.material_from_primitive(&primitive)?);
        }
        let keys = self.mesh_library.register_many(meshes);
        self.mesh_keys.insert(mesh.index(), keys.clone());
        Ok(keys.into_iter().zip(materials).collect())
    }

    /// 重新读取一个网格所有 primitive 的材质（去重缓存只存了 MeshKey）。
    fn materials_for(
        &mut self,
        mesh: &gltf::Mesh<'_>,
    ) -> Result<Vec<(MeshKey, Material)>, LoaderError> {
        // 走与 register_mesh 相同的路径需要 &mut self（贴图注册），这里直接重新构造。
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
    fn material_from_primitive(
        &mut self,
        primitive: &gltf::Primitive<'_>,
    ) -> Result<Material, LoaderError> {
        // gltf::Material 总是存在（未指定时是默认材质，因子为 1）。
        let gltf_material = primitive.material();
        let pbr = gltf_material.pbr_metallic_roughness();
        let base_color_texture = match pbr.base_color_texture() {
            Some(info) => Some(self.register_texture(info.texture().source().index())?),
            None => None,
        };
        let metallic_roughness_texture = match pbr.metallic_roughness_texture() {
            Some(info) => Some(self.register_texture(info.texture().source().index())?),
            None => None,
        };
        let normal_texture = match gltf_material.normal_texture() {
            Some(info) => Some(self.register_texture(info.texture().source().index())?),
            None => None,
        };
        Ok(Material {
            base_color: pbr.base_color_factor(),
            base_color_texture,
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            metallic_roughness_texture,
            normal_texture,
        })
    }

    /// 注册 glTF 图片（按图片索引去重）。
    fn register_texture(&mut self, image_index: usize) -> Result<TextureKey, LoaderError> {
        if let Some(key) = self.texture_keys.get(&image_index) {
            return Ok(*key);
        }
        let image = self
            .images
            .get(image_index)
            .ok_or(LoaderError::MissingImage(image_index))?;
        let texture = gltf_image_to_texture(image)?;
        let key = self.texture_library.register_many([texture])[0];
        self.texture_keys.insert(image_index, key);
        Ok(key)
    }

    /// 把一个 primitive 转换成运行时 `Mesh`（顶点属性统一转 f32，索引转 u32）。
    fn mesh_from_primitive(&self, primitive: &gltf::Primitive<'_>) -> Result<Mesh, LoaderError> {
        let reader = primitive.reader(|buffer| self.buffers.get(buffer.index()).copied());
        let Some(positions) = reader.read_positions() else {
            return Err(LoaderError::MissingPositions);
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
            other => return Err(LoaderError::UnsupportedMode(other)),
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
fn gltf_image_to_texture(image: &gltf::image::Data) -> Result<Texture, LoaderError> {
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
        return Err(LoaderError::ImageDataLengthMismatch);
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

/// 加载过程中的错误。
#[derive(Debug)]
pub enum LoaderError {
    Gltf(gltf::Error),
    NoDefaultScene,
    MissingPositions,
    MissingImage(usize),
    ImageDataLengthMismatch,
    UnsupportedMode(Mode),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gltf(e) => write!(f, "glTF 解析失败：{e}"),
            Self::NoDefaultScene => write!(f, "glTF 文件没有默认场景"),
            Self::MissingPositions => write!(f, "primitive 缺少 POSITION 属性"),
            Self::MissingImage(index) => write!(f, "图片索引 {index} 不存在"),
            Self::ImageDataLengthMismatch => write!(f, "图片像素数据长度与尺寸不匹配"),
            Self::UnsupportedMode(mode) => write!(f, "不支持的 primitive 模式：{mode:?}"),
        }
    }
}

impl Error for LoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Gltf(e) => Some(e),
            _ => None,
        }
    }
}

impl From<gltf::Error> for LoaderError {
    fn from(error: gltf::Error) -> Self {
        Self::Gltf(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::core::mesh::MeshLibrary;

    /// 一个带位置/法线/UV/顶点色和索引的三角形 glTF（TRIANGLES 模式）。
    const TRIANGLE_JSON: &str = r#"{
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [ { "nodes": [ 0 ] } ],
        "nodes": [ { "mesh": 0 } ],
        "meshes": [ { "primitives": [ {
            "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "COLOR_0": 3 },
            "indices": 4
        } ] } ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.0] },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 4, "componentType": 5123, "count": 3, "type": "SCALAR" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 72, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 96, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 132, "byteLength": 6 }
        ],
        "buffers": [ { "byteLength": 138 } ]
    }"#;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// 把 JSON + 二进制拼成一个合法的 .glb 文件字节序列。
    fn glb_bytes(json: &str, bin: &[u8]) -> Vec<u8> {
        let mut json_pad = json.as_bytes().to_vec();
        while json_pad.len() % 4 != 0 {
            json_pad.push(b' ');
        }
        let bin_len = bin.len().div_ceil(4) * 4;
        let total = 12 + 8 + json_pad.len() + 8 + bin_len;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_pad.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json_pad);
        out.extend_from_slice(&(bin_len as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(bin);
        out.resize(total, 0);
        out
    }

    fn triangle_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        // 3 × vec3 位置（36 字节）
        bin.extend(f32_bytes(&[-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0]));
        // 3 × vec3 法线（36 字节）
        bin.extend(f32_bytes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
        // 3 × vec2 UV（24 字节）
        bin.extend(f32_bytes(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0]));
        // 3 × vec3 顶点色（36 字节）
        bin.extend(f32_bytes(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
        // 3 × u16 索引（6 字节）
        for index in [0u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        bin
    }

    #[test]
    fn load_triangle_glb() {
        let bin = triangle_bin();
        assert_eq!(bin.len(), 138);
        let bytes = glb_bytes(TRIANGLE_JSON, &bin);
        let path = std::env::temp_dir().join("live-in-backrooms-test-triangle.glb");
        std::fs::write(&path, &bytes).expect("写测试文件");

        let mut library = MeshLibrary::new();
        let mut textures = TextureLibrary::new();
        let scene = load_scene(&path, &mut library, &mut textures).expect("应能加载测试三角形");

        // 网格：3 个顶点、3 个索引，属性值原样转换。
        assert_eq!(library.len(), 1);
        let mesh = &library.meshes()[0];
        assert_eq!(mesh.vertices().len(), 3);
        assert_eq!(mesh.indices(), &[0, 1, 2]);
        assert_eq!(mesh.vertices()[0].position, [-0.5, -0.5, 0.0]);
        assert_eq!(mesh.vertices()[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(mesh.vertices()[0].tex_coord, [0.0, 0.0]);
        assert_eq!(mesh.vertices()[2].color, [0.0, 0.0, 1.0]);

        // 层级：1 个根容器 + 1 个 primitive 子节点，世界变换为单位。
        assert_eq!(scene.object_count(), 2);
        let roots: Vec<_> = scene.roots().collect();
        assert_eq!(roots.len(), 1);
        let root = roots[0].0;
        assert_eq!(scene.object(root).unwrap().kind, SceneObjectKind::Empty);
        let children: Vec<_> = scene.children_of(root).collect();
        assert_eq!(children.len(), 1);
        assert!(matches!(
            scene.object(children[0]).unwrap().kind,
            SceneObjectKind::Mesh(_)
        ));
        let (_, _, translation) = scene
            .world_transform(children[0])
            .unwrap()
            .to_scale_rotation_translation();
        assert_eq!(translation, Vec3::ZERO);
    }

    #[test]
    fn load_repo_test_glb() {
        let mut library = MeshLibrary::new();
        let mut textures = TextureLibrary::new();
        let scene = load_scene(
            Path::new("src/engine/asset/test.glb"),
            &mut library,
            &mut textures,
        )
        .expect("仓库内的测试资产应能加载");
        assert!(!library.is_empty());
        assert!(!textures.is_empty(), "PBR 样例应带基础色贴图");
        assert!(scene.object_count() > 0);
        // PBR 材质数据应完整：至少一个网格物体带金属度/粗糙度贴图和法线贴图。
        let pbr_material = scene.objects().find_map(|(_, object)| {
            let mat = &object.material;
            (object.mesh_key().is_some()
                && mat.metallic_roughness_texture.is_some()
                && mat.normal_texture.is_some())
            .then_some(mat)
        });
        assert!(
            pbr_material.is_some(),
            "test.glb 应带 metallic-roughness 和 normal 贴图"
        );
    }

    /// MikkTSpace：XY 平面三角形，UV 的 u 沿 +X，切线应为 (1,0,0,1)。
    #[test]
    fn compute_tangents_basic() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = [[0.0, 0.0, 1.0]; 3];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tangents = compute_tangents(&positions, &normals, &uvs, &[0, 1, 2]);
        assert_eq!(tangents[0], [1.0, 0.0, 0.0, 1.0]);
    }
}
