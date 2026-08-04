//! 资产加载模块：把 glTF 2.0 文件转换成运行时的网格资产与场景。
//!
//! 目前只处理“模型”部分：
//! - 节点层级与局部变换 → [`Scene`]（每个 glTF 节点一个容器物体，网格 primitive
//!   作为其子物体，保证“父节点动、子节点跟着动”的层级关系）；
//! - 每个 primitive → [`Mesh`] 注册进 [`MeshLibrary`]（按网格索引去重，多节点
//!   共享同一网格时不会重复上传）；
//! - 顶点属性按 glTF 2.0 语义读取：POSITION（必需）、NORMAL、TEXCOORD_0、
//!   COLOR_0，各种存储格式（f32 / 归一化整数）统一转换成运行时的 f32 布局。
//!
//! 相机、灯光、材质、动画、蒙皮暂不读取（`gltf::import` 返回的 images 也被忽略）。

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;

use glam::{Mat4, Quat, Vec3};
use gltf::mesh::Mode;
use gltf::scene::Transform as GltfTransform;

use crate::mesh::{Mesh, MeshKey, MeshLibrary, Vertex};
use crate::scene::{ObjectKey, Scene, SceneObject, SceneObjectKind};
use crate::transform::Transform;

/// 从 glTF 文件加载场景：网格资产注册进 `mesh_library`，返回带层级的 `Scene`。
pub fn load_scene(path: &Path, mesh_library: &mut MeshLibrary) -> Result<Scene, LoaderError> {
    let (document, buffers, _images) = gltf::import(path)?;
    let scene = document
        .default_scene()
        .ok_or(LoaderError::NoDefaultScene)?;

    let buffer_slices: Vec<&[u8]> = buffers.iter().map(|b| b.0.as_slice()).collect();
    let mut loader = Loader {
        mesh_library,
        buffers: &buffer_slices,
        mesh_keys: HashMap::new(),
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
    buffers: &'a [&'a [u8]],
    /// glTF 网格索引 → 已注册的 MeshKey 列表（每个 primitive 一个）。
    mesh_keys: HashMap<usize, Vec<MeshKey>>,
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
        let mesh_keys = match node.mesh() {
            Some(mesh) => Some(self.register_mesh(&mesh)?),
            None => None,
        };
        // 既没有网格也没有子节点的空节点：跳过。
        if mesh_keys.is_none() && children.is_empty() {
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
        if let Some(keys) = mesh_keys {
            for key in keys {
                let _ = scene.attach(
                    container,
                    SceneObject::new(SceneObjectKind::Mesh(key), Transform::IDENTITY),
                );
            }
        }
        for child in children {
            self.load_node(scene, child, Some(container))?;
        }
        Ok(())
    }

    /// 注册一个 glTF 网格（按网格索引去重），返回每个 primitive 对应的 MeshKey。
    fn register_mesh(&mut self, mesh: &gltf::Mesh<'_>) -> Result<Vec<MeshKey>, LoaderError> {
        if let Some(keys) = self.mesh_keys.get(&mesh.index()) {
            return Ok(keys.clone());
        }
        let mut meshes = Vec::new();
        for primitive in mesh.primitives() {
            meshes.push(self.mesh_from_primitive(&primitive)?);
        }
        let keys = self.mesh_library.register_many(meshes);
        self.mesh_keys.insert(mesh.index(), keys.clone());
        Ok(keys)
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

        let mut vertices = Vec::with_capacity(positions.len());
        for i in 0..positions.len() {
            vertices.push(Vertex {
                position: positions[i],
                normal: normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                tex_coord: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
                color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
            });
        }

        // 只支持三角形拓扑；条带/扇形在这里转换成三角形列表。
        let indices = match primitive.mode() {
            Mode::Triangles => indices,
            Mode::TriangleStrip => strip_to_triangles(&indices),
            Mode::TriangleFan => fan_to_triangles(&indices),
            other => return Err(LoaderError::UnsupportedMode(other)),
        };

        Ok(Mesh::new(vertices, indices))
    }
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
    UnsupportedMode(Mode),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gltf(e) => write!(f, "glTF 解析失败：{e}"),
            Self::NoDefaultScene => write!(f, "glTF 文件没有默认场景"),
            Self::MissingPositions => write!(f, "primitive 缺少 POSITION 属性"),
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
    use crate::mesh::MeshLibrary;

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
        let scene = load_scene(&path, &mut library).expect("应能加载测试三角形");

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
        let scene = load_scene(Path::new("src/asset/test.glb"), &mut library)
            .expect("仓库内的测试资产应能加载");
        assert_eq!(library.len(), 1);
        assert_eq!(scene.object_count(), 2); // 1 个容器节点 + 1 个 primitive 子节点
        let mesh = &library.meshes()[0];
        assert_eq!(mesh.vertices().len(), 524);
        assert_eq!(mesh.indices().len(), 3024);
    }
}
