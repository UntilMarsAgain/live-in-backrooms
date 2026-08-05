//! 资产加载测试：glTF 解析、切线计算、索引转换与端到端加载。

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
