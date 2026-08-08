//! 资产解析模块（共享）：把 glTF 2.0 文件解析成运行时的网格与贴图数据。
//!
//! 本模块只做**纯解析**，不接触 GPU 与资产管理器：
//! - [`parse_glb`]：一个 `.glb` 文件 → 文档（场景骨架）+ [`GlbAssets`]
//!   （网格/贴图数组，数组索引即 File 条目的定位信息）；
//! - [`GlbLoader`]：实现 [`AssetLoader`]，供资产管理器按 `GamePath` 从磁盘
//!   加载/重载（客户端用于上传 GPU，服务端用于碰撞等 CPU 数据）；
//! - 顶点属性按 glTF 2.0 语义读取：POSITION（必需）、NORMAL、TEXCOORD_0、
//!   COLOR_0，各种存储格式统一转换成运行时的 f32 布局；切线文件自带则用，
//!   否则按 MikkTSpace（Blender 同款算法）计算。
//!
//! 场景树组装与资产注册在客户端（lbr-client 的 asset 模块，需要 GPU 资产管理器）；
//! 相机、动画、蒙皮暂不读取。

use anyhow::{Context, Result, bail};
use gltf::mesh::Mode;

use super::core::asset::AssetLoader;
use super::core::game_path::GamePath;
use super::core::mesh::{Mesh, Vertex};
use super::core::resource::MergedResourceSpace;
use super::core::texture::Texture;

/// 把一个 glTF primitive 转换成运行时 `Mesh`（顶点属性统一转 f32，索引转 u32）。
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
/// [`GlbLoader`]（按文件加载）与客户端 `load_scene`（场景加载）共用同一解析，
/// 保证两类入口产出的条目顺序一致（`GlbAssets` 数组索引即 File 条目的 Extra）。
pub fn parse_glb(
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
#[derive(Debug, Default)]
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
