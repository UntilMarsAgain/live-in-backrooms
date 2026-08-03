//! 网格模块：CPU 侧的顶点与索引（面）数据。
//!
//! 目前只保存静态几何数据，GPU 缓冲区的创建在 `render` 模块完成。
//! 后续可以在这里扩展法线、UV 等顶点属性。

use bytemuck::{Pod, Zeroable};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

/// 顶点：位置 + 颜色。
///
/// 本阶段没有物体变换，位置即世界坐标。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// 顶点缓冲布局：描述 wgpu 如何从该顶点类型读取数据。
    ///
    /// 与顶点数据绑定，放在这里而不放在 render 模块，方便其他渲染路径复用。
    pub fn layout() -> VertexBufferLayout<'static> {
        const ATTRIBUTES: [VertexAttribute; 2] = [
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ];
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

/// 网格：顶点数组 + 索引数组（索引即"面"的定义）。
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// 示例三角形：红绿蓝三色。
    ///
    /// 索引按逆时针绕序组织（配合渲染管线的背面剔除约定）。
    pub fn triangle() -> Self {
        Self::new(
            vec![
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                },
            ],
            vec![0, 1, 2],
        )
    }

    /// 示例四边形：4 个顶点、2 个三角形面（6 个索引），逆时针绕序。
    ///
    /// 用来验证调色盘里存在顶点/面数量不同的网格时，区间绘制仍正确。
    pub fn quad() -> Self {
        Self::new(
            vec![
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.9, 0.85, 0.6],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.9, 0.85, 0.6],
                },
                Vertex {
                    position: [-0.5, 0.5, 0.0],
                    color: [0.9, 0.85, 0.6],
                },
                Vertex {
                    position: [0.5, 0.5, 0.0],
                    color: [0.9, 0.85, 0.6],
                },
            ],
            vec![0, 1, 2, 2, 1, 3],
        )
    }

    /// 示例立方体：6 个面、每面独立 4 个顶点（24 顶点、36 索引），六面六色。
    ///
    /// 顶点绕序从面外侧看为逆时针（配合背面剔除）；每面独立顶点是为了
    /// 以后挂法线和 UV 时每个面可以有自己的朝向。
    pub fn cube() -> Self {
        let s = 0.5;
        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        let mut push_face = |corners: [(f32, f32, f32); 4], color: [f32; 3]| {
            let base = vertices.len() as u32;
            for (x, y, z) in corners {
                vertices.push(Vertex {
                    position: [x, y, z],
                    color,
                });
            }
            // 四角按逆时针排列，两个三角形共用对角线（base, base+2）。
            indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);
        };

        // 六面（从外侧看逆时针）：+Z 红、-Z 绿、+X 蓝、-X 黄、+Y 青、-Y 品红。
        push_face(
            [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)],
            [1.0, 0.0, 0.0],
        );
        push_face(
            [(-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)],
            [0.0, 1.0, 0.0],
        );
        push_face(
            [(s, -s, -s), (s, s, -s), (s, s, s), (s, -s, s)],
            [0.0, 0.0, 1.0],
        );
        push_face(
            [(-s, -s, s), (-s, s, s), (-s, s, -s), (-s, -s, -s)],
            [1.0, 1.0, 0.0],
        );
        push_face(
            [(-s, s, s), (s, s, s), (s, s, -s), (-s, s, -s)],
            [0.0, 1.0, 1.0],
        );
        push_face(
            [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)],
            [1.0, 0.0, 1.0],
        );

        Self::new(vertices, indices)
    }
}
