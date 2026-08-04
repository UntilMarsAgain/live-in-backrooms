//! 网格模块：CPU 侧的顶点与索引（面）数据。
//!
//! 目前只保存静态几何数据，GPU 缓冲区的创建在 `render` 模块完成。
//! 后续可以在这里扩展法线、UV 等顶点属性。
//!
//! [`MeshLibrary`] 是全局网格资产库：只追加、永久持有、跨场景共享。
//! 句柄是库里的稠密编号（不删除因此稳定），GPU 侧把全部网格合并成
//! 一份顶点/索引缓冲永久驻留，新增网格时整体重传。

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

/// 网格句柄：在 [`MeshLibrary`] 中的稠密编号。
///
/// 库只追加不删除，因此编号稳定、不会复用；句柄即索引，渲染时一跳直达。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshKey(usize);

impl MeshKey {
    pub fn index(self) -> usize {
        self.0
    }
}

/// 全局网格资产库：只追加、永久持有。
#[derive(Debug, Default)]
pub struct MeshLibrary {
    meshes: Vec<Mesh>,
    /// 版本号：每次注册新网格 +1，供 GPU 侧判断是否需要重传。
    version: u64,
}

impl MeshLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册单个网格（内部走批量路径）。
    pub fn register(&mut self, mesh: Mesh) -> MeshKey {
        self.register_many([mesh])[0]
    }

    /// 批量注册：一次调用追加多个网格并返回各自的句柄。
    pub fn register_many(&mut self, meshes: impl IntoIterator<Item = Mesh>) -> Vec<MeshKey> {
        let start = self.meshes.len();
        self.meshes.extend(meshes);
        let keys: Vec<_> = (start..self.meshes.len()).map(MeshKey).collect();
        if !keys.is_empty() {
            self.version += 1;
        }
        keys
    }

    pub fn mesh(&self, key: MeshKey) -> Option<&Mesh> {
        self.meshes.get(key.0)
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// 全部资产（追加顺序，句柄编号即下标）。
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    /// 当前版本号：新增资产后 +1。
    pub fn version(&self) -> u64 {
        self.version
    }
}
