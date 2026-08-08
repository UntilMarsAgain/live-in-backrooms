//! 共享解析层测试：切线与索引转换等纯解析逻辑。

use super::*;

/// MikkTSpace：XY 平面三角形，UV 的 u 沿 +X，切线应为 (1,0,0,1)。
#[test]
fn compute_tangents_basic() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let normals = [[0.0, 0.0, 1.0]; 3];
    let uvs = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let tangents = compute_tangents(&positions, &normals, &uvs, &[0, 1, 2]);
    assert_eq!(tangents[0], [1.0, 0.0, 0.0, 1.0]);
}
