//! 轴对齐包围盒（AABB）：碰撞、剔除、尺寸查询共用的小数学类型。
//!
//! 局部 AABB 由 [`super::mesh::Mesh`] 从顶点计算并持有；世界 AABB =
//! 局部 AABB 经场景对象世界变换后的包围盒（见 [`super::super::scene::SceneTemplate`]
//! 的碰撞查询）。

use glam::{Mat4, Vec3};

use super::transform::Transform;

/// 轴对齐包围盒：`min` / `max` 两个对角点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// 空盒：不包含任何点（用于"尚无数据"的默认值）。
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// 从点集构建；空迭代器返回 [`Self::EMPTY`]。
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut out = Self::EMPTY;
        for p in points {
            out.min = out.min.min(p);
            out.max = out.max.max(p);
        }
        out
    }

    /// 以中心 + 半尺寸构建（常用于玩家等"盒子"碰撞体）。
    pub fn from_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self::new(center - half_extents, center + half_extents)
    }

    #[allow(dead_code)] // 公共数学 API：碰撞解析/剔除中心用，暂无调用方
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    #[allow(dead_code)] // 公共数学 API：资产尺寸校验用，暂无调用方
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// 是否为空盒（任一轴反向即空）。
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// 点是否在盒内（闭区间，含边界）。
    pub fn contains(&self, point: Vec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// 与另一个 AABB 是否相交（边界相接也算相交）。
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// 经 [`Transform`] 变换后的 AABB：8 个角点变换后重取包围盒。
    ///
    /// 旋转会让包围盒变大，这是 AABB 的正常行为；角点法对任意矩阵
    /// （含负缩放）都正确，比"只变换中心+半尺寸"健壮。
    pub fn transformed(&self, transform: &Transform) -> Self {
        self.transformed_by(&transform.to_mat4())
    }

    /// 经任意 4×4 矩阵变换后的 AABB。
    pub fn transformed_by(&self, mat: &Mat4) -> Self {
        let mut out = Self::EMPTY;
        for corner in [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ] {
            let p = mat.transform_point3(corner);
            out.min = out.min.min(p);
            out.max = out.max.max(p);
        }
        out
    }
}

impl Default for Aabb {
    /// 默认即空盒：`Mesh::default()` 等"尚无数据"的场合语义正确。
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn empty_contains_nothing_and_intersects_nothing() {
        let empty = Aabb::EMPTY;
        assert!(empty.is_empty());
        assert!(!empty.contains(Vec3::ZERO));
        assert!(!empty.intersects(&Aabb::from_half_extents(Vec3::ZERO, Vec3::ONE)));
    }

    #[test]
    fn contains_is_inclusive_on_boundary() {
        let aabb = Aabb::from_half_extents(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains(Vec3::ZERO));
        assert!(aabb.contains(Vec3::new(1.0, 1.0, 1.0))); // 边界上也算
        assert!(!aabb.contains(Vec3::new(1.01, 0.0, 0.0)));
    }

    #[test]
    fn intersects_axis_separated() {
        let a = Aabb::from_half_extents(Vec3::ZERO, Vec3::ONE);
        let touching = Aabb::from_half_extents(Vec3::new(2.0, 0.0, 0.0), Vec3::ONE);
        let apart = Aabb::from_half_extents(Vec3::new(2.1, 0.0, 0.0), Vec3::ONE);
        assert!(a.intersects(&touching), "边界相接算相交");
        assert!(!a.intersects(&apart));
    }

    #[test]
    fn transformed_rotated_box_grows() {
        // 边长 2 的立方体绕 Y 转 45°：对角方向 AABB 变大。
        let local = Aabb::from_half_extents(Vec3::ZERO, Vec3::ONE);
        let world = local.transformed(&Transform::new(
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            Vec3::ONE,
        ));
        let diagonal = 2.0_f32.sqrt();
        assert!((world.half_extents().x - diagonal).abs() < 1e-4);
        assert!((world.half_extents().z - diagonal).abs() < 1e-4);
        // 绕 Y 转不改变 Y 方向尺寸。
        assert!((world.half_extents().y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_points_tracks_min_max() {
        let aabb = Aabb::from_points([
            Vec3::new(1.0, -2.0, 3.0),
            Vec3::new(-4.0, 5.0, 0.5),
            Vec3::new(2.0, 2.0, -3.0),
        ]);
        assert_eq!(aabb.min, Vec3::new(-4.0, -2.0, -3.0));
        assert_eq!(aabb.max, Vec3::new(2.0, 5.0, 3.0));
    }
}
