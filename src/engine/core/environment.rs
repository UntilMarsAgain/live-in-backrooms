//! 环境模块：CPU 侧的环境光数据（HDRI 等距矩形图）。
//!
//! 每个关卡一份环境资产（机制上与天空盒同源），启动时一次性解码，
//! 由渲染器转换成 GPU 上的环境立方体贴图与辐照度图。

use std::error::Error;
use std::fmt;
use std::path::Path;

use glam::Vec3;

const PI: f32 = std::f32::consts::PI;

/// 环境贴图：Radiance HDR 解码后的线性 RGB 浮点像素。
#[derive(Debug, Clone)]
pub struct Environment {
    pub width: u32,
    pub height: u32,
    /// 行优先（从上到下、从左到右）的线性 RGB 像素。
    pub rgb: Vec<[f32; 3]>,
}

impl Environment {
    /// 从 Radiance HDR（.hdr）字节解码。
    pub fn from_hdr_bytes(bytes: &[u8]) -> Result<Self, EnvironmentError> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .map_err(EnvironmentError::Decode)?;
        let rgb32f = image.to_rgb32f();
        let (width, height) = rgb32f.dimensions();
        let rgb = rgb32f.pixels().map(|p| p.0).collect();
        Ok(Self {
            width,
            height,
            rgb,
        })
    }

    /// 从磁盘读取并解码 Radiance HDR 文件。
    pub fn from_hdr_file(path: &Path) -> Result<Self, EnvironmentError> {
        let bytes = std::fs::read(path).map_err(EnvironmentError::Io)?;
        Self::from_hdr_bytes(&bytes)
    }

    /// 等距矩形图 → 立方体贴图（CPU，6 层 RGBA32F）。
    ///
    /// 层序遵循 WebGPU 规范：0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z，
    /// 每层 `face_size×face_size` 行优先。上传前在 CPU 完成，
    /// 避免依赖 GPU 的 storage 纹理 / 浮点渲染目标等后端差异。
    pub fn to_cubemap(&self, face_size: u32) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; (face_size * face_size * 6) as usize];
        for face in 0..6u32 {
            for y in 0..face_size {
                for x in 0..face_size {
                    let u = (x as f32 + 0.5) / face_size as f32;
                    let v = (y as f32 + 0.5) / face_size as f32;
                    let dir = face_dir(face, u, v);
                    let (su, sv) = dir_to_equirect(dir);
                    let c = bilinear_sample(&self.rgb, self.width, self.height, su, sv);
                    let idx = ((face * face_size + y) * face_size + x) as usize;
                    out[idx] = [c[0], c[1], c[2], 1.0];
                }
            }
        }
        out
    }

    /// 立方体贴图 → 辐照度图（漫反射 IBL，CPU）。
    ///
    /// `cube` 是 [`Self::to_cubemap`] 的输出（6 层 `cube_face×cube_face` RGBA32F），
    /// 输出 `out_size×out_size×6`，每个纹素用余弦加权 Hammersley 半球采样。
    pub fn irradiance_map(
        cube: &[[f32; 4]],
        cube_face: u32,
        out_size: u32,
        samples: u32,
    ) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; (out_size * out_size * 6) as usize];
        for face in 0..6u32 {
            for y in 0..out_size {
                for x in 0..out_size {
                    let u = (x as f32 + 0.5) / out_size as f32;
                    let v = (y as f32 + 0.5) / out_size as f32;
                    let n = face_dir(face, u, v).normalize();
                    let (t, b, n_axis) = tangent_basis(n);
                    let mut acc = [0.0f32; 3];
                    for i in 0..samples {
                        let sample = cosine_sample_hemisphere(hammersley(i, samples));
                        let l = (t * sample.x + b * sample.y + n_axis * sample.z).normalize();
                        let n_dot_l = n.dot(l).max(0.0);
                        let c = sample_cube(cube, cube_face, l);
                        acc[0] += c[0] * n_dot_l;
                        acc[1] += c[1] * n_dot_l;
                        acc[2] += c[2] * n_dot_l;
                    }
                    let scale = PI / samples as f32;
                    let idx = ((face * out_size + y) * out_size + x) as usize;
                    out[idx] = [acc[0] * scale, acc[1] * scale, acc[2] * scale, 1.0];
                }
            }
        }
        out
    }
}

/// 立方体面内像素坐标 (u, v ∈ [0,1]) → 世界方向（与 environment.wgsl 中一致）。
fn face_dir(face: u32, u: f32, v: f32) -> Vec3 {
    let x = u * 2.0 - 1.0;
    let y = v * 2.0 - 1.0;
    match face {
        0 => Vec3::new(1.0, -y, -x),  // +X
        1 => Vec3::new(-1.0, -y, x),  // -X
        2 => Vec3::new(x, 1.0, y),    // +Y
        3 => Vec3::new(x, -1.0, -y),  // -Y
        4 => Vec3::new(x, -y, 1.0),   // +Z
        _ => Vec3::new(-x, -y, -1.0), // -Z
    }
}

/// 世界方向 → 等距矩形 UV（u: 经度 0..1，v: 纬度 0 顶 1 底）。
fn dir_to_equirect(dir: Vec3) -> (f32, f32) {
    let d = dir.normalize();
    let phi = d.z.atan2(d.x);
    let theta = d.y.clamp(-1.0, 1.0).acos();
    (0.5 + phi / (2.0 * PI), theta / PI)
}

/// 世界方向 → 立方体面与面内 UV。
fn dir_to_face_uv(d: Vec3) -> (u32, f32, f32) {
    let d = d.normalize();
    let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
    if ax >= ay && ax >= az {
        if d.x > 0.0 {
            (0, (1.0 - d.z / d.x) * 0.5, (1.0 - d.y / d.x) * 0.5)
        } else {
            (1, (1.0 + d.z / d.x) * 0.5, (1.0 - d.y / d.x) * 0.5)
        }
    } else if ay >= az {
        if d.y > 0.0 {
            (2, (1.0 + d.x / d.y) * 0.5, (1.0 + d.z / d.y) * 0.5)
        } else {
            (3, (1.0 + d.x / d.y) * 0.5, (1.0 - d.z / d.y) * 0.5)
        }
    } else if d.z > 0.0 {
        (4, (1.0 + d.x / d.z) * 0.5, (1.0 - d.y / d.z) * 0.5)
    } else {
        (5, (1.0 - d.x / d.z) * 0.5, (1.0 - d.y / d.z) * 0.5)
    }
}

/// 双线性采样（边缘 clamp）。`img` 行优先、每像素 3 个 f32。
fn bilinear_sample(img: &[[f32; 3]], width: u32, height: u32, u: f32, v: f32) -> [f32; 3] {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let x = u * width as f32 - 0.5;
    let y = v * height as f32 - 0.5;
    let x0 = x.floor().clamp(0.0, width as f32 - 1.0) as usize;
    let y0 = y.floor().clamp(0.0, height as f32 - 1.0) as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
    let at = |xx: usize, yy: usize| img[yy * width as usize + xx];
    let (c00, c10) = (at(x0, y0), at(x1, y0));
    let (c01, c11) = (at(x0, y1), at(x1, y1));
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let top = c00[k] * (1.0 - fx) + c10[k] * fx;
        let bottom = c01[k] * (1.0 - fx) + c11[k] * fx;
        out[k] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

/// 从立方体贴图（RGBA32F，6 层）按方向双线性采样。
fn sample_cube(cube: &[[f32; 4]], face_size: u32, dir: Vec3) -> [f32; 3] {
    let (face, u, v) = dir_to_face_uv(dir);
    let x = u * face_size as f32 - 0.5;
    let y = v * face_size as f32 - 0.5;
    let x0 = x.floor().clamp(0.0, face_size as f32 - 1.0) as usize;
    let y0 = y.floor().clamp(0.0, face_size as f32 - 1.0) as usize;
    let x1 = (x0 + 1).min(face_size as usize - 1);
    let y1 = (y0 + 1).min(face_size as usize - 1);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
    let at = |xx: usize, yy: usize| -> [f32; 3] {
        let idx = ((face as usize) * (face_size * face_size) as usize)
            + yy * face_size as usize
            + xx;
        [cube[idx][0], cube[idx][1], cube[idx][2]]
    };
    let (c00, c10) = (at(x0, y0), at(x1, y0));
    let (c01, c11) = (at(x0, y1), at(x1, y1));
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let top = c00[k] * (1.0 - fx) + c10[k] * fx;
        let bottom = c01[k] * (1.0 - fx) + c11[k] * fx;
        out[k] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

/// Hammersley 低差异序列。
fn hammersley(i: u32, count: u32) -> (f32, f32) {
    (
        i as f32 / count as f32,
        radical_inverse_vdc(i),
    )
}

fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = bits.rotate_right(16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * (1.0 / 4294967296.0)
}

/// 余弦加权半球采样。
fn cosine_sample_hemisphere((u, v): (f32, f32)) -> Vec3 {
    let phi = 2.0 * PI * u;
    let cos_theta = (1.0 - v).sqrt();
    let sin_theta = v.sqrt();
    Vec3::new(phi.cos() * sin_theta, cos_theta, phi.sin() * sin_theta)
}

/// 以法线为 +Z 构造正交基。
fn tangent_basis(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let up = if n.z.abs() > 0.999 { Vec3::X } else { Vec3::Z };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t, b, n)
}

/// 环境贴图加载错误。
#[derive(Debug)]
pub enum EnvironmentError {
    Io(std::io::Error),
    Decode(image::ImageError),
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "无法读取环境贴图文件：{e}"),
            Self::Decode(e) => write!(f, "环境贴图解码失败：{e}"),
        }
    }
}

impl Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Decode(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 仓库内如有测试 HDR（assets 目录不入库），验证解码结果合理。
    #[test]
    fn decodes_repo_hdr_if_present() {
        let path = Path::new("assets/environments/test.hdr");
        if !path.is_file() {
            return;
        }
        let env = Environment::from_hdr_file(path).expect("HDR 应能解码");
        assert!(env.width > 0 && env.height > 0);
        assert_eq!(env.rgb.len(), (env.width * env.height) as usize);
        // HDRI 不该全黑：至少有一个非零像素。
        assert!(env.rgb.iter().any(|p| p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0));
    }

    /// CPU 转换：2×1 红绿图 → 立方体贴图，所有面都应有非零数据。
    #[test]
    fn to_cubemap_writes_nonzero() {
        let env = Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        };
        let cube = env.to_cubemap(4);
        assert_eq!(cube.len(), 4 * 4 * 6);
        let max = cube.iter().fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
        assert!(max > 0.0, "立方体贴图转换输出全为零");
    }

    /// CPU 转换：辐照度图应有数据且数值有限（无 NaN/Inf）。
    #[test]
    fn irradiance_map_is_finite_and_positive() {
        // 全白环境：辐照度应均匀且为正。
        let env = Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        };
        let cube = env.to_cubemap(4);
        let irr = Environment::irradiance_map(&cube, 4, 2, 64);
        for p in &irr {
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
            assert!(p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0);
        }
    }
}
