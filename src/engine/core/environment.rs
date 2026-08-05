//! 环境模块：CPU 侧的环境光数据（等距矩形环境贴图）。
//!
//! 每个关卡一份环境资产（机制上与天空盒同源），启动时一次性解码，
//! 由渲染器转换成 GPU 上的环境立方体贴图与辐照度图。
//!
//! 输入格式：
//! - **HDR**（Radiance .hdr）：高动态范围，物理正确的 PBR 环境光；
//! - **LDR**（PNG/JPEG 等普通图片）：sRGB → 线性 + 曝光补偿，
//!   给无法制作 HDR 的模组作者当简易天空盒/环境光用；
//! - [`Environment::from_file`] 按文件内容自动识别两种格式。

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
    /// LDR 环境贴图的默认曝光（自动识别路径使用；显式 LDR 接口可自定）。
    ///
    /// LDR 值域 0..1 且对比度低，乘曝光能部分补偿动态范围损失；
    /// 保守取 1.0，模组作者可按需调高。
    pub const DEFAULT_LDR_EXPOSURE: f32 = 1.0;

    /// 从 Radiance HDR（.hdr）字节解码。
    pub fn from_hdr_bytes(bytes: &[u8]) -> Result<Self, EnvironmentError> {
        let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Hdr)
            .map_err(EnvironmentError::Decode)?;
        let rgb32f = image.to_rgb32f();
        let (width, height) = rgb32f.dimensions();
        let rgb = rgb32f.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, rgb })
    }

    /// 从磁盘读取并解码 Radiance HDR 文件。
    #[allow(dead_code)] // 显式入口：调用方明确指定 HDR 时用；App 默认走自动识别
    pub fn from_hdr_file(path: &Path) -> Result<Self, EnvironmentError> {
        let bytes = std::fs::read(path).map_err(EnvironmentError::Io)?;
        Self::from_hdr_bytes(&bytes)
    }

    /// 从文件按后缀加载：`.hdr`（不区分大小写）走 Radiance 解码，
    /// 其余（PNG/JPEG 等）按 LDR 处理（默认曝光）。
    pub fn from_file(path: &Path) -> Result<Self, EnvironmentError> {
        let bytes = std::fs::read(path).map_err(EnvironmentError::Io)?;
        let is_hdr = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("hdr"));
        if is_hdr {
            Self::from_hdr_bytes(&bytes)
        } else {
            Self::from_ldr_bytes(&bytes, Self::DEFAULT_LDR_EXPOSURE)
        }
    }

    /// 从字节按内容（magic bytes）识别格式：HDR 走 Radiance，其余按 LDR（默认曝光）。
    #[allow(dead_code)] // 显式入口：无文件名的字节流用；App 默认走 from_file 后缀识别
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, EnvironmentError> {
        match image::guess_format(bytes) {
            Ok(image::ImageFormat::Hdr) => Self::from_hdr_bytes(bytes),
            _ => Self::from_ldr_bytes(bytes, Self::DEFAULT_LDR_EXPOSURE),
        }
    }

    /// 从 LDR 图片字节（PNG/JPEG 等）构造环境：sRGB → 线性 × 曝光。
    pub fn from_ldr_bytes(bytes: &[u8], exposure: f32) -> Result<Self, EnvironmentError> {
        let image = image::load_from_memory(bytes).map_err(EnvironmentError::Decode)?;
        Self::from_ldr_image(&image, exposure)
    }

    /// 从磁盘读取 LDR 图片并构造环境。
    #[allow(dead_code)] // 显式入口：需要自定义曝光时用；App 默认走自动识别
    pub fn from_ldr_file(path: &Path, exposure: f32) -> Result<Self, EnvironmentError> {
        let bytes = std::fs::read(path).map_err(EnvironmentError::Io)?;
        Self::from_ldr_bytes(&bytes, exposure)
    }

    /// 从已解码的图片构造环境：8-bit sRGB 像素 → 线性 RGB × 曝光。
    ///
    /// LDR 图片本身是 sRGB 语义，直接当线性用会让画面偏暗偏灰；
    /// 线性化后再乘曝光，才能和 HDR 路径共用同一套天空盒/IBL 流程。
    pub fn from_ldr_image(
        image: &image::DynamicImage,
        exposure: f32,
    ) -> Result<Self, EnvironmentError> {
        let rgb8 = image.to_rgb8();
        let (width, height) = rgb8.dimensions();
        let rgb = rgb8
            .pixels()
            .map(|p| {
                [
                    srgb_to_linear(p.0[0] as f32 / 255.0) * exposure,
                    srgb_to_linear(p.0[1] as f32 / 255.0) * exposure,
                    srgb_to_linear(p.0[2] as f32 / 255.0) * exposure,
                ]
            })
            .collect();
        Ok(Self { width, height, rgb })
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

    /// 立方体贴图 → 镜面预过滤 mip 链（镜面 IBL，CPU）。
    ///
    /// 每层 `out_size>>mip`，roughness = mip / (mip_count-1)，GGX 重要性采样；
    /// 与 environment.wgsl 的 `prefilter` 计算着色器使用同一套数学。
    pub fn prefilter_map(
        cube: &[[f32; 4]],
        cube_face: u32,
        out_size: u32,
        mip_count: u32,
        samples: u32,
    ) -> Vec<Vec<[f32; 4]>> {
        let mut mips = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let size = out_size >> mip;
            let roughness = mip as f32 / (mip_count.max(2) - 1) as f32;
            let mut face_pixels = vec![[0.0f32; 4]; (size * size * 6) as usize];
            for face in 0..6u32 {
                for y in 0..size {
                    for x in 0..size {
                        let u = (x as f32 + 0.5) / size as f32;
                        let v = (y as f32 + 0.5) / size as f32;
                        let n = face_dir(face, u, v).normalize();
                        let r = n; // 预过滤假设 view = normal（标准 split-sum 近似）。
                        let basis = tangent_basis(n);
                        let mut acc = [0.0f32; 3];
                        let mut weight = 0.0f32;
                        for i in 0..samples {
                            let h = importance_sample_ggx(hammersley(i, samples), roughness);
                            let h_world =
                                (basis.0 * h.x + basis.1 * h.y + basis.2 * h.z).normalize();
                            let l = (2.0 * r.dot(h_world) * h_world - r).normalize();
                            let n_dot_l = n.dot(l).max(0.0);
                            if n_dot_l > 0.0 {
                                let c = sample_cube(cube, cube_face, l);
                                acc[0] += c[0] * n_dot_l;
                                acc[1] += c[1] * n_dot_l;
                                acc[2] += c[2] * n_dot_l;
                                weight += n_dot_l;
                            }
                        }
                        let value = if weight > 1e-5 {
                            [acc[0] / weight, acc[1] / weight, acc[2] / weight, 1.0]
                        } else {
                            [0.0, 0.0, 0.0, 1.0]
                        };
                        let idx = ((face * size + y) * size + x) as usize;
                        face_pixels[idx] = value;
                    }
                }
            }
            mips.push(face_pixels);
        }
        mips
    }

    /// 生成 BRDF 积分查找表（split-sum 第二项，CPU）。
    ///
    /// 尺寸 `size×size`：x = NdotV，y = roughness；输出 RGBA（a, b 存前两通道），
    /// 与 environment.wgsl 的 `brdf_lut` 计算着色器使用同一套数学。
    pub fn brdf_lut(size: u32, samples: u32) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; (size * size) as usize];
        let n = Vec3::Z;
        for y in 0..size {
            for x in 0..size {
                let n_dot_v = (x as f32 + 0.5) / size as f32;
                let roughness = (y as f32 + 0.5) / size as f32;
                let v = Vec3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v);
                let mut a = 0.0f32;
                let mut b = 0.0f32;
                for i in 0..samples {
                    let h = importance_sample_ggx(hammersley(i, samples), roughness);
                    let l = (2.0 * v.dot(h) * h - v).normalize();
                    let n_dot_l = l.z.max(0.0);
                    let n_dot_h = h.z.max(0.0);
                    let v_dot_h = v.dot(h).max(0.0);
                    if n_dot_l > 0.0 {
                        let g = geometry_smith(n, v, l, roughness);
                        let g_vis = g * v_dot_h / (n_dot_h * n_dot_v + 0.0001);
                        let fc = (1.0 - v_dot_h).powf(5.0);
                        a += (1.0 - fc) * g_vis;
                        b += fc * g_vis;
                    }
                }
                out[(y * size + x) as usize] = [a / samples as f32, b / samples as f32, 0.0, 1.0];
            }
        }
        out
    }
}

/// sRGB 编码值 → 线性亮度（精确曲线）。
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
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
        let idx =
            ((face as usize) * (face_size * face_size) as usize) + yy * face_size as usize + xx;
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
    (i as f32 / count as f32, radical_inverse_vdc(i))
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

/// GGX 重要性采样：返回以法线为 +Z 的切线空间半向量（与 environment.wgsl 一致）。
fn importance_sample_ggx((u, v): (f32, f32), roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * u;
    let cos_theta = ((1.0 - v) / (1.0 + (a * a - 1.0) * v)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

/// GGX Schlick 几何项（BRDF LUT 积分用，k = a²/2）。
fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let k = roughness * roughness / 2.0;
    n_dot_x / (n_dot_x * (1.0 - k) + k)
}

/// GGX Smith 几何项（视角 × 光照两个方向）。
fn geometry_smith(n: Vec3, v: Vec3, l: Vec3, roughness: f32) -> f32 {
    geometry_schlick_ggx(n.dot(v).max(0.0), roughness)
        * geometry_schlick_ggx(n.dot(l).max(0.0), roughness)
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
        assert!(
            env.rgb
                .iter()
                .any(|p| p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0)
        );
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
        let max = cube
            .iter()
            .fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
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

    /// CPU 转换：镜面预过滤 mip 链应有数据且数值有限（无 NaN/Inf）。
    #[test]
    fn prefilter_map_is_finite_and_positive() {
        // 全白环境：预过滤结果应与环境同色（各向同性、为正）。
        let env = Environment {
            width: 2,
            height: 1,
            rgb: vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        };
        let cube = env.to_cubemap(8);
        let mips = Environment::prefilter_map(&cube, 8, 4, 3, 128);
        assert_eq!(mips.len(), 3);
        for mip in &mips {
            for p in mip {
                assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
                assert!(p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0);
            }
        }
    }

    /// CPU 转换：BRDF LUT 数值有限且在 [0, 1] 附近（无越界/NaN）。
    #[test]
    fn brdf_lut_is_finite_and_bounded() {
        let lut = Environment::brdf_lut(32, 128);
        assert_eq!(lut.len(), 32 * 32);
        for p in &lut {
            assert!(p[0].is_finite() && p[1].is_finite());
            // split-sum 的 a/b 项都在 [0, 1] 区间附近。
            assert!((0.0..=1.5).contains(&p[0]));
            assert!((0.0..=1.5).contains(&p[1]));
        }
    }

    /// LDR 解码：sRGB 中灰（128）应线性化为约 0.216，且曝光可放大。
    #[test]
    fn from_ldr_image_linearizes_and_applies_exposure() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            2,
            2,
            image::Rgb([128, 128, 128]),
        ));
        let env = Environment::from_ldr_image(&img, 1.0).expect("LDR 应能构造");
        assert_eq!((env.width, env.height), (2, 2));
        // 128/255 ≈ 0.502，sRGB → linear ≈ 0.216。
        let linear = env.rgb[0][0];
        assert!((linear - 0.216).abs() < 0.01, "线性化偏差过大：{linear}");

        let env2 = Environment::from_ldr_image(&img, 2.0).expect("LDR 应能构造");
        assert!(
            (env2.rgb[0][0] - 2.0 * linear).abs() < 1e-5,
            "曝光应线性放大"
        );
    }

    /// 自动识别：`from_bytes` 按内容识别（PNG → LDR、HDR → HDR）；
    /// `from_file` 按后缀识别（.png → LDR、.hdr → HDR）。
    #[test]
    fn from_bytes_auto_detects_format() {
        // PNG：用 image 编码器生成一张 2×1 的字节。
        let png = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            2,
            1,
            image::Rgb([255, 0, 0]),
        ));
        let mut png_bytes = Vec::new();
        png.write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .expect("PNG 编码应成功");
        let env = Environment::from_bytes(&png_bytes).expect("PNG 自动识别应成功");
        // LDR 路径：红色 255 → 线性 1.0。
        assert!((env.rgb[0][0] - 1.0).abs() < 1e-4, "PNG 红色应线性化为 1.0");

        // from_file 按后缀：临时 .png 文件走 LDR。
        let tmp = std::env::temp_dir().join("env_from_file_test.png");
        std::fs::write(&tmp, &png_bytes).expect("写临时 PNG");
        let env = Environment::from_file(&tmp).expect(".png 后缀应走 LDR");
        assert!((env.rgb[0][0] - 1.0).abs() < 1e-4);
        let _ = std::fs::remove_file(&tmp);

        // HDR：仓库内如有测试 HDR，验证 from_file 按 .hdr 后缀走 HDR 路径。
        let path = Path::new("assets/environments/test.hdr");
        if path.is_file() {
            let env = Environment::from_file(path).expect("HDR 自动识别应成功");
            let max = env
                .rgb
                .iter()
                .fold(0.0f32, |m, p| m.max(p[0]).max(p[1]).max(p[2]));
            assert!(max > 1.0, "HDR 应有 >1 的高动态范围值（区别于 LDR）");
        }
    }
}
