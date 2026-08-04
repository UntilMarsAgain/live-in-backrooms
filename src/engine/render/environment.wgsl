// 环境贴图管线：
//   1) `equirect_to_cubemap` 计算入口：等距矩形（HDRI）→ 环境立方体贴图；
//   2) `irradiance` 计算入口：环境立方体贴图 → 辐照度图（漫反射 IBL）；
//   3) 天空盒顶点/片元：全屏三角形 + 逆 view_proj 反推视线方向。
//
// GPU 转换（入口 1、2）只在 Vulkan/Metal 等 storage 数组纹理可靠的后端启用；
// GL 后端回退到 CPU 转换（core::environment），此时入口 1、2 不参与执行。
// 立方体面层序遵循 WebGPU 规范：0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z。

const PI: f32 = 3.141592653589793;

// 两个计算入口共用：`size` = 输出贴图每面尺寸；`sample_count` 仅辐照度入口使用。
struct EnvParams {
    size: u32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    inverse_view_proj: mat4x4<f32>,
}

// ---- 等距矩形 → 立方体贴图 ----
@group(0) @binding(0) var<uniform> convert_params: EnvParams;
@group(0) @binding(1) var equirect_tex: texture_2d<f32>;
@group(0) @binding(2) var equirect_sampler: sampler;
@group(0) @binding(3) var env_output: texture_storage_2d_array<rgba32float, write>;

// ---- 立方体贴图 → 辐照度图 ----
@group(0) @binding(0) var<uniform> irradiance_params: EnvParams;
@group(0) @binding(1) var env_cube: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var irradiance_output: texture_storage_2d_array<rgba32float, write>;

// ---- 天空盒 ----
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var skybox_tex: texture_cube<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

// 立方体面内像素坐标 (u, v ∈ [0,1]) → 世界方向。
fn face_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let x = u * 2.0 - 1.0;
    let y = v * 2.0 - 1.0;
    switch face {
        case 0u: {  // +X
            return vec3<f32>(1.0, -y, -x);
        }
        case 1u: {  // -X
            return vec3<f32>(-1.0, -y, x);
        }
        case 2u: {  // +Y
            return vec3<f32>(x, 1.0, y);
        }
        case 3u: {  // -Y
            return vec3<f32>(x, -1.0, -y);
        }
        case 4u: {  // +Z
            return vec3<f32>(x, -y, 1.0);
        }
        default: {  // -Z
            return vec3<f32>(-x, -y, -1.0);
        }
    }
}

// 世界方向 → 等距矩形 UV（u: 经度 0..1，v: 纬度 0 顶 1 底）。
fn dir_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), theta / PI);
}

@compute @workgroup_size(8, 8)
fn equirect_to_cubemap(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= convert_params.size || gid.y >= convert_params.size || gid.z >= 6u) {
        return;
    }
    // 取像素中心采样，避免面边缘出现锯齿。
    let u = (f32(gid.x) + 0.5) / f32(convert_params.size);
    let v = (f32(gid.y) + 0.5) / f32(convert_params.size);
    let dir = face_dir(gid.z, u, v);
    let color = textureSampleLevel(equirect_tex, equirect_sampler, dir_to_equirect(dir), 0.0);
    textureStore(env_output, vec2<i32>(gid.xy), i32(gid.z), vec4<f32>(color.rgb, 1.0));
}

// 低差异序列：Hammersley 点集（i / n, 位反转的 Van der Corput）。
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, count: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(count), radical_inverse_vdc(i));
}

// 余弦加权半球采样（u 的分布满足 p(ω) ∝ cosθ）。
fn cosine_sample_hemisphere(u: vec2<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * u.x;
    let cos_theta = sqrt(1.0 - u.y);
    let sin_theta = sqrt(u.y);
    return vec3<f32>(cos(phi) * sin_theta, cos_theta, sin(phi) * sin_theta);
}

// 以法线为 +Z 构造正交基（避免与法线平行的轴）。
fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    let up = select(
        vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.z) > 0.999,
    );
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

@compute @workgroup_size(8, 8)
fn irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= irradiance_params.size || gid.y >= irradiance_params.size || gid.z >= 6u) {
        return;
    }
    let u = (f32(gid.x) + 0.5) / f32(irradiance_params.size);
    let v = (f32(gid.y) + 0.5) / f32(irradiance_params.size);
    let n = normalize(face_dir(gid.z, u, v));
    let basis = tangent_basis(n);

    // 余弦加权蒙特卡洛：E(n) = ∫ L cosθ dω ≈ π * avg。
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < irradiance_params.sample_count; i = i + 1u) {
        let sample = cosine_sample_hemisphere(hammersley(i, irradiance_params.sample_count));
        let l = normalize(basis * sample);
        let n_dot_l = max(dot(n, l), 0.0);
        acc += textureSampleLevel(env_cube, env_sampler, l, 0.0).rgb * n_dot_l;
    }
    let irradiance_value = PI * acc / f32(irradiance_params.sample_count);
    textureStore(
        irradiance_output,
        vec2<i32>(gid.xy),
        i32(gid.z),
        vec4<f32>(irradiance_value, 1.0),
    );
}

// ---- 天空盒 ----

struct SkyboxOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn skybox_vs_main(@builtin(vertex_index) index: u32) -> SkyboxOutput {
    var out: SkyboxOutput;
    // 全屏三角形：覆盖整个 NDC，无顶点缓冲。
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    out.clip_position = vec4<f32>(positions[index], 1.0, 1.0);
    out.ndc = positions[index];
    return out;
}

@fragment
fn skybox_fs_main(in: SkyboxOutput) -> @location(0) vec4<f32> {
    // NDC 远平面点经逆矩阵回到世界空间，即视线方向。
    let p = camera.inverse_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(p.xyz / p.w);
    let radiance = textureSampleLevel(skybox_tex, skybox_sampler, dir, 0.0).rgb;
    // 曝光 + 指数色调映射（输出线性色，由 sRGB 交换链完成编码）。
    let exposure = 1.0;
    let mapped = vec3<f32>(1.0) - exp(-exposure * radiance);
    return vec4<f32>(mapped, 1.0);
}
