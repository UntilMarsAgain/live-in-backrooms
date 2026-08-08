// 环境贴图管线：
//   1) `equirect_to_cubemap` 计算入口：等距矩形（HDRI）→ 环境立方体贴图；
//   2) `irradiance` 计算入口：环境立方体贴图 → 辐照度图（漫反射 IBL）；
//   3) 天空盒顶点/片元：全屏三角形 + 逆 view_proj 反推视线方向；
//      输出原始辐射值（×曝光），色调映射由最后的 blit pass 统一完成。
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

// 环境参数 uniform（与 Rust 侧 EnvironmentParams 布局兼容）：
// `intensity` 字段在此处作为全局曝光值使用；
// `agx_min_ev` / `agx_max_ev` 为 AgX 色调映射的 EV 窗口（中间灰 0.18 锚定）。
struct EnvironmentParams {
    intensity: f32, // 曝光值
    agx_min_ev: f32,
    agx_max_ev: f32,
    _pad0: u32,
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
@group(1) @binding(2) var<uniform> environment_params: EnvironmentParams;

// ============================================================================
// 立方体面坐标工具函数（计算管线与天空盒共用逻辑保持一致）
// ============================================================================

// 立方体面内像素坐标 (u, v ∈ [0,1]) → 世界方向。
fn face_dir(face: u32, u: f32, v: f32) -> vec3<f32> {
    let x = u * 2.0 - 1.0;
    let y = v * 2.0 - 1.0;
    // WGSL switch 选择器必须用括号包裹
    switch (face) {
        case 0u: { return vec3<f32>(1.0, -y, -x); }  // +X
        case 1u: { return vec3<f32>(-1.0, -y, x); }  // -X
        case 2u: { return vec3<f32>(x, 1.0, y); }    // +Y
        case 3u: { return vec3<f32>(x, -1.0, -y); }  // -Y
        case 4u: { return vec3<f32>(x, -y, 1.0); }   // +Z
        default: { return vec3<f32>(-x, -y, -1.0); } // -Z
    }
}

// 世界方向 → 等距矩形 UV（u: 经度 0..1，v: 纬度 0 顶 1 底）。
fn dir_to_equirect(dir: vec3<f32>) -> vec2<f32> {
    let d = normalize(dir);
    let phi = atan2(d.z, d.x);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), theta / PI);
}

// ============================================================================
// Compute: Equirectangular → Cubemap
// ============================================================================

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

// ============================================================================
// Compute: Cubemap → Irradiance Map (Diffuse IBL)
// ============================================================================

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
    // WGSL select 需要 bool 条件；用 > 代替 step 返回浮点的行为
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
        // 'sample' 是 WGSL 保留关键字，改用 wi
        let wi = cosine_sample_hemisphere(hammersley(i, irradiance_params.sample_count));
        let l = normalize(basis * wi);
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

// ============================================================================
// Compute: Cubemap → Prefiltered Map (Specular IBL)
// ============================================================================

// GGX 重要性采样：返回以法线为 +Z 的切线空间半向量。
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// GGX Schlick 几何项（BRDF LUT 积分用，k = a²/2）。
fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    return geometry_schlick_ggx(max(dot(n, v), 0.0), roughness)
        * geometry_schlick_ggx(max(dot(n, l), 0.0), roughness);
}

struct PrefilterParams {
    size: u32,
    mip: u32,
    mip_count: u32,
    sample_count: u32,
}

@group(0) @binding(0) var<uniform> prefilter_params: PrefilterParams;
@group(0) @binding(1) var prefilter_env: texture_cube<f32>;
@group(0) @binding(2) var prefilter_sampler: sampler;
@group(0) @binding(3) var prefilter_output: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= prefilter_params.size || gid.y >= prefilter_params.size || gid.z >= 6u) {
        return;
    }
    let mip = prefilter_params.mip;
    let mip_count = prefilter_params.mip_count;
    let roughness = f32(mip) / f32(max(mip_count - 1u, 1u));
    let u = (f32(gid.x) + 0.5) / f32(prefilter_params.size);
    let v = (f32(gid.y) + 0.5) / f32(prefilter_params.size);
    let n = normalize(face_dir(gid.z, u, v));
    let r = n; // 预过滤假设 view = normal（标准 split-sum 近似）。
    let basis = tangent_basis(n);

    var acc = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < prefilter_params.sample_count; i = i + 1u) {
        let h = normalize(basis * importance_sample_ggx(hammersley(i, prefilter_params.sample_count), roughness));
        let l = normalize(2.0 * dot(r, h) * h - r);
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0) {
            acc += textureSampleLevel(prefilter_env, prefilter_sampler, l, 0.0).rgb * n_dot_l;
            weight += n_dot_l;
        }
    }
    let value = select(vec3<f32>(0.0), acc / max(weight, 1e-5), weight > 0.0);
    textureStore(
        prefilter_output,
        vec2<i32>(gid.xy),
        i32(gid.z),
        vec4<f32>(value, 1.0),
    );
}

// ============================================================================
// Compute: BRDF Integration LUT (Specular IBL, split-sum 第二项)
// ============================================================================

const BRDF_LUT_SIZE: u32 = 128u;
const BRDF_LUT_SAMPLES: u32 = 1024u;

@group(0) @binding(0) var brdf_lut_output: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn brdf_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= BRDF_LUT_SIZE || gid.y >= BRDF_LUT_SIZE) {
        return;
    }
    let n_dot_v = (f32(gid.x) + 0.5) / f32(BRDF_LUT_SIZE);
    let roughness = (f32(gid.y) + 0.5) / f32(BRDF_LUT_SIZE);
    let v = vec3<f32>(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    for (var i = 0u; i < BRDF_LUT_SAMPLES; i = i + 1u) {
        let h = normalize(importance_sample_ggx(hammersley(i, BRDF_LUT_SAMPLES), roughness));
        let l = normalize(2.0 * dot(v, h) * h - v);
        let n_dot_l = max(l.z, 0.0);
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        if (n_dot_l > 0.0) {
            let g = geometry_smith(n, v, l, roughness);
            let g_vis = g * v_dot_h / (n_dot_h * n_dot_v + 0.0001);
            let fc = pow(1.0 - v_dot_h, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    textureStore(
        brdf_lut_output,
        vec2<i32>(gid.xy),
        vec4<f32>(a / f32(BRDF_LUT_SAMPLES), b / f32(BRDF_LUT_SAMPLES), 0.0, 1.0),
    );
}

// ============================================================================
// Skybox Render Pipeline
// ============================================================================

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

    // 应用曝光（复用 environment_params.intensity 作为曝光值）
    let exposed = radiance * environment_params.intensity;
    // 输出原始辐射值（线性 HDR）；色调映射由最后的 blit pass 统一完成。
    return vec4<f32>(exposed, 1.0);
}
