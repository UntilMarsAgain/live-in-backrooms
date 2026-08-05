// 相机 uniform：内存布局与 Rust 侧 CameraUniform 保持一致（144 字节）。
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    inverse_view_proj: mat4x4<f32>,
}

// 物体数据：模型矩阵 + 法线矩阵 + PBR 材质参数。
struct ObjectData {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _pad0: f32,
    _pad1: f32,
}

// 光源数组：方向光 / 点光 / 面光。
const MAX_LIGHTS: u32 = 8u;
struct LightData {
    kind: u32, // 0=方向光 1=点光 2=面光
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    direction: vec3<f32>, // 方向光/面光：行进方向，即局部 -Z 经世界旋转
    _pad3: f32,
    position: vec3<f32>, // 点光/面光：世界位置
    _pad4: f32,
    color: vec3<f32>,
    intensity: f32,
    size: vec2<f32>, // 面光面板尺寸（当前近似未直接使用）
    _pad5: f32,
    _pad6: f32,
}
struct Lights {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lights: array<LightData, MAX_LIGHTS>,
}

// 环境参数：`intensity` = IBL 环境光强度（0 = 纯手动布光，1 = 满环境光）。
struct EnvironmentParams {
    intensity: f32,
    agx_min_ev: f32,
    agx_max_ev: f32,
    _pad0: u32,
}

const PI: f32 = 3.141592653589793;

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> object_data: ObjectData;
@group(2) @binding(0) var<uniform> lights: Lights;
@group(3) @binding(0) var base_color_tex: texture_2d<f32>;
@group(3) @binding(1) var base_color_sampler: sampler;
@group(3) @binding(2) var metallic_roughness_tex: texture_2d<f32>;
@group(3) @binding(3) var normal_tex: texture_2d<f32>;
@group(4) @binding(0) var irradiance_tex: texture_cube<f32>;
@group(4) @binding(1) var environment_tex: texture_cube<f32>;
@group(4) @binding(2) var environment_sampler: sampler;
@group(4) @binding(3) var<uniform> environment_params: EnvironmentParams;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) tex_coord: vec2<f32>,
    @location(4) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) tex_coord: vec2<f32>,
    @location(4) world_tangent: vec4<f32>,
}

// ---------- PBR BRDF ----------

// GGX 法线分布函数。
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX 几何遮挡（单方向）。
fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

fn geometry_smith(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_l, roughness) * geometry_schlick_ggx(n_dot_v, roughness);
}

// Schlick 菲涅尔。
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

// AgX Tone Mapping（与 Blender/Filament/three.js 同源实现，天空盒同一算法）。
// 参考: https://github.com/mrdoob/three.js（tonemapping_pars_fragment）
// 保证物体与背景亮度/对比一致；输出线性 [0,1]，由 sRGB 交换链完成编码。

// 线性 sRGB ↔ 线性 Rec.2020（AgX 在 Rec.2020 基色上工作）。
// WGSL mat3x3 按列填充，以下每 3 个数为一列（r0, r1, r2）。
const SRGB_TO_REC2020: mat3x3<f32> = mat3x3<f32>(
    0.6274, 0.0691, 0.0164,
    0.3293, 0.9195, 0.0880,
    0.0433, 0.0113, 0.8956,
);
const REC2020_TO_SRGB: mat3x3<f32> = mat3x3<f32>(
    1.6605, -0.1246, -0.0182,
    -0.5876, 1.1329, -0.1006,
    -0.0728, -0.0083, 1.1187,
);

// AgX Inset / Outset 色域矩阵。
const AGX_INSET_MAT: mat3x3<f32> = mat3x3<f32>(
    0.856627153315983, 0.137318972929847, 0.11189821299995,
    0.0951212405381588, 0.761241990602591, 0.0767994186031903,
    0.0482516061458583, 0.101439036467562, 0.811302368396859,
);
const AGX_OUTSET_MAT: mat3x3<f32> = mat3x3<f32>(
    1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
    -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
    -0.016493938717834573, -0.016493938717834257, 1.2519364065950405,
);

// AgX Default 对比曲线（7 项多项式拟合，非 smoothstep；误差平方 ≈ 3.7e-6）。
fn agx_default_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232;
}

// AgX Default 色调映射（输入/输出均为线性 sRGB）。
// EV 窗口来自 uniform（默认 -12.47393 / 4.026069，即中间灰上下 -10 ~ +6.5 EV），
// 场景可覆盖，这是 AgX 相对 ACES 的核心优势：每个层级配置自己的动态范围窗口。
fn agx_tone_map(color: vec3<f32>) -> vec3<f32> {
    // 1. 线性 sRGB → Rec.2020 → AgX 基础色域。
    var c = AGX_INSET_MAT * (SRGB_TO_REC2020 * color);

    // 2. Log2 编码 + 归一化（EV 窗口以中间灰 0.18 为锚点）。
    c = max(c, vec3<f32>(1e-10));
    c = log2(c);
    c = (c - vec3<f32>(environment_params.agx_min_ev))
        / vec3<f32>(environment_params.agx_max_ev - environment_params.agx_min_ev);
    c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));

    // 3. AgX Default 对比曲线。
    c = agx_default_contrast(c);

    // 4. Outset 矩阵 + 2.2 线性化 → 转回线性 sRGB。
    c = AGX_OUTSET_MAT * c;
    c = pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
    return REC2020_TO_SRGB * c;
}

// ACES Filmic Tone Mapping（Narkowicz 2015 拟合）。
// 保留作对比；当前用修正后的 AgX。
fn aces_filmic(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + b)) / (color * (c * color + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_position = object_data.model * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.world_position = world_position.xyz;
    out.world_normal = object_data.normal_matrix * input.normal;
    out.world_tangent = vec4<f32>(object_data.normal_matrix * input.tangent.xyz, input.tangent.w);
    out.color = input.color;
    out.tex_coord = input.tex_coord;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 基础色 = 贴图 × 材质因子 × 顶点色（glTF 组合规则）。
    let tex_color = textureSample(base_color_tex, base_color_sampler, input.tex_coord);
    let albedo = tex_color.rgb * object_data.base_color.rgb * input.color;

    // 金属度 / 粗糙度（glTF metallic-roughness：B=金属度，G=粗糙度）。
    let mr = textureSample(metallic_roughness_tex, base_color_sampler, input.tex_coord);
    let metallic = mr.b * object_data.metallic;
    let roughness = max(mr.g * object_data.roughness, 0.04);

    // 法线贴图：切线空间 → 世界空间（Gram-Schmidt 正交化 T）。
    let n_world = normalize(input.world_normal);
    let tangent_color = textureSample(normal_tex, base_color_sampler, input.tex_coord).rgb;
    let tangent_normal = normalize(tangent_color * 2.0 - 1.0);
    let t = normalize(input.world_tangent.xyz - dot(input.world_tangent.xyz, n_world) * n_world);
    let b = cross(n_world, t) * input.world_tangent.w;
    let n = normalize(t * tangent_normal.x + b * tangent_normal.y + n_world * tangent_normal.z);

    // PBR：GGX 镜面反射 + Schlick 菲涅尔；漫反射按金属度削减。
    let v = normalize(camera.position - input.world_position);
    let n_dot_v = max(dot(n, v), 0.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // IBL 漫反射：辐照度图已含 π 因子（E(n) = π * avg），这里除以 π 恢复物理量。
    // Phase 1 暂无镜面 IBL（预过滤环境图 + BRDF LUT），金属材质的环境高光暂缺。
    let irradiance = textureSampleLevel(irradiance_tex, environment_sampler, n, 0.0).rgb;
    let k_d_ambient = (vec3<f32>(1.0) - f0) * (1.0 - metallic);
    let ambient_diffuse = k_d_ambient * albedo / PI * irradiance * environment_params.intensity;
    var color = ambient_diffuse;

    for (var i = 0u; i < lights.count; i = i + 1u) {
        let light = lights.lights[i];
        // 光照方向与衰减按类型计算。
        // light.direction 对方向光/面光统一是行进方向（光源 → 场景）；
        // 着色时 l 需要"表面 → 光源"，方向光因此取反。
        var l: vec3<f32>;
        var attenuation: f32;
        if (light.kind == 0u) {
            l = -normalize(light.direction);
            attenuation = light.intensity;
        } else if (light.kind == 1u) {
            let delta = light.position - input.world_position;
            l = normalize(delta);
            let dist2 = dot(delta, delta);
            attenuation = light.intensity / (dist2 + 0.0001);
        } else {
            let delta = light.position - input.world_position;
            l = normalize(delta);
            let dist2 = dot(delta, delta);
            // 面光近似：朗伯发射面板（沿发射方向余弦分布）+ 平方反比。
            let to_panel = normalize(input.world_position - light.position);
            let panel = max(dot(to_panel, normalize(light.direction)), 0.0);
            attenuation = light.intensity * panel / (dist2 + 0.0001);
        }
        let n_dot_l = max(dot(n, l), 0.0);
        if (n_dot_l > 0.0 && attenuation > 0.0) {
            let h = normalize(v + l);
            let n_dot_h = max(dot(n, h), 0.0);
            let v_dot_h = max(dot(v, h), 0.0);
            let d = distribution_ggx(n_dot_h, roughness);
            let g = geometry_smith(n_dot_l, n_dot_v, roughness);
            let f = fresnel_schlick(v_dot_h, f0);
            let specular = d * g * f / (4.0 * n_dot_l * n_dot_v + 0.0001);
            let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);
            color +=
                (k_d * albedo / PI + specular) *
                light.color *
                attenuation *
                n_dot_l;
        }
    }
    return vec4<f32>(agx_tone_map(color), 1.0);
}
