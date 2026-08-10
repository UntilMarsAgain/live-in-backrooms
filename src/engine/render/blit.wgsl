// 色调映射 blit pass：把 HDR 中间目标采样出来，做 AgX 色调映射后写交换链。
//
// 场景 pass（网格 + 天空盒）输出**原始辐射值**（线性 HDR，可 >1），
// 不做任何压缩；显示前的最后一步在这里统一映射，全帧只做一次。
// 这样后处理（Bloom / SSR / SSAO 等）可以在 HDR 值上工作，且天空盒与
// 物体走同一条色调映射曲线。
//
// `exposure` = 全局曝光：场景 pass 输出未曝光的原始辐射值（天空盒、mesh
// IBL 都不乘），色调映射前在这里统一乘一次——天空盒与物体走同一曝光。
// `intensity`（IBL 系数）已被 mesh 着色器应用，blit 不读。
struct EnvironmentParams {
    intensity: f32,
    exposure: f32,
    agx_min_ev: f32,
    agx_max_ev: f32,
}

@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> environment_params: EnvironmentParams;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

// ============================================================================
// AgX Tone Mapping（与 Blender/Filament/three.js 同源实现）
// 参考: https://github.com/mrdoob/three.js（tonemapping_pars_fragment）
// ============================================================================

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

// 全屏三角形：覆盖整个 NDC，无顶点缓冲（index 0..2 由 draw(0..3) 提供）。
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    // 片元坐标（0..尺寸）→ UV：像素中心 +0.5 恰好对应纹素中心。
    let size = vec2<f32>(textureDimensions(hdr_tex));
    let uv = frag_pos.xy / size;
    // Bloom：把辉光结果加回原始辐射值（无 bloom 时绑定黑色纹理，贡献为 0）。
    let radiance = textureSampleLevel(hdr_tex, hdr_sampler, uv, 0.0).rgb
        + textureSampleLevel(bloom_tex, hdr_sampler, uv, 0.0).rgb;
    // 统一应用全局曝光，再做 AgX 色调映射；输出线性色由 sRGB 交换链编码。
    return vec4<f32>(agx_tone_map(radiance * environment_params.exposure), 1.0);
}
