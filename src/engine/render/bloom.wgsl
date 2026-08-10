// Bloom（辉光）后处理：从 HDR 目标提取高亮 → 多级下采样 → 逐级上采样合并。
//
// 插在"场景 pass → 色调映射 blit"之间（即 Blender 合成器 Glare 节点的位置）：
// 场景 pass 输出未曝光的原始辐射值，Bloom 在这里提取并扩散亮区，最终
// blit pass 把 bloom 结果加回 HDR 再统一曝光 + AgX 色调映射。
//
// pass 结构（每级尺寸减半，共 LEVELS 级）：
//   0. extract：threshold 截断高亮 → bloom[0]
//   1..LEVELS-1. downsample：2×2 平均 → bloom[i]
//   LEVELS-2..0. upsample：双线性放大上一级，blend add 合并到本级的原始高亮

struct BloomParams {
    threshold: f32,
    intensity: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: BloomParams;
@group(0) @binding(1) var source_tex: texture_2d<f32>;
@group(0) @binding(2) var source_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// 全屏三角形（3 顶点覆盖整个视口），UV 与片元坐标对应。
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = pos[index];
    return VsOut(vec4<f32>(p, 0.0, 1.0), p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5));
}

// 提取高亮：低于阈值的归零，高于的保留原始辐射值（可 >1）× 强度。
// 用 smoothstep 软阈值：在阈值附近平滑过渡，避免硬切断导致视角移动时
// 辉光强度跳变（闪烁）。
@fragment
fn extract_fs(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(source_tex, source_sampler, in.uv, 0.0).rgb;
    // 亮度按 max 通道估算（保留高亮通道），在 [threshold, threshold*1.5]
    // 区间内平滑过渡到全亮。
    let luma = max(max(c.r, c.g), c.b);
    let knee = smoothstep(params.threshold, params.threshold * 1.5, luma);
    let bright = c * knee * params.intensity;
    return vec4<f32>(bright, 1.0);
}

// 下采样：2×2 源纹素 **亮度加权**（Karis：越亮的纹素权重越低）。
// 偏移 ±0.5 纹素恰好覆盖四个相邻纹素中心；加权避免单个亮点在移动时
// 从一块跳到另一块导致辉光强度突变（时域闪烁）。
@fragment
fn downsample_fs(in: VsOut) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(source_tex));
    let texel = 1.0 / size;
    let a = textureSampleLevel(source_tex, source_sampler, in.uv + vec2<f32>(-0.5, -0.5) * texel, 0.0).rgb;
    let b = textureSampleLevel(source_tex, source_sampler, in.uv + vec2<f32>(0.5, -0.5) * texel, 0.0).rgb;
    let c = textureSampleLevel(source_tex, source_sampler, in.uv + vec2<f32>(-0.5, 0.5) * texel, 0.0).rgb;
    let d = textureSampleLevel(source_tex, source_sampler, in.uv + vec2<f32>(0.5, 0.5) * texel, 0.0).rgb;
    // Karis 亮度权重：越亮贡献越小（抑制 firefly 主导块均值）。
    let w_a = 1.0 / (1.0 + max(max(a.r, a.g), a.b));
    let w_b = 1.0 / (1.0 + max(max(b.r, b.g), b.b));
    let w_c = 1.0 / (1.0 + max(max(c.r, c.g), c.b));
    let w_d = 1.0 / (1.0 + max(max(d.r, d.g), d.b));
    let w_sum = w_a + w_b + w_c + w_d;
    let out = (a * w_a + b * w_b + c * w_c + d * w_d) / w_sum;
    return vec4<f32>(out, 1.0);
}

// 上采样：双线性放大（采样器线性过滤），目标帧缓冲 blend add —— 把上一级的
// 辉光叠加到本级的原始高亮上。
@fragment
fn upsample_fs(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(source_tex, source_sampler, in.uv, 0.0).rgb;
    return vec4<f32>(c, 1.0);
}
