// 相机 uniform：内存布局与 Rust 侧 CameraUniform 保持一致（80 字节）。
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

// 物体数据：模型矩阵 + 法线矩阵 + 材质基础色因子。
struct ObjectData {
    model: mat4x4<f32>,
    normal_matrix: mat3x3<f32>,
    base_color: vec4<f32>,
}

// 方向光数组：材质着色器在片元阶段遍历灯光累加光照。
const MAX_LIGHTS: u32 = 8u;
struct DirectionalLight {
    direction: vec3<f32>, // 世界空间方向：从表面指向光源
    intensity: f32,
    color: vec3<f32>,
    _pad: f32,
}
struct Lights {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lights: array<DirectionalLight, MAX_LIGHTS>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> object_data: ObjectData;
@group(2) @binding(0) var<uniform> lights: Lights;
@group(3) @binding(0) var base_color_tex: texture_2d<f32>;
@group(3) @binding(1) var base_color_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
    @location(3) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // 物体坐标 → 世界坐标（模型矩阵）→ 裁剪坐标（相机视图投影）。
    let world_position = object_data.model * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.world_position = world_position.xyz;
    out.world_normal = object_data.normal_matrix * input.normal;
    out.color = input.color;
    out.tex_coord = input.tex_coord;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    // 基础色 = 贴图采样 × 材质因子 × 顶点色（glTF 组合规则）。
    let tex_color = textureSample(base_color_tex, base_color_sampler, input.tex_coord);
    let albedo = tex_color.rgb * object_data.base_color.rgb * input.color;
    // 微弱环境光，避免背光面纯黑。
    var color = albedo * 0.08;
    // 基础光照：逐方向光累加 Lambert 漫反射。
    for (var i = 0u; i < lights.count; i = i + 1u) {
        let l = normalize(lights.lights[i].direction);
        let ndotl = max(dot(n, l), 0.0);
        color += albedo * lights.lights[i].color * lights.lights[i].intensity * ndotl;
    }
    return vec4<f32>(color, 1.0);
}
