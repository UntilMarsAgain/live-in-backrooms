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
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _pad0: f32,
    _pad1: f32,
}

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

// 每帧实际参与着色的灯光数（CPU 写入）；数组本体是运行时长度的 storage。
struct LightCount {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// 环境参数：`intensity` = IBL 环境光强度（0 = 纯手动布光，1 = 满环境光）；
// `exposure` 由 blit 统一应用，mesh 不读。
struct EnvironmentParams {
    intensity: f32,
    exposure: f32,
    agx_min_ev: f32,
    agx_max_ev: f32,
}

const PI: f32 = 3.141592653589793;

@group(0) @binding(0) var<uniform> camera: CameraUniform;
// 物体数据：全部物体一个 storage 数组，按实例索引（instance_index）取。
// 相比动态 uniform 的逐物体绑定：每帧只绑一次，无 256 字节对齐浪费。
@group(1) @binding(0) var<storage, read> object_data: array<ObjectData>;
@group(2) @binding(0) var<uniform> light_count: LightCount;
@group(2) @binding(1) var<storage, read> lights: array<LightData>;
@group(3) @binding(0) var base_color_tex: texture_2d<f32>;
@group(3) @binding(1) var base_color_sampler: sampler;
@group(3) @binding(2) var metallic_roughness_tex: texture_2d<f32>;
@group(3) @binding(3) var normal_tex: texture_2d<f32>;
@group(3) @binding(4) var emissive_tex: texture_2d<f32>;
@group(4) @binding(0) var irradiance_tex: texture_cube<f32>;
@group(4) @binding(1) var environment_tex: texture_cube<f32>;
@group(4) @binding(2) var environment_sampler: sampler;
@group(4) @binding(3) var<uniform> environment_params: EnvironmentParams;
@group(4) @binding(4) var prefiltered_tex: texture_cube<f32>;
@group(4) @binding(5) var brdf_lut_tex: texture_2d<f32>;

struct VertexInput {
    // 绘制时用实例区间 i..i+1 编码物体索引，这里即 object_data 数组下标。
    @builtin(instance_index) instance_index: u32,
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
    // flat：同一三角形所有片元拿到所属物体的索引（片元阶段取材质参数用）。
    @location(5) @interpolate(flat) object_index: u32,
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

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // instance_index = base_instance + 实例编号；绘制时每个物体用实例区间
    // i..i+1 编码，因此这里就是物体在 object_data 数组里的下标。
    let object = object_data[input.instance_index];
    out.object_index = input.instance_index;
    let world_position = object.model * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.world_position = world_position.xyz;
    out.world_normal = object.normal_matrix * input.normal;
    out.world_tangent = vec4<f32>(object.normal_matrix * input.tangent.xyz, input.tangent.w);
    out.color = input.color;
    out.tex_coord = input.tex_coord;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let object = object_data[input.object_index];
    // 基础色 = 贴图 × 材质因子 × 顶点色（glTF 组合规则）。
    let tex_color = textureSample(base_color_tex, base_color_sampler, input.tex_coord);
    let albedo = tex_color.rgb * object.base_color.rgb * input.color;

    // 金属度 / 粗糙度（glTF metallic-roughness：B=金属度，G=粗糙度）。
    let mr = textureSample(metallic_roughness_tex, base_color_sampler, input.tex_coord);
    let metallic = mr.b * object.metallic;
    let roughness = max(mr.g * object.roughness, 0.04);

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
    // 镜面 IBL 走 split-sum（预过滤环境图 + BRDF LUT），两者都乘环境强度。
    let irradiance = textureSampleLevel(irradiance_tex, environment_sampler, n, 0.0).rgb;
    let k_d_ambient = (vec3<f32>(1.0) - f0) * (1.0 - metallic);
    let ambient_diffuse = k_d_ambient * albedo / PI * irradiance * environment_params.intensity;
    var color = ambient_diffuse;

    // 镜面 IBL：预过滤环境图（按粗糙度选 mip）+ BRDF LUT（split-sum 第二项）。
    // 金属的 f0 = albedo（反射环境本色），非金属 f0 = 0.04。
    let max_reflection_lod = f32(textureNumLevels(prefiltered_tex)) - 1.0;
    let r = reflect(-v, n);
    let prefiltered = textureSampleLevel(
        prefiltered_tex,
        environment_sampler,
        r,
        roughness * max_reflection_lod,
    ).rgb;
    let brdf = textureSampleLevel(
        brdf_lut_tex,
        environment_sampler,
        vec2<f32>(n_dot_v, roughness),
        0.0,
    ).rg;
    let specular_ibl = prefiltered * (f0 * brdf.x + brdf.y);
    color += specular_ibl * environment_params.intensity;

    for (var i = 0u; i < light_count.count; i = i + 1u) {
        let light = lights[i];
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
    // 自发光：贴图（sRGB 自动转线性）× 因子；不受光照影响，直接叠加。
    // 因子为 0 或贴图为默认黑时贡献为 0（不发光）。
    let emissive = textureSample(emissive_tex, base_color_sampler, input.tex_coord).rgb
        * object_data[input.object_index].emissive.rgb;
    color += emissive;
    // 输出原始辐射值（线性 HDR，可 >1）；色调映射由最后的 blit pass 统一完成。
    return vec4<f32>(color, 1.0);
}
