// 相机 uniform：内存布局与 Rust 侧 CameraUniform 保持一致（80 字节）。
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

// 物体数据：模型矩阵。通过动态 uniform 偏移为每个物体绑定不同的矩阵。
struct ObjectData {
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> object_data: ObjectData;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // 物体坐标 → 世界坐标（模型矩阵）→ 裁剪坐标（相机视图投影）。
    let world_position = object_data.model * vec4<f32>(input.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
