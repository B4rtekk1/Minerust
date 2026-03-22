struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    csm_view_proj: array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_outline(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let half_width_px = model.uv.x;
    let side = model.normal.w;

    let clip_a = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    let clip_b = uniforms.view_proj * vec4<f32>(model.normal.xyz, 1.0);

    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;

    let aspect = 16.0 / 9.0;
    let dir = normalize((ndc_b - ndc_a) * vec2<f32>(aspect, 1.0));

    let perp = vec2<f32>(-dir.y, dir.x) / vec2<f32>(aspect, 1.0);

    let offset_ndc = perp * (half_width_px / 540.0) * clip_a.w;

    out.clip_position = vec4<f32>(clip_a.xy + offset_ndc * side, clip_a.zw);
    out.color = model.color;
    return out;
}

@fragment
fn fs_outline(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}