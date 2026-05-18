struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    csm_view_proj: array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
    screen_size: vec2<f32>,
    water_level: f32,
    _pad_water: f32,
    moon_position: vec3<f32>,
    _pad1_moon: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var scene_texture: texture_2d<f32>;

@group(0) @binding(2)
var composite_sampler: sampler;

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
);

var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(scene_texture, composite_sampler, in.uv);
}
