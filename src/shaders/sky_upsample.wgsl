@group(0) @binding(0)
var sky_texture: texture_2d<f32>;

@group(0) @binding(1)
var sky_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0), vec2<f32>(2.0, 1.0), vec2<f32>(0.0, -1.0));
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Linear filtering deliberately hides the lower-resolution target; this
    // pass is only a backdrop and is covered by all opaque scene geometry.
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(sky_texture, sky_sampler, uv, 0.0);
}
