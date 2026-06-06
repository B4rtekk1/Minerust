struct ShadowCascadeUniform {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> shadow_pass: ShadowCascadeUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_shadow(model: VertexInput) -> ShadowVertexOutput {
    var out: ShadowVertexOutput;
    out.clip_position = shadow_pass.light_view_proj * vec4<f32>(model.position, 1.0);
    return out;
}
