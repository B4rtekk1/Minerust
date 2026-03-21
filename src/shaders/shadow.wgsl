struct ShadowUniforms {
    cascade_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: ShadowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    // Depth-only shadow pass for terrain geometry. The visible terrain pass
    // does not apply any vertex deformation here, so the shadow map must use
    // the same positions to avoid swimming / offset shadows.
    return uniforms.cascade_view_proj * vec4<f32>(model.position, 1.0);
}
