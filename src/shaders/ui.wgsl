struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    reflection_mode: f32,
    moon_position:  vec3<f32>,
    _pad1_moon:     f32,
    moon_intensity: f32,
    wind_dir_x:     f32,
    wind_dir_z:     f32,
    wind_speed:     f32,
    rain_factor:    f32,
    sky_visibility: f32,
    menu_blur:      f32,
    _pad_uniforms:  f32,
};


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2)
var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tex_index: f32,
};

@vertex
fn vs_ui(model: VertexInput) -> VertexOutput {
    let r = f32((model.packed >> 21u) & 0xFu) / 15.0;
    let g = f32((model.packed >> 25u) & 0xFu) / 15.0;
    let b = f32((model.packed >> 29u) & 0x7u) / 7.0;
    let alpha_lo = (model.packed >> 13u) & 0xFu;
    let alpha_hi = (model.packed >> 17u) & 0xFu;
    let a = f32((alpha_hi << 4u) | alpha_lo) / 255.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.xy, 0.0, 1.0);
    out.color = vec4<f32>(r, g, b, a);
    out.uv = vec2<f32>(0.0, 0.0);
    out.tex_index = f32((model.packed >> 3u) & 0xFFu);
    return out;
}

@fragment
fn fs_ui(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
