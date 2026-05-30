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
    reflection_mode: f32,
    moon_position: vec3<f32>,
    _pad1_moon: f32,
    moon_intensity: f32,
    wind_dir_x: f32,
    wind_dir_z: f32,
    wind_speed: f32,
    _pad: f32,
    rain_factor: f32,
    shadows_enabled: f32,
    sky_visibility: f32,
    menu_blur: f32,
    _pad_menu0: f32,
    _pad_menu1: f32,
    _pad_menu2: f32,
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

fn sample_scene(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(scene_texture, composite_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

fn cover_uv(uv: vec2<f32>, blur: f32) -> vec2<f32> {
    if blur <= 0.001 {
        return uv;
    }

    let texture_size_u = textureDimensions(scene_texture);
    let texture_size = vec2<f32>(f32(texture_size_u.x), f32(texture_size_u.y));
    let screen_aspect = uniforms.screen_size.x / max(uniforms.screen_size.y, 1.0);
    let texture_aspect = texture_size.x / max(texture_size.y, 1.0);
    var covered = uv;

    if screen_aspect > texture_aspect {
        covered.y = 0.5 + (uv.y - 0.5) * (texture_aspect / screen_aspect);
    } else {
        covered.x = 0.5 + (uv.x - 0.5) * (screen_aspect / texture_aspect);
    }

    return covered;
}

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
    let blur = clamp(uniforms.menu_blur, 0.0, 1.0);
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let sample_uv = cover_uv(uv, blur);
    let texture_size_u = textureDimensions(scene_texture);
    let texture_size = vec2<f32>(f32(texture_size_u.x), f32(texture_size_u.y));
    let texel = vec2<f32>(1.0, 1.0) / max(texture_size, vec2<f32>(1.0, 1.0));

    var color = sample_scene(sample_uv);

    if blur > 0.001 {
        var blurred = color * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 4.0,  0.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-4.0,  0.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0,  4.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0, -4.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 5.5,  5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-5.5,  5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 5.5, -5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-5.5, -5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 10.0,  0.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-10.0,  0.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0,  10.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0, -10.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 14.0,  14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-14.0,  14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 14.0, -14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-14.0, -14.0)) * 0.03;

        color = mix(color, blurred * 0.82, blur);
    }

    return vec4<f32>(color, 1.0);
}
