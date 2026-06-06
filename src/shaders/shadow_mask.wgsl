struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_dir:        vec3<f32>,
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

struct ShadowUniforms {
    light_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    camera_forward: vec3<f32>,
    shadow_strength: f32,
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
@group(0) @binding(2) var<uniform> shadows: ShadowUniforms;
@group(0) @binding(3) var shadow_map: texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;

const POISSON_DISK: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.613392,  0.617481),
    vec2<f32>( 0.170019, -0.040254),
    vec2<f32>(-0.299417,  0.791925),
    vec2<f32>( 0.645680,  0.493210),
    vec2<f32>(-0.651784, -0.717887),
    vec2<f32>( 0.421003,  0.027070),
    vec2<f32>(-0.817194, -0.271096),
    vec2<f32>( 0.977050, -0.108615),
);

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
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

fn cascade_split(index: i32) -> f32 {
    switch index {
        case 0: { return shadows.cascade_splits.x; }
        case 1: { return shadows.cascade_splits.y; }
        case 2: { return shadows.cascade_splits.z; }
        default: { return shadows.cascade_splits.w; }
    }
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
        depth,
        1.0,
    );
    let world_h = uniforms.inv_view_proj * clip;
    return world_h.xyz / world_h.w;
}

fn reconstruct_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let dx = dpdx(world_pos);
    let dy = dpdy(world_pos);
    let raw_normal = cross(dy, dx);
    if dot(raw_normal, raw_normal) < 0.00000001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    var normal = normalize(raw_normal);
    let to_camera = normalize(uniforms.camera_pos - world_pos);
    if dot(normal, to_camera) < 0.0 {
        normal = -normal;
    }
    return normal;
}

fn sample_shadow_cascade(
    cascade: i32,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
) -> f32 {
    let ndotl = clamp(dot(normal, sun_dir), 0.0, 1.0);
    let normal_offset = normal * (0.025 + (1.0 - ndotl) * 0.035);
    let light_clip = shadows.light_view_proj[u32(cascade)] * vec4<f32>(world_pos + normal_offset, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let uv = light_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);

    if uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 || light_ndc.z <= 0.0 || light_ndc.z >= 1.0 {
        return 1.0;
    }

    let texel = 1.0 / max(shadows.params.x, 1.0);
    let cascade_radius = shadows.params.z * (1.0 + f32(cascade) * 0.35) * texel;
    let depth_bias = max(0.00035, 0.0012 * (1.0 - ndotl));
    let compare_depth = light_ndc.z - depth_bias;

    var visibility = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let sample_uv = clamp(
            uv + POISSON_DISK[i] * cascade_radius,
            vec2<f32>(0.001),
            vec2<f32>(0.999),
        );
        visibility += textureSampleCompare(
            shadow_map,
            shadow_sampler,
            sample_uv,
            cascade,
            compare_depth,
        );
    }

    return visibility * 0.125;
}

fn cascaded_shadow_visibility(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
) -> f32 {
    if shadows.shadow_strength <= 0.001 || sun_dir.y <= 0.02 {
        return 1.0;
    }

    let view_depth = dot(world_pos - uniforms.camera_pos, shadows.camera_forward);
    if view_depth <= 0.0 || view_depth >= shadows.cascade_splits.w {
        return 1.0;
    }

    var cascade = 0;
    if view_depth > shadows.cascade_splits.x { cascade = 1; }
    if view_depth > shadows.cascade_splits.y { cascade = 2; }
    if view_depth > shadows.cascade_splits.z { cascade = 3; }

    var visibility = sample_shadow_cascade(cascade, world_pos, normal, sun_dir);

    if cascade < 3 {
        let split = cascade_split(cascade);
        var previous_split = 0.0;
        if cascade > 0 {
            previous_split = cascade_split(cascade - 1);
        }
        let blend_width = max((split - previous_split) * 0.12, 3.0);
        let blend = smoothstep(split - blend_width, split, view_depth);
        if blend > 0.0 {
            let next_visibility = sample_shadow_cascade(cascade + 1, world_pos, normal, sun_dir);
            visibility = mix(visibility, next_visibility, blend);
        }
    }

    let fade_start = shadows.cascade_splits.w * 0.86;
    let distance_fade = 1.0 - smoothstep(fade_start, shadows.cascade_splits.w, view_depth);
    let strength = shadows.shadow_strength * distance_fade;
    return mix(1.0, visibility, strength);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let depth_size = textureDimensions(scene_depth);
    let depth_px = clamp(
        vec2<i32>(uv * vec2<f32>(f32(depth_size.x), f32(depth_size.y))),
        vec2<i32>(0),
        vec2<i32>(i32(depth_size.x) - 1, i32(depth_size.y) - 1),
    );
    let depth = textureLoad(scene_depth, depth_px, 0).r;
    if depth >= 0.99999 {
        return 1.0;
    }

    let world_pos = reconstruct_world_position(uv, depth);
    let normal = reconstruct_normal(world_pos);
    return cascaded_shadow_visibility(world_pos, normal, normalize(uniforms.sun_dir));
}
