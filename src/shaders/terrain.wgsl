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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    params:          vec4<f32>,
};

@group(1) @binding(0) var<uniform> shadow_uniforms: ShadowUniforms;
@group(1) @binding(1) var shadow_map: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;

const SHADOW_FILTER_RADIUS_TEXELS: f32 = 1.25;
const SHADOW_EDGE_FADE_TEXELS: f32 = 24.0;
const SHADOW_RECEIVER_BIAS_SCALE: f32 = 1.35;
const SHADOW_GRAZING_BIAS_POWER: f32 = 1.6;
const SHADOW_MAX_DYNAMIC_BIAS: f32 = 0.0032;

fn fast_global_illumination(
    normal:          vec3<f32>,
    sun_dir:         vec3<f32>,
    day_factor:      f32,
    twilight_factor: f32,
) -> vec3<f32> {
    let sky_visibility    = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    let ground_visibility = clamp(0.5 - normal.y * 0.5, 0.0, 1.0);
    let side_visibility   = 1.0 - abs(normal.y);

    let sky_day      = vec3<f32>(0.50, 0.62, 0.78);
    let sky_twilight = vec3<f32>(0.96, 0.48, 0.24);
    let sky_night    = vec3<f32>(0.018, 0.024, 0.052);
    let sky_color = mix(
        mix(sky_night, sky_day, day_factor),
        sky_twilight,
        twilight_factor * 0.45,
    );

    let ground_night = vec3<f32>(0.020, 0.018, 0.020);
    let ground_day   = vec3<f32>(0.26, 0.24, 0.19);
    let grass_bleed  = vec3<f32>(0.08, 0.13, 0.06) * day_factor;
    let ground_color = mix(ground_night, ground_day, day_factor) + grass_bleed;

    let sky_energy    = mix(0.035, 0.27, day_factor) + twilight_factor * 0.08;
    let ground_energy = mix(0.010, 0.070, day_factor) + twilight_factor * 0.020;

    let sky_light    = sky_color * sky_energy * (0.35 + 0.65 * sky_visibility);
    let ground_light = ground_color
        * ground_energy
        * (ground_visibility + side_visibility * 0.35);

    let bounce_dir = normalize(vec3<f32>(-sun_dir.x, 0.32, -sun_dir.z));
    let wrap_bounce = pow(clamp(dot(normal, bounce_dir) * 0.5 + 0.5, 0.0, 1.0), 2.0);
    let sun_bounce = vec3<f32>(1.0, 0.78, 0.48)
        * wrap_bounce
        * side_visibility
        * day_factor
        * 0.045;

    return sky_light + ground_light + sun_bounce;
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) color:     vec3<f32>,
    @location(3) uv:        vec2<f32>,
    @location(4) tex_index: f32,
    @location(5) shadow_pos: vec4<f32>,
};

struct DepthVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let n_idx   = model.packed & 0x7u;
    let t_idx   = (model.packed >> 3u) & 0xFFu;
    let uv_idx  = (model.packed >> 11u) & 0x3u;
    let w_raw   = (model.packed >> 13u) & 0xFu;
    let h_raw   = (model.packed >> 17u) & 0xFu;
    let r       = f32((model.packed >> 21u) & 0xFu) / 15.0;
    let g       = f32((model.packed >> 25u) & 0xFu) / 15.0;
    let b       = f32((model.packed >> 29u) & 0x7u) / 7.0;

    let width  = f32(w_raw + 1u);
    let height = f32(h_raw + 1u);

    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0),
    );

    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos     = model.position;
    out.normal        = normals[n_idx % 6u];
    out.color         = vec3<f32>(r, g, b);
    out.shadow_pos    = shadow_uniforms.light_view_proj * vec4<f32>(model.position, 1.0);

    let raw_uv = uvs[uv_idx % 4u];
    out.uv = vec2<f32>(raw_uv.x * width, raw_uv.y * height);

    out.tex_index = f32(t_idx);
    return out;
}

@vertex
fn vs_depth(model: VertexInput) -> DepthVertexOutput {
    var out: DepthVertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

fn shadow_receiver_plane_bias(uv: vec2<f32>, depth: f32, sample_radius: f32) -> f32 {
    let shadow_coord = vec3<f32>(uv, depth);
    let dx = dpdx(shadow_coord);
    let dy = dpdy(shadow_coord);
    let determinant = dx.x * dy.y - dx.y * dy.x;
    let valid_plane = abs(determinant) > 0.000001;
    let safe_determinant = select(1.0, determinant, valid_plane);
    let inv_determinant = 1.0 / safe_determinant;

    let depth_du = (dy.y * dx.z - dx.y * dy.z) * inv_determinant;
    let depth_dv = (-dy.x * dx.z + dx.x * dy.z) * inv_determinant;
    let receiver_span = (abs(depth_du) + abs(depth_dv)) * sample_radius;

    return select(0.0, receiver_span * SHADOW_RECEIVER_BIAS_SCALE, valid_plane);
}

fn shadow_dynamic_depth_bias(
    uv: vec2<f32>,
    depth: f32,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
    sample_radius: f32,
) -> f32 {
    let sun_facing = clamp(dot(normal, sun_dir), 0.0, 1.0);
    let min_bias = shadow_uniforms.params.z;
    let grazing_bias =
        shadow_uniforms.params.w * pow(1.0 - sun_facing, SHADOW_GRAZING_BIAS_POWER);
    let receiver_bias = shadow_receiver_plane_bias(uv, depth, sample_radius);

    return min(min_bias + grazing_bias + receiver_bias, SHADOW_MAX_DYNAMIC_BIAS);
}

fn sun_shadow_visibility(shadow_pos: vec4<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let sun_facing = max(dot(normal, sun_dir), 0.0);
    let height_fade = smoothstep(0.04, 0.18, sun_dir.y);
    let light_ndc = shadow_pos.xyz / shadow_pos.w;
    let uv = light_ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let texel_size = shadow_uniforms.params.x;
    let radius = texel_size * SHADOW_FILTER_RADIUS_TEXELS;
    let dynamic_bias = shadow_dynamic_depth_bias(uv, light_ndc.z, normal, sun_dir, radius);

    if sun_facing <= 0.001 || height_fade <= 0.001 {
        return 1.0;
    }

    if uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 ||
       light_ndc.z <= 0.0 || light_ndc.z >= 1.0 {
        return 1.0;
    }

    let edge_distance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let edge_fade = smoothstep(texel_size * 2.0, texel_size * SHADOW_EDGE_FADE_TEXELS, edge_distance);
    if edge_fade <= 0.001 {
        return 1.0;
    }

    let strength = clamp(shadow_uniforms.params.y * height_fade * edge_fade, 0.0, 1.0);
    let depth_ref = light_ndc.z - dynamic_bias;

    let visibility =
        textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.75, -0.25) * radius, depth_ref) +
        textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.25, -0.75) * radius, depth_ref) +
        textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.25,  0.75) * radius, depth_ref) +
        textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.75,  0.25) * radius, depth_ref);
    let filtered_visibility = visibility * 0.25;

    return mix(1.0 - strength, 1.0, filtered_visibility);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = uniforms.sun_dir;
    let day_factor = clamp(sun_dir.y, 0.0, 1.0);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);
    let normal = in.normal;
    let sun_shadow = sun_shadow_visibility(in.shadow_pos, normal, sun_dir);

    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }

    let indirect_light = fast_global_illumination(
        normal,
        sun_dir,
        day_factor,
        twilight_factor,
    );

    let sun_color = mix(vec3<f32>(1.0, 0.78, 0.52), vec3<f32>(1.0, 0.96, 0.86), day_factor);
    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.62 * day_factor;
    let ambient_shadow = mix(0.72, 1.0, sun_shadow);
    let fill_shadow = mix(0.70, 1.0, sun_shadow);
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.045 * day_factor;

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let face_contrast = mix(0.82, 1.0, face_shade);
    let total_light =
        (indirect_light * ambient_shadow
            + sun_color * sun_diff * sun_shadow
            + vec3<f32>(0.58, 0.68, 0.82) * fill_diff * fill_shadow)
        * face_contrast;
    var lit = tex.rgb * total_light * in.color;

    let sunset_factor = 1.0 - abs(sun_dir.y);
    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        lit *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }

    let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);
    let is_underwater = uniforms.is_underwater > 0.5;

    var final_color = lit;

    if is_underwater {
        final_color *= vec3<f32>(0.4, 0.7, 1.0);
        let caustic = sin(in.world_pos.x * 0.5 + uniforms.time * 2.0)
                    * sin(in.world_pos.z * 0.5 + uniforms.time * 1.5) * 0.1 + 0.9;
        final_color *= caustic;
        final_color = mix(final_color, vec3<f32>(0.05, 0.15, 0.3),
                          clamp(dist / 24.0, 0.0, 1.0) * 0.5);
    }

    return vec4<f32>(final_color, 1.0);
}
