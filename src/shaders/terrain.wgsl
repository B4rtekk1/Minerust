struct Uniforms {
    view_proj:           mat4x4<f32>,
    inv_view_proj:       mat4x4<f32>,
    csm_view_proj:       array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos:          vec3<f32>,
    time:                f32,
    sun_position:        vec3<f32>,
    is_underwater:       f32,
    screen_size:         vec2<f32>,
    water_level:         f32,
    reflection_mode:     f32,
    moon_position:       vec3<f32>,
    _pad1_moon:          f32,
    moon_intensity:      f32,
    wind_dir_x:          f32,
    wind_dir_z:          f32,
    wind_speed:          f32,
    _pad:                f32,
    rain_factor:         f32,
    shadows_enabled:     f32,
    sky_visibility:      f32,
};

struct ShadowConfig {
    shadow_map_size: f32,
    pcf_samples:     u32,
}

struct TemporalShadowUniforms {
    prev_view_proj:    mat4x4<f32>,
    prev_camera_pos:   vec3<f32>,
    history_weight:    f32,
    prev_sun_position: vec3<f32>,
    history_valid:     f32,
}

@group(0) @binding(0) var<uniform> uniforms:       Uniforms;
@group(0) @binding(1) var texture_atlas:           texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler:         sampler;
@group(0) @binding(3) var shadow_map:              texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler:          sampler_comparison;
@group(0) @binding(5) var<uniform> shadow_config: ShadowConfig;

@group(1) @binding(0) var ssr_depth: texture_2d<f32>;

@group(2) @binding(0) var output_shadow: texture_storage_2d<r32float, write>;

@group(3) @binding(0) var shadow_mask:   texture_2d<f32>;
@group(3) @binding(1) var point_sampler: sampler;
@group(3) @binding(2) var<uniform> temporal_shadow: TemporalShadowUniforms;

const MAX_PCF_SAMPLES:  i32 = 32;
const TEMPORAL_SHADOW_CLAMP: f32 = 0.35;
const TAU: f32 = 6.28318530718;

fn shadow_hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn poisson_rotation(world_pos: vec3<f32>) -> f32 {
    return shadow_hash21(world_pos.xz) * TAU;
}

fn get_poisson_sample(idx: i32, rotation: f32) -> vec2<f32> {
    var disk = array<vec2<f32>, 32>(
        vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
        vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
        vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
        vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
        vec2<f32>(-0.65476012, -0.05147385), vec2<f32>( 0.18395645,  0.89721549),
        vec2<f32>(-0.09715394, -0.00673456), vec2<f32>( 0.53472400,  0.73356543),
        vec2<f32>(-0.45611231, -0.40212851), vec2<f32>(-0.57321081,  0.65476012),
        vec2<f32>(-0.97540200, -0.07113860), vec2<f32>(-0.92034700, -0.41142000),
        vec2<f32>(-0.88451800,  0.56804100), vec2<f32>(-0.81194500, -0.90521000),
        vec2<f32>(-0.53795000,  0.71666600), vec2<f32>(-0.42094200,  0.99127200),
        vec2<f32>(-0.26114700,  0.58848800), vec2<f32>(-0.14633600, -0.25919400),
        vec2<f32>(-0.13943900, -0.88866800), vec2<f32>( 0.01168860,  0.32639500),
        vec2<f32>( 0.03805660,  0.62547700), vec2<f32>( 0.06259350, -0.50853000),
        vec2<f32>( 0.16946900, -0.99725300), vec2<f32>( 0.35917200, -0.63371700),
        vec2<f32>( 0.74315600, -0.50517300), vec2<f32>( 0.86541300,  0.76372600),
    );
    let p = disk[idx];
    let s = sin(rotation);
    let c = cos(rotation);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn sample_cascade_pcf(
    world_pos:     vec3<f32>,
    cascade_idx:   i32,
    bias:          f32,
    rotation:      f32,
) -> f32 {
    let sp = uniforms.csm_view_proj[cascade_idx] * vec4<f32>(world_pos, 1.0);
    if sp.w <= 0.0 { return 1.0; }

    let sc = sp.xyz / sp.w;
    let uv = vec2<f32>(sc.x * 0.5 + 0.5, 1.0 - (sc.y * 0.5 + 0.5));
    if sc.z < 0.0 || sc.z > 1.0 { return 1.0; }

    let pcf_samples = min(i32(shadow_config.pcf_samples), MAX_PCF_SAMPLES);
    if pcf_samples <= 0 { return 1.0; }

    var shadow = 0.0;
    let shadow_map_size = max(shadow_config.shadow_map_size, 1.0);
    let cascade_filter_texels = array<f32, 4>(1.55, 2.00, 2.60, 3.30);
    let filter_radius = cascade_filter_texels[cascade_idx] / shadow_map_size;
    let texel = 1.0 / shadow_map_size;
    let depth_ref = clamp(sc.z - bias, 0.0, 1.0);
    let edge_dist = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let edge_fade = smoothstep(0.0, filter_radius + texel, edge_dist);

    for (var i = 0; i < pcf_samples; i++) {
        let suv = clamp(
            uv + get_poisson_sample(i, rotation) * filter_radius,
            vec2<f32>(texel),
            vec2<f32>(1.0 - texel),
        );
        shadow += textureSampleCompareLevel(shadow_map, shadow_sampler, suv, cascade_idx, depth_ref);
    }

    return mix(1.0, shadow / f32(pcf_samples), edge_fade);
}

fn select_cascade_with_blend(view_depth: f32) -> vec2<f32> {
    let bf = 0.10;
    let splits = array<f32, 3>(
        uniforms.csm_split_distances.x,
        uniforms.csm_split_distances.y,
        uniforms.csm_split_distances.z,
    );

    for (var i = 0; i < 3; i++) {
        let blend_start = splits[i] * (1.0 - bf);
        if view_depth < blend_start { return vec2<f32>(f32(i), 0.0); }
        if view_depth < splits[i] {
            let t = (view_depth - blend_start) / (splits[i] - blend_start);
            return vec2<f32>(f32(i), smoothstep(0.0, 1.0, t));
        }
    }
    return vec2<f32>(3.0, 0.0);
}

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

    let sky_light = sky_color * sky_energy * (0.35 + 0.65 * sky_visibility);
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

fn calculate_shadow(
    world_pos:  vec3<f32>,
    normal:     vec3<f32>,
    sun_dir:    vec3<f32>,
    view_depth: f32,
) -> f32 {
    if sun_dir.y < 0.05 { return 0.0; }

    let cos_t = max(dot(normal, sun_dir), 0.0);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));

    let rot = poisson_rotation(world_pos);

    let cb = select_cascade_with_blend(view_depth);
    let ci = i32(cb.x);

    let cascade_bias_scale = array<f32, 4>(1.0, 1.2, 1.45, 1.75);
    let base_bias = clamp(0.00035 + 0.0012 * sin_t / max(cos_t, 0.10), 0.00035, 0.0035);

    let shadow_a = sample_cascade_pcf(world_pos, ci, base_bias * cascade_bias_scale[ci], rot);

    if cb.y > 0.001 && ci < 3 {
        let next_ci = ci + 1;
        let shadow_b =
            sample_cascade_pcf(world_pos, next_ci, base_bias * cascade_bias_scale[next_ci], rot);
        return mix(shadow_a, shadow_b, cb.y);
    }
    return shadow_a;
}

fn sample_shadow_mask_bilinear(uv: vec2<f32>) -> f32 {
    let dims_u = textureDimensions(shadow_mask);
    let dims = vec2<f32>(dims_u);
    let clamped_uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let pos = clamp(
        clamped_uv * dims - vec2<f32>(0.5),
        vec2<f32>(0.0),
        dims - vec2<f32>(1.0),
    );
    let base = floor(pos);
    let frac_part = fract(pos);
    let max_coord = vec2<i32>(dims_u) - vec2<i32>(1);

    let p00 = clamp(vec2<i32>(base), vec2<i32>(0), max_coord);
    let p10 = clamp(p00 + vec2<i32>(1, 0), vec2<i32>(0), max_coord);
    let p01 = clamp(p00 + vec2<i32>(0, 1), vec2<i32>(0), max_coord);
    let p11 = clamp(p00 + vec2<i32>(1, 1), vec2<i32>(0), max_coord);

    let s00 = textureLoad(shadow_mask, p00, 0).r;
    let s10 = textureLoad(shadow_mask, p10, 0).r;
    let s01 = textureLoad(shadow_mask, p01, 0).r;
    let s11 = textureLoad(shadow_mask, p11, 0).r;

    let sx0 = mix(s00, s10, frac_part.x);
    let sx1 = mix(s01, s11, frac_part.x);
    return mix(sx0, sx1, frac_part.y);
}

fn sample_shadow_history(uv: vec2<f32>) -> f32 {
    return sample_shadow_mask_bilinear(uv);
}

fn temporal_shadow_accumulation(world_pos: vec3<f32>, current_shadow: f32) -> f32 {
    if temporal_shadow.history_valid < 0.5 || temporal_shadow.history_weight <= 0.001 {
        return current_shadow;
    }

    let prev_clip = temporal_shadow.prev_view_proj * vec4<f32>(world_pos, 1.0);
    if prev_clip.w <= 0.0 {
        return current_shadow;
    }

    let prev_ndc = prev_clip.xyz / prev_clip.w;
    if prev_ndc.z < 0.0 || prev_ndc.z > 1.0 {
        return current_shadow;
    }

    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 1.0 - (prev_ndc.y * 0.5 + 0.5));
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        return current_shadow;
    }

    let edge_dist = min(min(prev_uv.x, prev_uv.y), min(1.0 - prev_uv.x, 1.0 - prev_uv.y));
    let edge_fade = smoothstep(0.0, 0.03, edge_dist);
    let history = sample_shadow_history(prev_uv);
    let clamped_history = clamp(
        history,
        current_shadow - TEMPORAL_SHADOW_CLAMP,
        current_shadow + TEMPORAL_SHADOW_CLAMP,
    );

    return mix(current_shadow, clamped_history, temporal_shadow.history_weight * edge_fade);
}

fn sample_screen_shadow(screen_pos: vec4<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(shadow_mask));
    let uv = screen_pos.xy / max(dims, vec2<f32>(1.0));
    return clamp(sample_shadow_mask_bilinear(uv), 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn compute_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(ssr_depth);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    if (shadow_config.pcf_samples == 0u) {
        textureStore(output_shadow, gid.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let depth = textureLoad(ssr_depth, gid.xy, 0).r;
    if depth >= 0.999999 {
        textureStore(output_shadow, gid.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(tex_size);
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,
        depth,
        1.0,
    );
    let wp4 = uniforms.inv_view_proj * ndc;
    let world_pos = wp4.xyz / max(wp4.w, 1e-6);

    let view_depth = length(world_pos - uniforms.camera_pos);

    let sun_dir = normalize(uniforms.sun_position);

    var shadow_factor = 1.0;
    if (sun_dir.y > 0.0) {
        shadow_factor = calculate_shadow(world_pos, vec3<f32>(0.0, 1.0, 0.0), sun_dir, view_depth);
    }
    shadow_factor = temporal_shadow_accumulation(world_pos, shadow_factor);

    textureStore(output_shadow, gid.xy, vec4<f32>(shadow_factor, 0.0, 0.0, 0.0));
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:  vec3<f32>,
    @location(1) normal:     vec3<f32>,
    @location(2) color:      vec3<f32>,
    @location(3) uv:         vec2<f32>,
    @location(4) tex_index:  f32,
    @location(5) view_depth: f32,
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
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), // -X, +X
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), // -Y, +Y
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)  // -Z, +Z
    );

    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos     = model.position;
    out.normal        = normals[n_idx % 6u];
    out.color         = vec3<f32>(r, g, b);

    let raw_uv = uvs[uv_idx % 4u];
    out.uv = vec2<f32>(raw_uv.x * width, raw_uv.y * height);

    out.tex_index     = f32(t_idx);
    out.view_depth    = out.clip_position.w;
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(model.position, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }

    let sun_dir = normalize(uniforms.sun_position);

    let day_factor      = clamp(sun_dir.y, 0.0, 1.0);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);

    let normal = normalize(in.normal);
    var shadow = 1.0;
    if uniforms.shadows_enabled > 0.5 && sun_dir.y > 0.0 {
        shadow = sample_screen_shadow(in.clip_position);
    }

    let indirect_light = fast_global_illumination(
        normal,
        sun_dir,
        day_factor,
        twilight_factor,
    );

    let sun_color = mix(vec3<f32>(1.0, 0.78, 0.52), vec3<f32>(1.0, 0.96, 0.86), day_factor);
    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.62 * shadow * day_factor;
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0)
        * 0.045
        * day_factor;

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let face_contrast = mix(0.82, 1.0, face_shade);
    let total_light =
        (indirect_light + sun_color * sun_diff + vec3<f32>(0.58, 0.68, 0.82) * fill_diff)
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
