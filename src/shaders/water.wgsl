const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const SHADOW_MAP_SIZE: f32 = 2048.0;
const PCF_SAMPLES: i32 = 20;

const SSR_MAX_STEPS: i32 = 32;
const SSR_BINARY_STEPS: i32 = 4;
const SSR_MAX_DISTANCE: f32 = 60.0;
const SSR_THICKNESS_BASE: f32 = 0.005;
const SSR_EARLY_EXIT_CONFIDENCE: f32 = 0.92;
const SSR_EDGE_FADE: f32 = 0.08;

const LOD_NEAR: f32 = 0.0;
const LOD_FAR: f32 = 100.0;
const NORMAL_BLEND_DISTANCE: f32 = 100.0;
const NORMAL_BLEND_MIN: f32 = 0.3;
const SSR_FADE_DISTANCE: f32 = 150.0;

const WATER_LEVEL_OFFSET: f32 = 0.15;
const FRESNEL_R0: f32 = 0.02;
const REFRACTION_STRENGTH: f32 = 0.025;
const REFRACTION_MIX: f32 = 0.55;

const ABSORPTION_R: f32 = 0.08;
const ABSORPTION_G: f32 = 0.02;
const ABSORPTION_B: f32 = 0.01;
const SCATTER_COLOR: vec3<f32> = vec3<f32>(0.04, 0.16, 0.22);
const SHALLOW_COLOR: vec3<f32> = vec3<f32>(0.1, 0.4, 0.35);

const FOAM_THRESHOLD: f32 = 0.35;
const FOAM_INTENSITY: f32 = 0.75;
const FOAM_EDGE_WIDTH: f32 = 0.5;

const SSS_STRENGTH: f32 = 0.28;
const SSS_DISTORTION: f32 = 0.15;
const SSS_POWER: f32 = 3.0;
const SSS_COLOR: vec3<f32> = vec3<f32>(0.1, 0.6, 0.45);

const WATER_ROUGHNESS: f32 = 0.04;

const AMBIENT_DAY: f32 = 0.4;
const AMBIENT_NIGHT: f32 = 0.008;
const SHADOW_CONTRIBUTION: f32 = 0.6;

const UNDERWATER_VISIBILITY: f32 = 20.0;
const FOG_VISIBILITY_NIGHT: f32 = 20.0;
const FOG_VISIBILITY_DAY: f32 = 250.0;
const FOG_START_RATIO: f32 = 0.2;

const SHADOW_BASE_BIAS: f32 = 0.001;
const SHADOW_SLOPE_BIAS: f32 = 0.003;
const SHADOW_EDGE_FADE: f32 = 0.03;

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
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(3) var shadow_map: texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var ssr_color: texture_2d<f32>;
@group(0) @binding(6) var ssr_depth: texture_depth_2d;
@group(0) @binding(7) var ssr_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
    @location(5) view_depth: f32,
    @location(6) original_pos: vec2<f32>,
};

const WAVE_K:     array<f32, 6> = array(0.5236, 0.6832, 0.9975, 1.6988, 2.9920, 4.4880);
const WAVE_C:     array<f32, 6> = array(0.3461, 0.2280, 0.3130, 0.4325, 0.4531, 0.4434);
const WAVE_AMP:   array<f32, 6> = array(0.08,   0.045,  0.035,  0.018,  0.013,  0.01);
const WAVE_STEEP: array<f32, 6> = array(0.6,    0.5,    0.45,   0.35,   0.3,    0.25);
const WAVE_DIR_X: array<f32, 6> = array(0.9743, -0.7318, 0.4110, -0.8886, 0.1499, 0.6172);
const WAVE_DIR_Y: array<f32, 6> = array(0.2251, 0.6815, -0.9117, 0.4588, 0.9887, 0.7868);
const WAVE_PHASE: array<f32, 6> = array(0.0,    1.57,   3.14,   4.71,   2.09,   5.23);

struct GerstnerDualResult {
    displacement: vec3<f32>,
    normal: vec3<f32>,
    spark_normal: vec3<f32>,
    foam: f32,
    wave_height: f32,
}

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(q) * 43758.5453) * 2.0 - 1.0;
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let h = vec4(
        fract(dot(i,              vec2(127.1, 311.7)) * 43758.5453),
        fract(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7)) * 43758.5453),
        fract(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7)) * 43758.5453),
        fract(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7)) * 43758.5453)
    );
    return mix(mix(h.x, h.y, u.x), mix(h.z, h.w, u.x), u.y);
}

fn foam_noise_fast(p: vec2<f32>, time: f32) -> f32 {
    let t = time * 0.25;
    let n1 = value_noise2(p * 1.8 + vec2(t * 0.6,  t * 0.25));
    let n2 = value_noise2(p * 3.7 - vec2(t * 0.4,  t * 0.7));
    return n1 * 0.65 + n2 * 0.35;
}

fn gradient_noise_deriv(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);

    let ga = hash2(i);
    let gb = hash2(i + vec2(1.0, 0.0));
    let gc = hash2(i + vec2(0.0, 1.0));
    let gd = hash2(i + vec2(1.0, 1.0));

    let va = dot(ga, f);
    let vb = dot(gb, f - vec2(1.0, 0.0));
    let vc = dot(gc, f - vec2(0.0, 1.0));
    let vd = dot(gd, f - vec2(1.0, 1.0));

    let value = mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y);
    let deriv = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd)
              + du * (u.yx * (va - vb - vc + vd) + vec2(vb - va, vc - va));
    return vec3(value, deriv);
}

fn fbm_detail_deriv(p: vec2<f32>, time: f32) -> vec2<f32> {
    let t = time * 0.4;
    var freq_p = p + vec2(t * 0.7, t * 0.3);
    var d_total = vec2(0.0);
    var amplitude = 0.5;
    var freq_scale = 1.0;
    for (var i = 0; i < 3; i++) {
        let n = gradient_noise_deriv(freq_p);
        d_total += amplitude * freq_scale * n.yz;
        freq_p = freq_p * 2.17 + vec2(1.7, 3.2);
        freq_scale *= 2.17;
        amplitude *= 0.45;
    }
    return d_total;
}

fn calculate_gerstner_dual(pos: vec3<f32>, time: f32, camera_pos: vec3<f32>) -> GerstnerDualResult {
    let dist = length(pos.xz - camera_pos.xz);
    let lod_factor = 1.0 - clamp((dist - LOD_NEAR) / (LOD_FAR - LOD_NEAR), 0.0, 1.0);
    let smooth_lod = lod_factor * lod_factor * lod_factor;

    var result: GerstnerDualResult;
    result.displacement = vec3(0.0);
    result.normal = vec3(0.0, 1.0, 0.0);
    result.spark_normal = vec3(0.0, 1.0, 0.0);
    result.foam = 0.0;
    result.wave_height = 0.0;

    if smooth_lod < 0.005 {
        return result;
    }

    var x_offset: f32 = 0.0;
    var y_offset: f32 = 0.0;
    var z_offset: f32 = 0.0;
    var dx: f32 = 0.0;
    var dz: f32 = 0.0;
    var spark_dx: f32 = 0.0;
    var spark_dz: f32 = 0.0;
    var j_xx: f32 = 0.0;
    var j_zz: f32 = 0.0;
    var j_xz: f32 = 0.0;

    let p = pos.xz;
    var max_amplitude: f32 = 0.0;
    let compute_spark = dist < 80.0;

    for (var i: i32 = 0; i < 6; i++) {
        let w_k = WAVE_K[i];
        let w_dir = vec2(WAVE_DIR_X[i], WAVE_DIR_Y[i]);
        let f = w_k * (dot(w_dir, p) - WAVE_C[i] * time) + WAVE_PHASE[i];
        let intensity = WAVE_AMP[i] * smooth_lod;
        let sin_f = sin(f);
        let cos_f = cos(f);
        let Q = WAVE_STEEP[i] * smooth_lod;

        if i < 4 {
            let q_int_cos = Q * intensity * cos_f;
            x_offset -= q_int_cos * w_dir.x;
            z_offset -= q_int_cos * w_dir.y;
            y_offset += intensity * sin_f;
            max_amplitude += intensity;
        }

        let ka_sin = w_k * intensity * sin_f * Q;
        j_xx += w_dir.x * w_dir.x * ka_sin;
        j_zz += w_dir.y * w_dir.y * ka_sin;
        j_xz += w_dir.x * w_dir.y * ka_sin;

        let df = intensity * w_k * cos_f;
        dx += w_dir.x * df;
        dz += w_dir.y * df;

        if compute_spark {
            let spark_f = w_k * (dot(w_dir, p * 2.0) - WAVE_C[i] * time * 1.5) + WAVE_PHASE[i];
            let spark_df = intensity * w_k * cos(spark_f);
            spark_dx += w_dir.x * spark_df;
            spark_dz += w_dir.y * spark_df;
        }
    }

    let detail_blend = smooth_lod * clamp(1.0 - dist / 60.0, 0.0, 1.0);
    if detail_blend > 0.01 {
        let d = fbm_detail_deriv(p * 0.8, time) * 0.18 * detail_blend;
        dx += d.x;
        dz += d.y;
    }

    result.displacement = vec3(x_offset, y_offset, z_offset);
    result.normal = normalize(vec3(-dx, 1.0, -dz));
    if compute_spark {
        result.spark_normal = normalize(vec3(-spark_dx, 1.0, -spark_dz));
    }

    let jacobian = (1.0 - j_xx) * (1.0 - j_zz) - j_xz * j_xz;
    result.foam = clamp(1.0 - jacobian - FOAM_THRESHOLD, 0.0, 1.0) * FOAM_INTENSITY;

    if max_amplitude > 0.001 {
        result.wave_height = clamp(y_offset / max_amplitude, -1.0, 1.0);
    }

    return result;
}

@vertex
fn vs_water(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = model.position;
    out.original_pos = pos.xz;

    if model.normal.y > 0.5 {
        let waves = calculate_gerstner_dual(pos, uniforms.time, uniforms.camera_pos);
        pos.x += waves.displacement.x;
        pos.z += waves.displacement.z;
        pos.y += waves.displacement.y - WATER_LEVEL_OFFSET;
    } else {

        let top_edge = 1.0 - model.uv.y;
        if top_edge > 0.001 {
             let waves = calculate_gerstner_dual(pos, uniforms.time, uniforms.camera_pos);
             pos.x += waves.displacement.x * top_edge;
             pos.z += waves.displacement.z * top_edge;
             pos.y += (waves.displacement.y - WATER_LEVEL_OFFSET) * top_edge;
        }
    }
    out.clip_position = uniforms.view_proj * vec4(pos, 1.0);
    out.world_pos = pos;
    out.normal = model.normal.xyz;
    out.color = model.color.rgb;
    out.uv = model.uv;
    out.tex_index = model.tex_index;
    out.view_depth = out.clip_position.w;
    return out;
}

fn schlick_fresnel(cos_theta: f32, r0: f32) -> f32 {
    let x = 1.0 - cos_theta;
    let x2 = x * x;
    return r0 + (1.0 - r0) * x2 * x2 * x;
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn ggx_specular(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, roughness: f32) -> f32 {
    let half_vec = normalize(view_dir + light_dir);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let D = ggx_distribution(n_dot_h, roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);
    let F = schlick_fresnel(max(dot(half_vec, view_dir), 0.0), FRESNEL_R0);
    return (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);
}

fn calculate_sss(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>, wave_height: f32, thickness: f32) -> vec3<f32> {
    let sss_dir = normalize(light_dir + normal * SSS_DISTORTION);
    let sss_dot = pow(clamp(dot(view_dir, -sss_dir), 0.0, 1.0), SSS_POWER);
    let height_factor = clamp(wave_height * 0.5 + 0.5, 0.0, 1.0);
    let crest_thin = pow(height_factor, 1.5);
    return SSS_COLOR * sss_dot * SSS_STRENGTH * (0.3 + crest_thin * 0.7) * thickness;
}

fn calculate_absorption(depth: f32) -> vec3<f32> {
    return exp(-vec3(ABSORPTION_R, ABSORPTION_G, ABSORPTION_B) * depth);
}

fn world_space_noise(world_pos: vec3<f32>) -> f32 {
    let cell = floor(world_pos);
    return fract(sin(dot(cell.xz, vec2(127.1, 311.7))) * 43758.5453);
}

fn get_poisson_sample(idx: i32, rotation: f32) -> vec2<f32> {
    var p: vec2<f32>;
    switch (idx) {
        case 0:  { p = vec2(-0.94201624, -0.39906216); }
        case 1:  { p = vec2( 0.94558609, -0.76890725); }
        case 2:  { p = vec2(-0.094184101,-0.92938870); }
        case 3:  { p = vec2( 0.34495938,  0.29387760); }
        case 4:  { p = vec2(-0.91588581,  0.45771432); }
        case 5:  { p = vec2(-0.81544232, -0.87912464); }
        case 6:  { p = vec2(-0.38277543,  0.27676845); }
        case 7:  { p = vec2( 0.97484398,  0.75648379); }
        case 8:  { p = vec2( 0.44323325, -0.97511554); }
        case 9:  { p = vec2( 0.53742981, -0.47373420); }
        case 10: { p = vec2(-0.65476012, -0.051473853); }
        case 11: { p = vec2( 0.18395645,  0.89721549); }
        case 12: { p = vec2(-0.097153940,-0.006734560); }
        case 13: { p = vec2( 0.53472400,  0.73356543); }
        case 14: { p = vec2(-0.45611231, -0.40212851); }
        case 15: { p = vec2(-0.57321081,  0.65476012); }
        default: { p = vec2(0.0, 0.0); }
    }
    let s = sin(rotation);
    let c = cos(rotation);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn sample_cascade_pcf(
    world_pos: vec3<f32>,
    cascade_idx: i32,
    bias: f32,
    rotation_phi: f32,
    filter_radius: f32,
) -> f32 {
    let shadow_pos    = uniforms.csm_view_proj[cascade_idx] * vec4(world_pos, 1.0);
    let shadow_coords = shadow_pos.xyz / shadow_pos.w;
    let uv = vec2(shadow_coords.x * 0.5 + 0.5, 1.0 - (shadow_coords.y * 0.5 + 0.5));

    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 { return 1.0; }

    let edge_factor       = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)) / SHADOW_EDGE_FADE;
    let edge_shadow_blend = clamp(edge_factor, 0.0, 1.0);

    let receiver_depth = shadow_coords.z;
    var shadow: f32 = 0.0;
    for (var i: i32 = 0; i < PCF_SAMPLES; i++) {
        let offset = get_poisson_sample(i, rotation_phi) * filter_radius;
        let sample_uv = uv + offset;
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            shadow += 1.0;
        } else {
            shadow += textureSampleCompare(shadow_map, shadow_sampler, sample_uv, cascade_idx, receiver_depth - bias);
        }
    }
    shadow /= f32(PCF_SAMPLES);
    return mix(1.0, shadow, edge_shadow_blend);
}

fn select_cascade_with_blend(view_depth: f32) -> vec2<f32> {
    let bf = 0.10;
    let s0 = uniforms.csm_split_distances.x;
    let s1 = uniforms.csm_split_distances.y;
    let s2 = uniforms.csm_split_distances.z;

    if view_depth < s0 * (1.0 - bf) { return vec2(0.0, 0.0); }
    else if view_depth < s0 { return vec2(0.0, smoothstep(0.0, 1.0, (view_depth - s0*(1.0-bf)) / (s0*bf))); }

    if view_depth < s1 * (1.0 - bf) { return vec2(1.0, 0.0); }
    else if view_depth < s1 { return vec2(1.0, smoothstep(0.0, 1.0, (view_depth - s1*(1.0-bf)) / (s1*bf))); }

    if view_depth < s2 * (1.0 - bf) { return vec2(2.0, 0.0); }
    else if view_depth < s2 { return vec2(2.0, smoothstep(0.0, 1.0, (view_depth - s2*(1.0-bf)) / (s2*bf))); }

    return vec2(3.0, 0.0);
}

fn reconstruct_world_pos(screen_uv: vec2<f32>, ndc_depth: f32) -> vec3<f32> {
    let ndc = vec4(
        screen_uv.x * 2.0 - 1.0,
        (1.0 - screen_uv.y) * 2.0 - 1.0,
        ndc_depth,
        1.0
    );
    let world_h = uniforms.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_height = sun_dir.y;
    let day_factor = clamp(sun_height, 0.0, 1.0);
    let night_factor = clamp(-sun_height, 0.0, 1.0);
    let sunset_factor = 1.0 - abs(sun_height);

    let view_height = view_dir.y;
    let view_horizontal_vec = vec3(view_dir.x, 0.0, view_dir.z);
    let sun_horizontal_vec = vec3(sun_dir.x, 0.0, sun_dir.z);
    let v_len = length(view_horizontal_vec);
    let s_len = length(sun_horizontal_vec);

    var cos_angle_horizontal = 0.0;
    if v_len > 0.0001 && s_len > 0.0001 {
        cos_angle_horizontal = dot(view_horizontal_vec / v_len, sun_horizontal_vec / s_len);
    }
    let cos_angle_3d = dot(normalize(view_dir), normalize(sun_dir));

    let zenith_day   = vec3(0.25, 0.45, 0.85);
    let horizon_day  = vec3(0.65, 0.82, 0.98);
    let zenith_night = vec3(0.001, 0.001, 0.008);
    let horizon_night= vec3(0.015, 0.015, 0.03);

    let height_factor = clamp(view_height * 0.5 + 0.5, 0.0, 1.0);
    let curved_height = pow(height_factor, 0.8);

    var sky_color = mix(horizon_day, zenith_day, curved_height) * day_factor;
    sky_color += mix(horizon_night, zenith_night, height_factor) * night_factor;

    if sunset_factor > 0.01 && sun_height > -0.3 {
        let sun_proximity_3d   = max(0.0, cos_angle_3d);
        let sun_proximity_horiz= max(0.0, cos_angle_horizontal);
        let sun_proximity = mix(sun_proximity_horiz, sun_proximity_3d, 0.5);

        let glow_tight  = pow(sun_proximity_3d, 32.0);
        let glow_medium = pow(sun_proximity, 4.0);
        let glow_wide   = pow(sun_proximity, 1.5);

        let sunset_intensity = smoothstep(-0.2, 0.1, sun_height) * smoothstep(0.6, 0.0, sun_height);
        let horizon_band  = 1.0 - abs(view_height);
        let horizon_boost = pow(horizon_band, 0.5) * smoothstep(0.0, 0.1, v_len);

        var sunset_color = vec3(0.0);
        sunset_color += vec3(1.0, 0.7, 0.3) * glow_tight * 1.2;
        sunset_color += vec3(1.0, 0.4, 0.1) * glow_medium * 0.8 * horizon_boost;
        sunset_color += vec3(0.9, 0.2, 0.05) * glow_wide * 0.5 * horizon_boost;

        let opposite_glow = max(0.0, -cos_angle_horizontal) * 0.2;
        sunset_color += vec3(0.95, 0.5, 0.6) * opposite_glow * horizon_band * smoothstep(0.0, 0.1, v_len);
        sky_color = mix(sky_color, sky_color + sunset_color, sunset_intensity);
    }

    if day_factor > 0.1 {
        sky_color += vec3(1.0, 0.95, 0.9) * pow(max(0.0, cos_angle_3d), 128.0) * day_factor;
    }

    return clamp(sky_color, vec3(0.0), vec3(1.5));
}

fn ssr_trace(world_pos: vec3<f32>, reflect_dir: vec3<f32>) -> vec4<f32> {
    let dir = normalize(reflect_dir);
    var ray_world = world_pos + dir * 0.2;
    var prev_ray_world = ray_world;

    var hit_uv = vec2(0.0);
    var hit_confidence = 0.0;
    var found_hit = false;

    let inv_steps = 1.0 / f32(SSR_MAX_STEPS);

    for (var i: i32 = 0; i < SSR_MAX_STEPS; i++) {
        let t = f32(i) * inv_steps;
        let step_dist = 0.3 + t * t * 5.5;
        prev_ray_world = ray_world;
        ray_world += dir * step_dist;

        if length(ray_world - world_pos) > SSR_MAX_DISTANCE { break; }

        let ray_clip = uniforms.view_proj * vec4(ray_world, 1.0);
        if ray_clip.w <= 0.0 { break; }
        let ray_ndc = ray_clip.xyz / ray_clip.w;
        if any(ray_ndc.xy < vec2(-1.0)) || any(ray_ndc.xy > vec2(1.0)) { break; }

        let ray_uv = vec2(ray_ndc.x * 0.5 + 0.5, 0.5 - ray_ndc.y * 0.5);
        let scene_depth = textureSample(ssr_depth, ssr_sampler, ray_uv);
        let depth_diff = ray_ndc.z - scene_depth;
        let thickness = SSR_THICKNESS_BASE + step_dist * 0.1;

        if depth_diff > 0.0 && depth_diff < thickness {
            var lo = prev_ray_world;
            var hi = ray_world;
            for (var b: i32 = 0; b < SSR_BINARY_STEPS; b++) {
                let mid = (lo + hi) * 0.5;
                let mc = uniforms.view_proj * vec4(mid, 1.0);
                if mc.w <= 0.0 { break; }
                let mn = mc.xyz / mc.w;
                let mu = vec2(mn.x * 0.5 + 0.5, 0.5 - mn.y * 0.5);
                let ms = textureSample(ssr_depth, ssr_sampler, mu);
                if mn.z > ms { hi = mid; } else { lo = mid; }
            }

            let fc = uniforms.view_proj * vec4(hi, 1.0);
            if fc.w > 0.0 {
                let fn_ = fc.xyz / fc.w;
                let fu  = vec2(fn_.x * 0.5 + 0.5, 0.5 - fn_.y * 0.5);
                let fs  = textureSample(ssr_depth, ssr_sampler, fu);
                let fd  = abs(fn_.z - fs);
                if fd < thickness {
                    let hit_world = reconstruct_world_pos(fu, fs);
                    let is_above_water = hit_world.y > uniforms.water_level + 0.5;
                    let reflects_upward = dir.y > -0.05;
                    if !(reflects_upward && is_above_water) {
                        hit_uv = fu;
                        hit_confidence = 1.0 - fd / thickness;
                        found_hit = true;
                    }
                    break;
                }
            }
            break;
        }

        if hit_confidence > SSR_EARLY_EXIT_CONFIDENCE { break; }
    }

    if found_hit && hit_confidence > 0.05 {
        let edge_dist = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
        let edge_fade = smoothstep(0.0, SSR_EDGE_FADE, edge_dist);
        let final_conf = smoothstep(0.05, 0.9, hit_confidence) * edge_fade;
        if final_conf > 0.02 {
            return vec4(textureSample(ssr_color, ssr_sampler, hit_uv).rgb, final_conf);
        }
    }
    return vec4(0.0);
}

fn calculate_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
    view_depth: f32
) -> f32 {
    if sun_dir.y < 0.05 { return 0.0; }

    let cos_theta = max(dot(normal, sun_dir), 0.0);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let bias = SHADOW_BASE_BIAS + SHADOW_SLOPE_BIAS * sin_theta / max(cos_theta, 0.1);

    let noise        = world_space_noise(world_pos);
    let rotation_phi = noise * TWO_PI;
    let filter_radius = 5.0 / SHADOW_MAP_SIZE;

    let cb          = select_cascade_with_blend(view_depth);
    let cascade_idx = i32(cb.x);
    let blend       = cb.y;
    let cascade_bias_scale = 1.0 + f32(cascade_idx) * 0.5;
    let scaled_bias = bias * cascade_bias_scale;

    let shadow_a = sample_cascade_pcf(world_pos, cascade_idx, scaled_bias, rotation_phi, filter_radius);
    if blend > 0.001 && cascade_idx < 3 {
        let next_bias = bias * (1.0 + f32(cascade_idx + 1) * 0.5);
        let shadow_b = sample_cascade_pcf(world_pos, cascade_idx + 1, next_bias, rotation_phi, filter_radius);
        return mix(shadow_a, shadow_b, blend);
    }
    return shadow_a;
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {
    let to_camera      = uniforms.camera_pos - in.world_pos;
    let dist_to_camera = length(to_camera);
    let view_dir = to_camera / dist_to_camera;
    let sun_dir  = normalize(uniforms.sun_position);
    let day_factor   = clamp(sun_dir.y, 0.0, 1.0);
    let is_underwater = uniforms.is_underwater > 0.5;
    let inv_clip_w = 1.0 / in.clip_position.w;
    let clip_ndc   = in.clip_position.xyz * inv_clip_w;
    let screen_uv  = vec2(clip_ndc.x * 0.5 + 0.5, 0.5 - clip_ndc.y * 0.5);

    let original_world = vec3(in.original_pos.x, in.world_pos.y, in.original_pos.y);
    let waves = calculate_gerstner_dual(original_world, uniforms.time, uniforms.camera_pos);
    let normal_blend  = clamp(1.0 - dist_to_camera / NORMAL_BLEND_DISTANCE, NORMAL_BLEND_MIN, 1.0);
    let water_normal  = normalize(mix(in.normal, waves.normal, normal_blend));

    let cos_theta = max(dot(view_dir, water_normal), 0.0);
    let fresnel   = schlick_fresnel(cos_theta, FRESNEL_R0);

    let scene_ndc_depth = textureSample(ssr_depth, ssr_sampler, screen_uv);
    let scene_world_pos  = reconstruct_world_pos(screen_uv, scene_ndc_depth);
    let scene_lin_depth  = length(scene_world_pos - uniforms.camera_pos);
    let water_lin_depth  = dist_to_camera;
    let water_depth      = max(scene_lin_depth - water_lin_depth, 0.0);

    let absorption = calculate_absorption(water_depth);

    var base_water = textureSample(texture_atlas, texture_sampler, in.uv, i32(in.tex_index + 0.5)).rgb;
    let depth_color = mix(SHALLOW_COLOR, SCATTER_COLOR, clamp(water_depth * 0.1, 0.0, 1.0));
    base_water = mix(base_water, depth_color, clamp(water_depth * 0.2, 0.0, 0.75));

    let refract_scale  = REFRACTION_STRENGTH * (1.0 + clamp(water_depth * 0.04, 0.0, 0.4));
    let refract_offset = water_normal.xz * refract_scale * (1.0 - fresnel * 0.7);
    let use_refract    = scene_world_pos.y < uniforms.water_level + 0.2;
    let refract_uv     = clamp(screen_uv + select(vec2(0.0), refract_offset, use_refract), vec2(0.0), vec2(1.0));
    let refract_color  = textureSample(ssr_color, ssr_sampler, refract_uv).rgb * absorption;
    base_water = mix(base_water, refract_color * 0.75, REFRACTION_MIX * (1.0 - fresnel));

    var reflect_dir = reflect(-view_dir, water_normal);
    reflect_dir.y   = max(reflect_dir.y, 0.001);
    reflect_dir     = normalize(reflect_dir);
    let sky_color   = calculate_sky_color(reflect_dir, sun_dir);

    var reflection_color = sky_color;
    if i32(uniforms.reflection_mode) != 0 {
        let ssr_fade = clamp(1.0 - dist_to_camera / SSR_FADE_DISTANCE, 0.0, 1.0);
        if ssr_fade > 0.01 {
            let ssr_result = ssr_trace(in.world_pos, reflect_dir);
            let conf = smoothstep(0.05, 0.9, ssr_result.w);
            if conf > 0.02 {
                reflection_color = mix(sky_color, ssr_result.rgb, conf * 0.85 * ssr_fade);
            }
        }
    }

    var shadow = 1.0;
    let sun_up = sun_dir.y > 0.0;
    if sun_up {
        shadow = calculate_shadow(in.world_pos, in.normal, sun_dir, in.view_depth);
    }

    let ambient = mix(AMBIENT_NIGHT, AMBIENT_DAY, day_factor);
    var water_color = mix(base_water, reflection_color, fresnel);

    if day_factor > 0.05 {
        water_color += calculate_sss(water_normal, view_dir, sun_dir, waves.wave_height, day_factor) * shadow;
    }

    if sun_up {
        let sun_color = vec3(1.0, 0.98, 0.9) * shadow * day_factor;
        let n_dot_l = max(dot(water_normal, sun_dir), 0.0);
        var spec = ggx_specular(water_normal, view_dir, sun_dir, WATER_ROUGHNESS) * n_dot_l * 2.0;

        if dist_to_camera < 80.0 {
            let glitter = pow(max(dot(water_normal, waves.spark_normal), 0.0), 32.0) * 0.15;
            let spark_n_dot_l = max(dot(waves.spark_normal, sun_dir), 0.0);
            spec += ggx_specular(waves.spark_normal, view_dir, sun_dir, WATER_ROUGHNESS * 1.5) * spark_n_dot_l * 0.5;
            water_color += vec3(glitter * shadow * day_factor);
        }
        water_color += sun_color * spec;
    }

    let night_factor = clamp(-sun_dir.y, 0.0, 1.0);
    if night_factor > 0.2 {
        let moon_dir = normalize(vec3(0.3, 0.5, -0.8));
        let moon_n_dot_l = max(dot(water_normal, moon_dir), 0.0);
        water_color += vec3(0.7, 0.8, 1.0) * ggx_specular(water_normal, view_dir, moon_dir, WATER_ROUGHNESS * 2.0) * moon_n_dot_l * 0.3 * night_factor;
    }

    water_color *= (ambient + shadow * SHADOW_CONTRIBUTION * day_factor);

    let shoreline_foam = clamp(1.0 - water_depth / FOAM_EDGE_WIDTH, 0.0, 1.0);
    let total_foam = clamp(waves.foam + shoreline_foam * 0.55, 0.0, 1.0);
    if total_foam > 0.01 {
        let fn_val  = foam_noise_fast(in.world_pos.xz * 0.35, uniforms.time);
        let lit_foam = vec3(0.88, 0.92, 0.96) * (ambient + shadow * day_factor * 0.8);
        water_color = mix(water_color, lit_foam, clamp(total_foam * (0.55 + fn_val * 0.45), 0.0, 0.85));
    }

    let dist_xz = length(in.world_pos.xz - uniforms.camera_pos.xz);
    var visibility_range: f32;
    var fog_color_final: vec3<f32>;
    if is_underwater {
        visibility_range = UNDERWATER_VISIBILITY;
        fog_color_final  = vec3(0.03, 0.12, 0.25);
    } else {
        visibility_range = mix(FOG_VISIBILITY_NIGHT, FOG_VISIBILITY_DAY, day_factor);
        fog_color_final  = mix(vec3(0.001, 0.001, 0.008), sky_color, day_factor);
    }
    let fog_start  = visibility_range * FOG_START_RATIO;
    let visibility = clamp((visibility_range - dist_xz) / (visibility_range - fog_start), 0.0, 1.0);
    var final_color = mix(fog_color_final, water_color, visibility);

    if is_underwater {
        final_color *= vec3(0.4, 0.75, 1.0);
        let t  = uniforms.time;
        let c1 = sin(in.world_pos.x * 0.3 + t * 0.8) * sin(in.world_pos.z * 0.3 + t * 0.6);
        let c2 = sin(in.world_pos.x * 0.65 - t * 0.6) * sin(in.world_pos.z * 0.45 + t * 0.9);
        final_color *= pow(max(c1 * 0.6 + c2 * 0.4, 0.0), 0.6) * 0.35 + 0.8;
        final_color *= calculate_absorption(max(uniforms.water_level - in.world_pos.y, 0.0) * 0.5);
    }

    let depth_opacity = clamp(water_depth * 0.35, 0.0, 0.25);
    return vec4(final_color, select(0.65 + fresnel * 0.3 + depth_opacity, 0.92, is_underwater));
}