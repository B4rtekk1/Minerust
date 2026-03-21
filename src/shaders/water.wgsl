
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

const SSR_MAX_STEPS:     i32 = 48;
const SSR_BINARY_STEPS:  i32 = 8;
const SSR_MAX_DISTANCE:  f32 = 120.0;
const SSR_THICKNESS:     f32 = 0.06;
const SSR_EDGE_FADE:     f32 = 0.06;
const SSR_FADE_DISTANCE: f32 = 300.0;

const SHADOW_MAP_SIZE: f32 = 2048.0;
const PCF_SAMPLES:     i32 = 8;


const LOD_FAR: f32 = 300.0;

const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.04, 0.28, 0.38);
const WATER_COLOR_DEEP:    vec3<f32> = vec3<f32>(0.01, 0.10, 0.22);
const WATER_OPACITY:       f32 = 0.45;
const FRESNEL_R0:          f32 = 0.02;
const WATER_LEVEL_OFFSET:  f32 = 0.15;
const WATER_ROUGHNESS_MIN: f32 = 0.03;
const WATER_ROUGHNESS_MAX: f32 = 0.14;

const FOAM_THRESHOLD: f32 = 0.30;
const FOAM_INTENSITY: f32 = 0.70;

const FOG_NEAR: f32 = 0.0;
const FOG_FAR:  f32 = 200.0;


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
    moon_intensity:      f32,
    wind_dir:            vec2<f32>,
    wind_speed:          f32,
    _pad:                f32,
};

@group(0) @binding(0) var<uniform> uniforms:  Uniforms;
@group(0) @binding(3) var shadow_map:         texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler:     sampler_comparison;
@group(0) @binding(5) var ssr_color:          texture_2d<f32>;
@group(0) @binding(6) var ssr_depth:          texture_depth_2d;
@group(0) @binding(7) var ssr_sampler:        sampler;


struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) color:     vec4<f32>,
    @location(3) uv:        vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:    vec3<f32>,
    @location(1) wave_normal:  vec4<f32>,
    @location(2) uv:           vec2<f32>,
    @location(3) view_depth:   f32,
    @location(4) original_pos: vec2<f32>,
    @location(5) jacobian:     f32,
};

const SW_K:     array<f32, 4> = array(0.38,  0.84,  2.10,  5.60);
const SW_C:     array<f32, 4> = array(0.51,  0.35,  0.26,  0.19);
const SW_AMP:   array<f32, 4> = array(0.260, 0.130, 0.052, 0.022);
const SW_STEEP: array<f32, 4> = array(0.72,  0.58,  0.42,  0.26);
const SW_DX:    array<f32, 4> = array( 0.966,  0.342,  0.174, -0.500);
const SW_DZ:    array<f32, 4> = array( 0.259, -0.940,  0.985,  0.866);
const SW_PH:    array<f32, 4> = array( 0.000,  2.618,  0.785,  5.497);

const CP_K:     array<f32, 4> = array(14.0,  22.0,  38.0,  60.0);
const CP_C:     array<f32, 4> = array(0.14,  0.11,  0.09,  0.07);
const CP_AMP:   array<f32, 4> = array(0.0095, 0.0065, 0.0040, 0.0024);
const CP_STEEP: array<f32, 4> = array(0.55,  0.45,  0.32,  0.20);
const CP_DX:    array<f32, 4> = array( 0.707,  0.866, -0.500,  0.259);
const CP_DZ:    array<f32, 4> = array( 0.707,  0.500,  0.866, -0.966);
const CP_PH:    array<f32, 4> = array( 1.047,  3.665,  2.094,  4.712);

struct GerstnerResult {
    displacement: vec3<f32>,
    normal:       vec3<f32>,
    foam:         f32,
    jacobian:     f32,
}

fn smooth5(x: f32) -> f32 { return x * x * x * (x * (x * 6.0 - 15.0) + 10.0); }

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2(127.1, 311.7));
    q += dot(q, q + 19.19);
    return fract(q.x * q.y);
}

fn calculate_gerstner(pos: vec3<f32>, time: f32) -> GerstnerResult {
    let dist    = length(pos.xz - uniforms.camera_pos.xz);
    let lod     = clamp(1.0 - dist / LOD_FAR, 0.05, 1.0);

    var result: GerstnerResult;
    result.displacement = vec3(0.0);
    result.normal       = vec3(0.0, 1.0, 0.0);
    result.foam         = 0.0;
    result.jacobian     = 1.0;

    if lod < 0.01 { return result; }

    let wind_n = normalize(uniforms.wind_dir);
    let p      = pos.xz;
    let wsp    = uniforms.wind_speed;

    var x_off = 0.0; var y_off = 0.0; var z_off = 0.0;
    var dx = 0.0;    var dz = 0.0;
    var j_xx = 0.0;  var j_zz = 0.0;  var j_xz = 0.0;

    for (var i: i32 = 0; i < 4; i++) {
        let base_dir = vec2(SW_DX[i], SW_DZ[i]);
        let dmod     = normalize(mix(base_dir, wind_n, 0.35));
        let align    = max(dot(dmod, wind_n), 0.0);
        let amp      = SW_AMP[i] * lod * (0.65 + 0.35 * align);
        let steep    = SW_STEEP[i] * lod;

        let phase = SW_K[i] * dot(dmod, p) - SW_C[i] * time * wsp + SW_PH[i];
        let sf = sin(phase);
        let cf = cos(phase);

        let qic = steep * amp * cf;
        x_off -= qic * dmod.x;
        z_off -= qic * dmod.y;
        y_off += amp * sf;

        let df = amp * SW_K[i] * cf;
        dx += dmod.x * df;
        dz += dmod.y * df;

        let ka_sf_Q = SW_K[i] * amp * sf * steep;
        j_xx += dmod.x * dmod.x * ka_sf_Q;
        j_zz += dmod.y * dmod.y * ka_sf_Q;
        j_xz += dmod.x * dmod.y * ka_sf_Q;
    }

    let cap_lod = clamp(1.0 - dist / (LOD_FAR * 0.4), 0.0, 1.0);
    if cap_lod > 0.01 {
        for (var i: i32 = 0; i < 4; i++) {
            let dmod  = normalize(vec2(CP_DX[i], CP_DZ[i]));
            let amp   = CP_AMP[i] * cap_lod;
            let steep = CP_STEEP[i] * cap_lod;

            let phase = CP_K[i] * dot(dmod, p) - CP_C[i] * time * wsp + CP_PH[i];
            let sf = sin(phase);
            let cf = cos(phase);

            let qic = steep * amp * cf;
            x_off -= qic * dmod.x;
            z_off -= qic * dmod.y;
            y_off += amp * sf;

            let df = amp * CP_K[i] * cf;
            dx += dmod.x * df;
            dz += dmod.y * df;

            let ka_sf_Q = CP_K[i] * amp * sf * steep;
            j_xx += dmod.x * dmod.x * ka_sf_Q;
            j_zz += dmod.y * dmod.y * ka_sf_Q;
            j_xz += dmod.x * dmod.y * ka_sf_Q;
        }
    }

    result.displacement = vec3(x_off, y_off, z_off);
    result.normal       = normalize(vec3(-dx, 1.0, -dz));

    let jacobian    = (1.0 - j_xx) * (1.0 - j_zz) - j_xz * j_xz;
    result.jacobian = jacobian;
    result.foam     = clamp(1.0 - jacobian - FOAM_THRESHOLD, 0.0, 1.0) * FOAM_INTENSITY;

    return result;
}

@vertex
fn vs_water(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = model.position;
    out.original_pos = pos.xz;

    var wave_normal = model.normal.xyz;
    var foam_val    = 0.0;
    var jac_val     = 1.0;

    if model.normal.y > 0.5 {
        let w  = calculate_gerstner(pos, uniforms.time);
        pos   += w.displacement - vec3(0.0, WATER_LEVEL_OFFSET, 0.0);
        wave_normal = w.normal;
        foam_val    = w.foam;
        jac_val     = w.jacobian;
    }

    out.clip_position = uniforms.view_proj * vec4(pos, 1.0);
    out.world_pos     = pos;
    out.wave_normal   = vec4(wave_normal, foam_val);
    out.uv            = model.uv;
    out.view_depth    = out.clip_position.w;
    out.jacobian      = jac_val;
    return out;
}

fn schlick_fresnel(cos_theta: f32, r0: f32) -> f32 {
    let x  = 1.0 - cos_theta;
    let x2 = x * x;
    return r0 + (1.0 - r0) * x2 * x2 * x;
}

fn ggx_distribution(ndh: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn ggx_spec_simple(normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>, roughness: f32) -> f32 {
    let h   = normalize(view + light);
    let ndh = max(dot(normal, h), 0.0);
    let ndl = max(dot(normal, light), 0.0);
    return ggx_distribution(ndh, roughness) * ndl;
}

fn fbm_normal_perturb(p: vec2<f32>, t: f32) -> vec2<f32> {
    let s1 = 0.28 * sin(vec2(p.x * 3.1 + t * 0.9, p.y * 2.7 - t * 1.1));
    let s2 = 0.14 * sin(vec2(p.x * 6.3 - t * 1.7 + 1.3, p.y * 5.9 + t * 1.4 + 0.7));
    let s3 = 0.06 * sin(vec2(p.x * 13.1 + t * 2.6 + 2.7, p.y * 11.7 - t * 2.2 + 1.4));
    return s1 + s2 + s3;
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    let sz = vec2<i32>(uniforms.screen_size);
    let px = clamp(vec2<i32>(uv * uniforms.screen_size), vec2<i32>(0), sz - vec2<i32>(1));
    return textureLoad(ssr_depth, px, 0);
}

fn reconstruct_world(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d, 1.0);
    let wh  = uniforms.inv_view_proj * ndc;
    return wh.xyz / wh.w;
}

fn ssr_trace(world_pos: vec3<f32>, refl_dir: vec3<f32>) -> vec4<f32> {
    let dir  = normalize(refl_dir);
    var ray  = world_pos + dir * 0.3;
    var prev = ray;

    var hit_uv   = vec2(0.0);
    var hit_conf = 0.0;
    var found    = false;

    for (var i: i32 = 0; i < SSR_MAX_STEPS; i++) {
        let fi   = f32(i);
        let step = 0.3 + fi * fi * 0.009;
        prev = ray;
        ray += dir * step;

        if length(ray - world_pos) > SSR_MAX_DISTANCE { break; }

        let clip = uniforms.view_proj * vec4(ray, 1.0);
        if clip.w <= 0.0 { break; }
        let ndc = clip.xyz / clip.w;
        if any(abs(ndc.xy) > vec2(1.0)) { break; }

        let uv   = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let sd   = sample_depth(uv);
        let diff = ndc.z - sd;

        if diff > 0.0 && diff < SSR_THICKNESS {
            var lo = prev; var hi = ray;
            for (var b: i32 = 0; b < SSR_BINARY_STEPS; b++) {
                let mid = (lo + hi) * 0.5;
                let mc  = uniforms.view_proj * vec4(mid, 1.0);
                if mc.w <= 0.0 { break; }
                let mn  = mc.xyz / mc.w;
                let mu  = vec2(mn.x * 0.5 + 0.5, 0.5 - mn.y * 0.5);
                if mn.z > sample_depth(mu) { hi = mid; } else { lo = mid; }
            }
            let fc = uniforms.view_proj * vec4(hi, 1.0);
            if fc.w > 0.0 {
                let fn_ = fc.xyz / fc.w;
                let fu  = vec2(fn_.x * 0.5 + 0.5, 0.5 - fn_.y * 0.5);
                let fd  = abs(fn_.z - sample_depth(fu));
                if fd < SSR_THICKNESS {
                    hit_uv   = fu;
                    hit_conf = 1.0 - fd / SSR_THICKNESS;
                    found    = true;
                }
            }
            break;
        }
    }

    if found && hit_conf > 0.05 {
        let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
        let ef   = smoothstep(0.0, SSR_EDGE_FADE, edge);
        let fc   = smoothstep(0.05, 0.9, hit_conf) * ef;
        if fc > 0.02 {
            return vec4(textureSampleLevel(ssr_color, ssr_sampler, hit_uv, 0.0).rgb, fc);
        }
    }
    return vec4(0.0);
}

const POISSON8: array<vec2<f32>, 8> = array(
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.81544232, -0.87912464),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2(-0.65476012, -0.05147385),
);

fn calculate_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if sun_dir.y < 0.05 { return 0.0; }

    let shadow_pos = uniforms.csm_view_proj[0] * vec4(world_pos, 1.0);
    let sc         = shadow_pos.xyz / shadow_pos.w;
    let uv         = vec2(sc.x * 0.5 + 0.5, 1.0 - (sc.y * 0.5 + 0.5));

    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) { return 1.0; }

    let bias  = 0.003;
    let texel = 1.0 / SHADOW_MAP_SIZE;
    var acc   = 0.0;
    for (var i: i32 = 0; i < PCF_SAMPLES; i++) {
        let offset = POISSON8[i] * texel * 3.0;
        acc += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, 0, sc.z - bias);
    }
    return acc / f32(PCF_SAMPLES);
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {

    let to_camera = uniforms.camera_pos - in.world_pos;
    let dist      = length(to_camera);
    let view_dir  = to_camera / dist;
    let sun_dir   = normalize(uniforms.sun_position);
    let day       = clamp(sun_dir.y, 0.0, 1.0);
    let t         = uniforms.time;

    let inv_w     = 1.0 / in.clip_position.w;
    let clip_ndc  = in.clip_position.xyz * inv_w;
    let screen_uv = vec2(clip_ndc.x * 0.5 + 0.5, 0.5 - clip_ndc.y * 0.5);

    let wave_n_raw = in.wave_normal.xyz;
    let foam_val   = in.wave_normal.w;

    let perturb_blend = clamp(1.0 - dist / 120.0, 0.0, 1.0);
    var normal = wave_n_raw;
    if perturb_blend > 0.005 {
        let perturb = fbm_normal_perturb(in.world_pos.xz * 0.15, t) * perturb_blend;
        normal = normalize(wave_n_raw + vec3(perturb.x, 0.0, perturb.y));
    }

    let micro_blend = clamp(1.0 - dist / 50.0, 0.0, 1.0);
    if micro_blend > 0.005 {
        let mp = 0.08 * sin(vec2(
            in.world_pos.x * 11.3 + t * 2.1,
            in.world_pos.z * 9.7  - t * 1.8
        )) * micro_blend;
        normal = normalize(normal + vec3(mp.x, 0.0, mp.y));
    }


    let cos_theta = max(dot(view_dir, normal), 0.0);
    let fresnel   = schlick_fresnel(cos_theta, FRESNEL_R0);
    let grazing   = smoothstep(0.25, 0.98, 1.0 - cos_theta);

    let depth_t     = clamp(1.0 - cos_theta * 1.4, 0.0, 1.0);
    var water_color = mix(WATER_COLOR_SHALLOW, WATER_COLOR_DEEP, depth_t);

    let wave_pulse = clamp(in.world_pos.y * 1.4 + 0.5, 0.0, 1.0);
    water_color   *= mix(0.75, 1.18, wave_pulse);

    let shadow  = calculate_shadow(in.world_pos, sun_dir);
    let ambient = mix(0.01, 0.35, day);
    water_color *= ambient + shadow * 0.55 * day;

    var refl_dir   = reflect(-view_dir, normal);
    refl_dir.y     = max(refl_dir.y, 0.001);

    let sky_height = clamp(refl_dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sky_day    = mix(vec3(0.55, 0.75, 0.98), vec3(0.20, 0.40, 0.85), sky_height) * day;
    let sky_night  = mix(vec3(0.012, 0.012, 0.025), vec3(0.001, 0.001, 0.008), sky_height)
                     * (1.0 - day);
    var refl_color = sky_day + sky_night;

    if uniforms.reflection_mode != 0.0 {
        let ssr_fade = clamp(1.0 - dist / SSR_FADE_DISTANCE, 0.0, 1.0);
        if ssr_fade > 0.01 {
            let ssr  = ssr_trace(in.world_pos, refl_dir);
            let conf = smoothstep(0.05, 0.9, ssr.w) * ssr_fade;
            if conf > 0.02 {
                refl_color = mix(refl_color, ssr.rgb, conf * 0.85);
            }
        }
    }

    refl_color *= 1.0 - grazing * 0.18;
    let reflection_mix = clamp(fresnel * (0.72 - grazing * 0.22), 0.0, 0.75);
    water_color = mix(water_color, refl_color, reflection_mix);

    if sun_dir.y > 0.0 {
        let jac_rough   = clamp(1.0 - in.jacobian, 0.0, 1.0);
        let roughness   = mix(WATER_ROUGHNESS_MIN, WATER_ROUGHNESS_MAX, jac_rough);
        let spec        = ggx_spec_simple(normal, view_dir, sun_dir, roughness);
        let spec_color  = mix(vec3(1.0, 0.97, 0.88), vec3(1.0, 0.82, 0.55), 1.0 - day);
        water_color    += spec_color * spec * 1.6 * day * shadow;

        if uniforms.moon_intensity > 0.01 && day < 0.2 {
            let moon_dir  = normalize(uniforms.moon_position);
            let spec_moon = ggx_spec_simple(normal, view_dir, moon_dir, WATER_ROUGHNESS_MIN * 0.5);
            water_color  += vec3(0.82, 0.88, 1.0) * spec_moon * uniforms.moon_intensity
                            * (1.0 - day) * 0.6;
        }
    }

    if foam_val > 0.01 {
        let foam_pulse = mix(0.85, 1.0, sin(t * 3.7 + in.world_pos.x * 2.1) * 0.5 + 0.5);
        let lit_foam   = vec3(0.88, 0.92, 0.96) * (ambient + shadow * day * 0.7);
        water_color    = mix(water_color, lit_foam,
                             clamp(foam_val * 0.75 * foam_pulse, 0.0, 0.8));
    }


    let delta_xz = in.world_pos.xz - uniforms.camera_pos.xz;
    let dist_xz  = sqrt(dot(delta_xz, delta_xz));
    let fog_t    = clamp((dist_xz - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0.0, 1.0);
    let fog_col  = mix(vec3(0.001, 0.001, 0.008), refl_color, day);
    water_color  = mix(water_color, fog_col, fog_t * fog_t);

    let alpha = WATER_OPACITY + fresnel * 0.28;

    return vec4(water_color, clamp(alpha, 0.0, 1.0));
}
