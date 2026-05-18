const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

const PCF_SAMPLES:     i32 = 8;
const SHADOW_DISTANCE:    f32 = 96.0;
const SHADOW_DISTANCE_SQ: f32 = SHADOW_DISTANCE * SHADOW_DISTANCE;

const LOD_FAR: f32 = 300.0;

// Natural coastal palette: green-blue shallows -> cold deep water.
const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.075, 0.410, 0.455);
const WATER_COLOR_DEEP:    vec3<f32> = vec3<f32>(0.010, 0.072, 0.165);
const WATER_OPACITY:       f32 = 0.26;
const FRESNEL_R0:          f32 = 0.020;
const WATER_LEVEL_OFFSET:  f32 = 0.15;
const WATER_ROUGHNESS_MAX: f32 = 0.14;

const FOG_NEAR: f32 = 0.0;
const FOG_FAR:  f32 = 200.0;

struct ShadowConfig {
    shadow_map_size: f32,
}

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
    _pad_water:          f32,
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
    csm_shadow_rects:    array<vec4<f32>, 4>,
    csm_shadow_sizes:    vec4<f32>,
    shadow_sun_position: vec3<f32>,
    _pad_shadow_sun:     f32,
};

@group(0) @binding(0) var<uniform> uniforms:  Uniforms;
@group(0) @binding(3) var shadow_map:         texture_depth_2d;
@group(0) @binding(4) var shadow_sampler:     sampler_comparison;
@group(0) @binding(5) var resolved_depth:     texture_2d<f32>;
@group(0) @binding(6) var flow_map:           texture_2d<f32>;
@group(0) @binding(7) var flow_sampler:       sampler;


struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) packed:    u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:    vec3<f32>,
    @location(1) wave_normal:  vec3<f32>,
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
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2(127.1, 311.7));
    q += dot(q, q + 19.19);
    return fract(q.x * q.y);
}

fn sun_poisson_phase(sun_dir: vec3<f32>) -> f32 {
    let dir = normalize(sun_dir);
    return atan2(dir.z, max(dir.y, 0.001));
}

fn uniform_wind_dir() -> vec2<f32> {
    return vec2(uniforms.wind_dir_x, uniforms.wind_dir_z);
}

fn calculate_gerstner(pos: vec3<f32>, time: f32) -> GerstnerResult {
    let dist    = length(pos.xz - uniforms.camera_pos.xz);
    let lod     = clamp(1.0 - dist / LOD_FAR, 0.05, 1.0);

    var result: GerstnerResult;
    result.displacement = vec3(0.0);
    result.normal       = vec3(0.0, 1.0, 0.0);

    if lod < 0.01 { return result; }

    let wind_n = normalize(uniform_wind_dir());
    let p      = pos.xz;
    let wsp    = uniforms.wind_speed;

    var x_off = 0.0; var y_off = 0.0; var z_off = 0.0;
    var dx = 0.0;    var dz = 0.0;

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
        }
    }

    result.displacement = vec3(x_off, y_off, z_off);
    result.normal       = normalize(vec3(-dx, 1.0, -dz));

    return result;
}

@vertex
fn vs_water(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = model.position;

    let n_idx   = model.packed & 0x7u;

    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)
    );

    var wave_normal = normals[n_idx % 6u];
    if n_idx == 3u {
        let w = calculate_gerstner(pos, uniforms.time);
        // Keep water geometry locked to the voxel grid. Horizontal Gerstner
        // displacement opens gaps at shorelines and vertical displacement can
        // climb over adjacent land blocks; the wave motion is retained through
        // normals/refraction instead.
        pos.y -= WATER_LEVEL_OFFSET;
        wave_normal = w.normal;
    }

    out.clip_position = uniforms.view_proj * vec4(pos, 1.0);
    out.world_pos     = pos;
    out.wave_normal   = normalize(wave_normal);
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
    let a = sin(vec4(
        p.x * 3.1  + t * 0.9,   p.y * 2.7  - t * 1.1,
        p.x * 6.3  - t * 1.7 + 1.3, p.y * 5.9 + t * 1.4 + 0.7
    ));
    let b = sin(vec2(p.x * 13.1 + t * 2.6 + 2.7, p.y * 11.7 - t * 2.2 + 1.4));
    return a.xz * 0.28 + a.yw * 0.14 + b * 0.06;
}

fn sky_reflection_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, moon_intensity: f32) -> vec3<f32> {
    let sun_h = sun_dir.y;
    let view_h = max(view_dir.y, 0.0);

    let day = smoothstep(-0.15, 0.15, sun_h);
    let night = smoothstep(0.12, -0.12, sun_h);
    let dusk = 1.0 - smoothstep(0.0, 0.45, abs(sun_h));

    let day_zenith = vec3<f32>(0.08, 0.32, 0.72);
    let day_horizon = vec3<f32>(0.56, 0.82, 0.98);
    let night_zenith = vec3<f32>(0.001, 0.003, 0.012);
    let night_horizon = vec3<f32>(0.008, 0.012, 0.024);
    let sunset_horizon = vec3<f32>(1.0, 0.42, 0.12);
    let sunset_zenith = vec3<f32>(0.18, 0.12, 0.35);

    var sky = mix(day_horizon, day_zenith, pow(view_h, 0.65)) * day;
    sky += mix(night_horizon, night_zenith, pow(view_h, 0.55)) * night;

    let dusk_color = mix(sunset_horizon, sunset_zenith, pow(view_h, 0.75));
    let sun_prox = max(dot(view_dir, sun_dir), 0.0);
    sky += dusk_color * dusk * (0.45 * pow(sun_prox, 2.0) + 0.15);

    if moon_intensity > 0.01 {
        sky += vec3<f32>(0.20, 0.26, 0.40) * moon_intensity * night;
    }

    sky = sky / (sky + vec3<f32>(0.15));
    return sky * 1.15;
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    let sz = vec2<i32>(uniforms.screen_size);
    let px = clamp(vec2<i32>(uv * uniforms.screen_size), vec2<i32>(0), sz - vec2<i32>(1));
    return textureLoad(resolved_depth, px, 0).r;
}

fn reconstruct_world(uv: vec2<f32>, d: f32) -> vec3<f32> {
    let ndc = vec4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d, 1.0);
    let wh  = uniforms.inv_view_proj * ndc;
    return wh.xyz / wh.w;
}

fn beer_lambert(transmittance_coeff: vec3<f32>, distance: f32) -> vec3<f32> {
    // transmittance_coeff is per-meter absorption; exp(-sigma * d)
    return exp(-transmittance_coeff * distance);
}

fn flow_vector(world_xz: vec2<f32>, time: f32) -> vec2<f32> {
    let wind = normalize(uniform_wind_dir() + vec2(0.001, 0.001));
    let uv = world_xz * 0.018 + wind * time * 0.035 + vec2(time * 0.01, -time * 0.008);
    let s = textureSampleLevel(flow_map, flow_sampler, uv, 0.0).rg;
    return (s - vec2(0.5)) * 2.0;
}

const POISSON16: array<vec2<f32>, 16> = array(
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.81544232, -0.87912464),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2(-0.65476012, -0.05147385),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.53742981, -0.47373420),
    vec2( 0.18395645,  0.89721549),
    vec2(-0.09715394, -0.00673456),
    vec2( 0.53472400,  0.73356543),
    vec2(-0.45611231, -0.40212851),
    vec2(-0.57321081,  0.65476012),
);

fn calculate_shadow(world_pos: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    if uniforms.shadows_enabled < 0.5 { return 1.0; }
    if sun_dir.y < 0.05 { return 0.0; }
    let shadow_delta = world_pos.xz - uniforms.camera_pos.xz;
    if dot(shadow_delta, shadow_delta) > SHADOW_DISTANCE_SQ { return 1.0; }

    let shadow_pos = uniforms.csm_view_proj[0] * vec4(world_pos, 1.0);
    if shadow_pos.w <= 0.0 { return 1.0; }

    let sc         = shadow_pos.xyz / shadow_pos.w;
    let uv         = vec2(sc.x * 0.5 + 0.5, 1.0 - (sc.y * 0.5 + 0.5));

    if sc.z < 0.0 || sc.z > 1.0 { return 1.0; }

    let bias  = 0.003;
    let shadow_map_size = max(uniforms.csm_shadow_sizes[0], 1.0);
    let texel = 1.0 / shadow_map_size;
    let filter_radius = texel * 2.0;
    let edge_dist = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let edge_fade = smoothstep(0.0, filter_radius + texel, edge_dist);
    let rot   = hash21(world_pos.xz) * TAU + sun_poisson_phase(uniforms.shadow_sun_position);
    let s     = sin(rot);
    let c     = cos(rot);
    var acc   = 0.0;
    for (var i: i32 = 0; i < PCF_SAMPLES; i++) {
        let rect = uniforms.csm_shadow_rects[0];
        let p = POISSON16[i];
        let offset = vec2(
            p.x * c - p.y * s,
            p.x * s + p.y * c,
        ) * filter_radius;
        let suv = clamp(uv + offset, vec2(texel), vec2(1.0 - texel));
        let atlas_uv = rect.xy + suv * rect.zw;
        acc += textureSampleCompare(shadow_map, shadow_sampler, atlas_uv, clamp(sc.z - bias, 0.0, 1.0));
    }
    return mix(1.0, acc / f32(PCF_SAMPLES), edge_fade);
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {

    let to_camera = uniforms.camera_pos - in.world_pos;
    let dist      = length(to_camera);
    let view_dir  = to_camera / dist;
    let sun_dir   = uniforms.sun_position;
    let shadow_sun_dir = normalize(uniforms.shadow_sun_position);
    let day       = clamp(sun_dir.y, 0.0, 1.0);
    let t         = uniforms.time;

    let wave_n_raw = in.wave_normal;

    let perturb_blend = clamp(1.0 - dist / 92.0, 0.0, 1.0);
    var normal = wave_n_raw;
    if perturb_blend > 0.005 {
        let perturb = fbm_normal_perturb(in.world_pos.xz * 0.13, t) * perturb_blend * 0.82;
        normal = normalize(wave_n_raw + vec3(perturb.x, 0.0, perturb.y));
    }

    let micro_blend = clamp(1.0 - dist / 30.0, 0.0, 1.0);
    if micro_blend > 0.005 {
        let mp = 0.055 * sin(vec2(
            in.world_pos.x * 11.3 + t * 2.1,
            in.world_pos.z * 9.7  - t * 1.8
        )) * micro_blend;
        normal = normalize(normal + vec3(mp.x, 0.0, mp.y));
    }

    let cos_theta = max(dot(view_dir, normal), 0.0);
    let fresnel   = schlick_fresnel(cos_theta, FRESNEL_R0);
    let grazing   = smoothstep(0.25, 0.98, 1.0 - cos_theta);

    // --- Screen-space thickness (water -> opaque hit) ---
    // Use resolved opaque depth to estimate how much water the ray traverses.
    // This drives shallow/deep color, absorption and refraction near shorelines.
    // Pixel coords (same builtin as a lone @builtin(position) param; must not duplicate it).
    let uv = vec2<f32>(in.clip_position.x, in.clip_position.y) / uniforms.screen_size;
    let scene_depth = sample_depth(uv);
    var thickness = 30.0;
    if scene_depth < 0.999999 {
        let scene_world = reconstruct_world(uv, scene_depth);
        thickness = clamp(length(scene_world - in.world_pos), 0.0, 30.0);
    }
    let shore_w = 1.0 - smoothstep(0.06, 0.95, thickness);

    var flow = vec2(0.0);
    if shore_w > 0.001 {
        flow = flow_vector(in.world_pos.xz, t);
    }

    // Absorption coefficients (m^-1-ish). Red dies first; green/blue linger.
    let sigma_a = vec3<f32>(0.34, 0.080, 0.028);
    let trans   = beer_lambert(sigma_a, thickness);

    // Base "scattering" tint: shallow is brighter/greener, deep is darker/bluer.
    let depth_t = smoothstep(0.35, 7.5, thickness);
    var water_color = mix(WATER_COLOR_SHALLOW, WATER_COLOR_DEEP, depth_t);
    // Slight depth-dependent saturation shift (deep water reads cooler, shallows read clearer).
    water_color = mix(water_color * vec3(1.16, 1.14, 1.04), water_color * vec3(0.96, 1.02, 1.12), depth_t);

    let wave_pulse = clamp(in.world_pos.y * 1.4 + 0.5, 0.0, 1.0);
    water_color   *= mix(0.92, 1.16, wave_pulse);

    let shadow  = calculate_shadow(in.world_pos, shadow_sun_dir);
    let night_factor = 1.0 - day;
    let night_dark = vec3(0.0, 0.0, 0.01);
    let night_mix = clamp(night_factor * 1.5, 0.0, 1.0);
    water_color = mix(water_color, night_dark, night_mix);

    // Use a smoother normal for sky reflections. The detailed normal is still
    // used for specular lighting, but using it for grazing reflections turns
    // the sky fallback into large white patches.
    let refl_normal = normalize(mix(normal, wave_n_raw, grazing * 0.65));
    var refl_dir   = reflect(-view_dir, refl_normal);
    refl_dir.y     = max(refl_dir.y, 0.001);

    var refl_color = sky_reflection_color(refl_dir, sun_dir, uniforms.moon_intensity);
    refl_color = mix(refl_color, WATER_COLOR_DEEP, 0.22 + grazing * 0.18);
    refl_color *= 0.98 - grazing * 0.16;
    let reflection_mix = clamp(0.055 + fresnel * (0.88 - grazing * 0.10), 0.0, 0.72);

    // --- Surface transmission hint ---
    // Keep a small wave/flow brightness variation so water remains animated
    // without sampling the resolved scene color.
    let cos_v = cos_theta;
    let n_refract = wave_n_raw;
    let cos_v_refract = max(dot(view_dir, n_refract), 0.0);

    let dist_fade = clamp(1.0 - dist / 120.0, 0.0, 1.0);
    let thick_stab = smoothstep(0.25, 3.5, thickness);
    let distort_strength =
        (0.006 + 0.016 * (1.0 - cos_v_refract)) * dist_fade * (1.0 - 0.68 * thick_stab);
    let flow_refract = flow * (0.08 * shore_w * shore_w);
    let surface_variation = clamp(
        0.92 + dot(vec2(n_refract.x, -n_refract.z) + flow_refract, vec2(0.65, -0.35))
            * distort_strength
            * 18.0
            + shore_w * 0.08,
        0.78,
        1.12,
    );

    // Apply absorption to the transmitted water tint, then add in-scattered tint.
    let refracted = water_color * trans * surface_variation;
    let inscatter = water_color * (1.0 - trans) * (0.62 + 0.42 * day);

    // Thin-water sunlight transmission (view-dependent, strongest opposite Fresnel).
    let ndl = max(dot(normal, sun_dir), 0.0);
    let transmit = exp(-thickness * 0.38) * (1.0 - cos_v) * ndl * max(sun_dir.y, 0.0) * shadow;
    let sun_tint = mix(vec3(0.35, 0.72, 0.48), vec3(0.95, 0.92, 0.75), day);
    let sss = sun_tint * transmit * (0.28 + shore_w * 0.35);

    water_color = refracted + inscatter + sss;
    let effective_reflection_mix = clamp(reflection_mix, 0.0, 0.72);
    water_color = mix(water_color, refl_color, effective_reflection_mix);

    // Soft sun glint on reflection vector. Keep it low; strong highlights read as white patches.
    if sun_dir.y > 0.0 {
        let sun_v = max(dot(refl_dir, sun_dir), 0.0);
        let glint = pow(sun_v, 112.0) * day * shadow * (0.018 + shore_w * 0.010);
        water_color += vec3(1.0, 0.94, 0.82) * glint * effective_reflection_mix;
    }

    if sun_dir.y > 0.0 {
        // Rain makes the surface rougher and dims highlights.
        let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
        let roughness_base = mix(0.075, WATER_ROUGHNESS_MAX, rain * 0.85);
        let roughness   = roughness_base * mix(1.0, 0.90, 1.0 - day);
        let spec        = min(ggx_spec_simple(normal, view_dir, sun_dir, roughness), 1.0);
        let spec_color  = mix(vec3(1.0, 0.97, 0.88), vec3(1.0, 0.84, 0.58), 1.0 - day);
        water_color    += spec_color * spec * mix(0.055, 0.085, day) * shadow;

        if uniforms.moon_intensity > 0.01 && day < 0.2 {
            let moon_dir  = uniforms.moon_position;
            let spec_moon = min(ggx_spec_simple(normal, view_dir, moon_dir, 0.09), 1.0);
            water_color  += vec3(0.82, 0.88, 1.0) * spec_moon * uniforms.moon_intensity
                            * (1.0 - day) * 0.10;
        }
    }

    let fog_t    = clamp((dist - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0.0, 1.0);
    let fog_col  = mix(vec3(0.004, 0.010, 0.024), refl_color,
                       0.85 * day + 0.15 * (1.0 - day));
    water_color  = mix(water_color, fog_col, fog_t * fog_t);

    // Alpha increases with thickness (more water -> less see-through), plus Fresnel.
    let thickness_alpha = 1.0 - exp(-thickness * 0.13);
    let reflection_alpha = fresnel * 0.08;
    let alpha = clamp(WATER_OPACITY + thickness_alpha * 0.42 + fresnel * 0.22 + reflection_alpha, 0.02, 0.96);

    return vec4(water_color, clamp(alpha, 0.0, 1.0));
}
