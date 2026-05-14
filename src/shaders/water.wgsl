const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

const SSSR_MAX_STEPS:     i32 = 18;
const SSSR_MIN_STEPS:     i32 = 7;
const SSSR_BINARY_STEPS:  i32 = 5;
const SSSR_SAMPLE_COUNT:  i32 = 4;
const SSSR_MAX_DISTANCE:  f32 = 80.0;
const SSSR_MAX_DISTANCE_SQ: f32 = SSSR_MAX_DISTANCE * SSSR_MAX_DISTANCE;
const SSSR_THICKNESS:     f32 = 0.06;
const SSSR_EDGE_FADE:     f32 = 0.06;
const SSSR_FADE_DISTANCE: f32 = 180.0;
const SSSR_CONE_ANGLE:    f32 = 0.040;

const SHADOW_MAP_SIZE: f32 = 2048.0;
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
const WATER_ROUGHNESS_MIN: f32 = 0.03;
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
    reflection_mode:     f32,
    moon_position:       vec3<f32>,
    moon_intensity:      f32,
    wind_dir:            vec2<f32>,
    wind_speed:          f32,
    _pad:                f32,
    rain_factor:         f32,
    shadows_enabled:     f32,
};

@group(0) @binding(0) var<uniform> uniforms:  Uniforms;
@group(0) @binding(3) var shadow_map:         texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler:     sampler_comparison;
@group(0) @binding(5) var ssr_color:          texture_2d<f32>;
@group(0) @binding(6) var ssr_depth:          texture_2d<f32>;
@group(0) @binding(7) var ssr_sampler:        sampler;
@group(0) @binding(8) var flow_map:           texture_2d<f32>;
@group(0) @binding(9) var flow_sampler:       sampler;


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

fn calculate_gerstner(pos: vec3<f32>, time: f32) -> GerstnerResult {
    let dist    = length(pos.xz - uniforms.camera_pos.xz);
    let lod     = clamp(1.0 - dist / LOD_FAR, 0.05, 1.0);

    var result: GerstnerResult;
    result.displacement = vec3(0.0);
    result.normal       = vec3(0.0, 1.0, 0.0);

    if lod < 0.01 { return result; }

    let wind_n = normalize(uniforms.wind_dir);
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
    return textureLoad(ssr_depth, px, 0).r;
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
    let wind = normalize(uniforms.wind_dir + vec2(0.001, 0.001));
    let uv = world_xz * 0.018 + wind * time * 0.035 + vec2(time * 0.01, -time * 0.008);
    let s = textureSampleLevel(flow_map, flow_sampler, uv, 0.0).rg;
    return (s - vec2(0.5)) * 2.0;
}

fn sssr_hit_stability(uv: vec2<f32>, center_depth: f32) -> f32 {
    let texel = 1.0 / uniforms.screen_size;

    let depth_x0 = sample_depth(clamp(uv - vec2(texel.x, 0.0), vec2(0.0), vec2(1.0)));
    let depth_x1 = sample_depth(clamp(uv + vec2(texel.x, 0.0), vec2(0.0), vec2(1.0)));
    let depth_y0 = sample_depth(clamp(uv - vec2(0.0, texel.y), vec2(0.0), vec2(1.0)));
    let depth_y1 = sample_depth(clamp(uv + vec2(0.0, texel.y), vec2(0.0), vec2(1.0)));

    let max_delta = max(
        max(abs(center_depth - depth_x0), abs(center_depth - depth_x1)),
        max(abs(center_depth - depth_y0), abs(center_depth - depth_y1)),
    );

    // Thin cutout geometry like leaves produces sharp depth discontinuities.
    // Fade SSSR there to avoid reflections appearing detached from the water.
    return 1.0 - smoothstep(0.0015, 0.012, max_delta);
}

fn sssr_trace_ray(
    world_pos: vec3<f32>,
    refl_dir: vec3<f32>,
    water_view_dist: f32,
    max_steps: i32,
) -> vec4<f32> {
    let dir  = normalize(refl_dir);
    var ray  = world_pos + dir * 0.3;
    var prev = ray;
    var traveled = 0.3;

    var hit_uv   = vec2(0.0);
    var hit_conf = 0.0;
    var found    = false;

    for (var i: i32 = 0; i < SSSR_MAX_STEPS; i++) {
        if i >= max_steps { break; }

        let fi   = f32(i);
        let step = 0.3 + fi * fi * 0.009;
        prev = ray;
        ray += dir * step;
        traveled += step;

        if traveled * traveled > SSSR_MAX_DISTANCE_SQ { break; }

        let clip = uniforms.view_proj * vec4(ray, 1.0);
        if clip.w <= 0.0 { break; }
        let ndc = clip.xyz / clip.w;
        if any(abs(ndc.xy) > vec2(1.0)) { break; }

        let uv   = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let sd   = sample_depth(uv);
        let diff = ndc.z - sd;

        if diff > 0.0 && diff < SSSR_THICKNESS {
            var lo = prev; var hi = ray;
            for (var b: i32 = 0; b < SSSR_BINARY_STEPS; b++) {
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
                let hit_depth = sample_depth(fu);
                let fd  = abs(fn_.z - hit_depth);
                if fd < SSSR_THICKNESS {
                    let hit_world = reconstruct_world(fu, hit_depth);
                    let hit_view_delta = hit_world - uniforms.camera_pos;
                    let hit_view_dist_sq = dot(hit_view_delta, hit_view_delta);
                    let min_hit_dist = max(water_view_dist - 0.35, 0.0);

                    // The opaque depth buffer contains foreground occluders too.
                    // If the SSR candidate is closer to the camera than this water
                    // fragment, it is usually an object standing between the camera
                    // and the water, not something the water surface should reflect.
                    if hit_view_dist_sq >= min_hit_dist * min_hit_dist {
                        hit_uv   = fu;
                        hit_conf = (1.0 - fd / SSSR_THICKNESS) * sssr_hit_stability(fu, hit_depth);
                        found    = true;
                    }
                }
            }
            break;
        }
    }

    if found && hit_conf > 0.05 {
        let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
        let ef   = smoothstep(0.0, SSSR_EDGE_FADE, edge);
        let fc   = smoothstep(0.05, 0.9, hit_conf) * ef;
        if fc > 0.02 {
            return vec4(textureSampleLevel(ssr_color, ssr_sampler, hit_uv, 0.0).rgb, fc);
        }
    }
    return vec4(0.0);
}

fn sssr_trace(
    world_pos: vec3<f32>,
    refl_dir: vec3<f32>,
    water_view_dist: f32,
    max_steps: i32,
    roughness: f32,
    seed: vec2<f32>,
) -> vec4<f32> {
    let base_dir = normalize(refl_dir);
    let up_hint = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(base_dir.y) > 0.92);
    let tangent = normalize(cross(up_hint, base_dir));
    let bitangent = cross(base_dir, tangent);
    let cone = SSSR_CONE_ANGLE * mix(0.45, 1.65, clamp(roughness / WATER_ROUGHNESS_MAX, 0.0, 1.0));

    var color_acc = vec3(0.0);
    var weight_acc = 0.0;
    var confidence = 0.0;

    for (var s: i32 = 0; s < SSSR_SAMPLE_COUNT; s++) {
        let sf = f32(s);
        let rnd0 = hash21(seed + vec2(sf * 17.13, sf * 3.71) + uniforms.time * 0.037);
        let rnd1 = hash21(seed + vec2(sf * 5.31, sf * 23.17) - uniforms.time * 0.029);
        let angle = (sf + rnd0) * (TAU / f32(SSSR_SAMPLE_COUNT));
        let radius = sqrt((sf + rnd1) / f32(SSSR_SAMPLE_COUNT)) * cone;
        let jitter = (tangent * cos(angle) + bitangent * sin(angle)) * radius;
        let sample_dir = normalize(base_dir + jitter);
        let hit = sssr_trace_ray(world_pos, sample_dir, water_view_dist, max_steps);
        let weight = hit.w * mix(1.0, 0.72, sf / f32(SSSR_SAMPLE_COUNT));
        color_acc += hit.rgb * weight;
        weight_acc += weight;
        confidence = max(confidence, hit.w);
    }

    if weight_acc > 0.0001 {
        let coverage = smoothstep(0.035, 0.34, weight_acc);
        return vec4(color_acc / weight_acc, clamp(confidence * mix(0.55, 1.0, coverage), 0.0, 1.0));
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
    let texel = 1.0 / SHADOW_MAP_SIZE;
    let filter_radius = texel * 2.0;
    let edge_dist = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    let edge_fade = smoothstep(0.0, filter_radius + texel, edge_dist);
    let rot   = hash21(world_pos.xz) * TAU;
    let s     = sin(rot);
    let c     = cos(rot);
    var acc   = 0.0;
    for (var i: i32 = 0; i < PCF_SAMPLES; i++) {
        let p = POISSON8[i];
        let offset = vec2(
            p.x * c - p.y * s,
            p.x * s + p.y * c,
        ) * filter_radius;
        let suv = clamp(uv + offset, vec2(texel), vec2(1.0 - texel));
        acc += textureSampleCompare(shadow_map, shadow_sampler, suv, 0, clamp(sc.z - bias, 0.0, 1.0));
    }
    return mix(1.0, acc / f32(PCF_SAMPLES), edge_fade);
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {

    let to_camera = uniforms.camera_pos - in.world_pos;
    let dist      = length(to_camera);
    let view_dir  = to_camera / dist;
    let sun_dir   = uniforms.sun_position;
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

    let shadow  = calculate_shadow(in.world_pos, sun_dir);
    let night_factor = 1.0 - day;
    let night_dark = vec3(0.0, 0.0, 0.01);
    let night_mix = clamp(night_factor * 1.5, 0.0, 1.0);
    water_color = mix(water_color, night_dark, night_mix);

    // Use a smoother normal for reflections. The detailed normal is still used
    // for refraction/specular, but using it for grazing reflections turns sky
    // fallback into large white patches.
    let refl_normal = normalize(mix(normal, wave_n_raw, grazing * 0.65));
    var refl_dir   = reflect(-view_dir, refl_normal);
    refl_dir.y     = max(refl_dir.y, 0.001);

    var refl_color = sky_reflection_color(refl_dir, sun_dir, uniforms.moon_intensity);
    // Sky is only a fallback here. Keep it water-tinted so failed SSR rays do
    // not become white mirror islands at low camera angles.
    refl_color = mix(refl_color, WATER_COLOR_DEEP, 0.22 + grazing * 0.18);
    refl_color *= 0.98 - grazing * 0.16;
    let reflection_mix = clamp(0.055 + fresnel * (0.88 - grazing * 0.10), 0.0, 0.72);
    var ssr_confidence = 0.0;

    // Rust currently toggles reflection_mode between 0 and 1. Treat 1 as SSSR;
    // there is no separate planar reflection path in this shader.
    if uniforms.reflection_mode >= 1.0 {
        // Fade SSSR with distance and with "effective roughness" (rainy/rough water breaks SSSR).
        let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
        let roughness_base = mix(WATER_ROUGHNESS_MIN, WATER_ROUGHNESS_MAX, rain * 0.85);
        let rough_fade = 1.0 - smoothstep(0.06, 0.16, roughness_base);
        let ssr_importance = clamp((0.52 + reflection_mix * 1.85 + grazing * 0.36) * (1.0 - shore_w * 0.16), 0.0, 1.0);
        let ssr_fade = clamp((1.0 - dist / SSSR_FADE_DISTANCE) * rough_fade * ssr_importance, 0.0, 1.0);
        if ssr_fade > 0.01 {
            let step_lerp = clamp(ssr_importance * (1.0 - dist / SSSR_FADE_DISTANCE), 0.0, 1.0);
            let ssr_steps = i32(mix(f32(SSSR_MIN_STEPS), f32(SSSR_MAX_STEPS), step_lerp));
            let ssr  = sssr_trace(in.world_pos, refl_dir, dist, ssr_steps, roughness_base, in.world_pos.xz + uv * 37.0);
            let conf = smoothstep(0.05, 0.9, ssr.w) * ssr_fade;
            if conf > 0.02 {
                ssr_confidence = max(ssr_confidence, conf);
                refl_color = mix(refl_color, ssr.rgb, min(conf * 1.12, 1.0));
            }
        }
    }

    // --- Refraction (screen-space) ---
    // Use only the vertex wave normal for UV offsets. High-frequency normal/detail + flow
    // animates the sample every frame while the opaque texture is static → “double image”.
    let cos_v = cos_theta;
    let n_refract = wave_n_raw;
    let cos_v_refract = max(dot(view_dir, n_refract), 0.0);

    let dist_fade = clamp(1.0 - dist / 120.0, 0.0, 1.0);
    // Thicker water column → less screen-space wobble (seabed should not “swim”).
    let thick_stab = smoothstep(0.25, 3.5, thickness);
    let distort_strength =
        (0.006 + 0.016 * (1.0 - cos_v_refract)) * dist_fade * (1.0 - 0.68 * thick_stab);
    // Flow map only near shore; never full strength or the refracted layer slides over geometry.
    let flow_refract = flow * (0.08 * shore_w * shore_w);
    let distort = (vec2(n_refract.x, -n_refract.z) + flow_refract) * distort_strength;
    let chroma =
        (1.0 - cos_v_refract) * 0.00065 * dist_fade * (1.0 - 0.70 * thick_stab) * (1.0 + shore_w * 0.15);
    let refr_uv0 = clamp(uv + distort, vec2(0.0), vec2(1.0));
    // Far from camera, chromatic split is subpixel: fall back to a single tap.
    var refracted_scene = textureSampleLevel(ssr_color, ssr_sampler, refr_uv0, 0.0).rgb;
    if chroma > 0.00025 {
        let refr_uv_r = clamp(uv + distort + vec2(chroma, -chroma * 0.35), vec2(0.0), vec2(1.0));
        let refr_uv_b = clamp(uv + distort - vec2(chroma * 0.85, chroma * 0.2), vec2(0.0), vec2(1.0));
        let sr = textureSampleLevel(ssr_color, ssr_sampler, refr_uv_r, 0.0).r;
        let sg = refracted_scene.g;
        let sb = textureSampleLevel(ssr_color, ssr_sampler, refr_uv_b, 0.0).b;
        refracted_scene = vec3(sr, sg, sb);
    }

    // Apply absorption to the refracted color, then add in-scattered tint.
    let refracted = refracted_scene * trans;
    let inscatter = water_color * (1.0 - trans) * (0.62 + 0.42 * day);

    // Thin-water sunlight transmission (view-dependent, strongest opposite Fresnel).
    let ndl = max(dot(normal, sun_dir), 0.0);
    let transmit = exp(-thickness * 0.38) * (1.0 - cos_v) * ndl * max(sun_dir.y, 0.0) * shadow;
    let sun_tint = mix(vec3(0.35, 0.72, 0.48), vec3(0.95, 0.92, 0.75), day);
    let sss = sun_tint * transmit * (0.28 + shore_w * 0.35);

    water_color = refracted + inscatter + sss;
    let effective_reflection_mix = clamp(reflection_mix * mix(0.46, 1.12, ssr_confidence), 0.0, 0.78);
    water_color = mix(water_color, refl_color, effective_reflection_mix);

    // Soft sun glint on reflection vector. Keep it low; strong highlights read as white patches.
    if sun_dir.y > 0.0 {
        let sun_v = max(dot(refl_dir, sun_dir), 0.0);
        let glint = pow(sun_v, 112.0) * day * shadow * (0.018 + shore_w * 0.010);
        water_color += vec3(1.0, 0.94, 0.82) * glint * effective_reflection_mix;
    }

    if sun_dir.y > 0.0 {
        // Rain makes the surface rougher/dimmer highlights (and breaks up SSR).
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
    let alpha = clamp(WATER_OPACITY + thickness_alpha * 0.42 + fresnel * 0.22, 0.02, 0.92);

    return vec4(water_color, clamp(alpha, 0.0, 1.0));
}
