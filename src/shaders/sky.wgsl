struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    csm_view_proj: array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
    _screen_size: vec2<f32>,
    _water_level: f32,
    _reflection_mode: f32,
    moon_position: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

const TAU: f32 = 6.28318530718;
const PI: f32 = 3.14159265359;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_pos: vec2<f32>,
};

@vertex
fn vs_sky(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.xy, 0.9999, 1.0);
    out.ndc_pos = model.position.xy;
    return out;
}

fn get_view_direction(ndc_xy: vec2<f32>) -> vec3<f32> {
    let clip = vec4<f32>(ndc_xy, 1.0, 1.0);
    let world_h = uniforms.inv_view_proj * clip;
    return normalize(world_h.xyz / world_h.w - uniforms.camera_pos);
}

fn hash11(p: f32) -> f32 {
    return fract(sin(p * 127.1) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 0.75 * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 0.001), 1.5);
    return 1.5 * (1.0 - g2) * (1.0 + cos_theta * cos_theta) / ((2.0 + g2) * denom);
}

fn atmospheric_gradient(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let h = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sun_h = sun_dir.y;
    let day = smoothstep(-0.02, 0.22, sun_h);
    let night = smoothstep(0.10, -0.10, sun_h);
    let dusk = 1.0 - smoothstep(0.02, 0.34, abs(sun_h));

    let zenith_day = vec3<f32>(0.18, 0.44, 0.88);
    let horizon_day = vec3<f32>(0.66, 0.82, 0.98);
    let zenith_night = vec3<f32>(0.002, 0.005, 0.015);
    let horizon_night = vec3<f32>(0.010, 0.014, 0.034);
    let dusk_low = vec3<f32>(0.96, 0.40, 0.18);
    let dusk_high = vec3<f32>(0.24, 0.10, 0.28);

    var sky = mix(horizon_day, zenith_day, pow(h, 0.78)) * day;
    sky += mix(horizon_night, zenith_night, pow(h, 0.72)) * night;
    sky += mix(dusk_low, dusk_high, pow(h, 1.1)) * dusk * 0.52;

    let horizon_band = 1.0 - smoothstep(0.0, 0.32, abs(view_dir.y));
    sky += vec3<f32>(0.16, 0.22, 0.30) * horizon_band * (0.55 * day + 0.12 * dusk);

    return sky;
}

fn atmospheric_scatter(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_h = sun_dir.y;
    let day = smoothstep(-0.08, 0.10, sun_h);
    let dusk = 1.0 - smoothstep(0.0, 0.30, abs(sun_h));
    let night = smoothstep(0.12, -0.08, sun_h);

    let cos_theta = dot(view_dir, sun_dir);
    let mu = clamp(cos_theta, -1.0, 1.0);
    let rayleigh = rayleigh_phase(mu);
    let mie = mie_phase(mu, 0.78);

    let view_height = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
    let altitude = pow(view_height, 0.6);
    let horizon = 1.0 - smoothstep(0.0, 0.42, abs(view_dir.y));

    let rayleigh_color = vec3<f32>(0.14, 0.28, 0.62) * rayleigh;
    let mie_color = vec3<f32>(1.0, 0.72, 0.42) * mie;
    let dusk_tint = mix(vec3<f32>(1.0, 0.40, 0.16), vec3<f32>(0.60, 0.28, 0.50), altitude);

    var scatter = rayleigh_color * (0.45 * day + 0.12 * dusk);
    scatter += mie_color * (0.30 * day + 0.45 * dusk) * horizon;
    scatter += dusk_tint * dusk * horizon * 0.20;
    scatter += vec3<f32>(0.010, 0.012, 0.020) * night * altitude;

    return scatter;
}

fn sun_glow(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_h = sun_dir.y;
    let above = smoothstep(-0.08, 0.02, sun_h);
    if above <= 0.001 {
        return vec3<f32>(0.0);
    }

    let cos_s = clamp(dot(view_dir, sun_dir), 0.0, 1.0);
    let core = pow(cos_s, 160.0);
    let inner = pow(cos_s, 36.0);
    let outer = pow(cos_s, 8.0);
    let halo = pow(cos_s, 2.5) * 0.03;
    let horizon_boost = 1.0 - smoothstep(-0.12, 0.42, sun_h);

    let warm = mix(vec3<f32>(1.0, 0.82, 0.54), vec3<f32>(1.0, 0.97, 0.93), smoothstep(0.12, 0.75, sun_h));
    return warm * (core * 1.8 + inner * 0.52 + outer * 0.12 + halo * 0.8)
        * above * (0.65 + 0.45 * horizon_boost);
}

fn cloud_noise(p: vec2<f32>) -> f32 {
    let a = sin(dot(p, vec2<f32>(1.3, 1.7)) + 0.3) * 0.5 + 0.5;
    let b = sin(dot(p, vec2<f32>(2.1, 1.2)) + 1.7) * 0.5 + 0.5;
    let c = sin(dot(p, vec2<f32>(3.7, 2.9)) + 2.6) * 0.5 + 0.5;
    let d = sin(dot(p, vec2<f32>(5.4, 4.1)) + 4.2) * 0.5 + 0.5;
    return a * 0.34 + b * 0.26 + c * 0.22 + d * 0.18;
}

fn cloud_layer(view_dir: vec3<f32>, sun_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let sun_h = sun_dir.y;
    let day = smoothstep(-0.03, 0.20, sun_h);
    let dusk = 1.0 - smoothstep(0.04, 0.30, abs(sun_h));
    if day <= 0.001 && dusk <= 0.001 {
        return vec3<f32>(0.0);
    }

    let band = pow(clamp(1.0 - abs(view_dir.y) * 1.18, 0.0, 1.0), 1.7);
    if band <= 0.001 {
        return vec3<f32>(0.0);
    }

    let lon = atan2(view_dir.z, view_dir.x) / TAU + 0.5;
    let lat = view_dir.y * 0.5 + 0.5;
    let p = vec2<f32>(lon * 5.2 + time * 0.0025, lat * 2.2);
    let drift = vec2<f32>(time * 0.008, time * 0.003);

    let n = cloud_noise(p + drift * 0.18);
    let n2 = cloud_noise(p * 1.8 - drift * 0.8);
    let coverage = smoothstep(0.60, 0.90, n * 0.70 + n2 * 0.30);
    let wisps = smoothstep(0.52, 0.78, cloud_noise(p * 2.9 + drift * 1.3));
    let layer = coverage * band * band;
    if layer <= 0.001 {
        return vec3<f32>(0.0);
    }

    let sun_light = pow(max(dot(view_dir, sun_dir), 0.0), 12.0) * 0.30;
    let horizon_warm = vec3<f32>(1.0, 0.74, 0.50);
    let cloud_day = vec3<f32>(0.90, 0.93, 0.97);
    let cloud_shadow = vec3<f32>(0.48, 0.54, 0.65);
    let dusk_tint = mix(vec3<f32>(1.0, 0.64, 0.42), vec3<f32>(0.78, 0.36, 0.48), lat);

    var tint = mix(cloud_shadow, cloud_day, day);
    tint = mix(tint, dusk_tint, dusk * 0.75);
    tint += horizon_warm * (1.0 - smoothstep(0.0, 0.20, abs(view_dir.y))) * dusk * 0.16;

    return tint * layer * (0.24 + 0.34 * day + sun_light * 0.20) * (0.82 + wisps * 0.18);
}

fn horizon_haze(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let band = pow(1.0 - smoothstep(0.0, 0.36, abs(view_dir.y)), 1.4);
    let sun_h = sun_dir.y;
    let dusk = 1.0 - smoothstep(0.0, 0.28, abs(sun_h));
    let day = smoothstep(-0.02, 0.18, sun_h);
    let warm = vec3<f32>(0.98, 0.58, 0.30);
    let cool = vec3<f32>(0.20, 0.32, 0.48);
    return mix(cool, warm, dusk) * band * (0.10 + 0.05 * day);
}

fn star_field(view_dir: vec3<f32>, time: f32, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 {
        return vec3<f32>(0.0);
    }

    let p = floor(view_dir * 420.0);
    let h = hash21(p.xz + vec2<f32>(p.y, p.y * 0.37));
    if h < 0.978 {
        return vec3<f32>(0.0);
    }

    let twinkle = 0.75 + 0.25 * sin(time * mix(1.5, 4.0, hash11(h * 73.0)) + h * TAU);
    let horizon = clamp(view_dir.y * 2.2 + 0.22, 0.0, 1.0);
    let brightness = smoothstep(0.978, 0.9995, h) * twinkle * horizon * night_factor;

    let warm_star = vec3<f32>(1.00, 0.90, 0.74);
    let cool_star = vec3<f32>(0.78, 0.88, 1.00);
    let tint = mix(warm_star, cool_star, hash11(h * 91.7));

    return tint * brightness * 1.25;
}

fn moon_disk(view_dir: vec3<f32>, moon_dir: vec3<f32>, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 {
        return vec3<f32>(0.0);
    }

    let cos_m = dot(view_dir, moon_dir);
    if cos_m < 0.9985 {
        let halo = pow(max(cos_m, 0.0), 64.0) * 0.03 + pow(max(cos_m, 0.0), 12.0) * 0.008;
        return vec3<f32>(0.52, 0.58, 0.76) * halo * night_factor;
    }

    var right = cross(moon_dir, vec3<f32>(0.0, 1.0, 0.0));
    if length(right) < 0.01 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    right = normalize(right);
    let up = normalize(cross(right, moon_dir));

    let disk_uv = vec2<f32>(dot(view_dir - moon_dir, right), dot(view_dir - moon_dir, up)) / 0.0045;
    let disk_r2 = dot(disk_uv, disk_uv);
    if disk_r2 > 1.0 {
        return vec3<f32>(0.0);
    }

    let limb = sqrt(max(0.0, 1.0 - disk_r2));
    let limb_darkening = mix(0.58, 1.0, limb);
    let disk_mask = 1.0 - smoothstep(0.95, 1.0, disk_r2);
    let surface_tint = mix(
        vec3<f32>(0.92, 0.88, 0.74),
        vec3<f32>(0.85, 0.91, 0.98),
        clamp(moon_dir.y * 2.0, 0.0, 1.0),
    );

    let halo = vec3<f32>(0.52, 0.58, 0.76)
        * (pow(max(cos_m, 0.0), 20.0) * 0.014 + pow(max(cos_m, 0.0), 72.0) * 0.045);

    return (surface_tint * limb_darkening * disk_mask + halo) * night_factor;
}

fn underwater_color(ndc_pos: vec2<f32>, view_dir: vec3<f32>, time: f32) -> vec4<f32> {
    let uv = ndc_pos * 0.5 + 0.5;
    let ripples = sin(uv.x * 12.0 + time * 1.8) * sin(uv.y * 8.5 - time * 1.2);
    let caustics = ripples * 0.03 + sin((uv.x + uv.y) * 16.0 + time * 0.7) * 0.015;

    let depth_fog = mix(vec3<f32>(0.012, 0.075, 0.22), vec3<f32>(0.05, 0.20, 0.36), uv.y);
    let up = max(0.0, view_dir.y);
    let light_rays = up * up * (sin(uv.x * 18.0 + time * 2.0) * 0.5 + 0.5) * 0.10;
    let tint = vec3<f32>(0.18, 0.42, 0.58) + vec3<f32>(0.08, 0.12, 0.14) * caustics;

    return vec4<f32>((depth_fog + tint + light_rays) * (1.0 + caustics), 1.0);
}

@fragment
fn fs_sky(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = get_view_direction(in.ndc_pos);

    if uniforms.is_underwater > 0.5 {
        return underwater_color(in.ndc_pos, view_dir, uniforms.time);
    }

    let sun_dir = normalize(uniforms.sun_position);
    let moon_dir = normalize(uniforms.moon_position);
    let sun_h = sun_dir.y;
    let night = clamp(-sun_h * 4.0 - 0.05, 0.0, 1.0);
    let dusk = 1.0 - smoothstep(0.0, 0.30, abs(sun_h));

    var sky = atmospheric_gradient(view_dir, sun_dir);
    sky += atmospheric_scatter(view_dir, sun_dir);
    sky += horizon_haze(view_dir, sun_dir);
    sky += sun_glow(view_dir, sun_dir);
    sky += cloud_layer(view_dir, sun_dir, uniforms.time);
    sky += star_field(view_dir, uniforms.time, night);
    sky += moon_disk(view_dir, moon_dir, night);

    if dusk > 0.01 {
        let warm_band = pow(max(0.0, 1.0 - abs(view_dir.y)), 2.6) * dusk;
        sky += vec3<f32>(1.0, 0.40, 0.16) * warm_band * 0.08;
    }

    sky *= vec3<f32>(0.98, 0.99, 1.02);
    sky = clamp(sky, vec3<f32>(0.0), vec3<f32>(1.8));

    return vec4<f32>(sky, 1.0);
}
