struct Uniforms {
    view_proj:          mat4x4<f32>,
    inv_view_proj:      mat4x4<f32>,
    csm_view_proj:      array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos:         vec3<f32>,
    time:               f32,
    sun_position:       vec3<f32>,
    is_underwater:      f32,
    _screen_size:       vec2<f32>,
    _water_level:       f32,
    _reflection_mode:   f32,
    moon_position:      vec3<f32>,
    _pad1:              f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas:   texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(3) var shadow_map:      texture_depth_2d;
@group(0) @binding(4) var shadow_sampler:  sampler_comparison;


const PI:        f32 = 3.14159265359;
const INV_16PI:  f32 = 0.01989436789;
const INV_8PI:   f32 = 0.03978873578;
const TWO_PI:    f32 = 6.28318530718;


struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) color:     vec4<f32>,
    @location(3) uv:        vec2<f32>,
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
    let clip     = vec4<f32>(ndc_xy, 1.0, 1.0);
    let world_h  = uniforms.inv_view_proj * clip;
    return normalize(world_h.xyz / world_h.w - uniforms.camera_pos);
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.3))) * 43758.5453);
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 * INV_16PI * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2    = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 0.0001), 1.5);
    return 3.0 * INV_8PI * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
           / ((2.0 + g2) * denom);
}

fn mie_normalized(cos_theta: f32, g: f32) -> f32 {
    return mie_phase(cos_theta, g) / mie_phase(1.0, g);
}

fn star_field(view_dir: vec3<f32>, time: f32) -> vec3<f32> {
    let scale    = 300.0;
    let p        = floor(view_dir * scale);
    let h        = hash3(p);

    let tier_dim    = step(0.970, h) * (1.0 - step(0.997, h));
    let tier_bright = step(0.997, h);
    let brightness  = tier_dim * 0.45 + tier_bright * 1.0;
    if brightness < 0.001 { return vec3<f32>(0.0); }

    let horizon_fade = clamp(view_dir.y * 3.0 + 0.5, 0.0, 1.0);

    let twinkle_phase = h * TWO_PI;
    let twinkle_speed = mix(1.5, 4.0, fract(h * 73.156));
    let twinkle = 0.75 + 0.25 * sin(time * twinkle_speed + twinkle_phase);

    let warm_star = vec3<f32>(1.00, 0.88, 0.72);
    let cool_star = vec3<f32>(0.78, 0.88, 1.00);
    let star_color = mix(warm_star, cool_star, fract(h * 17.31));

    return star_color * brightness * twinkle * horizon_fade;
}

fn moon_color(view_dir: vec3<f32>, moon_dir: vec3<f32>,
              sun_dir: vec3<f32>, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 { return vec3<f32>(0.0); }

    let cos_m = dot(view_dir, moon_dir);
    if cos_m < 0.998 {
        if cos_m < 0.985 { return vec3<f32>(0.0); }
        let halo_w = pow(max(0.0, cos_m), 12.0)  * 0.012;
        let halo_i = pow(max(0.0, cos_m), 64.0)  * 0.05;
        return vec3<f32>(0.50, 0.55, 0.72) * (halo_w + halo_i) * night_factor;
    }

    let right   = normalize(cross(moon_dir, vec3<f32>(0.0, 1.0, 0.0)));
    let up      = normalize(cross(right, moon_dir));
    let moon_r  = 0.004;
    let disk_uv = vec2<f32>(dot(view_dir - moon_dir, right),
                            dot(view_dir - moon_dir, up)) / moon_r;
    let disk_r2 = dot(disk_uv, disk_uv);
    if disk_r2 > 1.0 { return vec3<f32>(0.0); }

    let limb = sqrt(max(0.0, 1.0 - disk_r2));
    let limb_darkening = mix(0.55, 1.0, limb);

    let cos_phase  = dot(moon_dir, sun_dir);
    let sun_disk_x = dot(sun_dir, right);
    let sun_disk_y = dot(sun_dir, up);
    let lit_side   = sun_disk_x * disk_uv.x + sun_disk_y * disk_uv.y;
    let terminator = smoothstep(-0.04, 0.04, lit_side + cos_phase * 0.05);

    let earthshine = (1.0 - terminator) * 0.035 * vec3<f32>(0.25, 0.35, 0.60);

    let altitude_t   = clamp(moon_dir.y * 2.0, 0.0, 1.0);
    let moon_surface = mix(vec3<f32>(0.92, 0.86, 0.72),
                           vec3<f32>(0.90, 0.92, 0.98),
                           altitude_t);

    let disk_color = moon_surface * limb_darkening * terminator + earthshine;

    let halo_w = pow(max(0.0, cos_m), 12.0) * 0.012;
    let halo_i = pow(max(0.0, cos_m), 64.0) * 0.05;
    let halo   = vec3<f32>(0.50, 0.55, 0.72) * (halo_w + halo_i);

    return (disk_color + halo) * night_factor;
}

struct SkyParams {
    day_factor:    f32,
    night_factor:  f32,
    sunset_factor: f32,
    golden_hour:   f32,
    twilight:      f32,
    cos_theta:     f32,
    cos_azimuth:   f32,
    view_altitude: f32,
    view_horiz_len: f32,
    height_factor: f32,
    curved_height: f32,
};

fn make_sky_params(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> SkyParams {
    var p: SkyParams;
    let sh = sun_dir.y;

    p.day_factor    = clamp(sh * 5.0, 0.0, 1.0);
    p.night_factor  = clamp(-sh * 4.0 - 0.2, 0.0, 1.0);
    p.sunset_factor = smoothstep(0.30, 0.0, abs(sh));
    p.golden_hour   = smoothstep(0.18, 0.0, sh) * smoothstep(-0.10, 0.12, sh);
    p.twilight      = smoothstep(0.0, -0.15, sh) * smoothstep(-0.25, -0.04, sh);

    p.view_altitude  = view_dir.y;
    let vh_raw       = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    p.view_horiz_len = length(vh_raw);
    let vh           = vh_raw / max(p.view_horiz_len, 0.0001);
    let sh_raw       = vec3<f32>(sun_dir.x, 0.0, sun_dir.z);
    let sh_dir       = sh_raw / max(length(sh_raw), 0.0001);

    p.cos_theta   = dot(view_dir, sun_dir);
    p.cos_azimuth = dot(vh, sh_dir);

    p.height_factor = clamp(p.view_altitude * 0.5 + 0.5, 0.0, 1.0);
    p.curved_height = pow(p.height_factor, 0.75);

    return p;
}

fn sky_base(p: SkyParams) -> vec3<f32> {
    let day   = mix(vec3<f32>(0.60, 0.78, 0.96), vec3<f32>(0.18, 0.38, 0.82), p.curved_height);
    let night = mix(vec3<f32>(0.010, 0.012, 0.025), vec3<f32>(0.001, 0.002, 0.010), p.height_factor);
    return day * p.day_factor + night * p.night_factor;
}

fn sky_rayleigh(p: SkyParams) -> vec3<f32> {
    let r = rayleigh_phase(p.cos_theta);
    let scatter = vec3<f32>(0.28, 0.52, 1.00) * r * 0.15;
    return scatter * p.curved_height * p.day_factor;
}

fn sky_sunset_glow(p: SkyParams) -> vec3<f32> {
    if p.sunset_factor < 0.01 { return vec3<f32>(0.0); }

    let intensity = smoothstep(-0.14, 0.06, p.cos_theta - 0.0)
                  * p.sunset_factor;

    let prox_3d = max(0.0, p.cos_theta);
    let prox_az = max(0.0, p.cos_azimuth);
    let prox    = mix(prox_az, prox_3d, 0.5);

    let mie_s = mie_normalized(p.cos_theta, 0.76);

    let glow_core = pow(prox_3d, 90.0);
    let glow_near = pow(prox,    10.0);
    let glow_wide = pow(prox,     3.5);

    let horiz_band  = pow(1.0 - abs(p.view_altitude), 1.2);
    let horiz_boost = horiz_band * smoothstep(0.0, 0.12, p.view_horiz_len);

    var c = vec3<f32>(0.0);
    c += vec3<f32>(1.00, 0.90, 0.50) * glow_core * 1.5;
    c += vec3<f32>(1.00, 0.45, 0.10) * glow_near * 0.9 * horiz_boost;
    c += vec3<f32>(0.85, 0.20, 0.08) * glow_wide * 0.35 * horiz_boost;
    c += vec3<f32>(1.00, 0.65, 0.25) * clamp(mie_s, 0.0, 1.0) * 0.4;

    return c * intensity;
}

fn sky_golden_hour_tint(sky: vec3<f32>, p: SkyParams) -> vec3<f32> {
    if p.golden_hour < 0.01 { return sky; }
    return sky * mix(vec3<f32>(1.0), vec3<f32>(1.10, 0.90, 0.68), p.golden_hour * 0.45);
}

fn sky_twilight(p: SkyParams) -> vec3<f32> {
    if p.twilight < 0.01 { return vec3<f32>(0.0); }

    let band = vec3<f32>(0.14, 0.10, 0.30) * (1.0 - p.height_factor)
             + vec3<f32>(0.06, 0.04, 0.18) * p.height_factor;

    let arch = pow(max(0.0, -p.cos_azimuth), 4.0)
             * pow(1.0 - abs(p.view_altitude), 2.0)
             * p.twilight * 0.25;

    var c = band * 0.5 * p.twilight * 0.6;
    c    += vec3<f32>(0.50, 0.25, 0.38) * arch;
    return c;
}

fn sky_haze(p: SkyParams) -> vec3<f32> {
    let haze_amount = pow(1.0 - abs(p.view_altitude), 5.0) * p.day_factor * 0.55;
    let haze_color  = mix(vec3<f32>(0.78, 0.85, 0.95),
                          vec3<f32>(0.92, 0.88, 0.82), p.golden_hour);
    return haze_color * 0.18 * haze_amount;
}

fn sky_horizon_fog(sky: vec3<f32>, p: SkyParams) -> vec3<f32> {
    let fog_depth = clamp(smoothstep(0.05, -0.30, p.view_altitude), 0.0, 1.0);
    let fog_color = mix(vec3<f32>(0.68, 0.72, 0.78),
                        vec3<f32>(0.04, 0.04, 0.08), p.night_factor);
    return mix(sky, fog_color, fog_depth * 0.7);
}

fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, moon_dir: vec3<f32>) -> vec3<f32> {
    let p = make_sky_params(view_dir, sun_dir);

    var sky = sky_base(p);
    sky    += sky_rayleigh(p);

    let glow = sky_sunset_glow(p);
    sky      = mix(sky, sky + glow, p.sunset_factor);

    sky = sky_golden_hour_tint(sky, p);
    sky += sky_twilight(p);
    sky += sky_haze(p);
    sky  = sky_horizon_fog(sky, p);

    if p.night_factor > 0.01 {
        let star_rgb = star_field(view_dir, uniforms.time);
        sky += star_rgb * p.night_factor * 0.9;
    }

    sky += moon_color(view_dir, moon_dir, sun_dir, p.night_factor);

    return clamp(sky, vec3<f32>(0.0), vec3<f32>(2.0));
}

fn underwater_color(ndc_pos: vec2<f32>, view_dir: vec3<f32>, time: f32) -> vec4<f32> {
    let uv = ndc_pos * 0.5 + 0.5;

    const FREQ_A: f32 = 9.0;  const FREQ_B: f32  = 7.0;
    const FREQ_C: f32 = 14.0; const FREQ_D: f32  = 11.0;
    const FREQ_E: f32 = 8.0;
    let w1 = sin(uv.x * FREQ_A + time * 1.6) * sin(uv.y * FREQ_B + time * 1.3);
    let w2 = sin(uv.x * FREQ_C - time * 2.1 + 0.5) * sin(uv.y * FREQ_D + time * 0.9);
    let w3 = sin((uv.x + uv.y) * FREQ_E + time * 1.1);
    let caustics = (w1 * 0.5 + w2 * 0.3 + w3 * 0.2) * 0.035 + 0.04;

    let depth_fog = mix(vec3<f32>(0.015, 0.07, 0.22),
                        vec3<f32>(0.04,  0.18, 0.38), uv.y);

    let up           = max(0.0, view_dir.y);
    let ray_shimmer  = sin(uv.x * 20.0 + time * 3.0) * 0.5 + 0.5;
    let light_rays   = up * up * ray_shimmer * 0.12 * vec3<f32>(0.15, 0.30, 0.20);

    let surface_glint = pow(up, 4.0)
                      * (sin(uv.x * 40.0 + time * 5.0) * 0.5 + 0.5)
                      * 0.25;
    let surface_color = vec3<f32>(0.20, 0.50, 0.60) * surface_glint;

    let dist_t   = clamp(length(ndc_pos) * 0.5, 0.0, 1.0);
    let chroma   = vec3<f32>(
        exp(-dist_t * 3.0),
        exp(-dist_t * 1.8),
        exp(-dist_t * 0.6),
    );

    var result = (depth_fog + caustics + light_rays + surface_color) * chroma;
    return vec4<f32>(result, 1.0);
}

@fragment
fn fs_sky(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = get_view_direction(in.ndc_pos);

    if uniforms.is_underwater > 0.5 {
        return underwater_color(in.ndc_pos, view_dir, uniforms.time);
    }

    let sun_dir  = uniforms.sun_position;
    let moon_dir = uniforms.moon_position;

    return vec4<f32>(calculate_sky_color(view_dir, sun_dir, moon_dir), 1.0);
}
