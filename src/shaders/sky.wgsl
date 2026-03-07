/// Procedural Sky Shader — Enhanced
///
/// Renders a highly realistic atmospheric sky with:
/// - Physically-based Rayleigh + Mie atmospheric scattering
/// - Accurate day/night/twilight blending with color temperature shifts
/// - Procedural star field with twinkling
/// - Moon disk with halo
/// - Localized, vibrant sunrise/sunset centered precisely on the sun
/// - Proper horizon haze and ground fog
/// - HDR color layering for golden hour / twilight
/// - Smooth sun disk with physical Mie halo
/// - Enhanced underwater mode with caustics and light rays
///
/// Optimized for fullscreen quad rendering (sky dome replacement)

struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    csm_view_proj: array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
    // Padding fields to match the shared Rust Uniforms layout
    _screen_size: vec2<f32>,
    _water_level: f32,
    _reflection_mode: f32,
    moon_position: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Dummy bindings to keep bind group layout identical across shaders
@group(0) @binding(1)
var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2)
var texture_sampler: sampler;
@group(0) @binding(3)
var shadow_map: texture_depth_2d;
@group(0) @binding(4)
var shadow_sampler: sampler_comparison;

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

// ─────────────────────────────────────────────
// VERTEX SHADER
// ─────────────────────────────────────────────

@vertex
fn vs_sky(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.xy, 0.9999, 1.0);
    out.ndc_pos = model.position.xy;
    return out;
}

// ─────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────

fn get_view_direction(ndc_xy: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(ndc_xy, 1.0, 1.0);
    let world_pos_hom = uniforms.inv_view_proj * ndc;
    let world_pos = world_pos_hom.xyz / world_pos_hom.w;
    return normalize(world_pos - uniforms.camera_pos);
}

// ─────────────────────────────────────────────
// ATMOSPHERIC SCATTERING PHASES
// ─────────────────────────────────────────────

/// Rayleigh phase function — responsible for blue sky
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * 3.14159265)) * (1.0 + cos_theta * cos_theta);
}

/// Mie phase function — responsible for sun halo/glare (g=0 isotropic, g→1 forward peak)
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 0.0001), 1.5);
    return (3.0 / (8.0 * 3.14159265)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta))
        / ((2.0 + g2) * denom);
}

// ─────────────────────────────────────────────
// STAR FIELD
// ─────────────────────────────────────────────

/// Hash function for pseudo-random star placement on sphere
fn hash3(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.3))) * 43758.5453);
}

/// Procedural star field with per-star twinkle
fn star_field(view_dir: vec3<f32>, time: f32) -> f32 {
    // Quantize direction to create discrete star cells
    let scale = 300.0;
    let p = floor(view_dir * scale);
    let h = hash3(p);

    // Render stars across the whole sky including below horizon
    let horizon_fade = clamp(view_dir.y + 1.0, 0.0, 1.0);

    // Rare bright stars + common dim stars
    let brightness = mix(0.4, 1.0, step(0.997, h)) * step(0.970, h);
    if brightness == 0.0 { return 0.0; }

    // Per-star twinkle with unique phase
    let twinkle_phase = h * 6.2831;
    let twinkle_speed = mix(1.5, 4.0, hash3(p +     0.5));
    let twinkle = 0.75 + 0.25 * sin(time * twinkle_speed + twinkle_phase);

    return brightness * twinkle * horizon_fade;
}

// ─────────────────────────────────────────────
// MOON
// ─────────────────────────────────────────────

/// Moon disk + halo contribution
fn moon_color(view_dir: vec3<f32>, moon_dir: vec3<f32>, night_factor: f32) -> vec3<f32> {
    if night_factor < 0.01 { return vec3<f32>(0.0); }

    let cos_m = dot(view_dir, moon_dir);

    // Hard disk edge
    let disk = smoothstep(0.9994, 0.9998, cos_m);

    // Soft diffuse halo around moon
    let halo_wide  = pow(max(0.0, cos_m), 12.0)  * 0.015;
    let halo_inner = pow(max(0.0, cos_m), 64.0)  * 0.06;

    let moon_white = vec3<f32>(0.88, 0.90, 0.96);
    let halo_tint  = vec3<f32>(0.50, 0.55, 0.72);

    return (moon_white * disk + halo_tint * (halo_wide + halo_inner)) * night_factor;
}

// ─────────────────────────────────────────────
// MAIN SKY COLOR
// ─────────────────────────────────────────────

fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, moon_dir: vec3<f32>) -> vec3<f32> {
    let sun_height   = sun_dir.y;
    let view_altitude = view_dir.y;

    // ── Time-of-day factors ──────────────────────────────────────────────────
    let day_factor    = clamp(sun_height * 5.0, 0.0, 1.0);          // sharp day onset
    let night_factor  = clamp(-sun_height * 4.0 - 0.2, 0.0, 1.0);  // delayed night
    let sunset_factor = smoothstep(0.30, 0.0, abs(sun_height));      // narrower transition

    // Golden-hour blend (tight window just before/after horizon)
    let golden_hour   = smoothstep(0.18, 0.0, sun_height) * smoothstep(-0.10, 0.12, sun_height);

    // Twilight (sun just below horizon — purple/indigo)
    let twilight      = smoothstep(0.0, -0.15, sun_height) * smoothstep(-0.25, -0.04, sun_height);

    // ── Azimuth helpers ──────────────────────────────────────────────────────
    let view_horiz_raw = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let view_horiz_len = length(view_horiz_raw);
    let view_horiz     = view_horiz_raw / max(view_horiz_len, 0.0001);
    let sun_horiz      = normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z));
    let cos_azimuth    = dot(view_horiz, sun_horiz);
    let cos_theta      = dot(view_dir, sun_dir);   // full 3    -D angle to sun

    // ── Base sky gradient ────────────────────────────────────────────────────
    let day_zenith   = vec3<f32>(0.18, 0.38, 0.82);
    let day_horizon  = vec3<f32>(0.60, 0.78, 0.96);
    let night_zenith = vec3<f32>(0.001, 0.002, 0.010);
    let night_horizon= vec3<f32>(0.010, 0.012, 0.025);

    let height_factor  = clamp(view_altitude * 0.5 + 0.5, 0.0, 1.0);
    let curved_height  = pow(height_factor, 0.75);

    var sky = mix(day_horizon, day_zenith, curved_height) * day_factor;
    sky    += mix(night_horizon, night_zenith, height_factor) * night_factor;

    // ── Rayleigh scattering tint ─────────────────────────────────────────────
    // Adds blue wavelength dominance at zenith, slightly warmer at horizon
    let rayleigh = rayleigh_phase(cos_theta);
    let rayleigh_color = vec3<f32>(0.38, 0.60, 1.0) * rayleigh * 0.15 * day_factor;
    sky += rayleigh_color * curved_height;

    // ── Sunrise / Sunset (localized, HDR) ───────────────────────────────────
    if sunset_factor > 0.01 && sun_height > -0.20 {
        let sunset_intensity = smoothstep(-0.14, 0.06, sun_height) * smoothstep(0.40, 0.0, sun_height);

        let proximity_3d  = max(0.0, cos_theta);
        let proximity_az  = max(0.0, cos_azimuth);
        let proximity     = mix(proximity_az, proximity_3d, 0.5);

        // Mie forward-scattering for physically correct glow cone
        let mie_sun = mie_phase(cos_theta, 0.76) / mie_phase(1.0, 0.76); // normalize

        let glow_core = pow(proximity_3d, 90.0);           // tight white-yellow kernel
        let glow_near = pow(proximity,    10.0);           // orange band — tighter
        let glow_wide = pow(proximity,    3.5);            // red outer wash — less spread

        let horizon_strength = pow(1.0 - abs(view_altitude), 1.2);
        let horizon_boost    = horizon_strength * smoothstep(0.0, 0.12, view_horiz_len);

        var sunset_color = vec3<f32>(0.0);
        sunset_color += vec3<f32>(1.00, 0.90, 0.50) * glow_core * 1.5;                      // white-gold core
        sunset_color += vec3<f32>(1.00, 0.45, 0.10) * glow_near * 0.9 * horizon_boost;      // orange
        sunset_color += vec3<f32>(0.85, 0.20, 0.08) * glow_wide * 0.35 * horizon_boost;     // deep red
        sunset_color += vec3<f32>(1.00, 0.65, 0.25) * clamp(mie_sun, 0.0, 1.0) * 0.4 * sunset_intensity; // Mie halo

        sky = mix(sky, sky + sunset_color, sunset_intensity);
    }

    // ── Golden hour color temperature shift ─────────────────────────────────
    if golden_hour > 0.01 {
        let warm_tint = vec3<f32>(1.10, 0.90, 0.68);
        sky *= mix(vec3<f32>(1.0), warm_tint, golden_hour * 0.45);
    }

    // ── Twilight — indigo/purple band above darkening horizon ───────────────
    if twilight > 0.01 {
        let twilight_color = vec3<f32>(0.14, 0.10, 0.30) * (1.0 - height_factor)
                           + vec3<f32>(0.06, 0.04, 0.18) * height_factor;
        sky = mix(sky, sky + twilight_color * 0.5, twilight * 0.6);

        // Faint pink anti-twilight arch on opposite side from sun
        let anti_sun = max(0.0, -cos_azimuth);
        let arch = pow(anti_sun, 4.0) * pow(1.0 - abs(view_altitude), 2.0) * twilight * 0.25;
        sky += vec3<f32>(0.50, 0.25, 0.38) * arch;
    }

    // The visible sun disk/corona is rendered by the dedicated sun shader.
    // Keep only atmospheric scattering in sky to avoid doubling the sun.

    // ── Horizon haze (aerosol scattering) ───────────────────────────────────
    let haze = pow(1.0 - abs(view_altitude), 5.0) * day_factor;
    let haze_color = mix(vec3<f32>(0.78, 0.85, 0.95), vec3<f32>(0.92, 0.88, 0.82), golden_hour);
    sky = mix(sky, sky + haze_color * 0.18, haze * 0.55);

    // ── Ground / below-horizon fog ───────────────────────────────────────────
    {
        let fog_depth = clamp(smoothstep(0.05, -0.30, view_altitude), 0.0, 1.0);
        let fog_day   = vec3<f32>(0.68, 0.72, 0.78);
        let fog_night = vec3<f32>(0.04, 0.04, 0.08);
        let fog_color = mix(fog_day, fog_night, night_factor);
        sky = mix(sky, fog_color, fog_depth * 0.7);
    }

    // ── Stars ────────────────────────────────────────────────────────────────
    if night_factor > 0.01 {
        let stars = star_field(view_dir, uniforms.time);
        // Color variation: some stars are warm, some cool
        let star_h = hash3(floor(view_dir * 300.0) + 1.0);
        let star_col = mix(vec3<f32>(1.0, 0.85, 0.70), vec3<f32>(0.75, 0.88, 1.0), star_h);
        sky += star_col * stars * night_factor * 0.9;
    }

    // ── Moon ─────────────────────────────────────────────────────────────────
    sky += moon_color(view_dir, moon_dir, night_factor);

    // Allow slight overbright for bloom pipeline
    return clamp(sky, vec3<f32>(0.0), vec3<f32>(2.0));
}

// ─────────────────────────────────────────────
// UNDERWATER
// ─────────────────────────────────────────────

fn underwater_color(ndc_pos: vec2<f32>, view_dir: vec3<f32>, time: f32) -> vec4<f32> {
    let t  = time;
    let uv = ndc_pos * 0.5 + 0.5;

    // Multi-layer caustics
    let w1 = sin(uv.x * 9.0 + t * 1.6) * sin(uv.y * 7.0 + t * 1.3);
    let w2 = sin(uv.x * 14.0 - t * 2.1 + 0.5) * sin(uv.y * 11.0 + t * 0.9);
    let w3 = sin((uv.x + uv.y) * 8.0 + t * 1.1);
    let caustics = (w1 * 0.5 + w2 * 0.3 + w3 * 0.2) * 0.035 + 0.04;

    // Depth fog — darker above (looking up toward surface from deep)
    let depth_fog = mix(vec3<f32>(0.015, 0.07, 0.22), vec3<f32>(0.04, 0.18, 0.38), uv.y);

    // Upward light ray contribution
    let up_factor = max(0.0, view_dir.y);
    let ray_shimmer = sin(uv.x * 20.0 + t * 3.0) * 0.5 + 0.5;
    let light_ray = up_factor * up_factor * ray_shimmer * 0.12 * vec3<f32>(0.15, 0.30, 0.20);

    // Surface shimmer when looking directly up
    let surface_glint = pow(up_factor, 4.0) * (sin(uv.x * 40.0 + t * 5.0) * 0.5 + 0.5) * 0.25;
    let surface_color = vec3<f32>(0.20, 0.50, 0.60) * surface_glint;

    let result = depth_fog + caustics + light_ray + surface_color;
    return vec4<f32>(result, 1.0);
}

// ─────────────────────────────────────────────
// FRAGMENT SHADER
// ─────────────────────────────────────────────

@fragment
fn fs_sky(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_dir = get_view_direction(in.ndc_pos);

    if uniforms.is_underwater > 0.5 {
        return underwater_color(in.ndc_pos, view_dir, uniforms.time);
    }

    let sun_dir  = normalize(uniforms.sun_position);
    let moon_dir = normalize(uniforms.moon_position);

    let sky_color = calculate_sky_color(view_dir, sun_dir, moon_dir);

    return vec4<f32>(sky_color, 1.0);
}
