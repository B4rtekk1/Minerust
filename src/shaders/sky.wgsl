/// Procedural Sky Shader
///
/// Renders a highly realistic atmospheric sky with:
/// - Accurate day/night blending
/// - Localized, vibrant sunrise/sunset centered precisely on the sun
/// - Proper horizon glow and color banding
/// - Smooth sun disk halo
/// - Special underwater mode with subtle animated water tint
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
    // Padding to match other shaders' bind group layout (zero cost)
    _pad1: vec2<f32>,
    _pad2: f32,
    _pad3: f32,
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
    @location(0) position: vec3<f32>,  // Expected: fullscreen quad (-1..1, -1..1, 0)
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_pos: vec2<f32>,
};

/// Vertex shader for fullscreen sky quad
/// Places sky at far plane so it renders behind all world geometry
@vertex
fn vs_sky(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position.xy, 0.9999, 1.0);
    out.ndc_pos = model.position.xy;
    return out;
}

/// Reconstruct world-space view direction from NDC coordinates
fn get_view_direction(ndc_xy: vec2<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(ndc_xy, 1.0, 1.0);
    let world_pos_hom = uniforms.inv_view_proj * ndc;
    let world_pos = world_pos_hom.xyz / world_pos_hom.w;
    return normalize(world_pos - uniforms.camera_pos);
}

/// Calculate atmospheric sky color with localized sunrise/sunset
fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_height = sun_dir.y;
    
    // Time-of-day blending factors
    let day_factor = clamp(sun_height, 0.0, 1.0);
    let night_factor = clamp(-sun_height, 0.0, 1.0);
    let sunset_factor = 1.0 - abs(sun_height);
    
    // View altitude (how high we're looking)
    let view_altitude = view_dir.y;
    
    // Horizontal projection for azimuth alignment
    let view_horiz_raw = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let view_horiz_len = length(view_horiz_raw);
    let view_horiz = view_horiz_raw / max(view_horiz_len, 0.0001);

    let sun_horiz = normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z));
    let cos_azimuth = dot(view_horiz, sun_horiz);
    
    // Precise 3D angle to sun
    let cos_theta = dot(view_dir, sun_dir);
    
    // Base sky gradients
    let day_zenith = vec3<f32>(0.25, 0.45, 0.85);
    let day_horizon = vec3<f32>(0.65, 0.82, 0.98);
    let night_zenith = vec3<f32>(0.002, 0.002, 0.010);
    let night_horizon = vec3<f32>(0.015, 0.015, 0.030);

    let height_factor = clamp(view_altitude * 0.5 + 0.5, 0.0, 1.0);
    let curved_height = pow(height_factor, 0.8);

    var sky = mix(day_horizon, day_zenith, curved_height) * day_factor;
    sky += mix(night_horizon, night_zenith, height_factor) * night_factor;
    
    // --- SUNRISE / SUNSET ---
    if sunset_factor > 0.01 && sun_height > -0.3 {
        let sunset_intensity = smoothstep(-0.2, 0.1, sun_height) * smoothstep(0.6, 0.0, sun_height);
        
        // Proximity measures
        let proximity_3d = max(0.0, cos_theta);
        let proximity_azimuth = max(0.0, cos_azimuth);
        let proximity = mix(proximity_azimuth, proximity_3d, 0.5);
        
        // Color layers with different falloffs
        let glow_core = pow(proximity_3d, 64.0);           // Tight yellow/white core
        let glow_near = pow(proximity, 8.0);              // Orange band
        let glow_wide = pow(proximity, 2.5);              // Red outer band
        
        // Horizon enhancement
        let horizon_strength = pow(1.0 - abs(view_altitude), 0.7);
        let horizon_boost = horizon_strength * smoothstep(0.0, 0.15, view_horiz_len);

        var sunset_color = vec3<f32>(0.0);
        sunset_color += vec3<f32>(1.0, 0.85, 0.4) * glow_core * 1.5;           // Bright core
        sunset_color += vec3<f32>(1.0, 0.45, 0.1) * glow_near * 1.2 * horizon_boost;
        sunset_color += vec3<f32>(0.9, 0.25, 0.1) * glow_wide * 0.8 * horizon_boost;
        
        // (Removed opposite_glow to prevent "double sun" effect)

        sky = mix(sky, sky + sunset_color, sunset_intensity);
    }
    
    // Daytime sun halo (subtle, not overblown)
    if day_factor > 0.1 {
        let sun_halo = pow(max(0.0, cos_theta), 256.0) * day_factor * 2.0;
        sky += vec3<f32>(1.0, 0.98, 0.9) * sun_halo;
    }

    return clamp(sky, vec3<f32>(0.0), vec3<f32>(1.5)); // Allow slight overbright for bloom
}

/// Fragment shader - final sky color per pixel
@fragment
fn fs_sky(in: VertexOutput) -> @location(0) vec4<f32> {
    // Underwater: render as dark murky water instead of sky
    if uniforms.is_underwater > 0.5 {
        let noise = sin(in.ndc_pos.x * 4.0 + uniforms.time * 1.8) * 0.5 + 0.5;
        let wave = sin(in.ndc_pos.y * 3.0 + uniforms.time * 2.2) * noise * 0.08;
        let underwater_tint = vec3<f32>(0.03, 0.10, 0.25) + vec3<f32>(wave);
        return vec4<f32>(underwater_tint, 1.0);
    }

    let sun_dir = normalize(uniforms.sun_position);
    let view_dir = get_view_direction(in.ndc_pos);

    let sky_color = calculate_sky_color(view_dir, sun_dir);

    return vec4<f32>(sky_color, 1.0);
}