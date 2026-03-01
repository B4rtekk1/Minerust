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
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_sun(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let sun_dir = normalize(uniforms.sun_position);
    // Position far away to simulate infinite distance
    let sun_world_pos = uniforms.camera_pos + sun_dir * 450.0;
    
    // Construct orthonormal basis for billboarding
    let forward = -sun_dir;
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    var right = normalize(cross(world_up, forward));
    if length(right) < 0.01 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    let up = cross(forward, right);

    // Significantly increased size to prevent clipping at edges
    let size = 250.0;

    let offset = right * model.position.x * size + up * model.position.y * size;
    let world_pos = sun_world_pos + offset;

    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    // Force to background
    out.clip_position.z = out.clip_position.w * 0.99999;
    out.uv = model.uv;

    return out;
}

// --- Procedural Helpers ---

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

fn get_corona(uv: vec2<f32>, time: f32) -> f32 {
    let angle = atan2(uv.y, uv.x);
    let d = length(uv);
    let n = noise(vec2<f32>(angle * 3.0, time * 2.0)) * 0.5 + 0.5;
    let corona = pow(1.0 - clamp(d * (1.1 - n * 0.3), 0.0, 1.0), 4.0);
    return corona;
}

fn get_rays(uv: vec2<f32>, time: f32) -> f32 {
    let angle = atan2(uv.y, uv.x);
    let dist = length(uv);
    
    // Rotating primary rays
    var rays = pow(abs(sin(angle * 4.0 + time * 0.2)), 12.0);
    // Smaller secondary spikes
    rays += pow(abs(sin(angle * 12.0 - time * 0.1)), 16.0) * 0.4;

    // Steeper falloff to reach 0 before the quad edge (reaches 0 at dist=0.85)
    rays *= pow(1.0 - clamp(dist / 0.85, 0.0, 1.0), 3.0);
    return rays;
}

@fragment
fn fs_sun(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 2.0 - 1.0;
    let dist = length(uv);
    
    // Hard radial mask to ensure no square edges are visible
    let radial_mask = smoothstep(1.0, 0.8, dist);
    if radial_mask <= 0.0 {
        discard;
    }

    let sun_height = normalize(uniforms.sun_position).y;
    
    // Dynamic color based on sun height
    let sunset_factor = clamp(1.0 - abs(sun_height), 0.0, 1.0);

    let sun_color_midday = vec3<f32>(1.2, 1.1, 0.9);
    let sun_color_sunset = vec3<f32>(1.5, 0.4, 0.1);
    let color_core = mix(sun_color_midday, sun_color_sunset, pow(sunset_factor, 1.5));
    
    // 1. Solar Disk
    let disk_radius = 0.03;
    if dist < disk_radius {
        let n = dist / disk_radius;
        let limb_darkening = pow(1.0 - n * n, 0.5);
        let disk_color = color_core * (1.5 + limb_darkening * 2.5);
        return vec4<f32>(disk_color, 1.0);
    }

    // 2. Multi-layered Glow
    var final_color = vec3<f32>(0.0);
    var final_alpha = 0.0;

    // Corona (shimmering inner ring)
    let corona = get_corona(uv * 12.0, uniforms.time);
    final_color += color_core * corona * 0.8 * radial_mask;
    final_alpha = max(final_alpha, corona * 0.9 * radial_mask);

    // Inner bloom (intense)
    let inner_bloom = pow(1.0 - clamp((dist - disk_radius) * 6.0, 0.0, 1.0), 4.0);
    final_color += color_core * inner_bloom * 2.5 * radial_mask;
    final_alpha = max(final_alpha, inner_bloom * radial_mask);

    // Outer atmospheric glow (larger at sunset)
    let outer_radius = mix(0.5, 0.9, sunset_factor);
    let outer_bloom = pow(1.0 - clamp(dist / outer_radius, 0.0, 1.0), 8.0);
    final_color += mix(color_core, vec3<f32>(1.0, 0.2, 0.0), sunset_factor) * outer_bloom * 1.5 * radial_mask;
    final_alpha = max(final_alpha, outer_bloom * 0.6 * radial_mask);

    // 3. Diffraction Rays
    let rays = get_rays(uv, uniforms.time);
    final_color += color_core * rays * 3.0 * radial_mask;
    final_alpha = clamp(final_alpha + rays * 0.7 * radial_mask, 0.0, 1.0);

    // Underwater tinting
    if uniforms.is_underwater > 0.5 {
        final_color *= vec3<f32>(0.1, 0.4, 0.8);
        final_alpha *= 0.3;
    }

    if final_alpha < 0.001 {
        discard;
    }

    return vec4<f32>(final_color, final_alpha);
}


