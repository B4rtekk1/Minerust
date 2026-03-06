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
    // Keep sun closer to camera so its billboard appears larger on screen.
    let sun_world_pos = uniforms.camera_pos + sun_dir * 180.0;

    // Construct orthonormal basis for billboarding
    let forward = -sun_dir;
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    var right = normalize(cross(world_up, forward));
    if length(right) < 0.01 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    let up = cross(forward, right);

    // Significantly larger stylized sun disk.
    let size = 220.0;

    let offset = right * model.position.x * size + up * model.position.y * size;
    let world_pos = sun_world_pos + offset;

    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    // Force to background
    out.clip_position.z = out.clip_position.w * 0.99999;
    out.uv = model.uv;

    return out;
}

@fragment
fn fs_sun(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 2.0 - 1.0;
    let dist = length(uv);

    // Fade billboard near edges to keep it invisible as a quad.
    let edge_fade = 1.0 - smoothstep(0.92, 1.0, dist);

    let sun_height = normalize(uniforms.sun_position).y;
    let above_horizon = smoothstep(-0.05, 0.02, sun_height);
    let sun_visibility = above_horizon * edge_fade;
    if sun_visibility <= 0.001 {
        discard;
    }

    // Atmospheric extinction: blue/green attenuate more at low sun elevations.
    let sunset_factor = clamp(1.0 - abs(sun_height), 0.0, 1.0);
    let air_mass = 1.0 / max(0.10, sun_height + 0.13);
    let extinction = exp(-vec3<f32>(0.022, 0.052, 0.115) * air_mass * 0.9);
    let base_color = vec3<f32>(1.20, 1.14, 1.05) * extinction;
    let reddening = clamp((air_mass - 1.0) * 0.35, 0.0, 1.0);

    // Solar disk with soft edge and mild limb darkening.
    let disk_radius = 0.025;
    let disk_mask = smoothstep(disk_radius + 0.0015, disk_radius - 0.0015, dist);
    let disk_n = clamp(dist / disk_radius, 0.0, 1.0);
    let limb_darkening = mix(1.08, 0.90, pow(disk_n, 1.55));
    let disk_color = base_color * limb_darkening;

    // Physically-inspired Mie-like halo: bright aureole + broad atmospheric glow.
    let inner_halo = exp(-pow(dist / (disk_radius * 2.2), 2.0));
    let outer_radius = mix(0.10, 0.16, sunset_factor);
    let outer_halo = exp(-pow(dist / outer_radius, 2.0));
    let halo_tint = mix(base_color, vec3<f32>(1.0, 0.72, 0.50), reddening * 0.8);

    var final_color = vec3<f32>(0.0);
    final_color += disk_color * disk_mask * 1.75;
    final_color += halo_tint * inner_halo * 0.16;
    final_color += halo_tint * outer_halo * 0.06;

    var final_alpha = 0.0;
    final_alpha += disk_mask;
    final_alpha += inner_halo * 0.12;
    final_alpha += outer_halo * 0.05;

    // Underwater tinting
    if uniforms.is_underwater > 0.5 {
        final_color *= vec3<f32>(0.1, 0.4, 0.8);
        final_alpha *= 0.3;
    }

    final_color *= sun_visibility;
    final_alpha = clamp(final_alpha * sun_visibility, 0.0, 1.0);

    if final_alpha < 0.001 {
        discard;
    }

    return vec4<f32>(final_color, final_alpha);
}
