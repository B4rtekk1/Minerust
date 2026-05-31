const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.075, 0.410, 0.455);
const WATER_COLOR_DEEP:    vec3<f32> = vec3<f32>(0.010, 0.072, 0.165);
const WATER_OPACITY:       f32 = 0.30;
const FRESNEL_R0:          f32 = 0.020;
const WATER_LEVEL_OFFSET:  f32 = 0.15;

const FOG_NEAR: f32 = 0.0;
const FOG_FAR:  f32 = 180.0;

struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    reflection_mode: f32,
    moon_position:  vec3<f32>,
    _pad1_moon:     f32,
    moon_intensity: f32,
    wind_dir_x:     f32,
    wind_dir_z:     f32,
    wind_speed:     f32,
    rain_factor:    f32,
    sky_visibility: f32,
    menu_blur:      f32,
    _pad_uniforms:  f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(8) var scene_color: texture_2d<f32>;
@group(0) @binding(10) var scene_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:   vec3<f32>,
    @location(1) wave_normal: vec3<f32>,
};

fn face_normal(index: u32) -> vec3<f32> {
    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)
    );
    return normals[index % 6u];
}

fn cheap_water_normal(world_xz: vec2<f32>, time: f32) -> vec3<f32> {
    let wind = normalize(vec2<f32>(uniforms.wind_dir_x, uniforms.wind_dir_z) + vec2<f32>(0.001, 0.001));
    let cross_wind = vec2<f32>(-wind.y, wind.x);
    let speed = max(uniforms.wind_speed, 0.15);

    let phase_a = dot(world_xz, wind) * 0.55 - time * speed * 0.70;
    let phase_b = dot(world_xz, cross_wind) * 1.15 + time * speed * 0.42;
    let slope = wind * (cos(phase_a) * 0.115) + cross_wind * (cos(phase_b) * 0.045);

    return normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
}

@vertex
fn vs_water(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = model.position;

    let normal_index = model.packed & 0x7u;
    var normal = face_normal(normal_index);

    if normal_index == 3u {
        pos.y -= WATER_LEVEL_OFFSET;
        normal = cheap_water_normal(pos.xz, uniforms.time);
    }

    out.clip_position = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.wave_normal = normal;
    return out;
}

fn schlick_fresnel(cos_theta: f32) -> f32 {
    let x = 1.0 - cos_theta;
    let x2 = x * x;
    return FRESNEL_R0 + (1.0 - FRESNEL_R0) * x2 * x2 * x;
}

fn pow8(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    return x4 * x4;
}

fn sky_reflection_color(view_dir: vec3<f32>, sun_dir: vec3<f32>, moon_intensity: f32) -> vec3<f32> {
    let day = smoothstep(-0.10, 0.20, sun_dir.y);
    let night = 1.0 - day;
    let horizon = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);

    let day_sky = mix(vec3<f32>(0.52, 0.78, 0.96), vec3<f32>(0.08, 0.30, 0.70), horizon);
    let night_sky = mix(vec3<f32>(0.006, 0.010, 0.024), vec3<f32>(0.001, 0.003, 0.012), horizon);
    var sky = mix(night_sky, day_sky, day);

    let sun_glow = max(dot(view_dir, sun_dir), 0.0);
    sky += vec3<f32>(1.0, 0.58, 0.24) * pow8(sun_glow) * day * 0.20;
    sky += vec3<f32>(0.18, 0.23, 0.36) * moon_intensity * night;

    return sky;
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {
    let to_camera = uniforms.camera_pos - in.world_pos;
    let dist = max(length(to_camera), 0.001);
    let view_dir = to_camera / dist;
    let normal = normalize(in.wave_normal);
    let sun_dir = normalize(uniforms.sun_position);
    let day = smoothstep(-0.08, 0.24, sun_dir.y);

    let uv = clamp(in.clip_position.xy / uniforms.screen_size, vec2<f32>(0.0), vec2<f32>(1.0));
    let cos_theta = clamp(dot(view_dir, normal), 0.0, 1.0);
    let fresnel = schlick_fresnel(cos_theta);

    let dist_fade = clamp(1.0 - dist / 120.0, 0.0, 1.0);
    let distortion = vec2<f32>(normal.x, -normal.z) * (0.004 + 0.006 * fresnel) * dist_fade;
    let scene = textureSampleLevel(scene_color, scene_sampler, clamp(uv + distortion, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;

    let depth_tint = clamp(dist / 150.0, 0.0, 1.0);
    var water_color = mix(WATER_COLOR_SHALLOW, WATER_COLOR_DEEP, depth_tint);
    water_color *= mix(0.55, 1.0, day);
    water_color = mix(scene * water_color, water_color, 0.58);

    let refl_dir = normalize(reflect(-view_dir, normal));
    let sky = sky_reflection_color(refl_dir, sun_dir, uniforms.moon_intensity);
    let reflection_mix = clamp(0.055 + fresnel * 0.55, 0.0, 0.58);
    water_color = mix(water_color, sky, reflection_mix);

    if sun_dir.y > 0.0 {
        let glint_dir = max(dot(reflect(-sun_dir, normal), view_dir), 0.0);
        let glint = pow8(pow8(glint_dir)) * day * (0.055 + fresnel * 0.10);
        water_color += vec3<f32>(1.0, 0.92, 0.72) * glint;
    }

    let fog_t = clamp((dist - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0.0, 1.0);
    let fog_color = mix(vec3<f32>(0.004, 0.010, 0.024), sky, 0.65 * day + 0.10);
    water_color = mix(water_color, fog_color, fog_t * fog_t);

    let alpha = clamp(WATER_OPACITY + fresnel * 0.26 + (1.0 - dist_fade) * 0.10, 0.18, 0.72);
    return vec4<f32>(water_color, alpha);
}
