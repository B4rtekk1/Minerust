const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.030, 0.360, 0.410);
const WATER_COLOR_DEEP:    vec3<f32> = vec3<f32>(0.004, 0.045, 0.115);
const FRESNEL_R0:          f32 = 0.020;
const WATER_LEVEL_OFFSET:  f32 = 0.15;

const FOG_NEAR: f32 = 150.0;
const FOG_FAR:  f32 = 600.0;

struct Uniforms {
    view_proj:       mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    camera_pos:      vec3<f32>,
    time:            f32,
    sun_position:    vec3<f32>,
    is_underwater:   f32,
    screen_size:     vec2<f32>,
    water_level:     f32,
    reflection_mode: f32,
    moon_position:   vec3<f32>,
    _pad1_moon:      f32,
    moon_intensity:  f32,
    wind_dir_x:      f32,
    wind_dir_z:      f32,
    wind_speed:      f32,
    rain_factor:     f32,
    sky_visibility:  f32,
    menu_blur:       f32,
    _pad_uniforms:   f32,
    prev_view_proj:  mat4x4<f32>,
    prev_time:       f32,
    frame_index:     u32,
    sssr_history_valid: u32,
    _pad_sssr:       u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(8) var scene_color: texture_2d<f32>;
@group(0) @binding(9) var scene_depth: texture_2d<f32>;
@group(0) @binding(10) var scene_sampler: sampler;
@group(0) @binding(13) var sssr_reflection: texture_2d<f32>;

struct PackedQuad { origin_and_face: u32, size_material_ao: u32, color_flags: u32, _reserved: u32, }
struct SubchunkMeta { world_origin: vec4<i32>, draw_data: vec4<u32>, }
@group(1) @binding(0) var<storage, read> quads: array<PackedQuad>;
@group(1) @binding(1) var<storage, read> subchunks: array<SubchunkMeta>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:   vec3<f32>,
    @location(1) wave_normal: vec3<f32>,
    @location(2) previous_clip: vec4<f32>,
};

struct ScreenProjection {
    uv: vec2<f32>,
    depth: f32,
    valid: f32,
};

struct ReflectionHit {
    color: vec3<f32>,
    confidence: f32,
};

fn face_normal(index: u32) -> vec3<f32> {
    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)
    );
    return normals[index % 6u];
}

fn wind_basis() -> mat2x2<f32> {
    let wind = normalize(vec2<f32>(uniforms.wind_dir_x, uniforms.wind_dir_z) + vec2<f32>(0.001));
    return mat2x2<f32>(wind, vec2<f32>(-wind.y, wind.x));
}

// Four overlapping Gerstner-like wave bands keep the surface organic without
// requiring an external normal map. The same low-frequency bands are used for
// vertex displacement and for the base normal, so highlights follow the waves.
fn wave_height(world_xz: vec2<f32>) -> f32 {
    let basis = wind_basis();
    let wind = basis[0];
    let cross_wind = basis[1];
    let speed = max(uniforms.wind_speed, 0.15);
    let t = uniforms.time * speed;

    let a = sin(dot(world_xz, wind) * 0.42 - t * 0.72) * 0.030;
    let b = sin(dot(world_xz, cross_wind) * 0.83 + t * 0.51 + 1.7) * 0.018;
    let c = sin(dot(world_xz, normalize(wind + cross_wind * 0.55)) * 1.45 - t * 0.34) * 0.009;
    return a + b + c;
}

fn geometric_wave_normal(world_xz: vec2<f32>) -> vec3<f32> {
    let basis = wind_basis();
    let wind = basis[0];
    let cross_wind = basis[1];
    let diagonal = normalize(wind + cross_wind * 0.55);
    let speed = max(uniforms.wind_speed, 0.15);
    let t = uniforms.time * speed;

    var slope = wind * (cos(dot(world_xz, wind) * 0.42 - t * 0.72) * 0.030 * 0.42);
    slope += cross_wind * (cos(dot(world_xz, cross_wind) * 0.83 + t * 0.51 + 1.7) * 0.018 * 0.83);
    slope += diagonal * (cos(dot(world_xz, diagonal) * 1.45 - t * 0.34) * 0.009 * 1.45);
    return normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
}

// Adds two short wave bands per fragment. They affect reflections and glints,
// but not geometry, giving close water fine detail without visible tessellation.
fn detailed_wave_normal(world_xz: vec2<f32>, base_normal: vec3<f32>, distance_to_camera: f32) -> vec3<f32> {
    let basis = wind_basis();
    let wind = basis[0];
    let cross_wind = basis[1];
    let speed = max(uniforms.wind_speed, 0.15);
    let t = uniforms.time * speed;

    let detail_fade = 1.0 - smoothstep(45.0, 150.0, distance_to_camera);
    var slope = vec2<f32>(-base_normal.x, -base_normal.z) / max(abs(base_normal.y), 0.15);
    slope += wind * cos(dot(world_xz, wind + cross_wind * 0.31) * 3.15 - t * 1.23) * 0.035 * detail_fade;
    slope += cross_wind * cos(dot(world_xz, cross_wind - wind * 0.27) * 5.70 + t * 0.91) * 0.018 * detail_fade;
    return normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
}

fn quad_position(quad: PackedQuad, corner: u32, subchunk_id: u32) -> vec3<f32> {
    let origin = vec3<f32>(f32(quad.origin_and_face & 0x3fu), f32((quad.origin_and_face >> 6u) & 0x3fu), f32((quad.origin_and_face >> 12u) & 0x3fu)) * 0.5;
    let face = (quad.origin_and_face >> 18u) & 0x7u;
    let width = f32((quad.size_material_ao & 0x1fu) + 1u) * 0.5;
    let height = f32(((quad.size_material_ao >> 5u) & 0x1fu) + 1u) * 0.5;
    let default_corners = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let alternate_corners = array<u32, 6>(0u, 1u, 3u, 1u, 2u, 3u);
    let i = select(default_corners[corner], alternate_corners[corner], ((quad.color_flags >> 9u) & 1u) != 0u);
    var p = origin;
    if face == 0u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(0,0,width),vec3(0,height,width),vec3(0,height,0))[i]; }
    if face == 1u { p += array<vec3<f32>, 4>(vec3(0,0,width),vec3(0,0,0),vec3(0,height,0),vec3(0,height,width))[i]; }
    if face == 2u { p += array<vec3<f32>, 4>(vec3(0,0,width),vec3(0,0,0),vec3(height,0,0),vec3(height,0,width))[i]; }
    if face == 3u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(0,0,width),vec3(height,0,width),vec3(height,0,0))[i]; }
    if face == 4u { p += array<vec3<f32>, 4>(vec3(width,0,0),vec3(0,0,0),vec3(0,height,0),vec3(width,height,0))[i]; }
    if face == 5u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(width,0,0),vec3(width,height,0),vec3(0,height,0))[i]; }
    return vec3<f32>(subchunks[subchunk_id].world_origin.xyz) + p;
}

@vertex
fn vs_water(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> VertexOutput {
    var out: VertexOutput;
    let quad = quads[vertex_id / 6u];
    var pos = quad_position(quad, vertex_id % 6u, subchunk_id);

    let normal_index = (quad.origin_and_face >> 18u) & 0x7u;
    var normal = face_normal(normal_index);

    if normal_index == 3u {
        pos.y -= WATER_LEVEL_OFFSET;
        pos.y += wave_height(pos.xz);
        normal = geometric_wave_normal(pos.xz);
    }

    out.clip_position = uniforms.view_proj * vec4<f32>(pos, 1.0);
    // Wave displacement is evaluated with the former phase too, so temporal
    // SSSR tracks moving crests instead of treating them as disocclusions.
    let saved_time = uniforms.time;
    let phase_delta = (uniforms.prev_time - saved_time) * max(uniforms.wind_speed, 0.15);
    let previous_pos = pos + vec3<f32>(0.0, sin(dot(pos.xz, wind_basis()[0]) * 0.42 - saved_time * max(uniforms.wind_speed, 0.15) * 0.72 + phase_delta) * 0.030 - sin(dot(pos.xz, wind_basis()[0]) * 0.42 - saved_time * max(uniforms.wind_speed, 0.15) * 0.72) * 0.030, 0.0);
    out.previous_clip = uniforms.prev_view_proj * vec4<f32>(previous_pos, 1.0);
    out.world_pos = pos;
    out.wave_normal = normal;
    return out;
}

struct SurfaceOutput {
    @location(0) normal: vec4<f32>, @location(1) material: vec4<f32>,
    @location(2) depth: vec4<f32>, @location(3) motion: vec4<f32>,
};

// A compact forward G-buffer for SSSR only.  It is deliberately not a
// deferred renderer: opaque terrain remains on its existing forward path.
@fragment
fn fs_surface(in: VertexOutput) -> SurfaceOutput {
    var out: SurfaceOutput;
    let v = normalize(uniforms.camera_pos - in.world_pos);
    var n = detailed_wave_normal(in.world_pos.xz, normalize(in.wave_normal), length(uniforms.camera_pos - in.world_pos));
    if dot(n, v) < 0.0 { n = -n; }
    let roughness = clamp(mix(0.06, 0.18, 1.0 - max(dot(v, n), 0.0)) + uniforms.rain_factor * 0.18, 0.02, 0.5);
    let cur = in.clip_position.xy / max(in.clip_position.w, 0.0001);
    let prev = in.previous_clip.xy / max(in.previous_clip.w, 0.0001);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.material = vec4<f32>(roughness, 1.0, 0.0, 1.0);
    out.depth = vec4<f32>(in.clip_position.z / max(in.clip_position.w, 0.0001), 0.0, 0.0, 1.0);
    out.motion = vec4<f32>((cur - prev) * 0.5, 0.0, 1.0);
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

fn project_to_screen(world_pos: vec3<f32>) -> ScreenProjection {
    let clip = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    if clip.w <= 0.0001 {
        return ScreenProjection(vec2<f32>(0.0), 1.0, 0.0);
    }

    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = select(0.0, 1.0,
        uv.x > 0.001 && uv.x < 0.999 && uv.y > 0.001 && uv.y < 0.999 && ndc.z > 0.0 && ndc.z < 1.0);
    return ScreenProjection(uv, ndc.z, inside);
}

fn load_scene_depth(uv: vec2<f32>) -> f32 {
    let dimensions = textureDimensions(scene_depth);
    let max_coord = vec2<i32>(dimensions) - vec2<i32>(1);
    let coord = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), max_coord);
    return textureLoad(scene_depth, coord, 0).r;
}

fn reconstruct_world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = uniforms.inv_view_proj * ndc;
    return world.xyz / max(abs(world.w), 0.0001);
}

fn sample_reflection_color(uv: vec2<f32>, roughness: f32) -> vec3<f32> {
    let texel = 1.0 / max(uniforms.screen_size, vec2<f32>(1.0));
    let radius = 0.65 + roughness * 2.0;
    var color = textureSampleLevel(scene_color, scene_sampler, uv, 0.0).rgb * 0.72;
    color += textureSampleLevel(scene_color, scene_sampler, clamp(uv + vec2<f32>(texel.x, 0.0) * radius, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * 0.07;
    color += textureSampleLevel(scene_color, scene_sampler, clamp(uv - vec2<f32>(texel.x, 0.0) * radius, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * 0.07;
    color += textureSampleLevel(scene_color, scene_sampler, clamp(uv + vec2<f32>(0.0, texel.y) * radius, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * 0.07;
    color += textureSampleLevel(scene_color, scene_sampler, clamp(uv - vec2<f32>(0.0, texel.y) * radius, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb * 0.07;

    // Preserve recognizable reflected silhouettes instead of washing them into
    // the blue water tint. A slight contrast lift makes SSR immediately legible.
    return max((color - vec3<f32>(0.5)) * 1.10 + vec3<f32>(0.5), vec3<f32>(0.0));
}

// Screen-space ray march against the opaque depth buffer. Rays that leave the
// screen or miss geometry return zero confidence and naturally fall back to sky.
fn trace_screen_reflection(origin: vec3<f32>, ray_dir: vec3<f32>, roughness: f32) -> ReflectionHit {
    if uniforms.reflection_mode < 0.5 {
        return ReflectionHit(vec3<f32>(0.0), 0.0);
    }

    var travel = 0.20;
    var previous_travel = travel;
    var hit_uv = vec2<f32>(0.0);
    var hit = false;

    for (var i: i32 = 0; i < 28; i = i + 1) {
        previous_travel = travel;
        travel += 0.22 + f32(i) * 0.090;
        let projected = project_to_screen(origin + ray_dir * travel);
        if projected.valid < 0.5 {
            break;
        }

        let opaque_depth = load_scene_depth(projected.uv);
        let thickness = 0.0015 + travel * 0.00016;
        if opaque_depth < 0.9999 && projected.depth >= opaque_depth - thickness {
            // Four binary refinement steps keep silhouettes stable without the
            // cost of a much denser primary march.
            var low = previous_travel;
            var high = travel;
            for (var refinement: i32 = 0; refinement < 4; refinement = refinement + 1) {
                let middle = (low + high) * 0.5;
                let refined = project_to_screen(origin + ray_dir * middle);
                let refined_depth = load_scene_depth(refined.uv);
                if refined.depth >= refined_depth - thickness {
                    high = middle;
                    hit_uv = refined.uv;
                } else {
                    low = middle;
                }
            }
            hit = true;
            break;
        }
    }

    if !hit {
        return ReflectionHit(vec3<f32>(0.0), 0.0);
    }

    let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.015, 0.12, edge);
    let distance_fade = 1.0 - smoothstep(34.0, 55.0, travel);
    // SSR should be visually decisive when a reliable hit exists. Squaring the
    // inverse retains soft transitions while boosting medium-confidence hits.
    let raw_confidence = edge_fade * distance_fade;
    let confidence = 1.0 - (1.0 - raw_confidence) * (1.0 - raw_confidence);
    return ReflectionHit(sample_reflection_color(hit_uv, roughness), confidence);
}

fn sky_reflection_color(reflection_dir: vec3<f32>, sun_dir: vec3<f32>, moon_dir: vec3<f32>) -> vec3<f32> {
    let day = smoothstep(-0.12, 0.22, sun_dir.y);
    let horizon = pow(clamp(reflection_dir.y * 0.5 + 0.5, 0.0, 1.0), 0.65);

    let day_horizon = vec3<f32>(0.62, 0.78, 0.88);
    let day_zenith = vec3<f32>(0.055, 0.22, 0.58);
    let night_horizon = vec3<f32>(0.018, 0.028, 0.055);
    let night_zenith = vec3<f32>(0.0015, 0.003, 0.012);
    var sky = mix(mix(night_horizon, night_zenith, horizon), mix(day_horizon, day_zenith, horizon), day);

    let sun_alignment = max(dot(reflection_dir, sun_dir), 0.0);
    let sun_core = pow8(pow8(sun_alignment));
    let sun_halo = pow8(sun_alignment) * 0.22;
    sky += vec3<f32>(1.0, 0.76, 0.42) * (sun_core * 4.0 + sun_halo) * day;

    let moon_alignment = max(dot(reflection_dir, moon_dir), 0.0);
    sky += vec3<f32>(0.58, 0.68, 0.92) * pow8(pow8(moon_alignment)) * uniforms.moon_intensity * 1.8;
    return sky;
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {
    let to_camera = uniforms.camera_pos - in.world_pos;
    let distance_to_camera = max(length(to_camera), 0.001);
    let view_dir = to_camera / distance_to_camera;

    var normal = detailed_wave_normal(in.world_pos.xz, normalize(in.wave_normal), distance_to_camera);
    if dot(normal, view_dir) < 0.0 {
        normal = -normal;
    }

    let sun_dir = normalize(uniforms.sun_position);
    let moon_dir = normalize(uniforms.moon_position);
    let day = smoothstep(-0.10, 0.22, sun_dir.y);
    let cos_theta = clamp(dot(view_dir, normal), 0.0, 1.0);
    let fresnel = schlick_fresnel(cos_theta);
    let roughness = mix(0.06, 0.18, 1.0 - cos_theta) + uniforms.rain_factor * 0.18;

    let screen_uv = clamp(in.clip_position.xy / uniforms.screen_size, vec2<f32>(0.0), vec2<f32>(1.0));
    let distance_fade = 1.0 - smoothstep(60.0, 175.0, distance_to_camera);
    let distortion = normal.xz * vec2<f32>(1.0, -1.0) * (0.006 + 0.009 * fresnel) * distance_fade;
    let refract_uv = clamp(screen_uv + distortion, vec2<f32>(0.002), vec2<f32>(0.998));

    let background_depth = load_scene_depth(refract_uv);
    let background_world = reconstruct_world_position(refract_uv, background_depth);
    var water_depth = length(background_world - in.world_pos);
    if background_depth >= 0.9999 {
        water_depth = 32.0;
    }
    water_depth = clamp(water_depth, 0.0, 32.0);

    let refracted_scene = textureSampleLevel(scene_color, scene_sampler, refract_uv, 0.0).rgb;
    let depth_factor = 1.0 - exp(-water_depth * 0.16);
    var absorption_color = mix(WATER_COLOR_SHALLOW, WATER_COLOR_DEEP, smoothstep(0.0, 14.0, water_depth));
    absorption_color *= mix(0.42, 1.0, day);
    var water_color = mix(refracted_scene, absorption_color, clamp(depth_factor * 0.88 + 0.06, 0.0, 0.94));

    let reflection_dir = normalize(reflect(-view_dir, normal));
    let sky_reflection = sky_reflection_color(reflection_dir, sun_dir, moon_dir);
    let reflection_pixel = clamp(vec2<i32>(screen_uv * uniforms.screen_size), vec2<i32>(0), vec2<i32>(uniforms.screen_size) - 1);
    let sssr = textureLoad(sssr_reflection, reflection_pixel, 0);
    let ssr_weight = clamp(sssr.a, 0.0, 1.0);
    let reflected_scene = mix(sky_reflection * 0.82, sssr.rgb, ssr_weight);

    let reflection_strength = clamp(0.24 + fresnel * 0.74, 0.0, 0.96);
    if uniforms.reflection_mode >= 0.5 {
        let visible_reflection = max(reflection_strength, ssr_weight * 0.62);
        water_color = mix(water_color, reflected_scene, visible_reflection);
    }

    // A narrow moving sun path and a softer halo produce the characteristic
    // broken glitter seen on small wind waves.
    if sun_dir.y > 0.0 {
        let specular_alignment = max(dot(reflect(-sun_dir, normal), view_dir), 0.0);
        let sharp_glint = pow8(pow8(specular_alignment));
        let soft_glint = pow8(specular_alignment) * 0.16;
        water_color += vec3<f32>(1.0, 0.88, 0.62) * (sharp_glint * 2.4 + soft_glint) * day;
    }

    let crest = smoothstep(0.022, 0.050, wave_height(in.world_pos.xz));
    water_color += vec3<f32>(0.06, 0.14, 0.16) * crest * distance_fade * 0.22;

    let shore = 1.0 - smoothstep(0.18, 1.65, water_depth);
    let foam_wave = sin(in.world_pos.x * 2.35 + uniforms.time * 1.15)
        + sin(in.world_pos.z * 2.73 - uniforms.time * 0.92)
        + sin((in.world_pos.x + in.world_pos.z) * 1.34 + uniforms.time * 0.54);
    let foam_breakup = smoothstep(0.10, 1.45, foam_wave);
    let top_surface = smoothstep(0.55, 0.94, normal.y);
    let foam_amount = shore * mix(0.38, 0.92, foam_breakup) * top_surface * distance_fade;
    let foam_color = mix(vec3<f32>(0.42, 0.63, 0.66), vec3<f32>(0.86, 0.93, 0.91), day);
    water_color = mix(water_color, foam_color, foam_amount * 0.72);

    if uniforms.is_underwater > 0.5 {
        water_color = mix(water_color, WATER_COLOR_DEEP, 0.34);
    }

    let fog_amount = smoothstep(FOG_NEAR, FOG_FAR, distance_to_camera);
    let fog_color = mix(vec3<f32>(0.003, 0.008, 0.022), sky_reflection, day * 0.52 + 0.08);
    water_color = mix(water_color, fog_color, fog_amount * fog_amount);

    let alpha = clamp(
        0.38 + depth_factor * 0.25 + fresnel * 0.24 + fog_amount * 0.08 + foam_amount * 0.26,
        0.34,
        0.92,
    );
    return vec4<f32>(max(water_color, vec3<f32>(0.0)), alpha);
}
