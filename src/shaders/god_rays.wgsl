// Ray-marched volumetric light shafts ("god rays").
//
// Runs at a quarter of the surface resolution per axis: every texel marches
// the view ray through height fog and, at each sample, traces a ray towards
// the sun (or moon) through the voxel shadow grid. Lit samples in-scatter
// light, occluded ones do not, which produces visible shafts behind trees,
// cliffs and cave openings. The composite pass depth-aware upsamples the result.

struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    _pad_water:     f32,
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

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Full-resolution, single-sampled scene depth.
@group(0) @binding(1)
var scene_depth: texture_2d<f32>;

@group(0) @binding(2)
var<storage, read> shadow_bits: array<u32>;

struct ShadowGrid {
    params: vec4<f32>,
    slots: array<vec4<i32>, 256>,
};

@group(0) @binding(3)
var<storage, read> shadow_grid: ShadowGrid;

@group(0) @binding(4)
var god_rays_out: texture_storage_2d<rgba16float, write>;

const SHADOW_SLOT_WORDS: u32 = 2320u;
const SHADOW_BIT_WORDS: u32 = 2048u;
const SHADOW_COLUMN_WORDS: u32 = 8u;
const SHADOW_COLUMNS: u32 = 256u;
const SHADOW_TRACE_STEPS: i32 = 24;
const SHADOW_TRACE_DISTANCE: f32 = 64.0;
const SDF_DIM: i32 = 64;
const SDF_OFFSET: u32 = 593920u;
const SDF_HEADER_WORDS: u32 = 4u;
const SDF_DISTANCE_OFFSET: u32 = SDF_OFFSET + SDF_HEADER_WORDS;

const MARCH_STEPS: i32 = 16;
const MARCH_DISTANCE: f32 = 96.0;
const SCATTERING: f32 = 0.02;
const HG_G: f32 = 0.55;
const SHAFT_BASELINE: f32 = 0.7;
const SHADOW_DARKENING: f32 = 0.3;
const NEAR_SHAFT_FADE_START: f32 = 0.5;
const NEAR_SHAFT_FADE_END: f32 = 3.0;
const SHADOW_FADE_START: f32 = 2.0;
const SHADOW_FADE_END: f32 = 12.0;
// Final brightness of the shafts.
const INTENSITY: f32 = 2.6;

fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

fn shadow_box_exit(p: vec3<f32>, origin: vec3<f32>, inv_dir: vec3<f32>, positive: vec3<bool>,
                   moving: vec3<bool>, size: f32) -> f32 {
    let bound = floor(p.xz / size) * size + select(vec2<f32>(0.0), vec2<f32>(size), positive.xz);
    let t = select(vec2<f32>(1e30), (bound - origin.xz) * inv_dir.xz, moving.xz);
    return min(t.x, t.y);
}

fn shadow_sdf_distance(p: vec3<f32>, sdf_valid: bool, sdf_origin: vec2<i32>) -> u32 {
    if !sdf_valid {
        return 0u;
    }
    let brick = vec3<i32>(floor(p / vec3<f32>(4.0)));
    let local = brick.xz - sdf_origin;
    if brick.y < 0 || brick.y >= SDF_DIM || any(local < vec2<i32>(0)) || any(local >= vec2<i32>(SDF_DIM)) {
        return 0u;
    }
    let index = u32(local.x) + 64u * (u32(local.y) + 64u * u32(brick.y));
    return shadow_bits[SDF_DISTANCE_OFFSET + index];
}

fn shadow_sdf_box_exit(origin: vec3<f32>, inv_dir: vec3<f32>, positive: vec3<bool>, moving: vec3<bool>,
                       brick: vec3<i32>, radius: i32) -> f32 {
    let radius_f = f32(radius);
    let lower = (vec3<f32>(brick) - vec3<f32>(radius_f)) * 4.0;
    let upper = (vec3<f32>(brick) + vec3<f32>(radius_f + 1.0)) * 4.0;
    let bound = select(lower, upper, positive);
    let hit = (bound - origin) * inv_dir;
    let t = select(vec3<f32>(1e30), hit, moving);
    return min(min(t.x, t.y), t.z);
}

fn shadow_run_hit(base: u32, y0: u32, y1: u32) -> bool {
    var y = y0;
    loop {
        let word = y >> 5u;
        let last = min(y1, word * 32u + 31u);
        let count = (last & 31u) - (y & 31u) + 1u;
        let mask = select((1u << count) - 1u, 0xffffffffu, count == 32u) << (y & 31u);
        if (shadow_bits[base + word] & mask) != 0u {
            return true;
        }
        if last >= y1 {
            return false;
        }
        y = last + 1u;
    }
    return false;
}

// 1.0 when `pos` sees the light, 0.0 when a shadow-casting block is in the way.
// Same hierarchical (chunk → 4×4 tile → column) traversal as terrain.wgsl.
fn light_visible(pos: vec3<f32>, light_dir: vec3<f32>, max_t: f32) -> f32 {
    let origin = pos + light_dir * 0.04;
    let positive = light_dir > vec3<f32>(0.0);
    let moving = abs(light_dir) > vec3<f32>(1e-5);
    let inv_dir = 1.0 / select(vec3<f32>(1.0), light_dir, moving);
    let top = shadow_grid.params.z;
    let sdf_valid = shadow_bits[SDF_OFFSET + 2u] != 0u;
    let sdf_origin = vec2<i32>(bitcast<i32>(shadow_bits[SDF_OFFSET]), bitcast<i32>(shadow_bits[SDF_OFFSET + 1u]));
    var t = 0.0;
    var slot_index = -1;
    var slot = vec4<i32>(0);
    var slot_valid = false;

    for (var i = 0; i < SHADOW_TRACE_STEPS; i++) {
        let p = origin + light_dir * t;
        if p.y >= top || t > max_t {
            return 1.0;
        }
        let column = vec2<i32>(floor(p.xz));
        let chunk = column >> vec2<u32>(4u);
        let index = (chunk.x & 15) + (chunk.y & 15) * 16;
        if index != slot_index {
            slot_index = index;
            slot = shadow_grid.slots[index];
            slot_valid = slot.x == chunk.x && slot.y == chunk.y;
        }
        if !slot_valid {
            return 1.0;
        }
        let sdf_distance = shadow_sdf_distance(p, sdf_valid, sdf_origin);
        if sdf_distance > 0u {
            let brick = vec3<i32>(floor(p / vec3<f32>(4.0)));
            let radius = i32(sdf_distance) - 1;
            t = max(t, shadow_sdf_box_exit(origin, inv_dir, positive, moving, brick, radius)) + 1e-3;
            continue;
        }
        if p.y >= f32(slot.z) {
            t = shadow_box_exit(p, origin, inv_dir, positive, moving, 16.0) + 1e-3;
            continue;
        }

        let base = u32(index) * SHADOW_SLOT_WORDS;
        let local = column & vec2<i32>(15);
        let tile = u32((local.y >> 2u) * 4 + (local.x >> 2u));
        if p.y >= f32(shadow_bits[base + SHADOW_BIT_WORDS + SHADOW_COLUMNS + tile]) {
            t = shadow_box_exit(p, origin, inv_dir, positive, moving, 4.0) + 1e-3;
            continue;
        }

        let column_id = u32(local.y * 16 + local.x);
        let height = shadow_bits[base + SHADOW_BIT_WORDS + column_id] & 0xffffu;
        let t_out = shadow_box_exit(p, origin, inv_dir, positive, moving, 1.0);
        if p.y < f32(height) {
            let y0 = u32(floor(max(p.y, 0.0)));
            let y1 = max(min(u32(floor(origin.y + light_dir.y * (t_out - 1e-4))), height - 1u), y0);
            if shadow_run_hit(base + column_id * SHADOW_COLUMN_WORDS, y0, y1) {
                return 0.0;
            }
        }
        t = t_out + 1e-3;
    }
    // Out of steps: long, grazing traces at a low sun would otherwise turn
    // into false shadows across open terrain.
    return 1.0;
}

fn world_from_ndc(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let world_h = uniforms.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return world_h.xyz / select(world_h.w, 1e-5, abs(world_h.w) < 1e-5);
}

fn depth_distance(pixel: vec2<u32>, depth_size: vec2<u32>) -> f32 {
    let depth = textureLoad(scene_depth, vec2<i32>(pixel), 0).r;
    if depth >= 0.9999 {
        return MARCH_DISTANCE;
    }
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(depth_size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return min(length(world_from_ndc(ndc, depth) - uniforms.camera_pos), MARCH_DISTANCE);
}

fn march(texel: vec2<u32>, out_size: vec2<u32>) -> vec4<f32> {
    if uniforms.is_underwater > 0.5 || uniforms.menu_blur >= 0.999
        || (shadow_grid.params.x < 0.5 && shadow_grid.params.w < 0.5) {
        return vec4<f32>(vec3<f32>(0.0), MARCH_DISTANCE);
    }

    let sun_dir = normalize(uniforms.sun_position);
    let use_moon = sun_dir.y <= 0.02;
    let light_dir = select(sun_dir, normalize(uniforms.moon_position), use_moon);
    if light_dir.y <= 0.02 {
        return vec4<f32>(vec3<f32>(0.0), MARCH_DISTANCE);
    }
    // Shafts read best with a low sun; at noon they are kept but subtler.
    let sun_strength = smoothstep(0.02, 0.12, sun_dir.y)
        * mix(1.0, 0.75, smoothstep(0.35, 0.9, sun_dir.y));
    let moon_strength = clamp(uniforms.moon_intensity, 0.0, 1.0) * 0.25;
    let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
    let light_strength = select(sun_strength, moon_strength, use_moon) * (1.0 - 0.65 * rain);
    if light_strength <= 0.001 {
        return vec4<f32>(vec3<f32>(0.0), MARCH_DISTANCE);
    }

    let uv = (vec2<f32>(texel) + vec2<f32>(0.5)) / vec2<f32>(out_size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let depth_size = textureDimensions(scene_depth);
    let depth_pixel = min(vec2<u32>(uv * vec2<f32>(depth_size)), depth_size - vec2<u32>(1u));
    let depth_pixel_next = min(depth_pixel + vec2<u32>(1u), depth_size - vec2<u32>(1u));
    let march_distance = min(
        min(depth_distance(depth_pixel, depth_size), depth_distance(vec2<u32>(depth_pixel_next.x, depth_pixel.y), depth_size)),
        min(depth_distance(vec2<u32>(depth_pixel.x, depth_pixel_next.y), depth_size), depth_distance(depth_pixel_next, depth_size))
    );

    // Direction from a mid-depth point keeps sky texels stable; use the nearest
    // depth in a 2x2 footprint to avoid marching across foliage into the sky.
    let ray_dir = normalize(world_from_ndc(ndc, 0.5) - uniforms.camera_pos);
    if march_distance <= 0.05 {
        return vec4<f32>(vec3<f32>(0.0), march_distance);
    }

    let cos_theta = clamp(dot(ray_dir, light_dir), -1.0, 1.0);
    let g2 = HG_G * HG_G;
    let phase = (1.0 - g2) / (12.56637 * pow(max(1.0 + g2 - 2.0 * HG_G * cos_theta, 1e-3), 1.5))
        + 0.06;
    let sun_color = mix(vec3<f32>(1.0, 0.62, 0.32), vec3<f32>(1.0, 0.93, 0.80),
        smoothstep(0.06, 0.55, sun_dir.y));
    let light_color = select(sun_color, vec3<f32>(0.42, 0.56, 1.0), use_moon);

    let jitter = interleaved_gradient_noise(vec2<f32>(texel));
    let trace_distance = min(shadow_grid.params.y, SHADOW_TRACE_DISTANCE);
    var transmittance = 1.0;
    var shaft_scattering = 0.0;

    for (var i = 0; i < MARCH_STEPS; i++) {
        let fraction0 = f32(i) / f32(MARCH_STEPS);
        let fraction1 = f32(i + 1) / f32(MARCH_STEPS);
        let march_fraction0 = mix(fraction0, fraction0 * fraction0, 0.6);
        let march_fraction1 = mix(fraction1, fraction1 * fraction1, 0.6);
        let distance0 = march_fraction0 * march_distance;
        let distance1 = march_fraction1 * march_distance;
        let step_length = distance1 - distance0;
        let distance = distance0 + jitter * step_length;
        let sample_pos = uniforms.camera_pos + ray_dir * distance;
        let height = sample_pos.y - uniforms.water_level;
        let density = exp(-max(height, 0.0) / 48.0);
        let segment = 1.0 - exp(-density * SCATTERING * step_length);
        // Samples above every occluder are always lit; skip the trace.
        var visible = 1.0;
        if sample_pos.y < shadow_grid.params.z {
            visible = light_visible(sample_pos, light_dir, trace_distance);
        }
        let shadow_contrast = visible - SHAFT_BASELINE;
        let contribution_fade = select(
            smoothstep(SHADOW_FADE_START, SHADOW_FADE_END, distance) * SHADOW_DARKENING,
            smoothstep(NEAR_SHAFT_FADE_START, NEAR_SHAFT_FADE_END, distance),
            shadow_contrast >= 0.0
        );
        shaft_scattering += transmittance * segment * shadow_contrast * contribution_fade;
        transmittance *= 1.0 - segment;
    }

    return vec4<f32>(light_color * (shaft_scattering * phase * light_strength * INTENSITY), march_distance);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_size = textureDimensions(god_rays_out);
    if id.x >= out_size.x || id.y >= out_size.y {
        return;
    }
    let rays = march(id.xy, out_size);
    textureStore(god_rays_out, vec2<i32>(id.xy), rays);
}
