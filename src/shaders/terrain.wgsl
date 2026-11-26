struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_dir:        vec3<f32>,
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
    fog_far:        f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

// ── Voxel sun shadows ─────────────────────────────────────────────────────── //
// One bit per block for a 16×16-chunk toroidal window around the camera
// (see src/render/shadow_grid.rs). A pixel is in shadow when a ray from it
// towards the sun hits a filled cell. Sun rays always climb, so the ray
// walks block *columns* (a 2-D DDA): inside a column only y changes and the
// whole run of cells is one masked word test. A max-height pyramid (column,
// 4×4 tile, chunk) lets the ray cross any tile it is already above in one
// step.
struct ShadowGrid {
    /// x: enabled, y: trace distance, z: highest occluder (exclusive).
    params: vec4<f32>,
    /// Per slot: chunk x, chunk z, max y (exclusive), unused.
    slots: array<vec4<i32>, 256>,
};
/// Per slot: occupancy bits (8 words per column, bit = y), then 256 column
/// heights, then 16 tile heights.
@group(0) @binding(3) var<storage, read> shadow_bits: array<u32>;
@group(0) @binding(4) var<storage, read> shadow_grid: ShadowGrid;

const SHADOW_SLOT_WORDS: u32 = 2320u;
const SHADOW_BIT_WORDS: u32 = 2048u;
const SHADOW_COLUMN_WORDS: u32 = 8u;
const SHADOW_COLUMNS: u32 = 256u;
const SHADOW_MAX_ITERATIONS: i32 = 128;
const SHADOW_STEP_EPSILON: f32 = 1e-3;
const SDF_DIM: i32 = 64;
const SDF_OFFSET: u32 = 593920u;
const SDF_HEADER_WORDS: u32 = 4u;
const SDF_DISTANCE_OFFSET: u32 = SDF_OFFSET + SDF_HEADER_WORDS;

/// Ray parameter where the ray leaves the horizontal `size`-aligned square
/// containing `p`.
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

/// `true` if any of cells `y0..=y1` of the column starting at word `base` is
/// filled.
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

/// 1 when `pos` (on a face with normal `normal`) sees the sun, 0 in shadow.
/// `sun` must point upwards.
fn trace_sun(pos: vec3<f32>, normal: vec3<f32>, sun: vec3<f32>, max_t: f32) -> f32 {
    // Start in the open cell in front of the face.
    let origin = pos + normal * 0.01;
    let positive = sun > vec3<f32>(0.0);
    let moving = abs(sun) > vec3<f32>(1e-5);
    let inv_dir = 1.0 / select(vec3<f32>(1.0), sun, moving);
    let top = shadow_grid.params.z;
    let sdf_valid = shadow_bits[SDF_OFFSET + 2u] != 0u;
    let sdf_origin = vec2<i32>(bitcast<i32>(shadow_bits[SDF_OFFSET]), bitcast<i32>(shadow_bits[SDF_OFFSET + 1u]));

    var t = 0.0;
    var slot_index = -1;
    var slot = vec4<i32>(0);
    var slot_valid = false;
    for (var i = 0; i < SHADOW_MAX_ITERATIONS; i++) {
        let p = origin + sun * t;
        // Above every occluder in the grid: nothing left to hit.
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
        // Left the loaded window: nothing known to block the sun.
        if !slot_valid {
            return 1.0;
        }
        let sdf_distance = shadow_sdf_distance(p, sdf_valid, sdf_origin);
        if sdf_distance > 0u {
            let brick = vec3<i32>(floor(p / vec3<f32>(4.0)));
            let radius = i32(sdf_distance) - 1;
            t = max(t, shadow_sdf_box_exit(origin, inv_dir, positive, moving, brick, radius))
                + SHADOW_STEP_EPSILON;
            continue;
        }
        // Above the whole chunk: cross it in one step.
        if p.y >= f32(slot.z) {
            t = shadow_box_exit(p, origin, inv_dir, positive, moving, 16.0) + SHADOW_STEP_EPSILON;
            continue;
        }
        let base = u32(index) * SHADOW_SLOT_WORDS;
        let local = column & vec2<i32>(15);
        let tile = u32((local.y >> 2u) * 4 + (local.x >> 2u));
        // Above this 4×4 tile.
        if p.y >= f32(shadow_bits[base + SHADOW_BIT_WORDS + SHADOW_COLUMNS + tile]) {
            t = shadow_box_exit(p, origin, inv_dir, positive, moving, 4.0) + SHADOW_STEP_EPSILON;
            continue;
        }
        let column_id = u32(local.y * 16 + local.x);
        // Low 16 bits: height (the rest holds materials for water reflections).
        let height = shadow_bits[base + SHADOW_BIT_WORDS + column_id] & 0xffffu;
        let t_out = shadow_box_exit(p, origin, inv_dir, positive, moving, 1.0);
        if p.y < f32(height) {
            // The vertical run of cells crossed inside this column.
            let y0 = u32(floor(max(p.y, 0.0)));
            let y1 = max(min(u32(floor(origin.y + sun.y * (t_out - 1e-4))), height - 1u), y0);
            if shadow_run_hit(base + column_id * SHADOW_COLUMN_WORDS, y0, y1) {
                return 0.0;
            }
        }
        t = t_out + SHADOW_STEP_EPSILON;
    }
    return 1.0;
}

fn fast_global_illumination(
    normal:          vec3<f32>,
    sun_dir:         vec3<f32>,
    day_factor:      f32,
    twilight_factor: f32,
    sky_exposure:    f32,
) -> vec3<f32> {
    let face_sky_visibility = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    // `sky_exposure` is this surface's own world-space sky light, so caves,
    // overhangs and forest floors are dark no matter where the camera is.
    let area_sky_visibility = sky_exposure;
    let cave_ambient = mix(0.06, 1.0, area_sky_visibility);
    let ground_visibility = clamp(0.5 - normal.y * 0.5, 0.0, 1.0);
    let side_visibility   = 1.0 - abs(normal.y);

    let sky_day      = vec3<f32>(0.50, 0.62, 0.78);
    let sky_twilight = vec3<f32>(0.96, 0.48, 0.24);
    let sky_night    = vec3<f32>(0.030, 0.030, 0.034);
    let sky_color = mix(
        mix(sky_night, sky_day, day_factor),
        sky_twilight,
        twilight_factor * 0.45,
    );

    let ground_night = vec3<f32>(0.020, 0.018, 0.020);
    let ground_day   = vec3<f32>(0.26, 0.24, 0.19);
    let grass_bleed  = vec3<f32>(0.08, 0.13, 0.06) * day_factor;
    let ground_color = mix(ground_night, ground_day, day_factor) + grass_bleed;

    let sky_energy    = mix(0.035, 0.27, day_factor) + twilight_factor * 0.08;
    let ground_energy = mix(0.010, 0.070, day_factor) + twilight_factor * 0.020;

    let sky_light    = sky_color * sky_energy * (0.35 + 0.65 * face_sky_visibility) * cave_ambient;
    let ground_light = ground_color
        * ground_energy
        * (ground_visibility + side_visibility * 0.35)
        * mix(0.35, 1.0, area_sky_visibility);

    let bounce_dir = normalize(vec3<f32>(-sun_dir.x, 0.32, -sun_dir.z));
    let wrap_bounce = pow(clamp(dot(normal, bounce_dir) * 0.5 + 0.5, 0.0, 1.0), 2.0);
    let sun_bounce = vec3<f32>(1.0, 0.78, 0.48)
        * wrap_bounce
        * side_visibility
        * day_factor
        * area_sky_visibility
        * 0.045;

    let ambient = sky_light + ground_light + sun_bounce;
    let ambient_luma = dot(ambient, vec3<f32>(0.299, 0.587, 0.114));
    let night_neutralize = (1.0 - day_factor) * (1.0 - twilight_factor) * 0.72;
    return mix(ambient, vec3<f32>(ambient_luma), vec3<f32>(night_neutralize));
}

struct PackedQuad { origin_and_face: u32, size_material_ao: u32, light: u32, }
struct SubchunkMeta {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    terrain_draw_data: vec4<u32>,
    water_draw_data: vec4<u32>,
    terrain_far_draw_data: vec4<u32>,
    terrain_lod_draw_data: vec4<u32>,
    terrain_groups_a: vec4<u32>,
    terrain_groups_b: vec4<u32>,
    terrain_far_groups_a: vec4<u32>,
    terrain_far_groups_b: vec4<u32>,
    terrain_lod_groups_a: vec4<u32>,
    terrain_lod_groups_b: vec4<u32>,
}
@group(1) @binding(0) var<storage, read> quads: array<PackedQuad>;
@group(1) @binding(1) var<storage, read> subchunks: array<SubchunkMeta>;

// ── Climate tint ──────────────────────────────────────────────────────────── //
// Grass target colours (sRGB) indexed by `tint - 1`; rows frozen → hot,
// columns dry → wet. Mirrors `GRASS_TINTS` in src/core/biome.rs.
const GRASS_TINTS = array<vec3<f32>, 15>(
    vec3<f32>(0.58, 0.67, 0.60), vec3<f32>(0.52, 0.66, 0.60), vec3<f32>(0.47, 0.64, 0.60),
    vec3<f32>(0.50, 0.64, 0.40), vec3<f32>(0.40, 0.62, 0.38), vec3<f32>(0.33, 0.58, 0.40),
    vec3<f32>(0.62, 0.76, 0.32), vec3<f32>(0.47, 0.75, 0.29), vec3<f32>(0.32, 0.66, 0.27),
    vec3<f32>(0.72, 0.75, 0.30), vec3<f32>(0.54, 0.78, 0.26), vec3<f32>(0.28, 0.72, 0.22),
    vec3<f32>(0.82, 0.72, 0.36), vec3<f32>(0.66, 0.75, 0.28), vec3<f32>(0.22, 0.74, 0.18),
);
const LEAVES_TINT_SCALE = vec3<f32>(0.78, 0.86, 0.80);
// Linear luminance of the untinted atlas layers (grass top, grass side
// fringe, leaves), so the recoloured texel keeps the original brightness.
const GRASS_TOP_LUMA:  f32 = 0.2623;
const GRASS_SIDE_LUMA: f32 = 0.2303;
const LEAVES_LUMA:     f32 = 0.1359;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── Aerial perspective ────────────────────────────────────────────────────── //
// Distant terrain fades towards the horizon sky colour. A gentle exponential
// term gives depth at mid range; a steeper ramp near `fog_far` hides the edge
// of the loaded world, so chunks no longer pop in against a blank sky.
fn horizon_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let day   = smoothstep(-0.02, 0.22, sun_dir.y);
    let dusk  = 1.0 - smoothstep(0.02, 0.34, abs(sun_dir.y));
    let night = clamp(-sun_dir.y * 4.0 - 0.05, 0.0, 1.0);
    var color = vec3<f32>(0.70, 0.84, 0.98) * day
        + vec3<f32>(0.010, 0.014, 0.034) * night
        + vec3<f32>(0.96, 0.52, 0.30) * dusk * 0.55;
    // Forward scattering: haze glows warmer when looking towards the sun.
    let towards_sun = max(dot(normalize(vec3<f32>(view_dir.x, 0.0, view_dir.z)),
                              normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z) + vec3<f32>(1e-4))), 0.0);
    color += vec3<f32>(1.0, 0.72, 0.42) * pow(towards_sun, 6.0) * (0.10 * day + 0.35 * dusk);
    return color;
}

/// Haze colour and amount for a point, evaluated per vertex.
fn aerial_fog(world_pos: vec3<f32>) -> vec4<f32> {
    if uniforms.is_underwater > 0.5 {
        return vec4<f32>(0.0);
    }
    let to_frag = world_pos - uniforms.camera_pos;
    let dist = length(to_frag.xz);
    let fog_far = max(uniforms.fog_far, 64.0);
    let aerial = (1.0 - exp(-dist / (fog_far * 1.6))) * 0.55;
    let edge = smoothstep(fog_far * 0.55, fog_far * 0.96, dist);
    let amount = max(aerial, edge) * mix(0.25, 1.0, clamp(uniforms.sky_visibility, 0.0, 1.0));
    let haze = horizon_color(to_frag / max(length(to_frag), 1e-3), uniforms.sun_dir);
    return vec4<f32>(haze, amount);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv:        vec2<f32>,
    @location(2) @interpolate(flat) tex_index: u32,
    /// Climate tint palette index (0 = untinted).
    @location(3) @interpolate(flat) tint: u32,
    /// Linear climate colour and the base luminance it replaces.
    @location(4) @interpolate(flat) tint_color: vec4<f32>,
    /// Complete lighting (sky light, AO, sun, local tint), per vertex.
    @location(5) light: vec3<f32>,
    /// Aerial perspective: haze colour (rgb) and amount (a), per vertex.
    @location(6) fog: vec4<f32>,
    /// Direct sunlight reaching an unshadowed surface (rgb) and the cheap
    /// sky-light estimate of its visibility (a), used beyond shadow range.
    @location(7) sun: vec4<f32>,
    /// Face direction (index into NORMALS).
    @location(8) @interpolate(flat) normal_index: u32,
};

/// Indirect light and unshadowed direct sunlight of one vertex.
struct VertexLight {
    ambient: vec3<f32>,
    sun: vec4<f32>,
};

struct DepthVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

/// Lighting for a vertex. Every input is constant per face or varies
/// smoothly across it, so evaluating once per vertex (instead of per
/// pixel) is visually identical and removes the costly `pow`/`exp`/
/// `normalize` work from the fragment shader.
fn vertex_lighting(normal: vec3<f32>, color: vec3<f32>, ambient_occlusion: f32, sky_level: f32) -> VertexLight {
    let sun_dir = uniforms.sun_dir;
    let day_factor = clamp(sun_dir.y, 0.0, 1.0);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);

    // Light levels fall off exponentially, like perceived brightness: one
    // block into a cave is noticeably dimmer, ten blocks in is dark.
    let sky_exposure = pow(0.80, (1.0 - clamp(sky_level, 0.0, 1.0)) * 15.0);
    let indirect_light = fast_global_illumination(normal, sun_dir, day_factor, twilight_factor, sky_exposure);

    let sun_color = mix(vec3<f32>(1.0, 0.78, 0.52), vec3<f32>(1.0, 0.96, 0.86), day_factor);
    // Direct sunlight only reaches surfaces that are (nearly) open to the sky.
    let sun_exposure = smoothstep(0.35, 0.95, sky_exposure);
    // Direct sun is shadowed per pixel; `sun_exposure` is only the fallback
    // estimate beyond the traced range.
    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.62 * day_factor;
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.045 * day_factor * sun_exposure;

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let face_contrast = mix(0.82, 1.0, face_shade);
    let ambient_light =
        (indirect_light * ambient_occlusion
            + vec3<f32>(0.58, 0.68, 0.82) * fill_diff * mix(0.70, 1.0, ambient_occlusion))
        * face_contrast;
    let sun_light = sun_color * sun_diff * face_contrast;
    let total_light = ambient_light + sun_light * sun_exposure;
    let light_luma = dot(total_light, vec3<f32>(0.299, 0.587, 0.114));
    let tint_luma = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    let dark_tint_neutralize = smoothstep(0.16, 0.035, light_luma) * 0.65;
    let local_tint = mix(color, vec3<f32>(tint_luma), vec3<f32>(dark_tint_neutralize));
    var grade = local_tint;

    let sunset_factor = 1.0 - abs(sun_dir.y);
    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        grade *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }
    return VertexLight(ambient_light * grade, vec4<f32>(sun_light * grade, sun_exposure));
}

/// Linear climate target colour (rgb) and base luminance (a) for a layer.
fn climate_tint_color(tex_index: u32, tint: u32) -> vec4<f32> {
    if tint == 0u || tint > 15u {
        return vec4<f32>(0.0);
    }
    if tex_index == 8u {
        return vec4<f32>(srgb_to_linear(GRASS_TINTS[tint - 1u] * LEAVES_TINT_SCALE), LEAVES_LUMA);
    }
    let base = select(GRASS_TOP_LUMA, GRASS_SIDE_LUMA, tex_index == 1u);
    return vec4<f32>(srgb_to_linear(GRASS_TINTS[tint - 1u]), base);
}

// Retained for dynamic/player models, which still use the shared Vertex buffer.
struct LegacyVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed: u32,
};

const NORMALS = array<vec3<f32>, 6>(
    vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0),
);

@vertex
fn vs_legacy(model: LegacyVertexInput) -> VertexOutput {
    let n_idx = model.packed & 0x7u;
    let t_idx = (model.packed >> 3u) & 0xffu;
    let uv_idx = (model.packed >> 11u) & 0x3u;
    let ao_raw = (model.packed >> 21u) & 0x3u;
    let width = f32(((model.packed >> 13u) & 0xfu) + 1u);
    let height = f32(((model.packed >> 17u) & 0xfu) + 1u);
    let color = vec3<f32>(
        f32((model.packed >> 23u) & 0x7u) / 7.0,
        f32((model.packed >> 26u) & 0x7u) / 7.0,
        f32((model.packed >> 29u) & 0x7u) / 7.0,
    );
    let uvs = array<vec2<f32>, 4>(vec2(0., 0.), vec2(0., 1.), vec2(1., 1.), vec2(1., 0.));
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4(model.position, 1.0);
    out.world_pos = model.position;
    out.uv = uvs[uv_idx] * vec2(width, height);
    out.tex_index = t_idx;
    out.tint = 0u;
    out.tint_color = vec4<f32>(0.0);
    // Dynamic models (players, dropped items) sit next to the camera, so
    // the camera-area estimate is a fair stand-in for their sky light.
    let lighting = vertex_lighting(
        NORMALS[n_idx % 6u],
        color,
        mix(0.52, 1.0, f32(ao_raw) / 3.0),
        clamp(uniforms.sky_visibility, 0.0, 1.0),
    );
    out.light = lighting.ambient;
    out.sun = lighting.sun;
    out.normal_index = n_idx % 6u;
    out.fog = aerial_fog(model.position);
    return out;
}

fn emissive_glow(t_idx: u32) -> vec3<f32> {
    if t_idx == 42u {
        return vec3<f32>(1.0, 0.72, 0.42) * 0.8;
    }
    if t_idx == 43u {
        return vec3<f32>(1.0, 0.68, 0.28) * 0.8;
    }
    if t_idx == 44u {
        return vec3<f32>(1.0, 0.52, 0.28) * 0.75;
    }
    if t_idx == 45u {
        return vec3<f32>(0.62, 0.4, 1.0) * 0.75;
    }
    if t_idx == 37u {
        return vec3<f32>(1.0, 0.3, 0.08) * 0.25;
    }
    return vec3<f32>(0.0);
}

fn pulled_vertex(vertex_id: u32, draw_instance_id: u32) -> VertexOutput {
    let subchunk_id = draw_instance_id;
    // Indexed: 4 unique vertices per quad, triangulated by the shared index
    // pattern (0,1,2)(0,2,3). Rotating the corners by one turns that into
    // the other diagonal (1,2,3)(1,3,0), so AO-driven diagonal flips need no
    // second index pattern.
    let quad = quads[vertex_id / 4u];
    let diagonal = (quad.origin_and_face >> 30u) & 0x1u;
    let corner = (vertex_id + diagonal) % 4u;
    let n_idx = (quad.origin_and_face >> 18u) & 0x7u;
    let t_idx = (quad.size_material_ao >> 12u) & 0xffu;
    let width = f32(quad.size_material_ao & 0x3fu) * 0.5;
    let height = f32((quad.size_material_ao >> 6u) & 0x3fu) * 0.5;
    let ao_raw = (quad.size_material_ao >> (20u + corner * 2u)) & 0x3u;
    let color_bits = (quad.origin_and_face >> 21u) & 0x1ffu;
    let color = vec3<f32>(
        f32(color_bits & 0x7u),
        f32((color_bits >> 3u) & 0x7u),
        f32((color_bits >> 6u) & 0x7u),
    ) / 7.0;
    let local_origin = vec3<f32>(
        f32(quad.origin_and_face & 0x3fu), f32((quad.origin_and_face >> 6u) & 0x3fu), f32((quad.origin_and_face >> 12u) & 0x3fu),
    ) * 0.5;

    let edge_u = array<vec3<f32>, 6>(vec3(0.,0.,1.), vec3(0.,0.,-1.), vec3(0.,0.,-1.), vec3(0.,0.,1.), vec3(-1.,0.,0.), vec3(1.,0.,0.))[n_idx];
    let edge_v = array<vec3<f32>, 6>(vec3(0.,1.,0.), vec3(0.,1.,0.), vec3(1.,0.,0.), vec3(1.,0.,0.), vec3(0.,1.,0.), vec3(0.,1.,0.))[n_idx];
    let position_corners = array<vec2<f32>, 4>(vec2(0.,0.), vec2(1.,0.), vec2(1.,1.), vec2(0.,1.));
    let texture_corners = array<vec2<f32>, 4>(vec2(0.,1.), vec2(1.,1.), vec2(1.,0.), vec2(0.,0.));
    let local_position = local_origin + edge_u * (position_corners[corner].x * width) + edge_v * (position_corners[corner].y * height);
    let world_position = subchunks[subchunk_id].aabb_min.xyz + local_position;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos     = world_position;
    out.uv = texture_corners[corner] * vec2(width, height);
    out.tex_index = t_idx;
    out.tint = quad.size_material_ao >> 28u;
    out.tint_color = climate_tint_color(t_idx, out.tint);
    let normal = NORMALS[n_idx % 6u];
    let ambient_occlusion = mix(0.52, 1.0, f32(ao_raw) / 3.0);
    let sky_level = f32((quad.light >> (corner * 4u)) & 0xfu) / 15.0;
    let block_light_level = f32((quad.light >> (16u + corner * 4u)) & 0xfu) / 15.0;
    let lighting = vertex_lighting(normal, color, ambient_occlusion, sky_level);
    let warm_block_light = vec3<f32>(1.0, 0.72, 0.42)
        * pow(block_light_level, 1.35) * ambient_occlusion * 1.6;
    out.light = lighting.ambient + warm_block_light + emissive_glow(t_idx);
    out.sun = lighting.sun;
    out.normal_index = n_idx % 6u;
    out.fog = aerial_fog(world_position);
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> VertexOutput {
    return pulled_vertex(vertex_id, subchunk_id);
}

/// Recolours grass and leaves texels to the column's climate colour while
/// keeping the texture's per-texel brightness detail.
fn apply_climate_tint(color: vec3<f32>, in: VertexOutput) -> vec3<f32> {
    if in.tint == 0u || in.tint_color.a == 0.0 {
        return color;
    }
    if in.tex_index == 1u && color.g - max(color.r, color.b) < 0.04 {
        // Only the green fringe of the grass side; the dirt stays untouched.
        return color;
    }
    if in.tex_index != 0u && in.tex_index != 1u && in.tex_index != 8u {
        return color;
    }
    // An average texel maps exactly onto the target colour; brighter and
    // darker texels keep their relative contrast.
    return in.tint_color.rgb * (luminance(color) / in.tint_color.a);
}

/// Sun shadow ray from a point on the pixel's face.
fn shadow_ray(in: VertexOutput, pos: vec3<f32>) -> f32 {
    return trace_sun(pos, NORMALS[in.normal_index], uniforms.sun_dir, 256.0);
}

/// Everything after the texture fetch: the sun shadow ray (only for
/// surfaces facing the sun), the tint, the lighting multiply and fog.
fn shade(in: VertexOutput, texel: vec3<f32>) -> vec4<f32> {
    // Screen-space footprint of the pixel on its face, taken in uniform
    // control flow (needed for the edge anti-aliasing below).
    let footprint_x = dpdx(in.world_pos);
    let footprint_y = dpdy(in.world_pos);

    // Surfaces facing away from the sun (or lit so faintly that the
    // difference is invisible) and fully fogged pixels skip the ray.
    let wants_sun = max(in.sun.r, max(in.sun.g, in.sun.b)) > 0.01 && in.fog.a < 0.98;
    let params = shadow_grid.params;
    let dist = length(in.world_pos - uniforms.camera_pos);
    let traced_here = wants_sun && params.x > 0.5 && uniforms.sun_dir.y > 0.0 && dist < params.y;

    var traced = 1.0;
    if traced_here {
        traced = shadow_ray(in, in.world_pos);
    }
    // Shadow edges are exact, so one ray per pixel leaves hard stair-steps
    // that MSAA (geometry edges only) cannot smooth. Where the shadow
    // changes inside this 2×2 pixel quad, supersample the pixel with four
    // more rays on a rotated grid; everywhere else one ray is enough.
    let edge = fwidth(traced);
    if traced_here && edge > 0.0 {
        let o1 = footprint_x * 0.125 + footprint_y * 0.375;
        let o2 = footprint_x * -0.375 + footprint_y * 0.125;
        traced = (traced
            + shadow_ray(in, in.world_pos + o1)
            + shadow_ray(in, in.world_pos - o1)
            + shadow_ray(in, in.world_pos + o2)
            + shadow_ray(in, in.world_pos - o2)) * 0.2;
    }

    var light = in.light;
    if wants_sun {
        // Exact voxel shadows near the camera, fading into the sky-light
        // estimate at the edge of the traced range.
        var visibility = in.sun.a;
        if traced_here {
            visibility = mix(traced, in.sun.a, smoothstep(params.y * 0.8, params.y, dist));
        }
        light += in.sun.rgb * visibility;
    }
    // Self-lit materials stay visible in dark caves: magma glows, amethyst
    // gives off a faint shimmer.
    if in.tex_index == 37u {
        light = max(light, vec3<f32>(1.25, 1.1, 1.0));
    } else if in.tex_index == 39u {
        light = max(light, vec3<f32>(0.42, 0.36, 0.5));
    }
    var color = apply_climate_tint(texel, in) * light;
    color = mix(color, in.fog.rgb, in.fog.a);

    if uniforms.is_underwater > 0.5 {
        let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);
        color *= vec3<f32>(0.4, 0.7, 1.0);
        let caustic = sin(in.world_pos.x * 0.5 + uniforms.time * 2.0)
                    * sin(in.world_pos.z * 0.5 + uniforms.time * 1.5) * 0.1 + 0.9;
        color *= caustic;
        color = mix(color, vec3<f32>(0.05, 0.15, 0.3), clamp(dist / 24.0, 0.0, 1.0) * 0.5);
    }
    return vec4<f32>(color, 1.0);
}

/// Fully opaque blocks: no `discard`, so early depth testing stays on.
@fragment
fn fs_opaque(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index));
    return shade(in, tex.rgb);
}

/// Alpha-tested blocks (leaves, dead bushes) and dynamic models.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index));
    if tex.a < 0.5 { discard; }
    return shade(in, tex.rgb);
}
