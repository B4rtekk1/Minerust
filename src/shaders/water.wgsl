const WATER_COLOR_SHALLOW: vec3<f32> = vec3<f32>(0.075, 0.410, 0.455);
const WATER_COLOR_DEEP:    vec3<f32> = vec3<f32>(0.010, 0.072, 0.165);
const WATER_OPACITY:       f32 = 0.30;
const FRESNEL_R0:          f32 = 0.020;
// Reflected terrain gets a reflectance floor and a saturation boost so it
// stays readable even when looking steeply down at water.
const SSR_MIN_REFLECTANCE: f32 = 0.42;
const SSR_SATURATION:      f32 = 1.45;
const SSR_BRIGHTNESS:      f32 = 1.18;
// Reflection rays are traced for water within this distance of the camera
// and up to this length; further water reflects only the sky.
const REFLECTION_TRACE_DISTANCE: f32 = 192.0;
const REFLECTION_RAY_LENGTH:     f32 = 128.0;
const WATER_LEVEL_OFFSET:  f32 = 0.15;


struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    water_quality:  f32,
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
@group(0) @binding(8) var scene_color: texture_2d<f32>;
@group(0) @binding(10) var scene_sampler: sampler;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;
/// Full-resolution single-sample scene depth (level 0 is all that is used).
@group(0) @binding(13) var ssr_hiz: texture_2d<f32>;

// ── Voxel reflections ─────────────────────────────────────────────────────── //
// The sun-shadow occupancy grid (src/render/shadow_grid.rs) is shared with the
// terrain shader. A ray reflected off water always climbs, so it is traced
// with the same column walk + max-height pyramid as sun shadows: over open
// water with nothing around, it leaves through the top in a few steps.
struct ShadowGrid {
    params: vec4<f32>,
    slots: array<vec4<i32>, 256>,
};
@group(0) @binding(14) var<storage, read> shadow_bits: array<u32>;
@group(0) @binding(15) var<storage, read> shadow_grid: ShadowGrid;

const GRID_SLOT_WORDS: u32 = 2320u;
const GRID_BIT_WORDS: u32 = 2048u;
const GRID_COLUMN_WORDS: u32 = 8u;
const GRID_COLUMNS: u32 = 256u;
const GRID_MAX_ITERATIONS: i32 = 128;
const GRID_STEP_EPSILON: f32 = 1e-3;
const SDF_DIM: i32 = 64;
const SDF_OFFSET: u32 = 593920u;
const SDF_HEADER_WORDS: u32 = 4u;
const SDF_DISTANCE_OFFSET: u32 = SDF_OFFSET + SDF_HEADER_WORDS;

struct ReflectionHit {
    found: bool,
    /// Ray parameter of the hit (blocks along the ray).
    t: f32,
    /// Normal of the face that was hit.
    normal: vec3<f32>,
    /// Atlas layer showing what was hit.
    layer: u32,
};

/// Ray parameter where the ray leaves the horizontal `size`-aligned square
/// containing `p` (x), and the normal of the face it enters next (yzw).
fn grid_box_exit(p: vec3<f32>, origin: vec3<f32>, inv_dir: vec3<f32>, positive: vec3<bool>,
                 moving: vec3<bool>, size: f32) -> vec4<f32> {
    let bound = floor(p.xz / size) * size + select(vec2<f32>(0.0), vec2<f32>(size), positive.xz);
    let t = select(vec2<f32>(1e30), (bound - origin.xz) * inv_dir.xz, moving.xz);
    if t.x < t.y {
        return vec4<f32>(t.x, select(1.0, -1.0, positive.x), 0.0, 0.0);
    }
    return vec4<f32>(t.y, 0.0, 0.0, select(1.0, -1.0, positive.z));
}

fn grid_sdf_distance(p: vec3<f32>, sdf_valid: bool, sdf_origin: vec2<i32>) -> u32 {
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

fn grid_sdf_box_exit(origin: vec3<f32>, inv_dir: vec3<f32>, positive: vec3<bool>, moving: vec3<bool>,
                     brick: vec3<i32>, radius: i32) -> vec4<f32> {
    let radius_f = f32(radius);
    let lower = (vec3<f32>(brick) - vec3<f32>(radius_f)) * 4.0;
    let upper = (vec3<f32>(brick) + vec3<f32>(radius_f + 1.0)) * 4.0;
    let bound = select(lower, upper, positive);
    let hit = (bound - origin) * inv_dir;
    var exit_t = 1e30;
    var normal = vec3<f32>(0.0);
    if moving.x && hit.x < exit_t {
        exit_t = hit.x;
        normal = vec3<f32>(select(1.0, -1.0, positive.x), 0.0, 0.0);
    }
    if moving.y && hit.y < exit_t {
        exit_t = hit.y;
        normal = vec3<f32>(0.0, select(1.0, -1.0, positive.y), 0.0);
    }
    if moving.z && hit.z < exit_t {
        exit_t = hit.z;
        normal = vec3<f32>(0.0, 0.0, select(1.0, -1.0, positive.z));
    }
    return vec4<f32>(exit_t, normal);
}

/// Lowest filled cell among `y0..=y1` of the column starting at word
/// `base`, or 0xffffffff.
fn grid_first_hit(base: u32, y0: u32, y1: u32) -> u32 {
    var y = y0;
    loop {
        let word = y >> 5u;
        let last = min(y1, word * 32u + 31u);
        let count = (last & 31u) - (y & 31u) + 1u;
        let mask = select((1u << count) - 1u, 0xffffffffu, count == 32u) << (y & 31u);
        let bits = shadow_bits[base + word] & mask;
        if bits != 0u {
            return word * 32u + firstTrailingBit(bits);
        }
        if last >= y1 {
            return 0xffffffffu;
        }
        y = last + 1u;
    }
    return 0xffffffffu;
}

/// Traces a climbing ray through the voxel grid.
fn trace_reflection(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> ReflectionHit {
    var result: ReflectionHit;
    result.found = false;
    // `params.w`: the grid is being streamed for reflections.
    if shadow_grid.params.w < 0.5 || dir.y <= 0.0 {
        return result;
    }
    let positive = dir > vec3<f32>(0.0);
    let moving = abs(dir) > vec3<f32>(1e-5);
    let inv_dir = 1.0 / select(vec3<f32>(1.0), dir, moving);
    let top = shadow_grid.params.z;
    let sdf_valid = shadow_bits[SDF_OFFSET + 2u] != 0u;
    let sdf_origin = vec2<i32>(bitcast<i32>(shadow_bits[SDF_OFFSET]), bitcast<i32>(shadow_bits[SDF_OFFSET + 1u]));

    var t = 0.0;
    var entry_normal = vec3<f32>(0.0, -1.0, 0.0);
    var entered_side = false;
    var slot_index = -1;
    var slot = vec4<i32>(0);
    var slot_valid = false;
    for (var i = 0; i < GRID_MAX_ITERATIONS; i++) {
        let p = origin + dir * t;
        if p.y >= top || t > max_t {
            return result;
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
            return result;
        }
        let sdf_distance = grid_sdf_distance(p, sdf_valid, sdf_origin);
        if sdf_distance > 0u {
            let brick = vec3<i32>(floor(p / vec3<f32>(4.0)));
            let radius = i32(sdf_distance) - 1;
            let sdf_exit = grid_sdf_box_exit(origin, inv_dir, positive, moving, brick, radius);
            t = max(t, sdf_exit.x) + GRID_STEP_EPSILON;
            entry_normal = sdf_exit.yzw;
            entered_side = abs(entry_normal.y) < 0.5;
            continue;
        }
        var exit: vec4<f32>;
        let base = u32(index) * GRID_SLOT_WORDS;
        let local = column & vec2<i32>(15);
        let tile = u32((local.y >> 2u) * 4 + (local.x >> 2u));
        if p.y >= f32(slot.z) {
            exit = grid_box_exit(p, origin, inv_dir, positive, moving, 16.0);
        } else if p.y >= f32(shadow_bits[base + GRID_BIT_WORDS + GRID_COLUMNS + tile]) {
            exit = grid_box_exit(p, origin, inv_dir, positive, moving, 4.0);
        } else {
            let column_id = u32(local.y * 16 + local.x);
            let word = shadow_bits[base + GRID_BIT_WORDS + column_id];
            let height = word & 0xffffu;
            exit = grid_box_exit(p, origin, inv_dir, positive, moving, 1.0);
            if p.y < f32(height) {
                let y0 = u32(floor(max(p.y, 0.0)));
                let y1 = max(min(u32(floor(origin.y + dir.y * (exit.x - 1e-4))), height - 1u), y0);
                let hit_y = grid_first_hit(base + column_id * GRID_COLUMN_WORDS, y0, y1);
                if hit_y != 0xffffffffu {
                    result.found = true;
                    if hit_y == y0 && entered_side {
                        // Entered the column through a side face.
                        result.t = t;
                        result.normal = entry_normal;
                    } else {
                        // Rose into the cell through its bottom face.
                        result.t = (f32(hit_y) - origin.y) * inv_dir.y;
                        result.normal = vec3<f32>(0.0, -1.0, 0.0);
                    }
                    let near_top = hit_y + 2u >= height;
                    result.layer = select((word >> 24u) & 0xffu, (word >> 16u) & 0xffu, near_top);
                    return result;
                }
            }
        }
        t = exit.x + GRID_STEP_EPSILON;
        entry_normal = exit.yzw;
        entered_side = true;
    }
    return result;
}

/// Colour of a reflected hit: the lit scene when the point is visible on
/// screen, otherwise the average colour of its texture with simple lighting.
fn reflection_hit_color(hit: ReflectionHit, origin: vec3<f32>, dir: vec3<f32>, sun_dir: vec3<f32>, day: f32) -> vec3<f32> {
    let pos = origin + dir * hit.t;
    let clip = uniforms.view_proj * vec4<f32>(pos, 1.0);
    if clip.w > 0.1 {
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if all(uv > vec2<f32>(0.002)) && all(uv < vec2<f32>(0.998)) {
            let pixel = vec2<i32>(uv * uniforms.screen_size);
            let scene_depth = textureLoad(ssr_hiz, pixel, 0).r;
            let scene_h = uniforms.inv_view_proj * vec4<f32>(ndc.xy, scene_depth, 1.0);
            let scene_world = scene_h.xyz / scene_h.w;
            let hit_distance = length(pos - uniforms.camera_pos);
            let scene_distance = length(scene_world - uniforms.camera_pos);
            if abs(scene_distance - hit_distance) < 1.0 + hit_distance * 0.03 {
                return textureSampleLevel(scene_color, scene_sampler, uv, 0.0).rgb;
            }
        }
    }
    // Hidden from the camera: the texture's average colour (its smallest
    // mip level) lit by sky and sun.
    let albedo = textureSampleLevel(texture_atlas, texture_sampler, vec2<f32>(0.5), i32(hit.layer), 16.0).rgb;
    let sky = mix(0.08, 0.42, day) * (0.6 + 0.4 * max(hit.normal.y, 0.0));
    let sun = max(dot(hit.normal, sun_dir), 0.0) * 0.75 * day;
    return albedo * (sky + sun);
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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:   vec3<f32>,
    @location(1) wave_normal: vec3<f32>,
    /// Water depth below the surface in blocks (0 at the shore, 15 = deep).
    @location(2) water_depth: f32,
    /// 1 for the top surface, 0 for side faces.
    @location(3) @interpolate(flat) is_surface: u32,
};

// ── Realistic water ───────────────────────────────────────────────────────── //
// Wave shapes are baked once into tiling texture layers (see
// `src/render/water_waves.rs`): each texel holds (value, d/du, d/dv, extra).
// The surface is built from several samples of those layers, each rotated
// relative to the wind, scaled to a non-harmonic tile size and scrolled at
// its own speed, so the sum never visibly repeats. A slow macro layer warps
// the sampling position (bent crests) and modulates amplitude (wave groups).
//
// Peak height: Σ swell amplitude (0.094) × 1.25 group envelope × 1.15 depth
// scale ≈ 0.135, below WATER_LEVEL_OFFSET, so the surface never rises above
// the block top and never pokes through the shore.
@group(0) @binding(11) var wave_layers: texture_2d_array<f32>;
@group(0) @binding(12) var wave_sampler: sampler;

const LAYER_SWELL: i32 = 0;
const LAYER_RIPPLES: i32 = 1;
const LAYER_MACRO: i32 = 2;
// (tile size [blocks], angle from the wind [rad], scroll speed [blocks/s], amplitude)
// Two swell samples + one ripple sample + one macro sample (+ the refracted
// scene) are all the texture reads a water pixel needs.
const SWELLS = array<vec4<f32>, 2>(
    vec4<f32>(104.0,  0.00, 2.2, 0.058),
    vec4<f32>( 67.0,  0.55, 1.6, 0.036),
);
const RIPPLE = vec4<f32>(9.0, 0.40, 0.60, 0.034);
const MACRO_TILE: f32 = 300.0;
const WARP_BLOCKS: f32 = 5.0;
// Per-block absorption of light travelling through water (red is lost
// first, so shallows look turquoise and depths blue).
const ABSORPTION = vec3<f32>(0.26, 0.075, 0.050);
const DEEP_SCATTER = vec3<f32>(0.010, 0.070, 0.115);
const PI: f32 = 3.14159265;

fn wind_angle() -> f32 {
    return atan2(uniforms.wind_dir_z, uniforms.wind_dir_x + 1e-4);
}

fn rotate(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

fn layer_uv(p: vec2<f32>, tile: f32, angle: f32, scroll: f32) -> vec2<f32> {
    return rotate(p, -angle) / tile + vec2<f32>(scroll / tile, 0.0);
}

/// Converts a fetched texel to (value, ∂/∂x, ∂/∂z, extra) in world units.
fn to_world(t: vec4<f32>, tile: f32, angle: f32) -> vec4<f32> {
    return vec4<f32>(t.x, rotate(t.yz, angle) / tile, t.w);
}

/// Fragment-stage fetch: implicit derivatives pick the mip, so this must
/// stay in uniform control flow.
fn fetch_fs(layer: i32, p: vec2<f32>, tile: f32, angle: f32, scroll: f32) -> vec4<f32> {
    let t = textureSample(wave_layers, wave_sampler, layer_uv(p, tile, angle, scroll), layer);
    return to_world(t, tile, angle);
}

/// Vertex-stage fetch from the top mip.
fn fetch_vs(layer: i32, p: vec2<f32>, tile: f32, angle: f32, scroll: f32) -> vec4<f32> {
    let t = textureSampleLevel(wave_layers, wave_sampler, layer_uv(p, tile, angle, scroll), layer, 0.0);
    return to_world(t, tile, angle);
}

/// Macro texel → (warp offset xz [blocks], long-wave group, short-wave group).
/// The warp channels come back divided by the tile size (they share the
/// derivative slots), so they are scaled back up to blocks.
fn macro_params(m: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        m.yz * MACRO_TILE * WARP_BLOCKS,
        0.65 + 0.6 * smoothstep(-0.45, 0.45, m.x),
        0.65 + 0.6 * smoothstep(-0.45, 0.45, m.w),
    );
}

fn macro_uv_args(time: f32) -> vec3<f32> {
    return vec3<f32>(MACRO_TILE, 0.3, time * 0.8);
}

/// Swell (h, ∂h/∂x, ∂h/∂z) in the fragment stage.
fn swell_fs(p: vec2<f32>, time: f32) -> vec3<f32> {
    let a = macro_uv_args(time);
    let m = macro_params(fetch_fs(LAYER_MACRO, p, a.x, a.y, a.z));
    let q = p + m.xy;
    let base = wind_angle();
    let s0 = fetch_fs(LAYER_SWELL, q, SWELLS[0].x, base + SWELLS[0].y, time * SWELLS[0].z);
    let s1 = fetch_fs(LAYER_SWELL, q, SWELLS[1].x, base + SWELLS[1].y, time * SWELLS[1].z);
    return s0.xyz * SWELLS[0].w * m.z + s1.xyz * SWELLS[1].w * m.w;
}

/// Swell height in the vertex stage.
fn swell_vs(p: vec2<f32>, time: f32) -> f32 {
    let a = macro_uv_args(time);
    let m = macro_params(fetch_vs(LAYER_MACRO, p, a.x, a.y, a.z));
    let q = p + m.xy;
    let base = wind_angle();
    let s0 = fetch_vs(LAYER_SWELL, q, SWELLS[0].x, base + SWELLS[0].y, time * SWELLS[0].z);
    let s1 = fetch_vs(LAYER_SWELL, q, SWELLS[1].x, base + SWELLS[1].y, time * SWELLS[1].z);
    return s0.x * SWELLS[0].w * m.z + s1.x * SWELLS[1].w * m.w;
}

/// Ripples: (h, ∂h/∂x, ∂h/∂z, foam noise). The layer already holds three
/// baked octaves, so a single fetch suffices.
fn ripples(p: vec2<f32>, time: f32) -> vec4<f32> {
    let r = fetch_fs(LAYER_RIPPLES, p, RIPPLE.x, wind_angle() + RIPPLE.y, time * RIPPLE.z);
    return vec4<f32>(r.xyz * RIPPLE.w, r.w);
}

/// Wave amplitude scale for a given water depth: calm in the shallows so
/// the surface hugs the shore, fuller over open water.
fn depth_wave_scale(depth: f32) -> f32 {
    return smoothstep(0.4, 4.0, depth) * mix(0.85, 1.15, smoothstep(6.0, 14.0, depth));
}

/// Breaking-surf foam: irregular bands rolling towards the shore, torn up
/// by the ripple foam noise. Returns coverage 0..1.
fn shore_foam(depth: f32, time: f32, noise: f32, wobble: f32) -> f32 {
    let near_shore = 1.0 - smoothstep(0.35, 2.4, depth);
    let bands = sin(depth * 3.6 + time * 1.4 + wobble * 5.0) * 0.5 + 0.5;
    let edge = 1.0 - smoothstep(0.0, 0.6, depth + noise * 0.4);
    let coverage = near_shore * (bands * 0.6 + 0.4) * (0.55 + noise) + edge * 0.6;
    return smoothstep(0.45, 0.8, coverage);
}

/// GGX normal distribution for the sun highlight.
fn ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness * roughness * roughness;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

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
fn vs_water(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> VertexOutput {
    var out: VertexOutput;
    let quad = quads[vertex_id / 6u];
    let corner_id = vertex_id % 6u;
    let diagonal = (quad.origin_and_face >> 30u) & 0x1u;
    let regular = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u)[corner_id];
    let alternate = array<u32, 6>(0u, 1u, 3u, 1u, 2u, 3u)[corner_id];
    let corner = select(regular, alternate, diagonal == 1u);
    let normal_index = (quad.origin_and_face >> 18u) & 0x7u;
    let width = f32(quad.size_material_ao & 0x3fu) * 0.5;
    let height = f32((quad.size_material_ao >> 6u) & 0x3fu) * 0.5;
    let local_origin = vec3<f32>(f32(quad.origin_and_face & 0x3fu), f32((quad.origin_and_face >> 6u) & 0x3fu), f32((quad.origin_and_face >> 12u) & 0x3fu)) * 0.5;
    let edge_u = array<vec3<f32>, 6>(vec3(0.,0.,1.), vec3(0.,0.,-1.), vec3(0.,0.,-1.), vec3(0.,0.,1.), vec3(-1.,0.,0.), vec3(1.,0.,0.))[normal_index];
    let edge_v = array<vec3<f32>, 6>(vec3(0.,1.,0.), vec3(0.,1.,0.), vec3(1.,0.,0.), vec3(1.,0.,0.), vec3(0.,1.,0.), vec3(0.,1.,0.))[normal_index];
    let corner_uv = array<vec2<f32>, 4>(vec2(0.,0.), vec2(1.,0.), vec2(1.,1.), vec2(0.,1.))[corner];
    var pos = subchunks[subchunk_id].aabb_min.xyz + local_origin + edge_u * (corner_uv.x * width) + edge_v * (corner_uv.y * height);

    var normal = face_normal(normal_index);

    let depth = f32((quad.light >> (corner * 4u)) & 0xfu);
    if normal_index == 3u {
        pos.y -= WATER_LEVEL_OFFSET;
        normal = cheap_water_normal(pos.xz, uniforms.time);
        if uniforms.water_quality > 0.5 {
            // Displace by the swell; the exact per-pixel normal is computed
            // in the fragment shader.
            // Distant water only needs normals; skip displacement there.
            let fade = 1.0 - smoothstep(64.0, 160.0, length(pos.xz - uniforms.camera_pos.xz));
            if fade > 0.0 {
                pos.y += swell_vs(pos.xz, uniforms.time) * depth_wave_scale(depth) * fade;
            }
        }
    }

    out.clip_position = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.wave_normal = normal;
    out.water_depth = depth;
    out.is_surface = select(0u, 1u, normal_index == 3u);
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

// Same aerial perspective as `terrain.wgsl`, so distant water and land fade
// into the horizon together.
fn horizon_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let day   = smoothstep(-0.02, 0.22, sun_dir.y);
    let dusk  = 1.0 - smoothstep(0.02, 0.34, abs(sun_dir.y));
    let night = clamp(-sun_dir.y * 4.0 - 0.05, 0.0, 1.0);
    var color = vec3<f32>(0.70, 0.84, 0.98) * day
        + vec3<f32>(0.010, 0.014, 0.034) * night
        + vec3<f32>(0.96, 0.52, 0.30) * dusk * 0.55;
    let towards_sun = max(dot(normalize(vec3<f32>(view_dir.x, 0.0, view_dir.z)),
                              normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z) + vec3<f32>(1e-4))), 0.0);
    color += vec3<f32>(1.0, 0.72, 0.42) * pow(towards_sun, 6.0) * (0.10 * day + 0.35 * dusk);
    return color;
}

fn aerial_perspective(color: vec3<f32>, world_pos: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let to_frag = world_pos - uniforms.camera_pos;
    let dist = length(to_frag.xz);
    let fog_far = max(uniforms.fog_far, 64.0);
    let aerial = (1.0 - exp(-dist / (fog_far * 1.6))) * 0.55;
    let edge = smoothstep(fog_far * 0.55, fog_far * 0.96, dist);
    let amount = max(aerial, edge) * mix(0.25, 1.0, clamp(uniforms.sky_visibility, 0.0, 1.0));
    return mix(color, horizon_color(to_frag / max(length(to_frag), 1e-3), sun_dir), amount);
}

fn realistic_water(in: VertexOutput) -> vec4<f32> {
    let time = uniforms.time;
    let to_camera = uniforms.camera_pos - in.world_pos;
    let dist = max(length(to_camera), 0.001);
    let view_dir = to_camera / dist;
    let sun_dir = normalize(uniforms.sun_position);
    let day = smoothstep(-0.08, 0.24, sun_dir.y);
    let depth = max(in.water_depth, 0.0);
    let wave_scale = depth_wave_scale(depth);

    // Surface normal straight from the baked slopes. Mipmapping already
    // smooths distant ripples; the extra fade keeps them from shimmering.
    let p = in.world_pos.xz;
    let ripple_amount = (1.0 - smoothstep(30.0, 110.0, dist)) * mix(0.35, 1.0, smoothstep(0.2, 2.0, depth));
    let sw = swell_fs(p, time) * wave_scale;
    let rp = ripples(p, time);
    let slope = sw.yz + rp.yz * ripple_amount;
    let h0 = sw.x + rp.x * ripple_amount;
    let normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));

    let n_dot_v = clamp(dot(normal, view_dir), 0.0, 1.0);
    let fresnel = schlick_fresnel(n_dot_v);

    // Refraction: the scene behind, bent by the surface and dimmed by the
    // water it shines through (longer path at grazing angles).
    let uv = clamp(in.clip_position.xy / uniforms.screen_size, vec2<f32>(0.0), vec2<f32>(1.0));
    let bend = vec2<f32>(normal.x, -normal.z) * 0.055 * smoothstep(0.0, 3.0, depth) / (1.0 + dist * 0.04);
    let behind = textureSampleLevel(
        scene_color,
        scene_sampler,
        clamp(uv + bend, vec2<f32>(0.0), vec2<f32>(1.0)),
        0.0,
    ).rgb;
    let path = depth / max(view_dir.y, 0.3);
    let transmittance = exp(-ABSORPTION * path);
    let light_level = mix(0.18, 1.0, day) + uniforms.moon_intensity * 0.12;
    let inscatter = DEEP_SCATTER * light_level;
    var refracted = behind * transmittance + inscatter * (vec3<f32>(1.0) - transmittance);

    // Light scattered inside wave crests facing away from the sun.
    let crest = clamp(h0 / 0.08, 0.0, 1.0);
    let sun_flat = normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z) + vec3<f32>(1e-4, 0.0, 0.0));
    let backlit = pow(max(dot(-view_dir, sun_flat), 0.0) * 0.5 + 0.5, 4.0);
    refracted += vec3<f32>(0.05, 0.22, 0.20) * crest * backlit * day * 0.6;

    // Reflection of the sky.
    let refl_dir = normalize(reflect(-view_dir, normal));
    let sky = sky_reflection_color(refl_dir, sun_dir, uniforms.moon_intensity)
        * mix(0.35, 1.0, clamp(uniforms.sky_visibility, 0.0, 1.0));
    // Reflected terrain: a voxel ray march, only for water near the camera.
    var ssr = vec4<f32>(0.0);
    if dist < REFLECTION_TRACE_DISTANCE {
        let origin = in.world_pos + normal * 0.04;
        let hit = trace_reflection(origin, refl_dir, REFLECTION_RAY_LENGTH);
        if hit.found {
            let fade = (1.0 - smoothstep(REFLECTION_RAY_LENGTH * 0.8, REFLECTION_RAY_LENGTH, hit.t))
                * (1.0 - smoothstep(REFLECTION_TRACE_DISTANCE * 0.8, REFLECTION_TRACE_DISTANCE, dist));
            ssr = vec4<f32>(reflection_hit_color(hit, origin, refl_dir, sun_dir, day), fade);
        }
    }
    let ssr_luma = dot(ssr.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ssr_color = max(mix(vec3<f32>(ssr_luma), ssr.rgb, SSR_SATURATION), vec3<f32>(0.0)) * SSR_BRIGHTNESS;
    let reflected = mix(sky, ssr_color, ssr.a);
    let reflectance = mix(fresnel, max(fresnel, SSR_MIN_REFLECTANCE), ssr.a);
    var color = mix(refracted, reflected, reflectance);

    // Sun glitter: a GGX highlight that broadens with distance so far waves
    // keep a soft sheen instead of aliasing into noise.
    if sun_dir.y > -0.05 {
        let half_dir = normalize(sun_dir + view_dir);
        let roughness = mix(0.07, 0.22, smoothstep(20.0, 220.0, dist));
        let n_dot_h = max(dot(normal, half_dir), 0.0);
        let n_dot_l = max(dot(normal, sun_dir), 0.0);
        let spec = ggx(n_dot_h, roughness) * schlick_fresnel(max(dot(half_dir, view_dir), 0.0)) * n_dot_l;
        let sun_color = mix(vec3<f32>(1.0, 0.62, 0.32), vec3<f32>(1.0, 0.95, 0.86), smoothstep(0.0, 0.35, sun_dir.y));
        color += sun_color * min(spec, 60.0) * 0.30 * day;
    }

    // Foam along the shore and on the steepest crests.
    let foam = max(
        shore_foam(depth, time, rp.w, sw.x * 8.0),
        smoothstep(0.8, 1.0, crest) * smoothstep(0.1, 0.5, rp.w) * 0.7,
    );
    let foam_light = mix(0.12, 0.95, day) * mix(0.55, 1.0, max(normal.y, 0.0));
    color = mix(color, vec3<f32>(0.92, 0.96, 0.98) * foam_light, foam * 0.9);

    color = aerial_perspective(color, in.world_pos, sun_dir);
    // The refracted scene is composited here, so the result is opaque.
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_water(in: VertexOutput) -> @location(0) vec4<f32> {
    // The realistic path samples textures with implicit derivatives, so it
    // runs under a uniform condition only; side faces (rare) fall through.
    if uniforms.water_quality > 0.5 && uniforms.is_underwater < 0.5 {
        let realistic = realistic_water(in);
        if in.is_surface == 1u {
            return realistic;
        }
    }
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

    water_color = aerial_perspective(water_color, in.world_pos, sun_dir);

    let alpha = clamp(WATER_OPACITY + fresnel * 0.26 + (1.0 - dist_fade) * 0.10, 0.18, 0.72);
    return vec4<f32>(water_color, alpha);
}
