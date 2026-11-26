struct SubchunkMeta {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    terrain_draw_data: vec4<u32>,
    water_draw_data: vec4<u32>,
    terrain_far_draw_data: vec4<u32>,
    terrain_lod_draw_data: vec4<u32>,
    // Quads per terrain draw group: face directions −X, +X, −Y, +Y | −Z, +Z,
    // cutout (alpha-tested), unused. Groups are stored contiguously.
    terrain_groups_a: vec4<u32>,
    terrain_groups_b: vec4<u32>,
    terrain_far_groups_a: vec4<u32>,
    terrain_far_groups_b: vec4<u32>,
    terrain_lod_groups_a: vec4<u32>,
    terrain_lod_groups_b: vec4<u32>,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

/// Terrain is drawn indexed: 4 pulled vertices + 6 indices per quad, from a
/// shared static index pattern (`base_vertex` selects the first quad).
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct CullUniforms {
    view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_pos: vec3<f32>,
    subchunk_count: u32,
    hiz_size: vec2<f32>,
    screen_size: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> cull_uniforms: CullUniforms;

@group(0) @binding(1)
var<storage, read> subchunks: array<SubchunkMeta>;

/// Final terrain draws, sorted front to back by `sort_draws`.
@group(0) @binding(2)
var<storage, read_write> terrain_draw_commands: array<DrawIndexedIndirect>;

@group(0) @binding(3)
var<storage, read_write> terrain_visible_count: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> water_draw_commands: array<DrawIndirect>;

@group(0) @binding(5)
var<storage, read_write> water_visible_count: atomic<u32>;

@group(0) @binding(6)
var hiz_texture: texture_2d<f32>;

@group(0) @binding(7)
var hiz_sampler: sampler;

@group(0) @binding(8)
var<storage, read_write> cutout_draw_commands: array<DrawIndexedIndirect>;

@group(0) @binding(9)
var<storage, read_write> cutout_visible_count: atomic<u32>;

/// Visible terrain draws in arrival order, before sorting.
@group(0) @binding(10)
var<storage, read_write> terrain_unsorted: array<DrawIndexedIndirect>;

/// Distance bucket of each entry in `terrain_unsorted`.
@group(0) @binding(11)
var<storage, read_write> terrain_buckets: array<u32>;

/// [0, 8): draws per distance bucket; [8, 16): scatter cursors.
@group(0) @binding(12)
var<storage, read_write> sort_counters: array<atomic<u32>, 16>;

const SORT_BUCKETS: u32 = 8u;

/// Coarse front-to-back order: bucket 0 holds subchunks within 8 blocks of
/// the eye, and each further bucket doubles the range (16, 32, … 512+).
/// Drawing near geometry first lets early-Z reject what lies behind it.
fn distance_bucket(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> u32 {
    let cam = cull_uniforms.camera_pos;
    let outside = max(aabb_min - cam, max(cam - aabb_max, vec3<f32>(0.0)));
    let d = length(outside);
    return u32(clamp(floor(log2(max(d, 1.0) / 8.0) + 1.0), 0.0, f32(SORT_BUCKETS - 1u)));
}

/// Whether any face of direction `dir` inside the AABB can face the camera.
/// A face with normal −X lies on some plane x ≤ aabb_max.x and is visible
/// only from x below it, so the test is conservative per subchunk.
/// `camera_pos` must be the eye position; the small margin keeps faces that
/// are exactly level with the eye (and float rounding) on the safe side.
const DIRECTION_MARGIN: f32 = 0.05;

fn direction_visible(dir: u32, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let cam = cull_uniforms.camera_pos;
    let m = DIRECTION_MARGIN;
    switch dir {
        case 0u: { return cam.x < aabb_max.x + m; }
        case 1u: { return cam.x > aabb_min.x - m; }
        case 2u: { return cam.y < aabb_max.y + m; }
        case 3u: { return cam.y > aabb_min.y - m; }
        case 4u: { return cam.z < aabb_max.z + m; }
        default: { return cam.z > aabb_min.z - m; }
    }
}

fn indexed_draw(quads: u32, first_quad: u32, instance_id: u32) -> DrawIndexedIndirect {
    return DrawIndexedIndirect(quads * 6u, 1u, 0u, i32(first_quad * 4u), instance_id);
}

fn emit_terrain(quads: u32, first_quad: u32, instance_id: u32, bucket: u32) {
    let slot = atomicAdd(&terrain_visible_count, 1u);
    terrain_unsorted[slot] = indexed_draw(quads, first_quad, instance_id);
    terrain_buckets[slot] = bucket;
    atomicAdd(&sort_counters[bucket], 1u);
}

const FRUSTUM_CULL_MARGIN: f32 = 2.0;

fn aabb_vs_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: vec4<f32>) -> bool {
    let expanded_min = aabb_min - vec3<f32>(FRUSTUM_CULL_MARGIN);
    let expanded_max = aabb_max + vec3<f32>(FRUSTUM_CULL_MARGIN);
    let p = vec3<f32>(
        select(expanded_min.x, expanded_max.x, plane.x > 0.0),
        select(expanded_min.y, expanded_max.y, plane.y > 0.0),
        select(expanded_min.z, expanded_max.z, plane.z > 0.0),
    );
    return dot(plane.xyz, p) + plane.w >= 0.0;
}

fn is_frustum_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i++) {
        if !aabb_vs_plane(aabb_min, aabb_max, cull_uniforms.frustum_planes[i]) {
            return false;
        }
    }
    return true;
}

fn is_occlusion_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    if cull_uniforms.hiz_size.x < 1.0 {
        return true;
    }

    var min_uv  = vec2<f32>(1.0, 1.0);
    var max_uv  = vec2<f32>(0.0, 0.0);
    var min_z   = 1.0f;
    var any_behind = false;

    let corners = array<vec3<f32>, 8>(
        vec3<f32>(aabb_min.x, aabb_min.y, aabb_min.z),
        vec3<f32>(aabb_max.x, aabb_min.y, aabb_min.z),
        vec3<f32>(aabb_min.x, aabb_max.y, aabb_min.z),
        vec3<f32>(aabb_max.x, aabb_max.y, aabb_min.z),
        vec3<f32>(aabb_min.x, aabb_min.y, aabb_max.z),
        vec3<f32>(aabb_max.x, aabb_min.y, aabb_max.z),
        vec3<f32>(aabb_min.x, aabb_max.y, aabb_max.z),
        vec3<f32>(aabb_max.x, aabb_max.y, aabb_max.z),
    );

    for (var c = 0u; c < 8u; c++) {
        let clip = cull_uniforms.view_proj * vec4<f32>(corners[c], 1.0);
        if clip.w <= 0.0 {
            any_behind = true;
        } else {
            let ndc = clip.xyz / clip.w;
            let uv  = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
            min_uv  = min(min_uv, uv);
            max_uv  = max(max_uv, uv);
            min_z   = min(min_z, ndc.z);
        }
    }

    if any_behind { return true; }

    if max_uv.x <= 0.0 || min_uv.x >= 1.0 || max_uv.y <= 0.0 || min_uv.y >= 1.0 {
        return false;
    }

    let uv_lo = clamp(min_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_hi = clamp(max_uv, vec2<f32>(0.0), vec2<f32>(1.0));

    let max_mip_f   = f32(textureNumLevels(hiz_texture) - 1u);
    let pixel_dim   = (uv_hi - uv_lo) * cull_uniforms.hiz_size;
    let max_dim     = max(pixel_dim.x, pixel_dim.y);

    let mip_f = select(ceil(log2(max(max_dim, 1.0))), 0.0, max_dim < 1.0);
    let mip   = u32(clamp(mip_f, 0.0, max_mip_f));

    let mip_size = vec2<f32>(textureDimensions(hiz_texture, mip));
    let lo_px    = vec2<i32>(uv_lo * mip_size);
    let hi_px    = vec2<i32>(uv_hi * mip_size);
    let mip_max  = vec2<i32>(mip_size) - vec2<i32>(1);

    let t00 = textureLoad(hiz_texture, clamp(lo_px,                         vec2<i32>(0), mip_max), i32(mip)).r;
    let t10 = textureLoad(hiz_texture, clamp(vec2<i32>(hi_px.x, lo_px.y),  vec2<i32>(0), mip_max), i32(mip)).r;
    let t01 = textureLoad(hiz_texture, clamp(vec2<i32>(lo_px.x, hi_px.y),  vec2<i32>(0), mip_max), i32(mip)).r;
    let t11 = textureLoad(hiz_texture, clamp(hi_px,                         vec2<i32>(0), mip_max), i32(mip)).r;

    let occluder_z = max(max(t00, t10), max(t01, t11));

    if occluder_z <= 0.00001 {
        return true;
    }

    let nearest_z = clamp(min_z, 0.0, 1.0);
    return nearest_z <= occluder_z + 0.0001;
}

/// Beyond this horizontal distance (blocks) water waves are no longer
/// displaced (`vs_water` fades them out between 64 and 160 blocks).
const WATER_DETAIL_DISTANCE: f32 = 160.0;
/// Distance at which AO/light-tolerant terrain replaces the detailed mesh.
const TERRAIN_FAR_DISTANCE: f32 = 160.0;

/// Horizontal distance from the camera to the nearest point of a box.
fn horizontal_distance_to_box(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> f32 {
    let p = cull_uniforms.camera_pos.xz;
    let nearest = clamp(p, aabb_min.xz, aabb_max.xz);
    return length(p - nearest);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= cull_uniforms.subchunk_count {
        return;
    }

    let subchunk = subchunks[idx];

    if subchunk.terrain_draw_data.w == 0u
        && subchunk.terrain_far_draw_data.w == 0u
        && subchunk.terrain_lod_draw_data.w == 0u
        && subchunk.water_draw_data.w == 0u {
        return;
    }

    let aabb_min = subchunk.aabb_min.xyz;
    let aabb_max = subchunk.aabb_max.xyz;

    if !is_frustum_visible(aabb_min, aabb_max) {
        return;
    }

    if !is_occlusion_visible(aabb_min, aabb_max) {
        return;
    }

    let terrain_distance = horizontal_distance_to_box(aabb_min, aabb_max);
    var terrain_draw_data = subchunk.terrain_draw_data;
    var terrain_groups_a = subchunk.terrain_groups_a;
    var terrain_groups_b = subchunk.terrain_groups_b;
    if terrain_distance >= TERRAIN_FAR_DISTANCE && subchunk.terrain_far_draw_data.w != 0u {
        terrain_draw_data = subchunk.terrain_far_draw_data;
        terrain_groups_a = subchunk.terrain_far_groups_a;
        terrain_groups_b = subchunk.terrain_far_groups_b;
    }

    if terrain_draw_data.w != 0u {
        // One draw per face direction that can face the camera: roughly half
        // of all faces are skipped before vertex work and rasterization.
        let groups = array<u32, 7>(
            terrain_groups_a.x, terrain_groups_a.y,
            terrain_groups_a.z, terrain_groups_a.w,
            terrain_groups_b.x, terrain_groups_b.y,
            terrain_groups_b.z,
        );
        let bucket = distance_bucket(aabb_min, aabb_max);
        // `terrain_draw_data.y` is the first vertex of the 6-per-quad
        // stream; terrain draws are indexed and address quads directly.
        var first_quad = terrain_draw_data.y / 6u;
        for (var dir = 0u; dir < 6u; dir++) {
            let quads = groups[dir];
            if quads != 0u && direction_visible(dir, aabb_min, aabb_max) {
                emit_terrain(quads, first_quad, idx, bucket);
            }
            first_quad += quads;
        }
        if groups[6] != 0u {
            let slot = atomicAdd(&cutout_visible_count, 1u);
            cutout_draw_commands[slot] = indexed_draw(groups[6], first_quad, idx);
        }
    }
    if subchunk.water_draw_data.w != 0u {
        // Water is stored as near quads (1×1 surface tiles the waves
        // displace) followed by a merged copy (`.z` vertices). Waves fade
        // out by WATER_DETAIL_DISTANCE, so subchunks entirely beyond it draw
        // the merged copy: identical, with a fraction of the vertices.
        var vertex_count = subchunk.water_draw_data.x;
        var first_vertex = subchunk.water_draw_data.y;
        if subchunk.water_draw_data.z != 0u && terrain_distance > WATER_DETAIL_DISTANCE {
            first_vertex += vertex_count;
            vertex_count = subchunk.water_draw_data.z;
        }
        let slot = atomicAdd(&water_visible_count, 1u);
        water_draw_commands[slot].vertex_count   = vertex_count;
        water_draw_commands[slot].instance_count = 1u;
        water_draw_commands[slot].first_vertex   = first_vertex;
        water_draw_commands[slot].first_instance = idx;
    }
}

/// Second pass: scatter the visible terrain draws into bucket order.
/// Within a bucket the order is arbitrary; across buckets it is near → far.
@compute @workgroup_size(128)
fn sort_draws(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= atomicLoad(&terrain_visible_count) {
        return;
    }
    let bucket = terrain_buckets[i];
    var offset = 0u;
    for (var b = 0u; b < bucket; b++) {
        offset += atomicLoad(&sort_counters[b]);
    }
    let position = offset + atomicAdd(&sort_counters[SORT_BUCKETS + bucket], 1u);
    terrain_draw_commands[position] = terrain_unsorted[i];
}
