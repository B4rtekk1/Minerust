struct SubchunkMeta {
    world_origin: vec4<i32>,
    draw_data: vec4<u32>,
}

struct ClusterMeta {
    world_origin: vec4<i32>,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

struct CullUniforms {
    occlusion_view_proj: mat4x4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
    camera_pos: vec3<f32>,
    subchunk_count: u32,
    hiz_size: vec2<f32>,
    screen_size: vec2<f32>,
    cull_distance: f32,
    cluster_count: u32,
    occlusion_enabled: u32,
}

@group(0) @binding(0)
var<uniform> cull_uniforms: CullUniforms;

@group(0) @binding(1)
var<storage, read> subchunks: array<SubchunkMeta>;

@group(0) @binding(2)
var<storage, read_write> draw_commands: array<DrawIndirect>;

@group(0) @binding(3)
var<storage, read_write> visible_count: atomic<u32>;

@group(0) @binding(4)
var hiz_texture: texture_2d<f32>;

@group(0) @binding(5)
var hiz_sampler: sampler;

@group(0) @binding(6)
var<storage, read> clusters: array<ClusterMeta>;

@group(0) @binding(7)
var<storage, read> subchunk_clusters: array<u32>;

@group(0) @binding(8)
var<storage, read_write> cluster_visibility: array<u32>;

const FRUSTUM_CULL_MARGIN: f32 = 2.0;
const CLUSTER_EXTENT: vec3<f32> = vec3<f32>(32.0, 64.0, 32.0);
const WORKGROUP_SIZE: u32 = 128u;

// Portable workgroup-local compaction.  Keeping this in workgroup memory is
// intentionally the baseline path: subgroup operations are not available on
// every WebGPU backend.  A subgroup-specialized pipeline can replace just
// this scan when that capability is enabled without changing the output
// protocol below.
var<workgroup> scan_values: array<u32, WORKGROUP_SIZE>;
var<workgroup> workgroup_visible_base: u32;

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
    if cull_uniforms.occlusion_enabled == 0u || cull_uniforms.hiz_size.x < 1.0 {
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
        let clip = cull_uniforms.occlusion_view_proj * vec4<f32>(corners[c], 1.0);
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

fn is_within_cull_distance(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let camera_xz = cull_uniforms.camera_pos.xz;
    let nearest_xz = clamp(camera_xz, aabb_min.xz, aabb_max.xz);
    return distance(camera_xz, nearest_xz) <= cull_uniforms.cull_distance;
}

/// Coarse stage.  A failed node prevents all of its children from doing
/// frustum or Hi-Z work in `main` below.
@compute @workgroup_size(WORKGROUP_SIZE)
fn cull_clusters(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cluster_index = global_id.x;
    if cluster_index >= cull_uniforms.cluster_count {
        return;
    }
    let aabb_min = vec3<f32>(clusters[cluster_index].world_origin.xyz);
    let aabb_max = aabb_min + CLUSTER_EXTENT;
    let visible = is_within_cull_distance(aabb_min, aabb_max)
        && is_frustum_visible(aabb_min, aabb_max)
        && is_occlusion_visible(aabb_min, aabb_max);
    cluster_visibility[cluster_index] = select(0u, 1u, visible);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let idx = global_id.x;
    var is_visible = false;
    var subchunk: SubchunkMeta;

    // All invocations must reach the barriers below, including the tail of a
    // partially occupied workgroup, so visibility is represented as 0 or 1
    // instead of returning early.
    if idx < cull_uniforms.subchunk_count {
        subchunk = subchunks[idx];
        if subchunk.draw_data.w != 0u && cluster_visibility[subchunk_clusters[idx]] != 0u {
            // Children of a visible cluster are emitted directly.  The coarse
            // AABB is conservative, so this can only add harmless extra draws
            // near the cluster boundary; it cannot hide visible geometry.
            is_visible = true;
        }
    }

    scan_values[local_index] = select(0u, 1u, is_visible);
    workgroupBarrier();

    // Inclusive Hillis-Steele scan.  It is only seven synchronized shared
    // memory steps for the fixed 128-thread workgroup.
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset << 1u) {
        var addend = 0u;
        if local_index >= offset {
            addend = scan_values[local_index - offset];
        }
        workgroupBarrier();
        scan_values[local_index] += addend;
        workgroupBarrier();
    }

    if local_index == 0u {
        let local_visible_count = scan_values[WORKGROUP_SIZE - 1u];
        if local_visible_count != 0u {
            workgroup_visible_base = atomicAdd(&visible_count, local_visible_count);
        } else {
            workgroup_visible_base = 0u;
        }
    }
    workgroupBarrier();

    if is_visible {
        let local_slot = scan_values[local_index] - 1u;
        let slot = workgroup_visible_base + local_slot;
        draw_commands[slot].vertex_count   = subchunk.draw_data.x;
        draw_commands[slot].instance_count = 1u;
        draw_commands[slot].first_vertex   = subchunk.draw_data.y;
        draw_commands[slot].first_instance = subchunk.draw_data.z;
    }
}
