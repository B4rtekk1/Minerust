/// GPU Frustum Culling Compute Shader
///
/// Performs frustum culling on the GPU for all subchunks in parallel.
/// Visible subchunks are appended to the draw commands buffer.

struct SubchunkMeta {
    /// AABB min (xyz), padding in w
    aabb_min: vec4<f32>,
    /// AABB max (xyz), slot_index in w
    aabb_max: vec4<f32>,
    /// draw_data: index_count, first_index, base_vertex, enabled
    draw_data: vec4<u32>,
}

struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct CullUniforms {
    /// View-projection matrix for AABB projection
    view_proj: mat4x4<f32>,
    /// 6 frustum planes (each is vec4: xyz=normal, w=distance)
    frustum_planes: array<vec4<f32>, 6>,
    /// Camera position
    camera_pos: vec3<f32>,
    /// Number of active subchunks
    subchunk_count: u32,
    /// Hi-Z texture size
    hiz_size: vec2<f32>,
    /// Padding
    _padding: vec2<f32>,
}

/// Culling uniforms
@group(0) @binding(0)
var<uniform> cull_uniforms: CullUniforms;

/// All subchunk metadata (read-only)
@group(0) @binding(1)
var<storage, read> subchunks: array<SubchunkMeta>;

/// Output: visible draw commands
@group(0) @binding(2)
var<storage, read_write> draw_commands: array<DrawIndexedIndirect>;

/// Atomic counter for visible subchunks
@group(0) @binding(3)
var<storage, read_write> visible_count: atomic<u32>;

/// Hi-Z Depth Pyramid (Read-only texture with mips)
@group(0) @binding(4)
var hiz_texture: texture_2d<f32>;

/// Hi-Z Sampler
@group(0) @binding(5)
var hiz_sampler: sampler;

/// Test if an AABB is visible against a frustum plane
fn aabb_vs_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: vec4<f32>) -> bool {
    // Get the positive vertex (furthest along the plane normal)
    let p = vec3<f32>(
        select(aabb_min.x, aabb_max.x, plane.x > 0.0),
        select(aabb_min.y, aabb_max.y, plane.y > 0.0),
        select(aabb_min.z, aabb_max.z, plane.z > 0.0),
    );
    
    // If the positive vertex is behind the plane, AABB is fully outside
    return dot(plane.xyz, p) + plane.w >= 0.0;
}

/// Test if an AABB is inside the frustum
fn is_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    // Add margin for conservative culling
    let margin = vec3<f32>(2.0);
    let expanded_min = aabb_min - margin;
    let expanded_max = aabb_max + margin;
    
    // Test against all 6 frustum planes
    for (var i = 0u; i < 6u; i++) {
        if !aabb_vs_plane(expanded_min, expanded_max, cull_uniforms.frustum_planes[i]) {
            return false;
        }
    }
    return true;
}

/// Test if an AABB is occluded by the Hi-Z pyramid
/// Returns true if visible, false if occluded
fn is_occlusion_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    var min_uv = vec2<f32>(1.0, 1.0);
    var max_uv = vec2<f32>(0.0, 0.0);
    var min_z = 1.0;

    // Unroll the 8 corners manually
    // Corner 0: min, min, min
    var clip = cull_uniforms.view_proj * vec4<f32>(aabb_min.x, aabb_min.y, aabb_min.z, 1.0);
    if clip.w <= 0.0 { return true; }
    var ndc = clip.xyz / clip.w;
    var uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 1: max, min, min
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_max.x, aabb_min.y, aabb_min.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 2: min, max, min
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_min.x, aabb_max.y, aabb_min.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 3: max, max, min
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_max.x, aabb_max.y, aabb_min.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 4: min, min, max
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_min.x, aabb_min.y, aabb_max.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 5: max, min, max
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_max.x, aabb_min.y, aabb_max.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 6: min, max, max
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_min.x, aabb_max.y, aabb_max.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Corner 7: max, max, max
    clip = cull_uniforms.view_proj * vec4<f32>(aabb_max.x, aabb_max.y, aabb_max.z, 1.0);
    if clip.w <= 0.0 { return true; }
    ndc = clip.xyz / clip.w;
    uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    min_uv = min(min_uv, uv);
    max_uv = max(max_uv, uv);
    min_z = min(min_z, ndc.z);

    // Clamp UVs to screen and keep ordering stable
    min_uv = clamp(min_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    max_uv = clamp(max_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_min = min(min_uv, max_uv);
    let uv_max = max(min_uv, max_uv);

    // Calculate required mip level based on screen-space rectangle size
    let size = (uv_max - uv_min) * cull_uniforms.hiz_size;
    let max_dim = max(size.x, size.y);
    let safe_dim = max(max_dim, 1.0);
    let max_mip = textureNumLevels(hiz_texture) - 1u;
    let mip = min(u32(log2(safe_dim)), max_mip);

    // Sample four points in the Hi-Z buffer to be conservative
    let d0 = textureSampleLevel(hiz_texture, hiz_sampler, uv_min, f32(mip)).r;
    let d1 = textureSampleLevel(hiz_texture, hiz_sampler, uv_max, f32(mip)).r;
    let d2 = textureSampleLevel(hiz_texture, hiz_sampler, vec2<f32>(uv_min.x, uv_max.y), f32(mip)).r;
    let d3 = textureSampleLevel(hiz_texture, hiz_sampler, vec2<f32>(uv_max.x, uv_min.y), f32(mip)).r;

    // Hi-Z stores furthest depth (max for standard Z), so aggregate with max.
    let hiz_max_z = max(max(d0, d1), max(d2, d3));
    let nearest_z = clamp(min_z, 0.0, 1.0);

    // Fallback: if Hi-Z is effectively empty/uninitialized, skip occlusion culling.
    if hiz_max_z <= 0.00001 {
        return true;
    }
    
    // Visible if object's nearest point is not fully behind furthest known occluder.
    return nearest_z <= hiz_max_z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Bounds check
    if idx >= cull_uniforms.subchunk_count {
        return;
    }

    let subchunk = subchunks[idx];
    
    // Check if this slot is enabled
    if subchunk.draw_data.w == 0u {
        return;
    }

    let aabb_min = subchunk.aabb_min.xyz;
    let aabb_max = subchunk.aabb_max.xyz;
    
    // Frustum test first (cheap)
    if is_visible(aabb_min, aabb_max) {
        // Occlusion test (only for main pass, not shadow)
        // Note: shadow pass should skip this or pass identity Hi-Z
        if is_occlusion_visible(aabb_min, aabb_max) {
            // Atomically get slot in output array
            let slot = atomicAdd(&visible_count, 1u);
            
            // Write draw command
            draw_commands[slot].index_count = subchunk.draw_data.x;
            draw_commands[slot].instance_count = 1u;
            draw_commands[slot].first_index = subchunk.draw_data.y;
            draw_commands[slot].base_vertex = i32(subchunk.draw_data.z);
            draw_commands[slot].first_instance = 0u;
        }
    }
}
