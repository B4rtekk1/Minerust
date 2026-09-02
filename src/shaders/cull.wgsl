struct SubchunkMeta {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    terrain_draw_data: vec4<u32>,
    water_draw_data: vec4<u32>,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
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

@group(0) @binding(2)
var<storage, read_write> terrain_draw_commands: array<DrawIndirect>;

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

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= cull_uniforms.subchunk_count {
        return;
    }

    let subchunk = subchunks[idx];

    if subchunk.terrain_draw_data.w == 0u && subchunk.water_draw_data.w == 0u {
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

    if subchunk.terrain_draw_data.w != 0u {
        let slot = atomicAdd(&terrain_visible_count, 1u);
        terrain_draw_commands[slot].vertex_count   = subchunk.terrain_draw_data.x;
        terrain_draw_commands[slot].instance_count = 1u;
        terrain_draw_commands[slot].first_vertex   = subchunk.terrain_draw_data.y;
        terrain_draw_commands[slot].first_instance = idx;
    }
    if subchunk.water_draw_data.w != 0u {
        let slot = atomicAdd(&water_visible_count, 1u);
        water_draw_commands[slot].vertex_count   = subchunk.water_draw_data.x;
        water_draw_commands[slot].instance_count = 1u;
        water_draw_commands[slot].first_vertex   = subchunk.water_draw_data.y;
        water_draw_commands[slot].first_instance = idx;
    }
}
