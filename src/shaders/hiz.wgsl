/// Hi-Z Depth Downsample Shader
///
/// Takes the previous mip level and downsamples it to the next level 
/// by taking the MAXIMUM (furthest) depth in a 2x2 area for standard depth buffer.
/// This ensures a conservative occlusion test.
/// 
/// Assumptions: Standard depth (0=near, 1=far), clear(1.0), compare Less.
/// For reversed-Z, switch to min() and adjust game code accordingly.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(dst_tex);
    let src_size = textureDimensions(src_tex);  // Add for edge handling.

    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }

    let src_pos = vec2<i32>(id.xy * 2u);
    
    // Gather 4 samples with bounds check (clamp to edge).
    let d00 = textureLoad(src_tex, clamp(src_pos, vec2<i32>(0), vec2<i32>(src_size) - 1), 0).r;
    let d10 = textureLoad(src_tex, clamp(src_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(src_size) - 1), 0).r;
    let d01 = textureLoad(src_tex, clamp(src_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(src_size) - 1), 0).r;
    let d11 = textureLoad(src_tex, clamp(src_pos + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(src_size) - 1), 0).r;

    // For standard depth: furthest = MAX depth.
    let furthest_d = max(max(d00, d10), max(d01, d11));

    // Store only R channel.
    textureStore(dst_tex, vec2<i32>(id.xy), vec4<f32>(furthest_d, 0.0, 0.0, 1.0));
}