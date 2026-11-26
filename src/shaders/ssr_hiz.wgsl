@group(0) @binding(0) var source_depth: texture_depth_multisampled_2d;
@group(0) @binding(1) var resolved_depth: texture_storage_2d<r32float, write>;

const SAMPLE_COUNT: i32 = 4;

fn resolve_min_depth(coords: vec2<i32>, source_max: vec2<i32>) -> f32 {
    let pixel = clamp(coords, vec2<i32>(0), source_max);
    var depth = 1.0;
    for (var sample_index = 0; sample_index < SAMPLE_COUNT; sample_index++) {
        depth = min(depth, textureLoad(source_depth, pixel, sample_index));
    }
    return depth;
}

@compute @workgroup_size(16, 16)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(resolved_depth);
    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }

    let src_size = textureDimensions(source_depth);
    let src_max = vec2<i32>(src_size) - vec2<i32>(1);
    let depth = resolve_min_depth(vec2<i32>(id.xy), src_max);
    textureStore(resolved_depth, vec2<i32>(id.xy), vec4<f32>(depth, 0.0, 0.0, 1.0));
}
