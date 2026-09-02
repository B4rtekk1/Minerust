@group(0) @binding(0)
var msaa_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var hiz_seed: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var ssr_depth: texture_storage_2d<r32float, write>;

fn resolve_max_depth(coords: vec2<u32>) -> f32 {
    let s0 = textureLoad(msaa_depth, vec2<i32>(coords), 0);
    let s1 = textureLoad(msaa_depth, vec2<i32>(coords), 1);
    let s2 = textureLoad(msaa_depth, vec2<i32>(coords), 2);
    let s3 = textureLoad(msaa_depth, vec2<i32>(coords), 3);

    return max(max(s0, s1), max(s2, s3));
}

fn resolve_min_depth(coords: vec2<u32>) -> f32 {
    let s0 = textureLoad(msaa_depth, vec2<i32>(coords), 0);
    let s1 = textureLoad(msaa_depth, vec2<i32>(coords), 1);
    let s2 = textureLoad(msaa_depth, vec2<i32>(coords), 2);
    let s3 = textureLoad(msaa_depth, vec2<i32>(coords), 3);

    return min(min(s0, s1), min(s2, s3));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(msaa_depth);
    if id.x >= size.x || id.y >= size.y {
        return;
    }

    let max_depth = resolve_max_depth(id.xy);
    textureStore(hiz_seed, vec2<i32>(id.xy), vec4<f32>(max_depth, 0.0, 0.0, 1.0));
    let min_depth = resolve_min_depth(id.xy);
    textureStore(ssr_depth, vec2<i32>(id.xy), vec4<f32>(min_depth, 0.0, 0.0, 1.0));
}
