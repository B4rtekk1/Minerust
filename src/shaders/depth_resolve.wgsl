@group(0) @binding(0)
var msaa_depth: texture_depth_multisampled_2d;

@group(0) @binding(1)
var hiz_seed: texture_storage_2d<r32float, write>;

@group(0) @binding(2)
var ssr_depth: texture_storage_2d<r32float, write>;

fn resolve_min_max_depth(coords: vec2<u32>) -> vec2<f32> {
    let s0 = textureLoad(msaa_depth, vec2<i32>(coords), 0);
    let s1 = textureLoad(msaa_depth, vec2<i32>(coords), 1);
    let s2 = textureLoad(msaa_depth, vec2<i32>(coords), 2);
    let s3 = textureLoad(msaa_depth, vec2<i32>(coords), 3);

    let min_depth = min(min(s0, s1), min(s2, s3));
    let max_depth = max(max(s0, s1), max(s2, s3));
    return vec2<f32>(min_depth, max_depth);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(msaa_depth);
    if id.x >= size.x || id.y >= size.y {
        return;
    }

    let depths = resolve_min_max_depth(id.xy);
    textureStore(ssr_depth, vec2<i32>(id.xy), vec4<f32>(depths.x, 0.0, 0.0, 1.0));
    textureStore(hiz_seed, vec2<i32>(id.xy), vec4<f32>(depths.y, 0.0, 0.0, 1.0));
}
