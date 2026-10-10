@group(0) @binding(0)
var msaa_depth: texture_depth_multisampled_2d;

// This is the former Hi-Z mip 1 (half the surface dimensions), not a
// full-resolution intermediate. Water only samples opaque scene color.
@group(0) @binding(1)
var hiz_seed: texture_storage_2d<r32float, write>;

fn resolve_max_depth(coords: vec2<i32>, src_max: vec2<i32>) -> f32 {
    let pixel = clamp(coords, vec2<i32>(0), src_max);
    let s0 = textureLoad(msaa_depth, pixel, 0);
    let s1 = textureLoad(msaa_depth, pixel, 1);
    let s2 = textureLoad(msaa_depth, pixel, 2);
    let s3 = textureLoad(msaa_depth, pixel, 3);
    return max(max(s0, s1), max(s2, s3));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(hiz_seed);
    if id.x >= dst_size.x || id.y >= dst_size.y {
        return;
    }
    let src_size = textureDimensions(msaa_depth);
    let src_max = vec2<i32>(src_size) - vec2<i32>(1);
    let base = vec2<i32>(id.xy) * 2;
    var d = max(
        max(resolve_max_depth(base, src_max),
            resolve_max_depth(base + vec2<i32>(1, 0), src_max)),
        max(resolve_max_depth(base + vec2<i32>(0, 1), src_max),
            resolve_max_depth(base + vec2<i32>(1, 1), src_max))
    );

    // Preserve the overlapping 3-texel footprint of hiz.wgsl for odd sizes.
    // Omitting the last row/column could turn background into an occluder.
    if (src_size.x & 1u) != 0u {
        d = max(d, max(
            resolve_max_depth(base + vec2<i32>(2, 0), src_max),
            resolve_max_depth(base + vec2<i32>(2, 1), src_max)
        ));
    }
    if (src_size.y & 1u) != 0u {
        d = max(d, max(
            resolve_max_depth(base + vec2<i32>(0, 2), src_max),
            resolve_max_depth(base + vec2<i32>(1, 2), src_max)
        ));
    }
    if (src_size.x & 1u) != 0u && (src_size.y & 1u) != 0u {
        d = max(d, resolve_max_depth(base + vec2<i32>(2, 2), src_max));
    }
    textureStore(hiz_seed, vec2<i32>(id.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}
