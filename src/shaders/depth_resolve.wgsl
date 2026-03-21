override MSAA_SAMPLES: i32 = 4;

@group(0) @binding(0) var msaa_depth: texture_depth_multisampled_2d;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

struct ResolveOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: f32,
};

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> ResolveOutput {
    let coords = vec2<i32>(pos.xy);

    let d0 = textureLoad(msaa_depth, coords, 0);
    let d1 = textureLoad(msaa_depth, coords, 1);
    let d2 = textureLoad(msaa_depth, coords, 2);
    let d3 = textureLoad(msaa_depth, coords, 3);

    var out: ResolveOutput;
    out.depth = min(min(d0, d1), min(d2, d3));
    out.color = max(max(d0, d1), max(d2, d3));
    return out;
}