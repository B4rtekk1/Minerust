@group(0) @binding(0) var msaa_depth: texture_depth_multisampled_2d;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Fullscreen triangle trick (covers screen with 3 vertices)
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
    
    // Resolve MSAA depth: 
    // - min_depth (closest) for SSR/Refractions to ensure contact
    // - max_depth (furthest) for Hi-Z conservative occlusion culling
    var min_depth = 1.0;
    var max_depth = 0.0;
    for (var i = 0; i < 4; i++) {
        let d = textureLoad(msaa_depth, coords, i);
        min_depth = min(min_depth, d);
        max_depth = max(max_depth, d);
    }
    
    var out: ResolveOutput;
    out.depth = min_depth;
    out.color = max_depth;
    return out;
}
