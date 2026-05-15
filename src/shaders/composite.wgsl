struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    csm_view_proj: array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
    screen_size: vec2<f32>,
    water_level: f32,
    reflection_mode: f32,
    moon_position: vec3<f32>,
    _pad1_moon: f32,
};

struct PostProcessUniforms {
    render_size: vec2<f32>,
    output_size: vec2<f32>,
    sharpness: f32,
    fsr_enabled: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var scene_texture: texture_2d<f32>;

@group(0) @binding(2)
var composite_sampler: sampler;

@group(0) @binding(3)
var<uniform> post_process: PostProcessUniforms;

var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
);

var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn source_size_i() -> vec2<i32> {
    return vec2<i32>(textureDimensions(scene_texture));
}

fn clamp_px(px: vec2<i32>) -> vec2<i32> {
    let max_px = source_size_i() - vec2<i32>(1);
    return clamp(px, vec2<i32>(0), max_px);
}

fn load_scene(px: vec2<i32>) -> vec3<f32> {
    return textureLoad(scene_texture, clamp_px(px), 0).rgb;
}

fn catmull_rom_weight(x: f32) -> f32 {
    let ax = abs(x);
    if ax <= 1.0 {
        return (1.5 * ax - 2.5) * ax * ax + 1.0;
    }
    if ax < 2.0 {
        return ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0;
    }
    return 0.0;
}

fn sample_catmull_rom(src_pos: vec2<f32>) -> vec3<f32> {
    let base = vec2<i32>(floor(src_pos));
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var y = -1; y <= 2; y = y + 1) {
        let wy = catmull_rom_weight(src_pos.y - f32(base.y + y));
        for (var x = -1; x <= 2; x = x + 1) {
            let wx = catmull_rom_weight(src_pos.x - f32(base.x + x));
            let weight = wx * wy;
            color += load_scene(base + vec2<i32>(x, y)) * weight;
            weight_sum += weight;
        }
    }

    return color / max(weight_sum, 0.0001);
}

fn edge_strength(base_px: vec2<i32>) -> f32 {
    let c  = luma(load_scene(base_px));
    let l  = luma(load_scene(base_px + vec2<i32>(-1,  0)));
    let r  = luma(load_scene(base_px + vec2<i32>( 1,  0)));
    let u  = luma(load_scene(base_px + vec2<i32>( 0, -1)));
    let d  = luma(load_scene(base_px + vec2<i32>( 0,  1)));
    let ul = luma(load_scene(base_px + vec2<i32>(-1, -1)));
    let ur = luma(load_scene(base_px + vec2<i32>( 1, -1)));
    let dl = luma(load_scene(base_px + vec2<i32>(-1,  1)));
    let dr = luma(load_scene(base_px + vec2<i32>( 1,  1)));

    let axial = max(abs(r - l), abs(d - u));
    let diagonal = max(abs(dr - ul), abs(dl - ur)) * 0.65;
    let local = max(max(abs(c - l), abs(c - r)), max(abs(c - u), abs(c - d)));
    return smoothstep(0.025, 0.18, max(max(axial, diagonal), local));
}

fn sample_bilinear(src_pos: vec2<f32>, src_size: vec2<f32>) -> vec3<f32> {
    let half_texel = vec2<f32>(0.5) / src_size;
    let uv = clamp((src_pos + vec2<f32>(0.5)) / src_size, half_texel, vec2<f32>(1.0) - half_texel);
    return textureSampleLevel(scene_texture, composite_sampler, uv, 0.0).rgb;
}

fn rcas_sharpen(color: vec3<f32>, base_px: vec2<i32>, amount: f32) -> vec3<f32> {
    let l = load_scene(base_px + vec2<i32>(-1,  0));
    let r = load_scene(base_px + vec2<i32>( 1,  0));
    let u = load_scene(base_px + vec2<i32>( 0, -1));
    let d = load_scene(base_px + vec2<i32>( 0,  1));

    let blur = (l + r + u + d) * 0.25;
    let mn = min(min(l, r), min(u, d));
    let mx = max(max(l, r), max(u, d));
    let sharpened = color + (color - blur) * amount;

    return clamp(sharpened, mn - vec3<f32>(0.035), mx + vec3<f32>(0.035));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(scene_texture));
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));

    if post_process.fsr_enabled < 0.5 {
        return textureSampleLevel(scene_texture, composite_sampler, uv, 0.0);
    }

    let dst_size = max(post_process.output_size, vec2<f32>(1.0));
    let dst_px = in.position.xy - vec2<f32>(0.5);
    let src_pos = (dst_px + vec2<f32>(0.5)) * src_size / dst_size - vec2<f32>(0.5);
    let base_px = vec2<i32>(floor(src_pos));

    let cubic = sample_catmull_rom(src_pos);
    let bilinear = sample_bilinear(src_pos, src_size);
    let edge = edge_strength(base_px);

    var color = mix(cubic, bilinear, edge * 0.32);
    color = rcas_sharpen(color, base_px, post_process.sharpness);

    return vec4<f32>(max(color, vec3<f32>(0.0)), 1.0);
}
