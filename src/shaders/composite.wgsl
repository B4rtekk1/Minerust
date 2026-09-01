struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    reflection_mode: f32,
    moon_position:  vec3<f32>,
    _pad1_moon:     f32,
    moon_intensity: f32,
    wind_dir_x:     f32,
    wind_dir_z:     f32,
    wind_speed:     f32,
    rain_factor:    f32,
    sky_visibility: f32,
    menu_blur:      f32,
    _pad_uniforms:  f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var scene_texture: texture_2d<f32>;

@group(0) @binding(2)
var composite_sampler: sampler;

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

fn sample_scene(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(scene_texture, composite_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

fn cover_uv(uv: vec2<f32>, blur: f32) -> vec2<f32> {
    if blur <= 0.001 {
        return uv;
    }

    let texture_size_u = textureDimensions(scene_texture);
    let texture_size = vec2<f32>(f32(texture_size_u.x), f32(texture_size_u.y));
    let screen_aspect = uniforms.screen_size.x / max(uniforms.screen_size.y, 1.0);
    let texture_aspect = texture_size.x / max(texture_size.y, 1.0);
    var covered = uv;

    if screen_aspect > texture_aspect {
        covered.y = 0.5 + (uv.y - 0.5) * (texture_aspect / screen_aspect);
    } else {
        covered.x = 0.5 + (uv.x - 0.5) * (screen_aspect / texture_aspect);
    }

    return covered;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn bright_pass(color: vec3<f32>) -> vec3<f32> {
    let weight = smoothstep(0.68, 0.96, luminance(color));
    return color * weight;
}

fn screen_hash(pixel: vec2<f32>) -> f32 {
    return fract(sin(dot(pixel, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn color_grade(color_in: vec3<f32>, uv: vec2<f32>, blur: f32) -> vec3<f32> {
    var color = max(color_in, vec3<f32>(0.0));
    let sun_height = normalize(uniforms.sun_position).y;
    let daylight = smoothstep(-0.10, 0.28, sun_height);
    let world_exposure = mix(1.48, 1.16, daylight);
    color *= mix(world_exposure, 1.0, blur);
    let luma = luminance(color);

    // Preserve the atlas palette, but separate adjacent shades and gently
    // balance cool shadows against warm sunlight.
    let saturation = mix(1.075, 0.90, blur);
    color = mix(vec3<f32>(luma), color, saturation);
    color = (color - vec3<f32>(0.5)) * mix(1.025, 0.94, blur) + vec3<f32>(0.5);

    let shadow_weight = 1.0 - smoothstep(0.06, 0.42, luma);
    let highlight_weight = smoothstep(0.58, 0.96, luma);
    color *= mix(vec3<f32>(1.0), vec3<f32>(0.96, 0.985, 1.035), shadow_weight * 0.22);
    color *= mix(vec3<f32>(1.0), vec3<f32>(1.025, 1.005, 0.965), highlight_weight * 0.18);
    color += vec3<f32>(0.014, 0.016, 0.020) * shadow_weight * (1.0 - blur);

    // A restrained vignette draws the eye without making the corners visibly
    // black. It is softened further while the menu blur is active.
    var centered = uv * 2.0 - 1.0;
    centered.x *= uniforms.screen_size.x / max(uniforms.screen_size.y, 1.0);
    let vignette = 1.0 - smoothstep(0.42, 1.82, dot(centered, centered)) * mix(0.045, 0.025, blur);
    return clamp(color * vignette, vec3<f32>(0.0), vec3<f32>(1.0));
}

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let blur = clamp(uniforms.menu_blur, 0.0, 1.0);
    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let sample_uv = cover_uv(uv, blur);
    let texture_size_u = textureDimensions(scene_texture);
    let texture_size = vec2<f32>(f32(texture_size_u.x), f32(texture_size_u.y));
    let texel = vec2<f32>(1.0, 1.0) / max(texture_size, vec2<f32>(1.0, 1.0));

    var color = sample_scene(sample_uv);

    if uniforms.is_underwater > 0.5 && blur < 0.5 {
        // Very small wavelength separation at the edge of the lens sells the
        // underwater refraction without making the image difficult to read.
        let radial = (sample_uv - vec2<f32>(0.5)) * 1.6;
        let shift = radial * texel * (1.2 + 0.25 * sin(uniforms.time * 0.8));
        color.r = sample_scene(sample_uv + shift).r;
        color.b = sample_scene(sample_uv - shift).b;
        color = mix(color, color * vec3<f32>(0.83, 0.96, 1.08), 0.28);
    }

    // Compact cross-shaped bloom. The source target is LDR, so a soft
    // luminance threshold is used instead of relying on HDR values above 1.0.
    let bloom_radius = texel * 3.25;
    var bloom = bright_pass(color) * 0.28;
    bloom += bright_pass(sample_scene(sample_uv + vec2<f32>( bloom_radius.x, 0.0))) * 0.18;
    bloom += bright_pass(sample_scene(sample_uv + vec2<f32>(-bloom_radius.x, 0.0))) * 0.18;
    bloom += bright_pass(sample_scene(sample_uv + vec2<f32>(0.0,  bloom_radius.y))) * 0.18;
    bloom += bright_pass(sample_scene(sample_uv + vec2<f32>(0.0, -bloom_radius.y))) * 0.18;
    color += bloom * mix(0.105, 0.025, blur);

    if blur > 0.001 {
        var blurred = color * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 4.0,  0.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-4.0,  0.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0,  4.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0, -4.0)) * 0.08;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 5.5,  5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-5.5,  5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 5.5, -5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-5.5, -5.5)) * 0.07;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 10.0,  0.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-10.0,  0.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0,  10.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 0.0, -10.0)) * 0.05;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 14.0,  14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-14.0,  14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>( 14.0, -14.0)) * 0.03;
        blurred += sample_scene(sample_uv + texel * vec2<f32>(-14.0, -14.0)) * 0.03;

        color = mix(color, blurred * 0.82, blur);
    }

    color = color_grade(color, uv, blur);

    // One-LSB triangular-ish dither prevents visible banding in the smooth sky
    // gradient after conversion to the 8-bit swap-chain target.
    let pixel = floor(uv * max(uniforms.screen_size, vec2<f32>(1.0)));
    let dither = (screen_hash(pixel) - 0.5) / 255.0;
    color = clamp(color + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}
