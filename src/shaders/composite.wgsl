struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_position:   vec3<f32>,
    is_underwater:  f32,
    screen_size:    vec2<f32>,
    water_level:    f32,
    _pad_water:     f32,
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

// Quarter-resolution ray-marched light shafts written by god_rays.wgsl.
@group(0) @binding(3)
var god_rays_texture: texture_2d<f32>;

// Full-resolution linear depth used to keep the low-resolution shafts inside
// the same foreground/background surfaces during upsampling.
@group(0) @binding(4)
var scene_depth: texture_2d<f32>;

const GOD_RAYS_MAX_DISTANCE: f32 = 96.0;
const GOD_RAYS_DEPTH_EXP2_SCALE: f32 = 34.617359;

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

fn world_from_ndc(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let world_h = uniforms.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return world_h.xyz / select(world_h.w, 1e-5, abs(world_h.w) < 1e-5);
}

fn scene_view_distance(uv: vec2<f32>) -> f32 {
    let size = textureDimensions(scene_depth);
    let pixel = min(vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size)), size - vec2<u32>(1u));
    let depth = textureLoad(scene_depth, vec2<i32>(pixel), 0).r;
    if depth >= 0.9999 {
        return GOD_RAYS_MAX_DISTANCE;
    }
    let pixel_uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(size);
    let ndc = vec2<f32>(pixel_uv.x * 2.0 - 1.0, 1.0 - pixel_uv.y * 2.0);
    return min(length(world_from_ndc(ndc, depth) - uniforms.camera_pos), GOD_RAYS_MAX_DISTANCE);
}

fn load_god_rays(pixel: vec2<i32>, size: vec2<u32>) -> vec4<f32> {
    let clamped_pixel = clamp(pixel, vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    return textureLoad(god_rays_texture, clamped_pixel, 0);
}

// Wider bilateral upsample smooths quarter-resolution march noise while
// keeping shafts from bleeding across depth discontinuities such as leaves.
fn god_rays(uv: vec2<f32>, blur: f32) -> vec3<f32> {
    if uniforms.is_underwater > 0.5 || blur >= 0.999 {
        return vec3<f32>(0.0);
    }

    let sun_dir = normalize(uniforms.sun_position);
    let use_moon = sun_dir.y <= 0.02;
    let light_dir = select(sun_dir, normalize(uniforms.moon_position), use_moon);
    if light_dir.y <= 0.02 {
        return vec3<f32>(0.0);
    }
    let sun_strength = smoothstep(0.02, 0.12, sun_dir.y)
        * mix(1.0, 0.75, smoothstep(0.35, 0.9, sun_dir.y));
    let moon_strength = clamp(uniforms.moon_intensity, 0.0, 1.0) * 0.25;
    let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
    let light_strength = select(sun_strength, moon_strength, use_moon) * (1.0 - 0.65 * rain);
    if light_strength <= 0.001 {
        return vec3<f32>(0.0);
    }

    let size_u = textureDimensions(god_rays_texture);
    let ray_coord = uv * vec2<f32>(size_u) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(ray_coord));
    let blend = fract(ray_coord);
    let spatial_weights_x = vec4<f32>(
        (1.0 - blend.x) * 0.5,
        1.0 - blend.x * 0.5,
        0.5 + blend.x * 0.5,
        blend.x * 0.5,
    );
    let spatial_weights_y = vec4<f32>(
        (1.0 - blend.y) * 0.5,
        1.0 - blend.y * 0.5,
        0.5 + blend.y * 0.5,
        blend.y * 0.5,
    );
    let target_depth = scene_view_distance(uv);
    var weighted_rays = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var y = -1; y <= 2; y++) {
        let spatial_weight_y = spatial_weights_y[u32(y + 1)];
        for (var x = -1; x <= 2; x++) {
            let pixel = base + vec2<i32>(x, y);
            let ray = load_god_rays(pixel, size_u);
            let spatial_weight_x = spatial_weights_x[u32(x + 1)];
            let depth_inv_scale = 1.0 / max(max(ray.a, target_depth), 1.0);
            let depth_weight = exp2(-abs(ray.a - target_depth) * depth_inv_scale * GOD_RAYS_DEPTH_EXP2_SCALE);
            let weight = spatial_weight_x * spatial_weight_y * depth_weight;
            weighted_rays += ray.rgb * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 1e-4 {
        return weighted_rays / weight_sum * (1.0 - blur);
    }

    let ray_00 = load_god_rays(base, size_u).rgb;
    let ray_10 = load_god_rays(base + vec2<i32>(1, 0), size_u).rgb;
    let ray_01 = load_god_rays(base + vec2<i32>(0, 1), size_u).rgb;
    let ray_11 = load_god_rays(base + vec2<i32>(1, 1), size_u).rgb;
    let bilinear_rays = ray_00 * (1.0 - blend.x) * (1.0 - blend.y)
        + ray_10 * blend.x * (1.0 - blend.y)
        + ray_01 * (1.0 - blend.x) * blend.y
        + ray_11 * blend.x * blend.y;
    return bilinear_rays * (1.0 - blur);
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

    color += god_rays(uv, blur);
    color = color_grade(color, uv, blur);

    // One-LSB triangular-ish dither prevents visible banding in the smooth sky
    // gradient after conversion to the 8-bit swap-chain target.
    let pixel = floor(uv * max(uniforms.screen_size, vec2<f32>(1.0)));
    let dither = (screen_hash(pixel) - 0.5) / 255.0;
    color = clamp(color + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0));

    return vec4<f32>(color, 1.0);
}
