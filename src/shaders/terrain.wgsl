struct Uniforms {
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    camera_pos:     vec3<f32>,
    time:           f32,
    sun_dir:        vec3<f32>,
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

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    texel_size: vec2<f32>,
    _padding: vec2<f32>,
};

@group(1) @binding(0) var sun_shadow_map: texture_depth_2d;
@group(1) @binding(1) var sun_shadow_sampler: sampler_comparison;
@group(1) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;

const TERRAIN_FOG_NEAR: f32 = 180.0;
const TERRAIN_FOG_FAR:  f32 = 620.0;

fn hash31(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

// Rotated Poisson-disk PCF.  The radius is in shadow-map texels, keeping the
// penumbra stable when the player changes monitor resolution.
fn sun_shadow_visibility(world_pos: vec3<f32>, normal: vec3<f32>, sun_dir: vec3<f32>) -> f32 {
    let light_clip = shadow_uniforms.light_view_proj * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / max(light_clip.w, 0.00001);
    let uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 0.5 - light_ndc.y * 0.5);
    if any(uv <= vec2<f32>(0.001)) || any(uv >= vec2<f32>(0.999)) || light_ndc.z <= 0.0 || light_ndc.z >= 1.0 {
        return 1.0;
    }

    // Receiver-plane bias only counteracts depth quantisation.  Casters are
    // not moved, so block shadows stay attached instead of peter-panning.
    let bias = 0.00008 + (1.0 - max(dot(normal, sun_dir), 0.0)) * 0.00016;
    // Keep the kernel fixed in shadow-map space. Rotating it per fragment
    // produces attractive noise in stills but turns into visible crawling as
    // the light matrix changes by fractions of a texel.
    let poisson = array<vec2<f32>, 8>(
        vec2<f32>(-0.613,  0.617), vec2<f32>( 0.171, -0.040),
        vec2<f32>(-0.299, -0.428), vec2<f32>( 0.532,  0.343),
        vec2<f32>(-0.082, -0.814), vec2<f32>( 0.760, -0.582),
        vec2<f32>(-0.742, -0.169), vec2<f32>( 0.337,  0.734),
    );
    var visibility = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        visibility += textureSampleCompareLevel(
            sun_shadow_map,
            sun_shadow_sampler,
            uv + poisson[i] * shadow_uniforms.texel_size * 1.35,
            light_ndc.z - bias,
        );
    }
    return visibility * 0.125;
}

fn horizon_atmosphere(sun_dir: vec3<f32>, day: f32, twilight: f32) -> vec3<f32> {
    let night_haze = vec3<f32>(0.010, 0.016, 0.032);
    let day_haze = vec3<f32>(0.56, 0.70, 0.82);
    let sunset_haze = vec3<f32>(0.92, 0.40, 0.19);
    var haze = mix(night_haze, day_haze, day);
    haze = mix(haze, sunset_haze, twilight * 0.72);

    let overcast = vec3<f32>(0.34, 0.39, 0.45);
    haze = mix(haze, overcast, clamp(uniforms.rain_factor, 0.0, 1.0) * 0.72);
    return haze;
}

fn fast_global_illumination(
    normal:          vec3<f32>,
    sun_dir:         vec3<f32>,
    day_factor:      f32,
    twilight_factor: f32,
) -> vec3<f32> {
    let face_sky_visibility = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    // `sky_visibility` is sampled above the camera on the CPU. Without it,
    // every terrain fragment receives outdoor sky and sunlight, including
    // fragments in a cave below a solid ceiling.
    let area_sky_visibility = clamp(uniforms.sky_visibility, 0.0, 1.0);
    let cave_ambient = mix(0.08, 1.0, area_sky_visibility);
    let ground_visibility = clamp(0.5 - normal.y * 0.5, 0.0, 1.0);
    let side_visibility   = 1.0 - abs(normal.y);

    let sky_day      = vec3<f32>(0.50, 0.62, 0.78);
    let sky_twilight = vec3<f32>(0.96, 0.48, 0.24);
    let sky_night    = vec3<f32>(0.030, 0.030, 0.034);
    let sky_color = mix(
        mix(sky_night, sky_day, day_factor),
        sky_twilight,
        twilight_factor * 0.45,
    );

    let ground_night = vec3<f32>(0.020, 0.018, 0.020);
    let ground_day   = vec3<f32>(0.26, 0.24, 0.19);
    let grass_bleed  = vec3<f32>(0.08, 0.13, 0.06) * day_factor;
    let ground_color = mix(ground_night, ground_day, day_factor) + grass_bleed;

    let sky_energy    = mix(0.075, 0.36, day_factor) + twilight_factor * 0.10;
    let ground_energy = mix(0.024, 0.095, day_factor) + twilight_factor * 0.028;

    let sky_light    = sky_color * sky_energy * (0.35 + 0.65 * face_sky_visibility) * cave_ambient;
    let ground_light = ground_color
        * ground_energy
        * (ground_visibility + side_visibility * 0.35)
        * mix(0.35, 1.0, area_sky_visibility);

    let bounce_dir = normalize(vec3<f32>(-sun_dir.x, 0.32, -sun_dir.z));
    let wrap_bounce = pow(clamp(dot(normal, bounce_dir) * 0.5 + 0.5, 0.0, 1.0), 2.0);
    let sun_bounce = vec3<f32>(1.0, 0.78, 0.48)
        * wrap_bounce
        * side_visibility
        * day_factor
        * area_sky_visibility
        * 0.045;

    // A small exposure floor keeps textures legible in deep shade and caves.
    // It remains colored by the time of day, so night still reads as night.
    let ambient_floor = mix(
        vec3<f32>(0.026, 0.030, 0.040),
        vec3<f32>(0.090, 0.098, 0.108),
        day_factor,
    ) * mix(0.38, 1.0, area_sky_visibility);
    let ambient = sky_light + ground_light + sun_bounce + ambient_floor;
    let ambient_luma = dot(ambient, vec3<f32>(0.299, 0.587, 0.114));
    let night_neutralize = (1.0 - day_factor) * (1.0 - twilight_factor) * 0.72;
    return mix(ambient, vec3<f32>(ambient_luma), vec3<f32>(night_neutralize));
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) color:     vec3<f32>,
    @location(3) uv:        vec2<f32>,
    @location(4) tex_index: f32,
    @location(5) ambient_occlusion: f32,
};

struct DepthVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let n_idx   = model.packed & 0x7u;
    let t_idx   = (model.packed >> 3u) & 0xFFu;
    let uv_idx  = (model.packed >> 11u) & 0x3u;
    let w_raw   = (model.packed >> 13u) & 0xFu;
    let h_raw   = (model.packed >> 17u) & 0xFu;
    let ao_raw  = (model.packed >> 21u) & 0x3u;
    let r       = f32((model.packed >> 23u) & 0x7u) / 7.0;
    let g       = f32((model.packed >> 26u) & 0x7u) / 7.0;
    let b       = f32((model.packed >> 29u) & 0x7u) / 7.0;

    let width  = f32(w_raw + 1u);
    let height = f32(h_raw + 1u);

    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0),
    );

    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos     = model.position;
    out.normal        = normals[n_idx % 6u];
    out.color         = vec3<f32>(r, g, b);

    let raw_uv = uvs[uv_idx % 4u];
    out.uv = vec2<f32>(raw_uv.x * width, raw_uv.y * height);

    out.tex_index = f32(t_idx);
    out.ambient_occlusion = mix(0.52, 1.0, f32(ao_raw) / 3.0);
    return out;
}

@vertex
fn vs_depth(model: VertexInput) -> DepthVertexOutput {
    var out: DepthVertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> DepthVertexOutput {
    var out: DepthVertexOutput;
    out.clip_position = shadow_uniforms.light_view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_dir);
    let day_factor = smoothstep(-0.08, 0.22, sun_dir.y);
    let twilight_factor = (1.0 - smoothstep(0.12, 0.48, abs(sun_dir.y)))
        * smoothstep(-0.16, 0.02, sun_dir.y);
    let normal = in.normal;

    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }

    let indirect_light = fast_global_illumination(
        normal,
        sun_dir,
        day_factor,
        twilight_factor,
    );

    let sun_color = mix(vec3<f32>(1.0, 0.68, 0.38), vec3<f32>(1.0, 0.96, 0.86), day_factor);
    // Direct sunlight and sky fill cannot reach an area with a solid ceiling.
    let sun_exposure = smoothstep(0.0, 0.20, uniforms.sky_visibility);
    let direct_day = smoothstep(-0.04, 0.12, sun_dir.y);
    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.78 * direct_day * sun_exposure;
    let sun_shadow = sun_shadow_visibility(in.world_pos, normal, sun_dir);
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.045 * day_factor * sun_exposure;

    // Cool directional moonlight keeps silhouettes and block faces readable at
    // night without flattening caves with a global ambient term.
    let moon_dir = normalize(uniforms.moon_position);
    let moon_diff = max(dot(normal, moon_dir), 0.0)
        * uniforms.moon_intensity * sun_exposure * 0.22;
    let moon_ambient = vec3<f32>(0.075, 0.092, 0.132)
        * uniforms.moon_intensity * sun_exposure
        * (0.35 + 0.65 * clamp(normal.y * 0.5 + 0.5, 0.0, 1.0));

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let face_contrast = mix(0.82, 1.0, face_shade);
    let ambient_occlusion = in.ambient_occlusion;
    let total_light =
        (indirect_light * ambient_occlusion
            + sun_color * sun_diff * sun_shadow
            + vec3<f32>(0.58, 0.68, 0.82) * fill_diff * mix(0.70, 1.0, ambient_occlusion)
            + vec3<f32>(0.44, 0.57, 0.86) * moon_diff
            + moon_ambient * ambient_occlusion)
        * face_contrast;
    let light_luma = dot(total_light, vec3<f32>(0.299, 0.587, 0.114));
    let tint_luma = dot(in.color, vec3<f32>(0.299, 0.587, 0.114));
    let dark_tint_neutralize = smoothstep(0.16, 0.035, light_luma) * 0.65;
    let local_tint = mix(in.color, vec3<f32>(tint_luma), vec3<f32>(dark_tint_neutralize));
    // A tiny per-block variation breaks up large tiled surfaces while keeping
    // the pixel-art atlas crisp and recognizable.
    let block_cell = floor(in.world_pos - normal * 0.01);
    let material_variation = mix(0.965, 1.035, hash31(block_cell));
    var lit = tex.rgb * total_light * local_tint * material_variation;

    // Rain darkens upward-facing outdoor surfaces and adds a restrained sun
    // glint. It stays disabled in caves through the same sky-exposure signal.
    let rain = clamp(uniforms.rain_factor, 0.0, 1.0);
    let wetness = rain * sun_exposure * smoothstep(0.35, 0.95, normal.y);
    let view_dir = normalize(uniforms.camera_pos - in.world_pos);
    let half_dir = normalize(sun_dir + view_dir + vec3<f32>(0.0, 0.001, 0.0));
    let wet_specular = pow(max(dot(normal, half_dir), 0.0), 48.0)
        * wetness * direct_day * 0.28;
    lit *= mix(1.0, 0.84, wetness);
    lit += sun_color * wet_specular;

    let sunset_factor = 1.0 - abs(sun_dir.y);
    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        lit *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }

    let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);
    let is_underwater = uniforms.is_underwater > 0.5;

    var final_color = lit;

    if is_underwater {
        final_color *= vec3<f32>(0.4, 0.7, 1.0);
        let caustic_a = sin(in.world_pos.x * 0.72 + in.world_pos.z * 0.46 + uniforms.time * 1.8);
        let caustic_b = sin(in.world_pos.x * -0.38 + in.world_pos.z * 0.83 - uniforms.time * 1.35);
        let caustic = 0.88 + pow(abs(caustic_a + caustic_b) * 0.5, 5.0) * 0.24;
        final_color *= caustic;
        final_color = mix(final_color, vec3<f32>(0.05, 0.15, 0.3),
                          clamp(dist / 24.0, 0.0, 1.0) * 0.5);
    } else {
        // Distance haze hides chunk pop-in and visually joins the terrain with
        // the sky. Squaring the factor preserves nearby texture contrast.
        let fog_linear = smoothstep(TERRAIN_FOG_NEAR, TERRAIN_FOG_FAR, dist);
        let height_haze = 0.72 + 0.28 * (1.0 - smoothstep(70.0, 150.0, in.world_pos.y));
        let fog_amount = fog_linear * fog_linear * height_haze * sun_exposure;
        final_color = mix(
            final_color,
            horizon_atmosphere(sun_dir, day_factor, twilight_factor),
            fog_amount * 0.88,
        );
    }

    return vec4<f32>(final_color, 1.0);
}
