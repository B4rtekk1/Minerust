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
    fog_far:        f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

fn fast_global_illumination(
    normal:          vec3<f32>,
    sun_dir:         vec3<f32>,
    day_factor:      f32,
    twilight_factor: f32,
    sky_exposure:    f32,
) -> vec3<f32> {
    let face_sky_visibility = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    // `sky_exposure` is this surface's own world-space sky light, so caves,
    // overhangs and forest floors are dark no matter where the camera is.
    let area_sky_visibility = sky_exposure;
    let cave_ambient = mix(0.06, 1.0, area_sky_visibility);
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

    let sky_energy    = mix(0.035, 0.27, day_factor) + twilight_factor * 0.08;
    let ground_energy = mix(0.010, 0.070, day_factor) + twilight_factor * 0.020;

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

    let ambient = sky_light + ground_light + sun_bounce;
    let ambient_luma = dot(ambient, vec3<f32>(0.299, 0.587, 0.114));
    let night_neutralize = (1.0 - day_factor) * (1.0 - twilight_factor) * 0.72;
    return mix(ambient, vec3<f32>(ambient_luma), vec3<f32>(night_neutralize));
}

struct PackedQuad { origin_and_face: u32, size_material_ao: u32, light: u32, }
struct SubchunkMeta { aabb_min: vec4<f32>, aabb_max: vec4<f32>, terrain_draw_data: vec4<u32>, water_draw_data: vec4<u32>, }
@group(1) @binding(0) var<storage, read> quads: array<PackedQuad>;
@group(1) @binding(1) var<storage, read> subchunks: array<SubchunkMeta>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) color:     vec3<f32>,
    @location(3) uv:        vec2<f32>,
    @location(4) tex_index: f32,
    @location(5) ambient_occlusion: f32,
    @location(6) @interpolate(flat) tint: u32,
    /// Interpolated world-space sky light, 0 (none) .. 1 (open sky).
    @location(7) sky_light: f32,
};

struct DepthVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Retained for dynamic/player models, which still use the shared Vertex buffer.
struct LegacyVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed: u32,
};

@vertex
fn vs_legacy(model: LegacyVertexInput) -> VertexOutput {
    let n_idx = model.packed & 0x7u;
    let t_idx = (model.packed >> 3u) & 0xffu;
    let uv_idx = (model.packed >> 11u) & 0x3u;
    let ao_raw = (model.packed >> 21u) & 0x3u;
    let width = f32(((model.packed >> 13u) & 0xfu) + 1u);
    let height = f32(((model.packed >> 17u) & 0xfu) + 1u);
    let color = vec3<f32>(
        f32((model.packed >> 23u) & 0x7u) / 7.0,
        f32((model.packed >> 26u) & 0x7u) / 7.0,
        f32((model.packed >> 29u) & 0x7u) / 7.0,
    );
    let normals = array<vec3<f32>, 6>(
        vec3(-1., 0., 0.), vec3(1., 0., 0.), vec3(0., -1., 0.),
        vec3(0., 1., 0.), vec3(0., 0., -1.), vec3(0., 0., 1.),
    );
    let uvs = array<vec2<f32>, 4>(vec2(0., 0.), vec2(0., 1.), vec2(1., 1.), vec2(1., 0.));
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4(model.position, 1.0);
    out.world_pos = model.position;
    out.normal = normals[n_idx % 6u];
    out.color = color;
    out.uv = uvs[uv_idx] * vec2(width, height);
    out.tex_index = f32(t_idx);
    out.ambient_occlusion = mix(0.52, 1.0, f32(ao_raw) / 3.0);
    out.tint = 0u;
    // Dynamic models (players, dropped items) sit next to the camera, so
    // the camera-area estimate is a fair stand-in for their sky light.
    out.sky_light = clamp(uniforms.sky_visibility, 0.0, 1.0);
    return out;
}

fn pulled_vertex(vertex_id: u32, subchunk_id: u32) -> VertexOutput {
    let quad = quads[vertex_id / 6u];
    let corner_id = vertex_id % 6u;
    let diagonal = (quad.origin_and_face >> 30u) & 0x1u;
    let regular = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u)[corner_id];
    let alternate = array<u32, 6>(0u, 1u, 3u, 1u, 2u, 3u)[corner_id];
    let corner = select(regular, alternate, diagonal == 1u);
    let n_idx = (quad.origin_and_face >> 18u) & 0x7u;
    let t_idx = (quad.size_material_ao >> 12u) & 0xffu;
    let width = f32(quad.size_material_ao & 0x3fu) * 0.5;
    let height = f32((quad.size_material_ao >> 6u) & 0x3fu) * 0.5;
    let ao_raw = (quad.size_material_ao >> (20u + corner * 2u)) & 0x3u;
    let color_bits = (quad.origin_and_face >> 21u) & 0x1ffu;
    let r = f32(color_bits & 0x7u) / 7.0;
    let g = f32((color_bits >> 3u) & 0x7u) / 7.0;
    let b = f32((color_bits >> 6u) & 0x7u) / 7.0;
    let local_origin = vec3<f32>(
        f32(quad.origin_and_face & 0x3fu), f32((quad.origin_and_face >> 6u) & 0x3fu), f32((quad.origin_and_face >> 12u) & 0x3fu),
    ) * 0.5;

    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0),
    );

    let edge_u = array<vec3<f32>, 6>(vec3(0.,0.,1.), vec3(0.,0.,-1.), vec3(0.,0.,-1.), vec3(0.,0.,1.), vec3(-1.,0.,0.), vec3(1.,0.,0.))[n_idx];
    let edge_v = array<vec3<f32>, 6>(vec3(0.,1.,0.), vec3(0.,1.,0.), vec3(1.,0.,0.), vec3(1.,0.,0.), vec3(0.,1.,0.), vec3(0.,1.,0.))[n_idx];
    let position_corners = array<vec2<f32>, 4>(vec2(0.,0.), vec2(1.,0.), vec2(1.,1.), vec2(0.,1.));
    let texture_corners = array<vec2<f32>, 4>(vec2(0.,1.), vec2(1.,1.), vec2(1.,0.), vec2(0.,0.));
    let local_position = local_origin + edge_u * (position_corners[corner].x * width) + edge_v * (position_corners[corner].y * height);
    let world_position = subchunks[subchunk_id].aabb_min.xyz + local_position;

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos     = world_position;
    out.normal        = normals[n_idx % 6u];
    out.color         = vec3<f32>(r, g, b);

    out.uv = texture_corners[corner] * vec2(width, height);

    out.tex_index = f32(t_idx);
    out.ambient_occlusion = mix(0.52, 1.0, f32(ao_raw) / 3.0);
    out.tint = quad.size_material_ao >> 28u;
    out.sky_light = f32((quad.light >> (corner * 4u)) & 0xfu) / 15.0;
    return out;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> VertexOutput {
    return pulled_vertex(vertex_id, subchunk_id);
}

@vertex
fn vs_depth(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> DepthVertexOutput {
    var out: DepthVertexOutput;
    out.clip_position = pulled_vertex(vertex_id, subchunk_id).clip_position;
    return out;
}

// ── Climate tint ──────────────────────────────────────────────────────────── //
// Grass target colours (sRGB) indexed by `tint - 1`; rows frozen → hot,
// columns dry → wet. Mirrors `GRASS_TINTS` in src/core/biome.rs.
const GRASS_TINTS = array<vec3<f32>, 15>(
    vec3<f32>(0.58, 0.67, 0.60), vec3<f32>(0.52, 0.66, 0.60), vec3<f32>(0.47, 0.64, 0.60),
    vec3<f32>(0.50, 0.64, 0.40), vec3<f32>(0.40, 0.62, 0.38), vec3<f32>(0.33, 0.58, 0.40),
    vec3<f32>(0.62, 0.76, 0.32), vec3<f32>(0.47, 0.75, 0.29), vec3<f32>(0.32, 0.66, 0.27),
    vec3<f32>(0.72, 0.75, 0.30), vec3<f32>(0.54, 0.78, 0.26), vec3<f32>(0.28, 0.72, 0.22),
    vec3<f32>(0.82, 0.72, 0.36), vec3<f32>(0.66, 0.75, 0.28), vec3<f32>(0.22, 0.74, 0.18),
);
const LEAVES_TINT_SCALE = vec3<f32>(0.78, 0.86, 0.80);
// Linear luminance of the untinted atlas layers (grass top, grass side
// fringe, leaves), so the recoloured texel keeps the original brightness.
const GRASS_TOP_LUMA:  f32 = 0.2623;
const GRASS_SIDE_LUMA: f32 = 0.2303;
const LEAVES_LUMA:     f32 = 0.1359;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Recolours grass and leaves texels to the column's climate colour while
/// keeping the texture's per-texel brightness detail.
fn apply_climate_tint(color: vec3<f32>, tex_index: u32, tint: u32) -> vec3<f32> {
    if tint == 0u || tint > 15u {
        return color;
    }
    var tint_target = srgb_to_linear(GRASS_TINTS[tint - 1u]);
    var base_luma = GRASS_TOP_LUMA;
    if tex_index == 8u {
        tint_target = srgb_to_linear(GRASS_TINTS[tint - 1u] * LEAVES_TINT_SCALE);
        base_luma = LEAVES_LUMA;
    } else if tex_index == 1u {
        // Only the green fringe of the grass side; the dirt stays untouched.
        let greenness = color.g - max(color.r, color.b);
        if greenness < 0.04 {
            return color;
        }
        base_luma = GRASS_SIDE_LUMA;
    } else if tex_index != 0u {
        return color;
    }
    // An average texel maps exactly onto the target colour; brighter and
    // darker texels keep their relative contrast.
    return tint_target * (luminance(color) / base_luma);
}

// ── Aerial perspective ────────────────────────────────────────────────────── //
// Distant terrain fades towards the horizon sky colour. A gentle exponential
// term gives depth at mid range; a steeper ramp near `fog_far` hides the edge
// of the loaded world, so chunks no longer pop in against a blank sky.
fn horizon_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let day   = smoothstep(-0.02, 0.22, sun_dir.y);
    let dusk  = 1.0 - smoothstep(0.02, 0.34, abs(sun_dir.y));
    let night = clamp(-sun_dir.y * 4.0 - 0.05, 0.0, 1.0);
    var color = vec3<f32>(0.70, 0.84, 0.98) * day
        + vec3<f32>(0.010, 0.014, 0.034) * night
        + vec3<f32>(0.96, 0.52, 0.30) * dusk * 0.55;
    // Forward scattering: haze glows warmer when looking towards the sun.
    let towards_sun = max(dot(normalize(vec3<f32>(view_dir.x, 0.0, view_dir.z)),
                              normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z) + vec3<f32>(1e-4))), 0.0);
    color += vec3<f32>(1.0, 0.72, 0.42) * pow(towards_sun, 6.0) * (0.10 * day + 0.35 * dusk);
    return color;
}

fn aerial_perspective(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let to_frag = world_pos - uniforms.camera_pos;
    let dist = length(to_frag.xz);
    let fog_far = max(uniforms.fog_far, 64.0);
    let aerial = (1.0 - exp(-dist / (fog_far * 1.6))) * 0.55;
    let edge = smoothstep(fog_far * 0.55, fog_far * 0.96, dist);
    let amount = max(aerial, edge) * mix(0.25, 1.0, clamp(uniforms.sky_visibility, 0.0, 1.0));
    let haze = horizon_color(to_frag / max(length(to_frag), 1e-3), uniforms.sun_dir);
    return mix(color, haze, amount);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = uniforms.sun_dir;
    let day_factor = clamp(sun_dir.y, 0.0, 1.0);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);
    let normal = in.normal;

    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }
    let albedo = apply_climate_tint(tex.rgb, u32(in.tex_index + 0.5), in.tint);

    // Light levels fall off exponentially, like perceived brightness: one
    // block into a cave is noticeably dimmer, ten blocks in is dark.
    let sky_level = clamp(in.sky_light, 0.0, 1.0);
    let sky_exposure = pow(0.80, (1.0 - sky_level) * 15.0);
    let indirect_light = fast_global_illumination(
        normal,
        sun_dir,
        day_factor,
        twilight_factor,
        sky_exposure,
    );

    let sun_color = mix(vec3<f32>(1.0, 0.78, 0.52), vec3<f32>(1.0, 0.96, 0.86), day_factor);
    // Direct sunlight only reaches surfaces that are (nearly) open to the sky.
    let sun_exposure = smoothstep(0.35, 0.95, sky_exposure);
    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.62 * day_factor * sun_exposure;
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.045 * day_factor * sun_exposure;

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let face_contrast = mix(0.82, 1.0, face_shade);
    let ambient_occlusion = in.ambient_occlusion;
    let total_light =
        (indirect_light * ambient_occlusion
            + sun_color * sun_diff
            + vec3<f32>(0.58, 0.68, 0.82) * fill_diff * mix(0.70, 1.0, ambient_occlusion))
        * face_contrast;
    let light_luma = dot(total_light, vec3<f32>(0.299, 0.587, 0.114));
    let tint_luma = dot(in.color, vec3<f32>(0.299, 0.587, 0.114));
    let dark_tint_neutralize = smoothstep(0.16, 0.035, light_luma) * 0.65;
    let local_tint = mix(in.color, vec3<f32>(tint_luma), vec3<f32>(dark_tint_neutralize));
    var lit = albedo * total_light * local_tint;

    let sunset_factor = 1.0 - abs(sun_dir.y);
    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        lit *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }

    let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);
    let is_underwater = uniforms.is_underwater > 0.5;

    var final_color = lit;
    if !is_underwater {
        final_color = aerial_perspective(final_color, in.world_pos);
    }

    if is_underwater {
        final_color *= vec3<f32>(0.4, 0.7, 1.0);
        let caustic = sin(in.world_pos.x * 0.5 + uniforms.time * 2.0)
                    * sin(in.world_pos.z * 0.5 + uniforms.time * 1.5) * 0.1 + 0.9;
        final_color *= caustic;
        final_color = mix(final_color, vec3<f32>(0.05, 0.15, 0.3),
                          clamp(dist / 24.0, 0.0, 1.0) * 0.5);
    }

    return vec4<f32>(final_color, 1.0);
}
