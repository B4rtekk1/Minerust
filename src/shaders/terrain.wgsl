/// Terrain Rendering Shader
///
/// This shader handles the rendering of solid terrain blocks (grass, dirt, stone, etc.)
/// It includes support for:
/// - Texture Array based atlas sampling
/// - Cascaded Shadow Maps (CSM) with 4 cascades for stable, high-quality shadows
/// - Receiver-plane depth bias for accurate shadow edges on sloped surfaces
/// - Time-of-day based lighting (ambient, solar diffuse, secondary fill light)
/// - Biome-aware fog and atmospheric scattering
///

struct Uniforms {
    /// Projection * View matrix for the camera
    view_proj: mat4x4<f32>,
    /// Inverse of Project * View matrix for unprojecting
    inv_view_proj: mat4x4<f32>,
    /// CSM cascade view-projection matrices (4 cascades)
    csm_view_proj: array<mat4x4<f32>, 4>,
    /// View-space split distances for cascade selection
    csm_split_distances: vec4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    /// 1.0 if camera is underwater, 0.0 otherwise
    is_underwater: f32,
    _screen_size: vec2<f32>,
    _water_level: f32,
    _reflection_mode: f32,
    /// Moon direction (normalized) — used for night fog tint
    moon_position: vec3<f32>,
    _pad1_moon: f32,
};


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

/// Array of 2D textures containing block faces
@group(0) @binding(1)
var texture_atlas: texture_2d_array<f32>;
@group(0) @binding(2)
var texture_sampler: sampler;
/// Depth map array generated during the shadow pass (4 cascades)
@group(0) @binding(3)
var shadow_map: texture_depth_2d_array;
@group(0) @binding(4)
var shadow_sampler: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) tex_index: f32,
    @location(5) view_depth: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos = model.position;
    out.normal = model.normal.xyz;
    out.color = model.color.rgb;
    out.uv = model.uv;
    out.tex_index = model.tex_index;
    // Pass view-space depth for cascade selection
    out.view_depth = out.clip_position.w;
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    // Current cascade matrix is passed in view_proj when using dynamic offsets or dedicated buffer
    return uniforms.view_proj * vec4<f32>(model.position, 1.0);
}

const PI: f32 = 3.14159265359;
const SHADOW_MAP_SIZE: f32 = 2048.0;
const PCF_SAMPLES: i32 = 16;

/// Calculate sky color with localized sunrise/sunset gradient
fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_height = sun_dir.y;

    // Time-of-day factors
    let day_factor = clamp(sun_height, 0.0, 1.0);
    let night_factor = clamp(-sun_height, 0.0, 1.0);
    let sunset_factor = 1.0 - abs(sun_height);

    // Vertical gradient
    let view_height = view_dir.y;

    // Angle between view direction and sun direction
    let view_horizontal_vec = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let sun_horizontal_vec = vec3<f32>(sun_dir.x, 0.0, sun_dir.z);

    let v_len = length(view_horizontal_vec);
    let s_len = length(sun_horizontal_vec);

    var cos_angle_horizontal = 0.0;
    if v_len > 0.0001 && s_len > 0.0001 {
        cos_angle_horizontal = dot(view_horizontal_vec / v_len, sun_horizontal_vec / s_len);
    }

    // 3D angle to sun
    let cos_angle_3d = dot(normalize(view_dir), normalize(sun_dir));

    // --- BASE SKY COLORS (Unified) ---
    let zenith_day = vec3<f32>(0.25, 0.45, 0.85);
    let horizon_day = vec3<f32>(0.65, 0.82, 0.98);
    let zenith_night = vec3<f32>(0.001, 0.001, 0.008);
    let horizon_night = vec3<f32>(0.015, 0.015, 0.03);

    let height_factor = clamp(view_height * 0.5 + 0.5, 0.0, 1.0);
    let curved_height = pow(height_factor, 0.8);
    var sky_color = mix(horizon_day, zenith_day, curved_height) * day_factor;
    sky_color += mix(horizon_night, zenith_night, height_factor) * night_factor;

    // --- LOCALIZED SUNSET/SUNRISE EFFECT ---
    if sunset_factor > 0.01 && sun_height > -0.3 {
        let sunset_orange = vec3<f32>(1.0, 0.4, 0.1);
        let sunset_red = vec3<f32>(0.9, 0.2, 0.05);
        let sunset_yellow = vec3<f32>(1.0, 0.7, 0.3);
        let sunset_pink = vec3<f32>(0.95, 0.5, 0.6);

        let sun_proximity_3d = max(0.0, cos_angle_3d);
        let sun_proximity_horiz = max(0.0, cos_angle_horizontal);
        let sun_proximity = mix(sun_proximity_horiz, sun_proximity_3d, 0.5);

        let glow_tight = pow(sun_proximity_3d, 32.0);
        let glow_medium = pow(sun_proximity, 4.0);
        let glow_wide = pow(sun_proximity, 1.5);

        let sunset_intensity = smoothstep(-0.2, 0.1, sun_height) * smoothstep(0.6, 0.0, sun_height);

        let horizon_band = 1.0 - abs(view_height);
        let horizon_boost = pow(horizon_band, 0.5) * smoothstep(0.0, 0.1, v_len);

        var sunset_color = vec3<f32>(0.0);
        sunset_color += sunset_yellow * glow_tight * 1.2;
        sunset_color += sunset_orange * glow_medium * 0.8 * horizon_boost;
        sunset_color += sunset_red * glow_wide * 0.5 * horizon_boost;

        let opposite_glow = max(0.0, -cos_angle_horizontal) * 0.2;
        sunset_color += sunset_pink * opposite_glow * horizon_band * smoothstep(0.0, 0.1, v_len);

        sky_color = mix(sky_color, sky_color + sunset_color, sunset_intensity);
    }

    if day_factor > 0.1 {
        let sun_glow = pow(max(0.0, cos_angle_3d), 128.0) * day_factor;
        sky_color += vec3<f32>(1.0, 0.95, 0.9) * sun_glow;
    }

    return clamp(sky_color, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Receiver-plane depth bias for accurate shadow edges on sloped surfaces.
/// Derivatives must be computed in fs_main (uniform control flow) and passed in —
/// dpdx/dpdy are not allowed inside helper functions per the WGSL spec.
fn receiver_plane_depth_bias(
    dudx: vec2<f32>,
    dudy: vec2<f32>,
    dzdx: f32,
    dzdy: f32,
) -> f32 {
    // Solve for the depth gradient on the receiver plane
    let det = dudx.x * dudy.y - dudx.y * dudy.x;
    if abs(det) < 1e-6 {
        return 0.0;
    }
    let inv_det = 1.0 / det;
    let depth_gradient = vec2<f32>(
        (dzdx * dudy.y - dzdy * dudx.y) * inv_det,
        (dzdy * dudx.x - dzdx * dudy.x) * inv_det
    );

    // Maximum bias based on filter radius
    let texel_size = 1.0 / SHADOW_MAP_SIZE;
    let max_offset = texel_size * 2.0;

    return max_offset * (abs(depth_gradient.x) + abs(depth_gradient.y));
}

/// World-space stable noise — hashes the block cell of world_pos so the PCF
/// rotation stays consistent for a given surface point regardless of camera
/// movement or sun angle changes.  This eliminates temporal shimmer at shadow
/// edges caused by the screen-space rotation pattern shifting every frame.
fn world_space_noise(world_pos: vec3<f32>) -> f32 {
    // Snap to block grid so neighbouring fragments on the same block face share
    // the same rotation angle — removes sub-block variation that would cause
    // aliasing in the noise pattern.
    let cell = floor(world_pos);
    return fract(sin(dot(cell.xz, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn get_poisson_sample(idx: i32, rotation: f32) -> vec2<f32> {
    var p: vec2<f32>;
    switch (idx) {
        case 0: { p = vec2<f32>(-0.94201624, -0.39906216); }
        case 1: { p = vec2<f32>(0.94558609, -0.76890725); }
        case 2: { p = vec2<f32>(-0.094184101, -0.92938870); }
        case 3: { p = vec2<f32>(0.34495938, 0.29387760); }
        case 4: { p = vec2<f32>(-0.91588581, 0.45771432); }
        case 5: { p = vec2<f32>(-0.81544232, -0.87912464); }
        case 6: { p = vec2<f32>(-0.38277543, 0.27676845); }
        case 7: { p = vec2<f32>(0.97484398, 0.75648379); }
        case 8: { p = vec2<f32>(0.44323325, -0.97511554); }
        case 9: { p = vec2<f32>(0.53742981, -0.47373420); }
        case 10: { p = vec2<f32>(-0.65476012, -0.051473853); }
        case 11: { p = vec2<f32>(0.18395645, 0.89721549); }
        case 12: { p = vec2<f32>(-0.097153940, -0.006734560); }
        case 13: { p = vec2<f32>(0.53472400, 0.73356543); }
        case 14: { p = vec2<f32>(-0.45611231, -0.40212851); }
        case 15: { p = vec2<f32>(-0.57321081, 0.65476012); }
        default: { p = vec2<f32>(0.0, 0.0); }
    }
    let s = sin(rotation);
    let c = cos(rotation);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

/// Sample PCF shadow for one specific cascade.
/// Returns shadow factor in [0,1] (0 = fully shadowed, 1 = fully lit).
fn sample_cascade_pcf(
    world_pos: vec3<f32>,
    cascade_idx: i32,
    bias: f32,
    rotation_phi: f32,
    filter_radius: f32,
) -> f32 {
    let shadow_pos    = uniforms.csm_view_proj[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let shadow_coords = shadow_pos.xyz / shadow_pos.w;

    let uv = vec2<f32>(
        shadow_coords.x * 0.5 + 0.5,
        1.0 - (shadow_coords.y * 0.5 + 0.5)
    );

    // Outside frustum → fully lit
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0;
    }

    // Smooth edge fade to prevent hard cutoff at shadow map edges
    let edge_margin = 0.05;
    let edge_fade_x  = smoothstep(0.0, edge_margin, uv.x) * smoothstep(1.0, 1.0 - edge_margin, uv.x);
    let edge_fade_y  = smoothstep(0.0, edge_margin, uv.y) * smoothstep(1.0, 1.0 - edge_margin, uv.y);
    let edge_fade    = edge_fade_x * edge_fade_y;

    let receiver_depth = shadow_coords.z;

    var shadow: f32 = 0.0;
    for (var i: i32 = 0; i < PCF_SAMPLES; i++) {
        let offset = get_poisson_sample(i, rotation_phi) * filter_radius;
        let sample_uv = uv + offset;
        // Taps that fall outside the shadow map frustum are in lit space — return
        // 1.0 instead of clamping to an edge texel which gives wrong comparisons.
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            shadow += 1.0;
        } else {
            shadow += textureSampleCompare(
                shadow_map,
                shadow_sampler,
                sample_uv,
                cascade_idx,
                receiver_depth - bias
            );
        }
    }
    shadow /= f32(PCF_SAMPLES);

    return mix(1.0, shadow, edge_fade);
}

/// Select cascade and compute a blend weight at cascade boundaries.
/// Returns vec2(cascade_index_f32, blend_to_next_cascade [0..1]).
/// Blend zone is 10 % of each cascade's far distance —  eliminates the hard
/// seam that causes shadows to "jump" when a fragment straddles two cascades.
fn select_cascade_with_blend(view_depth: f32) -> vec2<f32> {
    let blend_fraction = 0.10; // size of blend zone relative to split distance

    // Per-cascade split distances from uniforms
    let split0 = uniforms.csm_split_distances.x;
    let split1 = uniforms.csm_split_distances.y;
    let split2 = uniforms.csm_split_distances.z;

    // Cascade 0 → 1
    let blend_start0 = split0 * (1.0 - blend_fraction);
    if view_depth < blend_start0 {
        return vec2<f32>(0.0, 0.0);
    } else if view_depth < split0 {
        let t = (view_depth - blend_start0) / (split0 - blend_start0);
        return vec2<f32>(0.0, smoothstep(0.0, 1.0, t));
    }

    // Cascade 1 → 2
    let blend_start1 = split1 * (1.0 - blend_fraction);
    if view_depth < blend_start1 {
        return vec2<f32>(1.0, 0.0);
    } else if view_depth < split1 {
        let t = (view_depth - blend_start1) / (split1 - blend_start1);
        return vec2<f32>(1.0, smoothstep(0.0, 1.0, t));
    }

    // Cascade 2 → 3
    let blend_start2 = split2 * (1.0 - blend_fraction);
    if view_depth < blend_start2 {
        return vec2<f32>(2.0, 0.0);
    } else if view_depth < split2 {
        let t = (view_depth - blend_start2) / (split2 - blend_start2);
        return vec2<f32>(2.0, smoothstep(0.0, 1.0, t));
    }

    // Cascade 3 — last cascade, no further blending
    return vec2<f32>(3.0, 0.0);
}

/// Percentage Closer Filtering (PCF) shadow calculation with rotated Vogel disk.
/// Uses pseudo-random rotation per pixel to break up banding into high-frequency noise.
/// Cascades are blended at their boundaries to eliminate seam jitter.
/// plane_bias is pre-computed in fs_main from derivatives and passed in.
fn calculate_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    sun_dir: vec3<f32>,
    view_depth: f32,
    plane_bias: f32,
) -> f32 {
    // Disable shadows if sun is below horizon
    if sun_dir.y < 0.05 {
        return 0.0;
    }

    // Adaptive bias based on surface slope
    let cos_theta  = max(dot(normal, sun_dir), 0.0);
    let sin_theta  = sqrt(1.0 - cos_theta * cos_theta);
    // Raised base from 0.001 → 0.003 to cover sub-texel depth variation
    // that survives texel-snapping when the sun rotates.
    let slope_bias = 0.003 + 0.004 * sin_theta / max(cos_theta, 0.1);
    let bias       = slope_bias + plane_bias;

    // Rotated Vogel disk PCF parameters
    // World-space noise ensures the rotation is stable for each surface point
    // regardless of camera position, preventing temporal flicker at shadow edges.
    let texel_size    = 1.0 / SHADOW_MAP_SIZE;
    let noise         = world_space_noise(world_pos);
    let rotation_phi  = noise * 2.0 * PI;
    let filter_radius = 5.0 * texel_size;

    // Cascade selection with smooth blend at boundaries
    let cb          = select_cascade_with_blend(view_depth);
    let cascade_idx = i32(cb.x);
    let blend       = cb.y;

    let shadow_a = sample_cascade_pcf(world_pos, cascade_idx, bias, rotation_phi, filter_radius);

    // Only pay the cost of a second cascade sample when we're in the blend zone
    if blend > 0.001 && cascade_idx < 3 {
        let shadow_b = sample_cascade_pcf(world_pos, cascade_idx + 1, bias, rotation_phi, filter_radius);
        return mix(shadow_a, shadow_b, blend);
    }

    return shadow_a;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // -------------------------------------------------------------------
    // All derivative calls MUST be at the top of fs_main, in uniform
    // control flow. dpdx/dpdy are illegal inside helper functions per the
    // WGSL spec — move them here and pass results as arguments instead.
    // -------------------------------------------------------------------

    // UV derivatives for atlas sampling — computed before fract() to avoid
    // mip selection artifacts at tile boundaries (the UV 1.0→0.0 jump would
    // otherwise produce huge dpdx/dpdy values and force the lowest mip).
    let ddx_uv = dpdx(in.uv);
    let ddy_uv = dpdy(in.uv);

    // Shadow UV + depth derivatives for receiver-plane depth bias.
    // We use cascade 0 as the reference projection; the bias is a small
    // scalar correction that is stable across cascades.
    let shadow_ref   = uniforms.csm_view_proj[0] * vec4<f32>(in.world_pos, 1.0);
    let shadow_uv    = vec2<f32>(
        shadow_ref.x / shadow_ref.w * 0.5 + 0.5,
        1.0 - (shadow_ref.y / shadow_ref.w * 0.5 + 0.5)
    );
    let shadow_depth = shadow_ref.z / shadow_ref.w;

    let dudx = dpdx(shadow_uv);
    let dudy = dpdy(shadow_uv);
    let dzdx = dpdx(shadow_depth);
    let dzdy = dpdy(shadow_depth);

    let plane_bias = receiver_plane_depth_bias(dudx, dudy, dzdx, dzdy);

    // -------------------------------------------------------------------

    let wrapped_uv  = fract(in.uv);
    let tex_sample  = textureSampleGrad(texture_atlas, texture_sampler, wrapped_uv, i32(in.tex_index + 0.5), ddx_uv, ddy_uv);

    // Alpha test (for leaves, etc.)
    if tex_sample.a < 0.5 {
        discard;
    }

    let tex_color = tex_sample.rgb;
    let sun_dir = normalize(uniforms.sun_position);

    // Calculate view direction from camera to this fragment (for localized sky gradient)
    let view_dir = normalize(in.world_pos - uniforms.camera_pos);

    // Horizontal view direction (y=0) used for fog color to avoid sunset bleeding
    // through terrain: fog color should reflect the horizon sky in the horizontal
    // direction toward the fragment, not the actual (possibly upward) view angle.
    let view_dir_horiz_raw = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let view_dir_horiz = select(
        normalize(view_dir_horiz_raw),
        view_dir,
        length(view_dir_horiz_raw) < 0.0001
    );

    // --- LIGHTING MODEL ---

    // Time-of-day factors
    let day_factor = clamp(sun_dir.y, 0.0, 1.0);
    let night_factor = clamp(-sun_dir.y, 0.0, 1.0);
    let sunset_factor = 1.0 - abs(sun_dir.y);
    // Twilight factor - active during sunrise/sunset transition
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);

    // Calculate sky color with localized sunset effect based on horizontal view direction.
    // Using view_dir_horiz (y=0) ensures fog color matches the horizon sky toward the
    // fragment rather than the angled view ray — this prevents the sunset glow from
    // bleeding through terrain when looking at distant mountains during sunset.
    let sky_color = calculate_sky_color(view_dir_horiz, sun_dir);

    // Primary solar shadow — pass pre-computed plane_bias from derivatives above
    var shadow = 1.0;
    if sun_dir.y > 0.0 {
        shadow = calculate_shadow(in.world_pos, in.normal, sun_dir, in.view_depth, plane_bias);
    }

    // Ambient light - add twilight boost during sunrise/sunset
    let ambient_day = 0.4;
    let ambient_night = 0.005;
    let ambient_twilight = 0.25; // Extra ambient during sunrise/sunset
    var ambient = mix(ambient_night, ambient_day, day_factor);
    ambient = max(ambient, ambient_twilight * twilight_factor);

    // Main sun diffuse component
    let sun_diffuse = max(dot(in.normal, sun_dir), 0.0) * 0.5 * shadow * day_factor;

    // Secondary "fill" light (from opposite side) to ground objects
    let fill_dir = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diffuse = max(dot(in.normal, fill_dir), 0.0) * 0.1 * day_factor;

    // Directional shading for block faces (mimic Minecraft look)
    var face_shade = 1.0;
    if abs(in.normal.y) > 0.5 {
        if in.normal.y > 0.0 {
            face_shade = 1.0; // Top
        } else {
            face_shade = 0.5; // Bottom
        }
    } else if abs(in.normal.x) > 0.5 {
        face_shade = 0.7; // X-sides
    } else {
        face_shade = 0.8; // Z-sides
    }

    let effective_face_shade = mix(1.0, face_shade, day_factor + 0.3);

    let lighting_simple = (ambient + sun_diffuse + fill_diffuse) * effective_face_shade;
    var lit_color = tex_color * lighting_simple;

    // Apply sunset tint to lit surfaces
    if sunset_factor > 0.4 && sun_dir.y > -0.1 && sun_dir.y < 0.25 {
        let tint_strength = smoothstep(0.4, 0.7, sunset_factor) * smoothstep(-0.1, 0.05, sun_dir.y) * smoothstep(0.25, 0.05, sun_dir.y);
        let sunset_tint = vec3<f32>(1.0, 0.88, 0.75);
        lit_color = lit_color * mix(vec3<f32>(1.0), sunset_tint, tint_strength * 0.35);
    }

    // --- FOG CALCULATION ---

    let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);

    // Check if underwater
    let is_underwater = uniforms.is_underwater > 0.5;

    // Visibility range depends on time of day and underwater state
    var visibility_range: f32;
    var fog_color: vec3<f32>;

    if is_underwater {
        // Underwater: very short visibility, blue-green tint
        visibility_range = 24.0;
        fog_color = vec3<f32>(0.05, 0.15, 0.3);
    } else {
        let visibility_night = 20.0;  // Much shorter at night - objects should disappear in darkness
        let visibility_day = 250.0;
        let visibility_twilight = 100.0;
        // Better visibility during twilight than pure night
        visibility_range = mix(visibility_night, visibility_day, day_factor);
        visibility_range = max(visibility_range, visibility_twilight * twilight_factor);

        // Fog color: at night, fog must match the night sky exactly to hide silhouettes
        let night_fog_color = vec3<f32>(0.001, 0.001, 0.008);  // Must match zenith_night color
        let twilight_blend = max(day_factor, twilight_factor * 0.7);
        fog_color = mix(night_fog_color, sky_color, twilight_blend);
    }

    let fog_start = visibility_range * 0.2;
    let fog_end = visibility_range;

    let visibility = clamp((fog_end - dist) / (fog_end - fog_start), 0.0, 1.0);

    var final_color = mix(fog_color, lit_color, visibility);

    // Apply underwater color filter
    if is_underwater {
        // Blue-green color shift
        let water_tint = vec3<f32>(0.4, 0.7, 1.0);
        final_color = final_color * water_tint;

        // Add subtle caustic-like brightness variations
        let caustic = sin(in.world_pos.x * 0.5 + uniforms.time * 2.0) * sin(in.world_pos.z * 0.5 + uniforms.time * 1.5) * 0.1 + 0.9;
        final_color = final_color * caustic;

        // Darken with depth (simulate light absorption)
        let depth_factor = clamp(dist / visibility_range, 0.0, 1.0);
        final_color = mix(final_color, fog_color, depth_factor * 0.5);
    }

    return vec4<f32>(final_color, 1.0);
}
