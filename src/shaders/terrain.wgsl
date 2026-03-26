struct Uniforms {
    view_proj:           mat4x4<f32>,
    inv_view_proj:       mat4x4<f32>,
    csm_view_proj:       array<mat4x4<f32>, 4>,
    csm_split_distances: vec4<f32>,
    camera_pos:          vec3<f32>,
    time:                f32,
    sun_position:        vec3<f32>,
    is_underwater:       f32,
};

struct ShadowConfig {
    shadow_map_size: f32,
    pcf_samples:     u32,
}

@group(0) @binding(0) var<uniform> uniforms:       Uniforms;
@group(0) @binding(1) var texture_atlas:           texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler:         sampler;
@group(0) @binding(3) var shadow_map:              texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler:          sampler_comparison;
@group(0) @binding(5) var<uniform> shadow_config: ShadowConfig;

// ==================== G-BUFFER (dla compute) ====================

@group(1) @binding(0) var gbuffer_world_pos:  texture_2d<f32>;
@group(1) @binding(1) var gbuffer_normal:     texture_2d<f32>;
@group(1) @binding(2) var gbuffer_view_depth: texture_2d<f32>;

// ==================== WYNIK COMPUTE ====================

@group(2) @binding(0) var output_shadow: texture_storage_2d<r32float, write>;

// ==================== FINALNY FRAGMENT (shadow mask) ====================

@group(3) @binding(0) var shadow_mask:   texture_2d<f32>;
@group(3) @binding(1) var point_sampler: sampler;   // nearest sampler

const PI:               f32 = 3.14159265359;
const MAX_PCF_SAMPLES:  i32 = 16;

// ====================== FUNKCJE POMOCNICZE ======================

fn world_space_noise(world_pos: vec3<f32>) -> f32 {
    return fract(sin(dot(world_pos.xz, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn get_poisson_sample(idx: i32, rotation: f32) -> vec2<f32> {
    var disk = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216), vec2<f32>( 0.94558609, -0.76890725),
        vec2<f32>(-0.09418410, -0.92938870), vec2<f32>( 0.34495938,  0.29387760),
        vec2<f32>(-0.91588581,  0.45771432), vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543,  0.27676845), vec2<f32>( 0.97484398,  0.75648379),
        vec2<f32>( 0.44323325, -0.97511554), vec2<f32>( 0.53742981, -0.47373420),
        vec2<f32>(-0.65476012, -0.05147385), vec2<f32>( 0.18395645,  0.89721549),
        vec2<f32>(-0.09715394, -0.00673456), vec2<f32>( 0.53472400,  0.73356543),
        vec2<f32>(-0.45611231, -0.40212851), vec2<f32>(-0.57321081,  0.65476012),
    );
    let p = disk[idx];
    let s = sin(rotation);
    let c = cos(rotation);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

fn sample_cascade_pcf(
    world_pos:     vec3<f32>,
    cascade_idx:   i32,
    bias:          f32,
    rotation:      f32,
    filter_radius: f32,
    is_last:       bool,
) -> f32 {
    let sp = uniforms.csm_view_proj[cascade_idx] * vec4<f32>(world_pos, 1.0);
    let sc = sp.xyz / sp.w;
    let uv = vec2<f32>(sc.x * 0.5 + 0.5, 1.0 - (sc.y * 0.5 + 0.5));

    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return select(0.0, 1.0, is_last);
    }

    let em = 0.05;
    let edge_fade = smoothstep(0.0, em, uv.x) * smoothstep(1.0, 1.0 - em, uv.x)
                  * smoothstep(0.0, em, uv.y) * smoothstep(1.0, 1.0 - em, uv.y);

    // POPRAWKA #2: ogranicz do MAX_PCF_SAMPLES, żeby nie wyjść poza tablicę Poissona
    let pcf_samples = min(i32(shadow_config.pcf_samples), MAX_PCF_SAMPLES);
    var shadow = 0.0;
    let shadow_dims = vec2<f32>(textureDimensions(shadow_map).xy);

    for (var i = 0; i < pcf_samples; i++) {
        let suv = uv + get_poisson_sample(i, rotation) * filter_radius;
        if any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0)) {
            shadow += select(0.0, 1.0, is_last);
        } else {
            // Compute cannot use comparison sampling, so we perform a manual
            // nearest-depth lookup and compare it explicitly.
            let texel = vec2<i32>(clamp(suv * shadow_dims, vec2<f32>(0.0), shadow_dims - vec2<f32>(1.0)));
            let depth = textureLoad(shadow_map, texel, cascade_idx, 0);
            if sc.z - bias <= depth {
                shadow += 1.0;
            }
        }
    }

    return mix(1.0, shadow / f32(pcf_samples), edge_fade);
}

fn select_cascade_with_blend(view_depth: f32) -> vec2<f32> {
    let bf = 0.10;
    let splits = array<f32, 3>(
        uniforms.csm_split_distances.x,
        uniforms.csm_split_distances.y,
        uniforms.csm_split_distances.z,
    );

    for (var i = 0; i < 3; i++) {
        let blend_start = splits[i] * (1.0 - bf);
        if view_depth < blend_start { return vec2<f32>(f32(i), 0.0); }
        if view_depth < splits[i] {
            let t = (view_depth - blend_start) / (splits[i] - blend_start);
            return vec2<f32>(f32(i), smoothstep(0.0, 1.0, t));
        }
    }
    return vec2<f32>(3.0, 0.0);
}

fn calculate_shadow(
    world_pos:  vec3<f32>,
    normal:     vec3<f32>,
    sun_dir:    vec3<f32>,
    view_depth: f32,
) -> f32 {
    if sun_dir.y < 0.05 { return 0.0; }

    let cos_t = max(dot(normal, sun_dir), 0.0);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));

    let bias = clamp(0.004 + 0.008 * sin_t / max(cos_t, 0.05), 0.004, 0.02);

    let rot = world_space_noise(world_pos) * 2.0 * PI;
    let shadow_map_size = max(shadow_config.shadow_map_size, 1.0);
    let radius = 3.0 / shadow_map_size;

    let cb = select_cascade_with_blend(view_depth);
    let ci = i32(cb.x);

    let shadow_a = sample_cascade_pcf(world_pos, ci, bias, rot, radius, ci == 3);

    if cb.y > 0.001 && ci < 3 {
        let shadow_b = sample_cascade_pcf(world_pos, ci + 1, bias, rot, radius, (ci + 1) == 3);
        return mix(shadow_a, shadow_b, cb.y);
    }
    return shadow_a;
}

// ====================== COMPUTE SHADER ======================

@compute @workgroup_size(8, 8, 1)
fn compute_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tex_size = textureDimensions(gbuffer_world_pos);
    if (gid.x >= tex_size.x || gid.y >= tex_size.y) {
        return;
    }

    // POPRAWKA #4: sprawdzamy alpha G-buffera — sky/brak geometrii daje shadow = 1.0
    let gbuffer_sample = textureLoad(gbuffer_world_pos, gid.xy, 0);
    if gbuffer_sample.a < 0.5 {
        textureStore(output_shadow, gid.xy, vec4<f32>(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let world_pos  = gbuffer_sample.xyz;
    let normal     = normalize(textureLoad(gbuffer_normal,     gid.xy, 0).xyz);
    let view_depth = textureLoad(gbuffer_view_depth, gid.xy, 0).r;

    let sun_dir = normalize(uniforms.sun_position);

    var shadow_factor = 1.0;
    if (sun_dir.y > 0.0) {
        shadow_factor = calculate_shadow(world_pos, normal, sun_dir, view_depth);
    }

    textureStore(output_shadow, gid.xy, vec4<f32>(shadow_factor, 0.0, 0.0, 0.0));
}

// ====================== VERTEX SHADER ======================

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) packed:   u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos:  vec3<f32>,
    @location(1) normal:     vec3<f32>,
    @location(2) color:      vec3<f32>,
    @location(3) uv:         vec2<f32>,
    @location(4) tex_index:  f32,
    @location(5) view_depth: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    let n_idx   = model.packed & 0x7u;
    let t_idx   = (model.packed >> 3u) & 0xFFu;
    let uv_idx  = (model.packed >> 11u) & 0x3u;
    let w_raw   = (model.packed >> 13u) & 0xFu;
    let h_raw   = (model.packed >> 17u) & 0xFu;
    let r       = f32((model.packed >> 21u) & 0xFu) / 15.0;
    let g       = f32((model.packed >> 25u) & 0xFu) / 15.0;
    let b       = f32((model.packed >> 29u) & 0x7u) / 7.0;

    let width  = f32(w_raw + 1u);
    let height = f32(h_raw + 1u);

    let normals = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), // -X, +X
        vec3<f32>(0.0, -1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), // -Y, +Y
        vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 0.0, 1.0)  // -Z, +Z
    );

    let uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );

    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos     = model.position;
    out.normal        = normals[n_idx % 6u];
    out.color         = vec3<f32>(r, g, b);

    // Apply greedy meshing UV scaling
    let raw_uv = uvs[uv_idx % 4u];
    out.uv = vec2<f32>(raw_uv.x * width, raw_uv.y * height);

    out.tex_index     = f32(t_idx);
    out.view_depth    = out.clip_position.w;
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(model.position, 1.0);
}

// ====================== FRAGMENT SHADER (lekki) ======================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }

    // Pobierz shadow z compute shadera
    let shadow_tex_size = vec2<f32>(textureDimensions(shadow_mask));
    let screen_uv = in.clip_position.xy / shadow_tex_size;
    let shadow = textureSampleLevel(shadow_mask, point_sampler, screen_uv, 0.0).r;

    let sun_dir = normalize(uniforms.sun_position);

    let day_factor      = clamp(sun_dir.y, 0.0, 1.0);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);

    let ambient = max(
        mix(0.005, 0.38, day_factor),
        0.18 * twilight_factor,
    );

    // POPRAWKA #5: normalizujemy in.normal przed dot product,
    // bo interpolacja między wierzchołkami może rozciągnąć wektor.
    let normal = normalize(in.normal);

    let sun_diff  = max(dot(normal, sun_dir), 0.0) * 0.55 * shadow * day_factor;
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.08 * day_factor;

    var face_shade: f32;
    if      abs(normal.y) > 0.5 { face_shade = select(0.5, 1.0, normal.y > 0.0); }
    else if abs(normal.x) > 0.5 { face_shade = 0.7; }
    else                        { face_shade = 0.8; }

    let total_light = (ambient + sun_diff + fill_diff) * face_shade;
    var lit = tex.rgb * total_light;

    // Sunset tint
    let sunset_factor = 1.0 - abs(sun_dir.y);
    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        lit *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }

    // Underwater + caustic + fog
    let dist = length(in.world_pos.xz - uniforms.camera_pos.xz);
    let is_underwater = uniforms.is_underwater > 0.5;

    var final_color = lit;

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
