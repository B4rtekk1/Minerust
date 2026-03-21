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

@group(0) @binding(0) var<uniform> uniforms:       Uniforms;
@group(0) @binding(1) var texture_atlas:           texture_2d_array<f32>;
@group(0) @binding(2) var texture_sampler:         sampler;
@group(0) @binding(3) var shadow_map:              texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler:          sampler_comparison;


struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) color:     vec4<f32>,
    @location(3) uv:        vec2<f32>,
    @location(4) tex_index: f32,
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
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.world_pos  = model.position;
    out.normal     = model.normal.xyz;
    out.color      = model.color.rgb;
    out.uv         = model.uv;
    out.tex_index  = model.tex_index;
    out.view_depth = out.clip_position.w;
    return out;
}

@vertex
fn vs_shadow(model: VertexInput) -> @builtin(position) vec4<f32> {
    return uniforms.view_proj * vec4<f32>(model.position, 1.0);
}


const PI:              f32 = 3.14159265359;
const SHADOW_MAP_SIZE: f32 = 2048.0;
const PCF_SAMPLES:     i32 = 16;


fn calculate_sky_color(view_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_height    = sun_dir.y;
    let day_factor    = clamp( sun_height, 0.0, 1.0);
    let night_factor  = clamp(-sun_height, 0.0, 1.0);
    let sunset_factor = 1.0 - abs(sun_height);
    let view_height   = view_dir.y;

    let view_h = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let sun_h  = vec3<f32>(sun_dir.x,  0.0, sun_dir.z);
    let v_len  = length(view_h);
    let s_len  = length(sun_h);
    var cos_horiz = 0.0;
    if v_len > 0.0001 && s_len > 0.0001 {
        cos_horiz = dot(view_h / v_len, sun_h / s_len);
    }
    let cos_3d = dot(normalize(view_dir), normalize(sun_dir));

    let curved_height = pow(clamp(view_height * 0.5 + 0.5, 0.0, 1.0), 0.8);
    var sky = mix(vec3<f32>(0.65, 0.82, 0.98), vec3<f32>(0.25, 0.45, 0.85), curved_height) * day_factor
            + mix(vec3<f32>(0.015, 0.015, 0.03), vec3<f32>(0.001, 0.001, 0.008),
                  clamp(view_height * 0.5 + 0.5, 0.0, 1.0)) * night_factor;

    if sunset_factor > 0.01 && sun_height > -0.3 {
        let prox      = mix(max(0.0, cos_horiz), max(0.0, cos_3d), 0.5);
        let horiz_b   = pow(1.0 - abs(view_height), 0.5) * smoothstep(0.0, 0.1, v_len);
        let intensity = smoothstep(-0.2, 0.1, sun_height) * smoothstep(0.6, 0.0, sun_height);

        var glow  = vec3<f32>(1.0, 0.7, 0.3)  * pow(max(0.0, cos_3d), 32.0) * 1.2;
        glow     += vec3<f32>(1.0, 0.4, 0.1)  * pow(prox, 4.0) * 0.8 * horiz_b;
        glow     += vec3<f32>(0.9, 0.2, 0.05) * pow(prox, 1.5) * 0.5 * horiz_b;
        glow     += vec3<f32>(0.95, 0.5, 0.6) * max(0.0, -cos_horiz) * 0.2
                    * (1.0 - abs(view_height)) * smoothstep(0.0, 0.1, v_len);

        sky = mix(sky, sky + glow, intensity);
    }

    if day_factor > 0.1 {
        sky += vec3<f32>(1.0, 0.95, 0.9) * pow(max(0.0, cos_3d), 128.0) * day_factor;
    }

    return clamp(sky, vec3<f32>(0.0), vec3<f32>(1.0));
}


fn world_space_noise(world_pos: vec3<f32>) -> f32 {
    let cell = floor(world_pos);
    return fract(sin(dot(cell.xz, vec2<f32>(127.1, 311.7))) * 43758.5453);
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
    let s = sin(rotation); let c = cos(rotation);
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

    let em        = 0.05;
    let edge_fade = smoothstep(0.0, em, uv.x) * smoothstep(1.0, 1.0 - em, uv.x)
                  * smoothstep(0.0, em, uv.y) * smoothstep(1.0, 1.0 - em, uv.y);

    var shadow = 0.0;
    for (var i = 0; i < PCF_SAMPLES; i++) {
        let suv = uv + get_poisson_sample(i, rotation) * filter_radius;
        if any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0)) {
            shadow += select(0.0, 1.0, is_last);
        } else {
            shadow += textureSampleCompare(
                shadow_map, shadow_sampler, suv, cascade_idx, sc.z - bias);
        }
    }

    return mix(1.0, shadow / f32(PCF_SAMPLES), edge_fade);
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

    let rot    = world_space_noise(world_pos) * 2.0 * PI;
    let radius = 5.0 / SHADOW_MAP_SIZE;

    let cb = select_cascade_with_blend(view_depth);
    let ci = i32(cb.x);

    let shadow_a = sample_cascade_pcf(world_pos, ci, bias, rot, radius, ci == 3);

    if cb.y > 0.001 && ci < 3 {
        let shadow_b = sample_cascade_pcf(world_pos, ci + 1, bias, rot, radius, (ci + 1) == 3);
        return mix(shadow_a, shadow_b, cb.y);
    }
    return shadow_a;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex = textureSample(texture_atlas, texture_sampler, fract(in.uv), i32(in.tex_index + 0.5));
    if tex.a < 0.5 { discard; }

    let sun_dir = normalize(uniforms.sun_position);

    let day_factor      = clamp( sun_dir.y, 0.0, 1.0);
    let night_factor    = clamp(-sun_dir.y, 0.0, 1.0);
    let sunset_factor   = 1.0 - abs(sun_dir.y);
    let twilight_factor = smoothstep(-0.1, 0.15, sun_dir.y) * smoothstep(0.4, 0.0, sun_dir.y);

    let sky_color = calculate_sky_color(normalize(in.world_pos - uniforms.camera_pos), sun_dir);

    var shadow = 1.0;
    if sun_dir.y > 0.0 {
        shadow = calculate_shadow(in.world_pos, in.normal, sun_dir, in.view_depth);
    }

    let ambient = max(
        mix(0.005, 0.38, day_factor),
        0.18 * twilight_factor,
    );

    let sun_diff  = max(dot(in.normal, sun_dir), 0.0) * 0.55 * shadow * day_factor;
    let fill_dir  = normalize(vec3<f32>(-sun_dir.x, 0.5, -sun_dir.z));
    let fill_diff = max(dot(in.normal, fill_dir), 0.0) * 0.08 * day_factor;

    var face_shade: f32;
    if      abs(in.normal.y) > 0.5 { face_shade = select(0.5, 1.0, in.normal.y > 0.0); }
    else if abs(in.normal.x) > 0.5 { face_shade = 0.7; }
    else                            { face_shade = 0.8; }


    let total_light = (ambient + sun_diff + fill_diff) * face_shade;
    var lit = tex.rgb * total_light;

    if sunset_factor > 0.3 && sun_dir.y > -0.2 {
        lit *= mix(vec3<f32>(1.0), vec3<f32>(1.0, 0.85, 0.7), sunset_factor * 0.5);
    }


    let dist          = length(in.world_pos.xz - uniforms.camera_pos.xz);
    let is_underwater = uniforms.is_underwater > 0.5;

    var vis_range: f32;
    var fog_color: vec3<f32>;

    if is_underwater {
        vis_range = 24.0;
        fog_color = vec3<f32>(0.05, 0.15, 0.3);
    } else {
        vis_range = max(mix(18.0, 250.0, day_factor), 80.0 * twilight_factor);
        let night_sky = vec3<f32>(0.001, 0.001, 0.008);
        fog_color = mix(night_sky, sky_color, max(day_factor, twilight_factor * 0.7));
    }

    let fog_start  = vis_range * 0.2;
    let visibility = clamp((vis_range - dist) / (vis_range - fog_start), 0.0, 1.0);
    var final_color = mix(fog_color, lit, visibility);


    if is_underwater {
        final_color *= vec3<f32>(0.4, 0.7, 1.0);
        let caustic  = sin(in.world_pos.x * 0.5 + uniforms.time * 2.0)
                     * sin(in.world_pos.z * 0.5 + uniforms.time * 1.5) * 0.1 + 0.9;
        final_color *= caustic;
        final_color  = mix(final_color, fog_color,
                           clamp(dist / vis_range, 0.0, 1.0) * 0.5);
    }

    return vec4<f32>(final_color, 1.0);
}
