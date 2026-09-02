struct Uniforms {
    view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>, time: f32, sun_position: vec3<f32>, is_underwater: f32,
    screen_size: vec2<f32>, water_level: f32, reflection_mode: f32,
    moon_position: vec3<f32>, _pad1_moon: f32, moon_intensity: f32,
    wind_dir_x: f32, wind_dir_z: f32, wind_speed: f32, rain_factor: f32,
    sky_visibility: f32, menu_blur: f32, _pad_uniforms: f32,
    prev_view_proj: mat4x4<f32>, prev_time: f32, frame_index: u32,
    sssr_history_valid: u32, _pad_sssr: u32,
};

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e * 2.0 - 1.0, 1.0 - abs(e.x * 2.0 - 1.0) - abs(e.y * 2.0 - 1.0));
    if n.z < 0.0 {
        // WGSL does not permit assignment to a swizzle (`n.xy`).
        let folded = (1.0 - abs(n.yx)) * sign(n.xy);
        n = vec3<f32>(folded, n.z);
    }
    return normalize(n);
}
fn reconstruct_world(uv: vec2<f32>, depth: f32, inv: mat4x4<f32>) -> vec3<f32> {
    let p = inv * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / max(p.w, 0.00001);
}
fn project(world: vec3<f32>, vp: mat4x4<f32>) -> vec3<f32> {
    let p = vp * vec4<f32>(world, 1.0);
    let n = p.xyz / max(p.w, 0.00001);
    return vec3<f32>(n.x * .5 + .5, .5 - n.y * .5, n.z);
}
fn hash2(p: vec2<u32>, frame: u32) -> vec2<f32> {
    let x = f32((p.x * 1973u + p.y * 9277u + frame * 26699u) & 65535u) / 65536.0;
    let y = f32((p.x * 3181u + p.y * 1013u + frame * 17389u) & 65535u) / 65536.0;
    return vec2<f32>(x, y);
}
