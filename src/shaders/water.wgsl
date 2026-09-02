// Minimalny shader wody: stały niebieski kolor z lekkim rozjaśnieniem fal
// oraz przezroczystością obsługiwaną przez alpha blending w `Water Pipeline`.
struct Uniforms {
    view_proj: mat4x4<f32>, inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>, time: f32,
    sun_dir: vec3<f32>, is_underwater: f32,
    screen_size: vec2<f32>, water_level: f32, _pad_water: f32,
    moon_position: vec3<f32>, _pad1_moon: f32,
    moon_intensity: f32, wind_dir_x: f32, wind_dir_z: f32, wind_speed: f32,
    rain_factor: f32, sky_visibility: f32, menu_blur: f32, _pad_uniforms: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct PackedQuad {
    origin_and_face: u32,
    size_material_ao: u32,
    color_flags: u32,
    _reserved: u32,
}
struct SubchunkMeta { world_origin: vec4<i32>, draw_data: vec4<u32>, }
@group(1) @binding(0) var<storage, read> quads: array<PackedQuad>;
@group(1) @binding(1) var<storage, read> subchunks: array<SubchunkMeta>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

fn quad_position(quad: PackedQuad, corner: u32, subchunk_id: u32) -> vec3<f32> {
    let origin = vec3<f32>(
        f32(quad.origin_and_face & 0x3fu),
        f32((quad.origin_and_face >> 6u) & 0x3fu),
        f32((quad.origin_and_face >> 12u) & 0x3fu),
    ) * 0.5;
    let face = (quad.origin_and_face >> 18u) & 0x7u;
    let width = f32((quad.size_material_ao & 0x1fu) + 1u) * 0.5;
    let height = f32(((quad.size_material_ao >> 5u) & 0x1fu) + 1u) * 0.5;
    let corners = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
    let alternate_corners = array<u32, 6>(0u, 1u, 3u, 1u, 2u, 3u);
    let i = select(corners[corner], alternate_corners[corner], ((quad.color_flags >> 9u) & 1u) != 0u);
    var p = origin;
    if face == 0u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(0,0,width),vec3(0,height,width),vec3(0,height,0))[i]; }
    if face == 1u { p += array<vec3<f32>, 4>(vec3(0,0,width),vec3(0,0,0),vec3(0,height,0),vec3(0,height,width))[i]; }
    if face == 2u { p += array<vec3<f32>, 4>(vec3(0,0,width),vec3(0,0,0),vec3(height,0,0),vec3(height,0,width))[i]; }
    if face == 3u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(0,0,width),vec3(height,0,width),vec3(height,0,0))[i]; }
    if face == 4u { p += array<vec3<f32>, 4>(vec3(width,0,0),vec3(0,0,0),vec3(0,height,0),vec3(width,height,0))[i]; }
    if face == 5u { p += array<vec3<f32>, 4>(vec3(0,0,0),vec3(width,0,0),vec3(width,height,0),vec3(0,height,0))[i]; }
    return vec3<f32>(subchunks[subchunk_id].world_origin.xyz) + p;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) subchunk_id: u32) -> VertexOutput {
    let quad = quads[vertex_id / 6u];
    let face = (quad.origin_and_face >> 18u) & 0x7u;
    let normals = array<vec3<f32>, 6>(
        vec3(-1,0,0), vec3(1,0,0), vec3(0,-1,0),
        vec3(0,1,0), vec3(0,0,-1), vec3(0,0,1),
    );
    let position = quad_position(quad, vertex_id % 6u, subchunk_id);
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(position, 1.0);
    out.world_pos = position;
    out.normal = normals[face % 6u];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let wave = sin((in.world_pos.x + in.world_pos.z) * 1.4 + uniforms.time * 1.5) * 0.04;
    let top_light = max(in.normal.y, 0.0) * 0.12;
    let water_blue = vec3<f32>(0.04, 0.34, 0.88) + vec3<f32>(wave + top_light);
    return vec4<f32>(water_blue, 0.62);
}
