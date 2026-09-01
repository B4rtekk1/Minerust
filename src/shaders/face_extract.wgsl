struct PackedQuad { origin_and_face: u32, size_material_ao: u32, color_flags: u32, reserved: u32, }
struct SubchunkMeta { world_origin: vec4<i32>, draw_data: vec4<u32>, }
struct ExtractUniforms { output_quad_offset: u32, quad_capacity: u32, subchunk_slot: u32, _padding: u32, }

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> quads: array<PackedQuad>;
@group(0) @binding(2) var<storage, read_write> subchunks: array<SubchunkMeta>;
@group(0) @binding(3) var<storage, read_write> face_counter: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: ExtractUniforms;

fn voxel_index(x: u32, y: u32, z: u32) -> u32 { return x * 324u + y * 18u + z; }
fn transparent(block: u32) -> bool { return block == 0u || block == 7u || block == 12u || block == 14u || block == 15u; }
fn visible(block: u32, neighbor: u32) -> bool {
    if neighbor == 0u { return true; }
    if block == 5u { return false; }
    if neighbor == 5u { return true; }
    if block == 7u && neighbor == 7u { return true; }
    return transparent(neighbor);
}
fn tex(block: u32, face: u32) -> u32 {
    if block == 1u { if face == 2u { return 2u; } if face == 3u { return 0u; } return 1u; }
    if block == 6u { if face == 2u || face == 3u { return 7u; } return 6u; }
    if block == 16u { if face == 0u || face == 1u { return 7u; } return 6u; }
    if block == 17u { if face == 4u || face == 5u { return 7u; } return 6u; }
    return array<u32, 18>(0u,0u,2u,3u,4u,5u,7u,8u,9u,10u,11u,12u,13u,14u,15u,7u,6u,6u)[block];
}
fn color(face: u32) -> u32 {
    // The packed colour is a local-light/GI tint, not material albedo (the
    // atlas provides that).  These are the fully-open equivalents of the CPU
    // face-GI model: down, sides, then upward-facing surfaces.
    if face == 2u { return 292u; } // 4, 4, 4
    if face == 3u { return 502u; } // 6, 6, 7
    return 438u;                   // 6, 6, 6
}
fn emit(x: u32, y: u32, z: u32, face: u32, block: u32) {
    let output = atomicAdd(&face_counter[0], 1u);
    if output >= params.quad_capacity { return; }
    var ox = (x - 1u) * 2u;
    var oy = (y - 1u) * 2u;
    var oz = (z - 1u) * 2u;
    if face == 1u { ox += 2u; }
    if face == 3u { oy += 2u; }
    if face == 5u { oz += 2u; }
    // AO=3 at every corner is the fully-open baseline.  A subsequent AO
    // extraction pass can replace these eight bits without changing layout.
    quads[params.output_quad_offset + output] = PackedQuad(ox | (oy << 6u) | (oz << 12u) | (face << 18u), 1u | (1u << 5u) | (tex(block, face) << 10u) | (0xffu << 18u), color(face), 0u);
}

@compute @workgroup_size(64)
fn extract_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= 4096u { return; }
    let x = id & 15u; let y = (id >> 4u) & 15u; let z = id >> 8u;
    let px = x + 1u; let py = y + 1u; let pz = z + 1u;
    let block = voxels[voxel_index(px, py, pz)];
    if block == 0u { return; }
    if visible(block, voxels[voxel_index(px - 1u, py, pz)]) { emit(px, py, pz, 0u, block); }
    if visible(block, voxels[voxel_index(px + 1u, py, pz)]) { emit(px, py, pz, 1u, block); }
    if visible(block, voxels[voxel_index(px, py - 1u, pz)]) { emit(px, py, pz, 2u, block); }
    if visible(block, voxels[voxel_index(px, py + 1u, pz)]) { emit(px, py, pz, 3u, block); }
    if visible(block, voxels[voxel_index(px, py, pz - 1u)]) { emit(px, py, pz, 4u, block); }
    if visible(block, voxels[voxel_index(px, py, pz + 1u)]) { emit(px, py, pz, 5u, block); }
}

@compute @workgroup_size(1)
fn finalize_faces() {
    let count = min(atomicLoad(&face_counter[0]), params.quad_capacity);
    subchunks[params.subchunk_slot].draw_data.x = count * 6u;
}
