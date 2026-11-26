struct ShadowGrid {
    params: vec4<f32>,
    slots: array<vec4<i32>, 256>,
};

struct SdfParams {
    origin_brick_xz: vec2<i32>,
    _pad: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> shadow_grid: ShadowGrid;
@group(0) @binding(1) var<storage, read_write> shadow_bits: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch_a: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch_b: array<u32>;
@group(0) @binding(4) var<uniform> sdf_params: SdfParams;

const SHADOW_SLOT_WORDS: u32 = 2320u;
const SHADOW_BIT_WORDS: u32 = 2048u;
const SHADOW_COLUMN_WORDS: u32 = 8u;
const SDF_DIM: u32 = 64u;
const SDF_RADIUS: i32 = 8;
const SDF_DISTANCE_CAP: u32 = 9u;
const SDF_OFFSET: u32 = 593920u;
const SDF_HEADER_WORDS: u32 = 4u;
const SDF_DISTANCE_OFFSET: u32 = SDF_OFFSET + SDF_HEADER_WORDS;
const SDF_CELL_COUNT: u32 = SDF_DIM * SDF_DIM * SDF_DIM;

fn sdf_index(brick: vec3<u32>) -> u32 {
    return brick.x + SDF_DIM * (brick.z + SDF_DIM * brick.y);
}

fn sdf_coords(index: u32) -> vec3<u32> {
    let x = index % SDF_DIM;
    let z = (index / SDF_DIM) % SDF_DIM;
    let y = index / (SDF_DIM * SDF_DIM);
    return vec3<u32>(x, y, z);
}

@compute @workgroup_size(64)
fn build_occupancy(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= SDF_CELL_COUNT {
        return;
    }
    let brick = sdf_coords(index);
    let block_x = (sdf_params.origin_brick_xz.x + i32(brick.x)) * 4;
    let block_z = (sdf_params.origin_brick_xz.y + i32(brick.z)) * 4;
    let chunk_x = block_x >> 4;
    let chunk_z = block_z >> 4;
    let slot_index = (chunk_x & 15) + (chunk_z & 15) * 16;
    let slot = shadow_grid.slots[slot_index];
    var occupied = slot.x != chunk_x || slot.y != chunk_z;

    let y = brick.y * 4u;
    let word_offset = y >> 5u;
    let mask = 0xFu << (y & 31u);
    for (var dz = 0u; dz < 4u; dz++) {
        for (var dx = 0u; dx < 4u; dx++) {
            if !occupied {
                let local_x = (block_x + i32(dx)) & 15;
                let local_z = (block_z + i32(dz)) & 15;
                let column = u32(local_z * 16 + local_x);
                let word = u32(slot_index) * SHADOW_SLOT_WORDS
                    + column * SHADOW_COLUMN_WORDS + word_offset;
                occupied = (shadow_bits[word] & mask) != 0u;
            }
        }
        if occupied {
            break;
        }
    }

    scratch_a[index] = select(SDF_DISTANCE_CAP, 0u, occupied);
}

@compute @workgroup_size(64)
fn distance_x(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= SDF_CELL_COUNT {
        return;
    }
    let brick = sdf_coords(index);
    var distance = SDF_DISTANCE_CAP;
    for (var offset = -SDF_RADIUS; offset <= SDF_RADIUS; offset++) {
        let x = i32(brick.x) + offset;
        var source = 0u;
        if x >= 0 && x < i32(SDF_DIM) {
            source = scratch_a[sdf_index(vec3<u32>(u32(x), brick.y, brick.z))];
        }
        distance = min(distance, max(u32(abs(offset)), source));
    }
    scratch_b[index] = distance;
}

@compute @workgroup_size(64)
fn distance_y(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= SDF_CELL_COUNT {
        return;
    }
    let brick = sdf_coords(index);
    var distance = SDF_DISTANCE_CAP;
    for (var offset = -SDF_RADIUS; offset <= SDF_RADIUS; offset++) {
        let y = i32(brick.y) + offset;
        var source = 0u;
        if y >= 0 && y < i32(SDF_DIM) {
            source = scratch_b[sdf_index(vec3<u32>(brick.x, u32(y), brick.z))];
        }
        distance = min(distance, max(u32(abs(offset)), source));
    }
    scratch_a[index] = distance;
}

@compute @workgroup_size(64)
fn distance_z(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= SDF_CELL_COUNT {
        return;
    }
    let brick = sdf_coords(index);
    var distance = SDF_DISTANCE_CAP;
    for (var offset = -SDF_RADIUS; offset <= SDF_RADIUS; offset++) {
        let z = i32(brick.z) + offset;
        var source = 0u;
        if z >= 0 && z < i32(SDF_DIM) {
            source = scratch_a[sdf_index(vec3<u32>(brick.x, brick.y, u32(z)))];
        }
        distance = min(distance, max(u32(abs(offset)), source));
    }
    shadow_bits[SDF_DISTANCE_OFFSET + index] = distance;
}