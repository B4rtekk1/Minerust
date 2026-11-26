//! Voxel sun-shadow grid.
//!
//! Sun shadows are ray-marched per pixel through the voxel world instead of
//! being rasterised into shadow maps. Because the world is made of unit
//! cubes, an exact shadow test is just "does a ray towards the sun hit a
//! filled cell?". No geometry is re-rendered from the sun, so shadows are
//! pixel-exact blocky voxel shadows with no acne, peter-panning or cascade
//! seams, and moving the sun costs nothing on the CPU.
//!
//! # Making the march cheap
//!
//! A sun ray always climbs, and the world is mostly a height field with
//! some caves and overhangs. The tracer exploits both:
//!
//! - **Column stepping.** The ray walks a 2-D DDA over block *columns*
//!   instead of a 3-D DDA over cells. Inside one column only `y` changes,
//!   so the whole vertical run of cells it crosses is tested with one or
//!   two masked word reads: each column's 256 occupancy bits are stored
//!   contiguously (8 words, bit = `y`).
//! - **Max-height pyramid.** Every slot stores the height of the highest
//!   occluder per column, per 4×4 tile and per chunk. A climbing ray that is
//!   already above a tile's maximum can never hit anything in it, so the
//!   tile is crossed in a single step; above the highest occluder of the
//!   whole grid the ray is lit at once. Caves and overhangs still work:
//!   below a column's maximum the real bits are tested.
//!
//! Typical rays finish in a handful of steps instead of walking every cell.
//!
//! # Layout
//!
//! The grid is a toroidal window: chunk column `(cx, cz)` always lives in
//! slot `(cx & 15, cz & 15)`, so when the player walks only the newly
//! entered columns are uploaded. Each slot records which column it holds;
//! the shader treats a slot holding another column as the end of the data
//! (lit). Per slot, one buffer holds [`BIT_WORDS`] occupancy words followed
//! by 256 column words and 16 tile heights (one `u32` each).
//!
//! A column word holds the column height in bits 0..16 and, for the water
//! reflection tracer (`water.wgsl`), two atlas layers describing what a ray
//! hitting the column sees: bits 16..24 the top of the highest block
//! ([`COLUMN_TOP_SHIFT`]) and bits 24..32 the side of a block a few blocks
//! lower ([`COLUMN_WALL_SHIFT`]), e.g. grass over a stone cliff.

use crate::constants::{CHUNK_SIZE, SUBCHUNK_HEIGHT, WORLD_HEIGHT};
use crate::core::block::BlockType;
use crate::core::chunk::{Chunk, SubChunkStorage};
use crate::world::World;

/// Chunk columns along each horizontal side of the grid.
pub const SHADOW_GRID_CHUNKS: i32 = 16;
/// Chunk slots in the grid.
pub const SHADOW_SLOTS: usize = (SHADOW_GRID_CHUNKS * SHADOW_GRID_CHUNKS) as usize;
/// Block columns per chunk.
const COLUMNS: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;
/// Occupancy words per block column (one bit per block of height).
pub const COLUMN_WORDS: usize = WORLD_HEIGHT as usize / 32;
/// Occupancy words per chunk slot.
pub const BIT_WORDS: usize = COLUMNS * COLUMN_WORDS;
/// 4×4-column tiles per chunk.
const TILES: usize = COLUMNS / 16;
/// All words of one slot: bits, column heights, tile heights.
pub const WORDS_PER_SLOT: usize = BIT_WORDS + COLUMNS + TILES;
/// Sun shadows are only traced for pixels within this many chunks of the
/// camera (fading out over the last fifth). Everything further away —
/// most of a 32-chunk view — uses the cheap sky-light estimate and never
/// starts a shadow ray.
pub const SHADOW_DISTANCE_CHUNKS: i32 = 6;
/// [`SHADOW_DISTANCE_CHUNKS`] in blocks.
pub const SHADOW_DISTANCE: f32 = (SHADOW_DISTANCE_CHUNKS * CHUNK_SIZE) as f32;
/// Columns uploaded per frame at most, nearest first, so streaming in a
/// whole new window never causes a hitch.
const UPLOADS_PER_FRAME: usize = 4;
/// Tag of a slot that holds no column.
const EMPTY_TAG: i32 = i32::MIN;
/// Mask of the height in a column word.
pub const COLUMN_HEIGHT_MASK: u32 = 0xffff;
/// Bit offset of the top-surface atlas layer in a column word.
pub const COLUMN_TOP_SHIFT: u32 = 16;
/// Bit offset of the wall atlas layer in a column word.
pub const COLUMN_WALL_SHIFT: u32 = 24;
/// Blocks below the top whose side texture stands for the column's walls.
const WALL_DEPTH: i32 = 4;
/// Step limit of a shadow ray (mirrors `SHADOW_MAX_ITERATIONS` in WGSL).
#[cfg(test)]
const SHADOW_MAX_ITERATIONS: usize = 128;
/// Nudge past a column or tile boundary (mirrors `SHADOW_STEP_EPSILON`).
#[cfg(test)]
const STEP_EPSILON: f32 = 1e-3;

const _: () = {
    assert!(SHADOW_GRID_CHUNKS.count_ones() == 1, "slot index uses `& (N - 1)`");
    assert!(CHUNK_SIZE == 16, "the shader indexes columns with `& 15` / `>> 4`");
    assert!(32 % SUBCHUNK_HEIGHT == 0, "a subchunk must fit inside one occupancy word");
    assert!(WORLD_HEIGHT % 32 == 0);
};

/// `true` if `block` stops direct sunlight.
pub fn casts_shadow(block: BlockType) -> bool {
    block.is_solid_opaque() || block.is_leaves() || block == BlockType::WoodStairs
}

/// [`casts_shadow`] for every block id, so the per-block test in
/// [`column_bits`] is a table load instead of a `match`.
fn shadow_table() -> &'static [bool; 256] {
    static TABLE: std::sync::OnceLock<[bool; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [false; 256];
        for block in BlockType::ALL {
            table[block as usize] = casts_shadow(block);
        }
        table
    })
}

/// Index of block column `(lx, lz)` within a slot.
#[inline]
pub fn column_index(lx: i32, lz: i32) -> usize {
    (lz * CHUNK_SIZE + lx) as usize
}

/// Index of the 4×4 tile holding column `(lx, lz)`.
#[inline]
fn tile_index(lx: i32, lz: i32) -> usize {
    ((lz >> 2) * 4 + (lx >> 2)) as usize
}

/// Occupancy and height pyramid of one chunk column, in slot layout.
pub struct ColumnBits {
    pub words: Vec<u32>,
    /// One past the highest occluding block (0 if none).
    pub max_y: i32,
}

impl ColumnBits {
    /// `true` if block `(lx, y, lz)` casts a shadow.
    pub fn occupied(&self, lx: i32, y: i32, lz: i32) -> bool {
        let word = self.words[column_index(lx, lz) * COLUMN_WORDS + (y / 32) as usize];
        word & (1 << (y % 32)) != 0
    }
}

/// Builds the occupancy bits and height pyramid of a chunk column.
pub fn column_bits(chunk: &Chunk) -> ColumnBits {
    let mut words = vec![0u32; WORDS_PER_SLOT];
    let table = shadow_table();
    let set = |words: &mut [u32], lx: i32, y: i32, lz: i32| {
        words[column_index(lx, lz) * COLUMN_WORDS + (y / 32) as usize] |= 1 << (y % 32);
    };
    for (sy, sub) in chunk.subchunks.iter().enumerate() {
        if sub.is_empty {
            continue;
        }
        let base_y = sy as i32 * SUBCHUNK_HEIGHT;
        match sub.storage() {
            SubChunkStorage::Uniform(block) => {
                if casts_shadow(*block) {
                    for lz in 0..CHUNK_SIZE {
                        for lx in 0..CHUNK_SIZE {
                            for ly in 0..SUBCHUNK_HEIGHT {
                                set(&mut words, lx, base_y + ly, lz);
                            }
                        }
                    }
                }
            }
            SubChunkStorage::Dense(blocks) => {
                // Dense order is x-major, then y, then z. A subchunk is 16
                // blocks tall and starts on a multiple of 16, so it fills
                // one half of one occupancy word per column.
                let word = (base_y / 32) as usize;
                let shift = base_y % 32;
                let mut index = 0;
                for lx in 0..CHUNK_SIZE {
                    for ly in 0..SUBCHUNK_HEIGHT {
                        let bit = 1u32 << (shift + ly);
                        for lz in 0..CHUNK_SIZE {
                            if table[blocks[index] as usize] {
                                words[column_index(lx, lz) * COLUMN_WORDS + word] |= bit;
                            }
                            index += 1;
                        }
                    }
                }
            }
        }
    }

    // Height pyramid: per column, per 4×4 tile, per chunk.
    let mut max_y = 0;
    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let column = column_index(lx, lz);
            let bits = &words[column * COLUMN_WORDS..(column + 1) * COLUMN_WORDS];
            let height = bits
                .iter()
                .enumerate()
                .rev()
                .find(|(_, w)| **w != 0)
                .map_or(0, |(i, w)| i as u32 * 32 + 32 - w.leading_zeros());
            let tile = BIT_WORDS + COLUMNS + tile_index(lx, lz);
            words[tile] = words[tile].max(height);
            max_y = max_y.max(height as i32);
            // What a reflected ray hitting this column sees: the highest
            // block's top, and the side of a block a little lower down.
            let mut materials = 0;
            if height > 0 {
                let top = chunk.get_block(lx, height as i32 - 1, lz);
                let wall_y = (height as i32 - 1 - WALL_DEPTH).max(0);
                let wall = match chunk.get_block(lx, wall_y, lz) {
                    block if casts_shadow(block) => block,
                    _ => top,
                };
                materials = (top.tex_top() as u32) << COLUMN_TOP_SHIFT
                    | (wall.tex_side() as u32) << COLUMN_WALL_SHIFT;
            }
            words[BIT_WORDS + column] = height | materials;
        }
    }
    ColumnBits { words, max_y }
}

/// Sum of the subchunk block-edit counters: changes only when a block of
/// the column changes (not when a neighbour's lighting re-meshes it).
fn column_version(chunk: &Chunk) -> u64 {
    chunk
        .subchunks
        .iter()
        .fold(chunk.subchunks.len() as u64, |acc, s| acc.wrapping_add(s.block_version as u64))
}

/// GPU layout of the grid header and slot table (`ShadowGrid` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridHeader {
    /// x: sun shadows enabled (0/1), y: shadow trace distance, z: highest
    /// occluder (exclusive) over all slots, w: water reflections enabled.
    params: [f32; 4],
    /// Per slot: chunk x, chunk z, max y (exclusive), unused.
    slots: [[i32; 4]; SHADOW_SLOTS],
}

#[derive(Clone, Copy)]
struct Slot {
    tag: (i32, i32),
    version: u64,
}

/// CPU mirror and GPU buffers of the shadow grid.
pub struct ShadowGrid {
    bits: wgpu::Buffer,
    header_buffer: wgpu::Buffer,
    header: Box<GridHeader>,
    slots: Vec<Slot>,
    header_dirty: bool,
    world_id: u64,
}

impl ShadowGrid {
    pub fn new(device: &wgpu::Device) -> Self {
        let bits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Grid Bits"),
            size: (SHADOW_SLOTS * WORDS_PER_SLOT * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut header = Box::new(GridHeader {
            params: [0.0; 4],
            slots: [[EMPTY_TAG, EMPTY_TAG, 0, 0]; SHADOW_SLOTS],
        });
        header.params[1] = SHADOW_DISTANCE;
        let header_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Grid Header"),
            size: std::mem::size_of::<GridHeader>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bits,
            header_buffer,
            header,
            slots: vec![Slot { tag: (EMPTY_TAG, EMPTY_TAG), version: 0 }; SHADOW_SLOTS],
            header_dirty: true,
            world_id: 0,
        }
    }

    pub fn bits_buffer(&self) -> &wgpu::Buffer {
        &self.bits
    }

    pub fn header_buffer(&self) -> &wgpu::Buffer {
        &self.header_buffer
    }

    /// Forgets every uploaded column (new or loaded world).
    pub fn clear(&mut self) {
        for (slot, meta) in self.slots.iter_mut().zip(self.header.slots.iter_mut()) {
            slot.tag = (EMPTY_TAG, EMPTY_TAG);
            *meta = [EMPTY_TAG, EMPTY_TAG, 0, 0];
        }
        self.header_dirty = true;
    }

    /// Slot of chunk column `(cx, cz)`.
    pub fn slot_index(cx: i32, cz: i32) -> usize {
        let mask = SHADOW_GRID_CHUNKS - 1;
        ((cx & mask) + (cz & mask) * SHADOW_GRID_CHUNKS) as usize
    }

    /// Streams changed columns around `(center_cx, center_cz)` to the GPU
    /// and updates the trace parameters. The grid is kept up to date while
    /// either sun shadows or voxel water reflections use it.
    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        world: &World,
        center_cx: i32,
        center_cz: i32,
        shadows: bool,
        reflections: bool,
    ) {
        let enabled = shadows || reflections;
        if world.instance_id != self.world_id {
            self.world_id = world.instance_id;
            self.clear();
        }
        let flags = [f32::from(u8::from(shadows)), f32::from(u8::from(reflections))];
        if self.header.params[0] != flags[0] || self.header.params[3] != flags[1] {
            self.header.params[0] = flags[0];
            self.header.params[3] = flags[1];
            self.header_dirty = true;
        }
        if enabled {
            let half = SHADOW_GRID_CHUNKS / 2;
            let mut pending: Vec<(i32, i32, i32, u64)> = Vec::new();
            for cx in center_cx - half..center_cx + half {
                for cz in center_cz - half..center_cz + half {
                    let index = Self::slot_index(cx, cz);
                    match world.chunks.get(&(cx, cz)) {
                        Some(chunk) => {
                            let version = column_version(chunk);
                            let slot = self.slots[index];
                            if slot.tag != (cx, cz) || slot.version != version {
                                let d = (cx - center_cx).abs().max((cz - center_cz).abs());
                                pending.push((d, cx, cz, version));
                            }
                        }
                        None if self.slots[index].tag != (EMPTY_TAG, EMPTY_TAG) => {
                            self.slots[index].tag = (EMPTY_TAG, EMPTY_TAG);
                            self.header.slots[index] = [EMPTY_TAG, EMPTY_TAG, 0, 0];
                            self.header_dirty = true;
                        }
                        None => {}
                    }
                }
            }
            pending.sort_unstable_by_key(|&(d, ..)| d);
            for &(_, cx, cz, version) in pending.iter().take(UPLOADS_PER_FRAME) {
                let index = Self::slot_index(cx, cz);
                let column = column_bits(&world.chunks[&(cx, cz)]);
                queue.write_buffer(
                    &self.bits,
                    (index * WORDS_PER_SLOT * 4) as u64,
                    bytemuck::cast_slice(&column.words),
                );
                self.slots[index] = Slot { tag: (cx, cz), version };
                self.header.slots[index] = [cx, cz, column.max_y, 0];
                self.header_dirty = true;
            }
        }
        if self.header_dirty {
            let max_y = self
                .header
                .slots
                .iter()
                .filter(|s| s[0] != EMPTY_TAG)
                .map(|s| s[2])
                .max()
                .unwrap_or(0);
            self.header.params[2] = max_y as f32;
            queue.write_buffer(&self.header_buffer, 0, bytemuck::bytes_of(&*self.header));
            self.header_dirty = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn column_bits_mark_occluders_and_heights() {
        let mut chunk = Chunk::new(0, 0);
        chunk.set_block_raw(3, 70, 5, BlockType::Stone);
        chunk.set_block_raw(1, 2, 1, BlockType::Leaves);
        chunk.set_block_raw(2, 90, 2, BlockType::Water);
        chunk.set_block_raw(4, 91, 4, BlockType::DeadBush);
        chunk.rebuild_metadata();
        let bits = column_bits(&chunk);
        assert!(bits.occupied(3, 70, 5));
        assert!(bits.occupied(1, 2, 1));
        assert!(!bits.occupied(2, 90, 2), "water lets the sun through");
        assert!(!bits.occupied(4, 91, 4), "plants do not shadow");
        assert!(!bits.occupied(3, 71, 5));
        assert_eq!(bits.max_y, 71);
        let column = |lx, lz| bits.words[BIT_WORDS + column_index(lx, lz)];
        assert_eq!(column(3, 5) & COLUMN_HEIGHT_MASK, 71);
        assert_eq!(column(1, 1) & COLUMN_HEIGHT_MASK, 3);
        assert_eq!(column(0, 0), 0);
        // Stone column: stone top; nothing below it, so the wall is stone too.
        assert_eq!(column(3, 5) >> COLUMN_TOP_SHIFT & 0xff, crate::constants::TEX_STONE as u32);
        assert_eq!(column(3, 5) >> COLUMN_WALL_SHIFT, crate::constants::TEX_STONE as u32);
        assert_eq!(column(1, 1) >> COLUMN_TOP_SHIFT & 0xff, crate::constants::TEX_LEAVES as u32);
        assert_eq!(bits.words[BIT_WORDS + COLUMNS + tile_index(3, 5)], 71);
        assert_eq!(bits.words[BIT_WORDS + COLUMNS + tile_index(1, 1)], 3);
    }

    #[test]
    fn neighbouring_columns_never_share_a_slot() {
        let mut seen = std::collections::HashSet::new();
        for cx in -8..8 {
            for cz in 100..116 {
                assert!(seen.insert(ShadowGrid::slot_index(cx, cz)));
            }
        }
        assert_eq!(seen.len(), SHADOW_SLOTS);
    }

    #[test]
    fn shader_constants_match() {
        let shader = include_str!("../shaders/terrain.wgsl");
        for (name, value) in [
            ("SHADOW_SLOT_WORDS", WORDS_PER_SLOT),
            ("SHADOW_BIT_WORDS", BIT_WORDS),
            ("SHADOW_COLUMN_WORDS", COLUMN_WORDS),
            ("SHADOW_COLUMNS", COLUMNS),
        ] {
            let line = format!("const {name}: u32 = {value}u;");
            assert!(shader.contains(&line), "terrain.wgsl is missing `{line}`");
        }
        assert!(shader.contains(&format!("slots: array<vec4<i32>, {}>", SHADOW_SLOTS)));
    }

    /// CPU copy of the GPU data: slot words keyed by chunk.
    struct CpuGrid {
        columns: HashMap<(i32, i32), ColumnBits>,
        top: f32,
    }

    impl CpuGrid {
        fn occupied(&self, x: i32, y: i32, z: i32) -> Option<bool> {
            let c = self.columns.get(&(x >> 4, z >> 4))?;
            Some((0..WORLD_HEIGHT).contains(&y) && c.occupied(x & 15, y, z & 15))
        }
    }

    /// Line-for-line port of `trace_sun` in terrain.wgsl.
    fn trace_columns(grid: &CpuGrid, pos: [f32; 3], normal: [f32; 3], sun: [f32; 3], max_t: f32, steps: &mut usize) -> f32 {
        let origin = [0, 1, 2].map(|i| pos[i] + normal[i] * 0.01);
        let moving = sun.map(|d| d.abs() > 1e-5);
        let positive = sun.map(|d| d > 0.0);
        let inv = [0, 1, 2].map(|i| if moving[i] { 1.0 / sun[i] } else { 1.0 });
        let box_exit = |p: [f32; 3], size: f32| -> f32 {
            let mut t = f32::INFINITY;
            for i in [0, 2] {
                if moving[i] {
                    let bound = (p[i] / size).floor() * size + if positive[i] { size } else { 0.0 };
                    t = t.min((bound - origin[i]) * inv[i]);
                }
            }
            t
        };
        let mut t = 0.0f32;
        for _ in 0..SHADOW_MAX_ITERATIONS {
            *steps += 1;
            let p = [0, 1, 2].map(|i| origin[i] + sun[i] * t);
            if p[1] >= grid.top || t > max_t {
                return 1.0;
            }
            let (x, z) = (p[0].floor() as i32, p[2].floor() as i32);
            let Some(c) = grid.columns.get(&(x >> 4, z >> 4)) else {
                return 1.0;
            };
            if p[1] >= c.max_y as f32 {
                t = box_exit(p, 16.0) + STEP_EPSILON;
                continue;
            }
            let (lx, lz) = (x & 15, z & 15);
            if p[1] >= c.words[BIT_WORDS + COLUMNS + tile_index(lx, lz)] as f32 {
                t = box_exit(p, 4.0) + STEP_EPSILON;
                continue;
            }
            let height = c.words[BIT_WORDS + column_index(lx, lz)] & COLUMN_HEIGHT_MASK;
            let t_out = box_exit(p, 1.0);
            if p[1] < height as f32 {
                let y0 = p[1].max(0.0).floor() as u32;
                let y1 = ((origin[1] + sun[1] * (t_out - 1e-4)).floor() as u32).min(height - 1).max(y0);
                if (y0..=y1).any(|y| c.occupied(lx, y as i32, lz)) {
                    return 0.0;
                }
            }
            t = t_out + STEP_EPSILON;
        }
        1.0
    }

    /// Plain 3-D DDA over every cell: the ground truth.
    fn trace_cells(grid: &CpuGrid, pos: [f32; 3], normal: [f32; 3], sun: [f32; 3], steps: &mut usize) -> f32 {
        let origin = [0, 1, 2].map(|i| pos[i] + normal[i] * 0.01);
        let mut cell = origin.map(|v| v.floor() as i32);
        let step = sun.map(|d| if d > 0.0 { 1 } else { -1 });
        let delta = sun.map(|d| if d.abs() > 1e-5 { (1.0 / d).abs() } else { f32::INFINITY });
        let mut t_max = [0, 1, 2].map(|i| {
            if sun[i].abs() <= 1e-5 {
                f32::INFINITY
            } else {
                (cell[i] as f32 + if sun[i] > 0.0 { 1.0 } else { 0.0 } - origin[i]) / sun[i]
            }
        });
        loop {
            *steps += 1;
            if cell[1] as f32 >= grid.top {
                return 1.0;
            }
            match grid.occupied(cell[0], cell[1], cell[2]) {
                None => return 1.0,
                Some(true) => return 0.0,
                Some(false) => {}
            }
            let axis = if t_max[0] < t_max[1] && t_max[0] < t_max[2] {
                0
            } else if t_max[1] < t_max[2] {
                1
            } else {
                2
            };
            cell[axis] += step[axis];
            t_max[axis] += delta[axis];
        }
    }

    #[test]
    fn column_tracer_matches_cell_by_cell_ray_march() {
        let generator = crate::world::ChunkGenerator::new(2026);
        let mut columns = HashMap::new();
        for cx in -3..3 {
            for cz in -3..3 {
                let mut chunk = generator.generate_chunk(cx, cz);
                chunk.rebuild_metadata();
                columns.insert((cx, cz), column_bits(&chunk));
            }
        }
        let top = columns.values().map(|c| c.max_y).max().unwrap() as f32;
        let grid = CpuGrid { columns, top };
        let normals = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let mut rng = 0x1234_5678u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng as f32 / u32::MAX as f32
        };
        let (mut total, mut mismatches, mut shadowed) = (0, 0, 0);
        let (mut cell_steps, mut column_steps) = (0usize, 0usize);
        for sun_angle in [0.35f32, 0.7, 1.1, 1.57, 2.3] {
            let len = (0.35f32 * 0.35 + 1.0).sqrt();
            let sun = [0.35 / len, sun_angle.sin() / len, sun_angle.cos() / len];
            for _ in 0..4000 {
                // A random open face of a random block near the middle.
                let (x, z) = ((next() * 64.0 - 32.0) as i32, (next() * 64.0 - 32.0) as i32);
                let normal = normals[(next() * 5.0) as usize % 5];
                // Every open face of this column in this direction.
                let faces: Vec<i32> = (1..grid.top as i32)
                    .filter(|&y| {
                        grid.occupied(x, y, z) == Some(true)
                            && grid.occupied(
                                x + normal[0] as i32,
                                y + normal[1] as i32,
                                z + normal[2] as i32,
                            ) == Some(false)
                    })
                    .collect();
                if faces.is_empty() {
                    continue;
                }
                let y = faces[(next() * faces.len() as f32) as usize % faces.len()];
                if normal[0] * sun[0] + normal[1] * sun[1] + normal[2] * sun[2] <= 0.0 {
                    continue;
                }
                // A point on that face.
                let (u, v) = (next() * 0.98 + 0.01, next() * 0.98 + 0.01);
                let pos = match normal {
                    [0.0, 1.0, 0.0] => [x as f32 + u, y as f32 + 1.0, z as f32 + v],
                    [1.0, _, _] => [x as f32 + 1.0, y as f32 + u, z as f32 + v],
                    [-1.0, _, _] => [x as f32, y as f32 + u, z as f32 + v],
                    [_, _, 1.0] => [x as f32 + u, y as f32 + v, z as f32 + 1.0],
                    _ => [x as f32 + u, y as f32 + v, z as f32],
                };
                let truth = trace_cells(&grid, pos, normal, sun, &mut cell_steps);
                let fast = trace_columns(&grid, pos, normal, sun, 1e9, &mut column_steps);
                total += 1;
                shadowed += usize::from(truth == 0.0);
                mismatches += usize::from(truth != fast);
            }
        }
        assert!(total > 2000 && shadowed > 200, "too few samples: {total}, {shadowed} shadowed");
        println!(
            "{total} rays, {shadowed} shadowed, {mismatches} mismatches; \
             average steps: cell DDA {:.1}, column tracer {:.1}",
            cell_steps as f32 / total as f32,
            column_steps as f32 / total as f32
        );
        assert!(column_steps * 3 < cell_steps, "the column tracer should take far fewer steps");
        assert!(
            mismatches * 1000 <= total,
            "{mismatches} of {total} rays disagree with the reference"
        );
    }
}
