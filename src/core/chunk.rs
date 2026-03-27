use glam::Vec3;

use crate::constants::*;
use crate::core::block::BlockType;
use crate::frustum::AABB;

/// A fixed-height vertical slice of a [`Chunk`].
///
/// The world is divided vertically into sub-chunks of height [`SUBCHUNK_HEIGHT`].
/// Each sub-chunk owns its block data, a dirty flag for mesh rebuilding, and
/// cached state flags used to skip unnecessary rendering work.
///
/// # Coordinate system
/// Local coordinates are in the range `[0, CHUNK_SIZE)` on X/Z and
/// `[0, SUBCHUNK_HEIGHT)` on Y. Out-of-bounds reads return [`BlockType::Air`];
/// out-of-bounds writes are silently ignored.
pub struct SubChunk {
    /// 3-D block array indexed as `blocks[x][y][z]` in local sub-chunk space.
    pub blocks: [[[BlockType; CHUNK_SIZE as usize]; SUBCHUNK_HEIGHT as usize]; CHUNK_SIZE as usize],

    /// `true` when every block in this sub-chunk is [`BlockType::Air`].
    ///
    /// Used to skip mesh generation and rendering entirely. Updated eagerly by
    /// [`SubChunk::set_block`] and authoritatively by [`SubChunk::check_empty`].
    pub is_empty: bool,

    /// `true` when the GPU mesh is out of date and needs to be rebuilt.
    ///
    /// Set to `true` on construction and on every [`SubChunk::set_block`] call.
    /// Cleared by the meshing system after a successful upload.
    pub mesh_dirty: bool,

    /// Number of solid-geometry indices in the current GPU mesh.
    ///
    /// Used by the render pass to issue the correct `draw_indexed` call.
    /// Zero when no mesh has been generated yet or the sub-chunk is empty.
    pub num_indices: u32,

    /// Number of water-geometry indices in the current GPU mesh.
    ///
    /// Kept separate from [`Self::num_indices`] because water is rendered in a
    /// dedicated translucent pass.
    pub num_water_indices: u32,

    /// Axis-aligned bounding box in world space.
    ///
    /// Used for frustum culling. Computed once in [`SubChunk::new`] and never
    /// updated because a sub-chunk's world position is immutable.
    pub aabb: AABB,

    /// `true` when every block in this sub-chunk is solid and opaque.
    ///
    /// When `true`, neighboring sub-chunks can skip rendering faces that are
    /// adjacent to this one. Updated by [`SubChunk::check_fully_opaque`].
    pub is_fully_opaque: bool,
}

impl SubChunk {
    /// Creates an empty sub-chunk at the given chunk-grid coordinates.
    ///
    /// All blocks are initialised to [`BlockType::Air`], `is_empty` is `true`,
    /// and `mesh_dirty` is `true` so the mesher will process it on first use.
    ///
    /// # Parameters
    /// - `chunk_x` / `chunk_z` — horizontal chunk-grid position.
    /// - `subchunk_y` — vertical sub-chunk index (0 = bottom of the world).
    pub fn new(chunk_x: i32, subchunk_y: i32, chunk_z: i32) -> Self {
        let world_x = chunk_x * CHUNK_SIZE;
        let world_y = subchunk_y * SUBCHUNK_HEIGHT;
        let world_z = chunk_z * CHUNK_SIZE;

        SubChunk {
            blocks: [[[BlockType::Air; CHUNK_SIZE as usize]; SUBCHUNK_HEIGHT as usize];
                CHUNK_SIZE as usize],
            is_empty: true,
            is_fully_opaque: false,
            mesh_dirty: true,
            num_indices: 0,
            num_water_indices: 0,
            aabb: AABB::new(
                Vec3::new(world_x as f32, world_y as f32, world_z as f32),
                Vec3::new(
                    (world_x + CHUNK_SIZE) as f32,
                    (world_y + SUBCHUNK_HEIGHT) as f32,
                    (world_z + CHUNK_SIZE) as f32,
                ),
            ),
        }
    }

    /// Returns the block at local position `(x, y, z)`.
    ///
    /// Returns [`BlockType::Air`] for any coordinate outside the valid range
    /// rather than panicking, making neighbor lookups safe without prior bounds checks.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x >= 0 && x < CHUNK_SIZE && y >= 0 && y < SUBCHUNK_HEIGHT && z >= 0 && z < CHUNK_SIZE {
            self.blocks[x as usize][y as usize][z as usize]
        } else {
            BlockType::Air
        }
    }

    /// Sets the block at local position `(x, y, z)` and marks the mesh dirty.
    ///
    /// Out-of-bounds writes are silently ignored. If `block` is not
    /// [`BlockType::Air`], `is_empty` is cleared to `false`; the flag is
    /// **not** re-set to `true` here — call [`SubChunk::check_empty`] when
    /// a full scan is needed (e.g. after bulk edits).
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if x >= 0 && x < CHUNK_SIZE && y >= 0 && y < SUBCHUNK_HEIGHT && z >= 0 && z < CHUNK_SIZE {
            self.blocks[x as usize][y as usize][z as usize] = block;
            self.mesh_dirty = true;
            if block != BlockType::Air {
                self.is_empty = false;
            }
        }
    }

    /// Scans all blocks and updates [`Self::is_empty`].
    ///
    /// Prefer this over relying solely on the incremental flag when blocks may
    /// have been removed in bulk (e.g. world generation overwriting an existing
    /// sub-chunk). Returns early on the first non-air block found.
    pub fn check_empty(&mut self) {
        self.is_empty = true;
        for x in 0..CHUNK_SIZE as usize {
            for y in 0..SUBCHUNK_HEIGHT as usize {
                for z in 0..CHUNK_SIZE as usize {
                    if self.blocks[x][y][z] != BlockType::Air {
                        self.is_empty = false;
                        return;
                    }
                }
            }
        }
    }

    /// Scans all blocks and updates [`Self::is_fully_opaque`].
    ///
    /// Sets the flag to `true` only if every block returns `true` from
    /// [`BlockType::is_solid_opaque`]. Returns early on the first non-opaque
    /// block found. Call after generation or bulk edits that may change opacity.
    pub fn check_fully_opaque(&mut self) {
        for x in 0..CHUNK_SIZE as usize {
            for y in 0..SUBCHUNK_HEIGHT as usize {
                for z in 0..CHUNK_SIZE as usize {
                    if !self.blocks[x][y][z].is_solid_opaque() {
                        self.is_fully_opaque = false;
                        return;
                    }
                }
            }
        }
        self.is_fully_opaque = true;
    }
}

/// A full-height vertical column of [`SubChunk`]s at a fixed `(x, z)` position.
///
/// A chunk spans [`WORLD_HEIGHT`] blocks vertically, divided into
/// [`NUM_SUBCHUNKS`] sub-chunks of [`SUBCHUNK_HEIGHT`] blocks each.
/// Horizontal coordinates are in chunk-local space `[0, CHUNK_SIZE)`;
/// vertical coordinates are in world space `[0, WORLD_HEIGHT)`.
pub struct Chunk {
    /// Ordered list of sub-chunks, bottom (index 0) to top (index `NUM_SUBCHUNKS - 1`).
    pub subchunks: Vec<SubChunk>,

    /// `true` if a player has placed or broken any block in this chunk.
    ///
    /// Used to prioritize saving and to distinguish generated chunks from
    /// player-modified ones.
    pub player_modified: bool,
}

impl Chunk {
    /// Creates a new chunk at chunk-grid position `(x, z)`.
    ///
    /// Allocates [`NUM_SUBCHUNKS`] sub-chunks, all initialized to air.
    pub fn new(x: i32, z: i32) -> Self {
        let mut subchunks = Vec::with_capacity(NUM_SUBCHUNKS as usize);
        for sy in 0..NUM_SUBCHUNKS {
            subchunks.push(SubChunk::new(x, sy, z));
        }
        Chunk {
            subchunks,
            player_modified: false,
        }
    }

    /// Returns the block at world-space column-local position `(x, y, z)`.
    ///
    /// `x` and `z` are in chunk-local space `[0, CHUNK_SIZE)`.
    /// Returns [`BlockType::Air`] when `y` is outside `[0, WORLD_HEIGHT)`.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= WORLD_HEIGHT {
            return BlockType::Air;
        }
        let subchunk_idx = (y / SUBCHUNK_HEIGHT) as usize;
        let local_y = y % SUBCHUNK_HEIGHT;
        self.subchunks[subchunk_idx].get_block(x, local_y, z)
    }

    /// Sets the block at world-space column-local position `(x, y, z)`.
    ///
    /// `x` and `z` are in chunk-local space `[0, CHUNK_SIZE)`.
    /// Silently ignores writes where `y` is outside `[0, WORLD_HEIGHT)`.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let subchunk_idx = (y / SUBCHUNK_HEIGHT) as usize;
        let local_y = y % SUBCHUNK_HEIGHT;
        self.subchunks[subchunk_idx].set_block(x, local_y, z, block);
    }
}