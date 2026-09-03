use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::render::quad::{PackedQuad, emit_packed_quad};
use crate::world::generator::ChunkGenerator;
use parking_lot::RwLock;
use rand::random;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::thread;

// ─────────────────────────────────────────────────────────────────────────────
// World
// ─────────────────────────────────────────────────────────────────────────────

/// The voxel world: a flat collection of [`Chunk`] columns together with the
/// generator and utilities needed to query and modify block data.
///
/// # Coordinate conventions
///
/// - **World space** – `(x, y, z)` in individual blocks.  Y is the vertical
///   axis; the valid range is `[0, WORLD_HEIGHT)`.
/// - **Chunk space** – `(cx, cz)` in chunks.  Each chunk column covers a
///   `CHUNK_SIZE × WORLD_HEIGHT × CHUNK_SIZE` block volume.
/// - **Local space** – `(lx, ly, lz)` within a single chunk or sub-chunk,
///   always in `[0, CHUNK_SIZE)` for XZ and `[0, SUBCHUNK_HEIGHT)` for Y.
///
/// Negative world coordinates are handled by floor division so that, e.g.,
/// block `(-1, y, 0)` belongs to chunk `(-1, 0)` and not chunk `(0, 0)`.
///
/// # Chunk storage
///
/// Chunks are stored in a [`FxHashMap`] keyed by `(cx, cz)`.  `FxHashMap`
/// uses a non-cryptographic hash optimized for small integer keys, which gives
/// a measurable speed improvement over the standard `HashMap` for the dense
/// lookup patterns of a voxel engine.
///
/// # Ownership
///
/// `World` owns both the chunk data and the [`ChunkGenerator`].  The generator
/// is kept here rather than in a separate thread pool so that synchronous,
/// single-call chunk generation (used during initialization and F9 world load)
/// remains simple.  Background chunk generation is delegated to
/// [`ChunkLoader`](crate::world::chunk_loader::ChunkLoader), which holds its
/// own generator clones.
pub struct World {
    /// All currently loaded chunk columns, keyed by `(cx, cz)`.
    pub chunks: FxHashMap<(i32, i32), Chunk>,

    /// Chunk coordinates at which the last unload sweep was triggered.
    /// Set to `i32::MIN` initially so the first call to
    /// `update_chunks_around_player` always runs regardless of player position.
    last_cleanup_cx: i32,
    last_cleanup_cz: i32,

    /// The seed used to initialize the terrain generator.  Stored so the world
    /// can be serialized (F5 save) and later restored with identical terrain.
    pub seed: u32,

    /// Terrain generator used for synchronous chunk generation.  Worker threads
    /// in `ChunkLoader` each hold their own clone of this generator.
    generator: ChunkGenerator,
}

const MESH_CACHE_PAD: usize = 1;
const MESH_CACHE_SIZE: usize = CHUNK_SIZE as usize + MESH_CACHE_PAD * 2;
const MESH_CACHE_HEIGHT: usize = SUBCHUNK_HEIGHT as usize + MESH_CACHE_PAD * 2;
const MESH_CACHE_LEN: usize = MESH_CACHE_SIZE * MESH_CACHE_HEIGHT * MESH_CACHE_SIZE;
const MESH_SKY_CACHE_LEN: usize = MESH_CACHE_SIZE * MESH_CACHE_SIZE;

#[derive(Clone)]
pub struct SubchunkMeshSnapshot {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub subchunk_y: i32,
    pub mesh_version: u64,
    pub has_blocks: bool,
    pub block_cache: [BlockType; MESH_CACHE_LEN],
    pub sky_height_cache: [i16; MESH_SKY_CACHE_LEN],
}

impl World {
    /// Creates a new world with the default seed (`2137`) and no loaded chunks.
    pub fn new() -> Self {
        let seed = random();
        Self::new_empty_with_seed(seed) //42
    }

    /// Creates a new empty world with the given `seed`.
    pub fn new_empty_with_seed(seed: u32) -> Self {
        World {
            chunks: FxHashMap::default(),
            last_cleanup_cx: i32::MIN,
            last_cleanup_cz: i32::MIN,
            seed,
            generator: ChunkGenerator::new(seed),
        }
    }

    /// Creates a new world with the given `seed` and pre-generates the initial
    /// chunk ring synchronously on the calling thread.
    ///
    /// The initial ring covers `[-RENDER_DISTANCE, RENDER_DISTANCE]` in both
    /// chunk X and Z, giving the player visible terrain immediately on spawn
    /// without waiting for the background `ChunkLoader`.
    pub fn new_with_seed(seed: u32) -> Self {
        let mut world = Self::new_empty_with_seed(seed);
        world.generate_chunks_in_radius(0, 0, RENDER_DISTANCE);
        world
    }

    /// Generates all chunks within `radius` of `(center_cx, center_cz)` on the
    /// calling thread.
    pub fn generate_chunks_in_radius(&mut self, center_cx: i32, center_cz: i32, radius: i32) {
        for cx in (center_cx - radius)..=(center_cx + radius) {
            for cz in (center_cz - radius)..=(center_cz + radius) {
                if !self.chunks.contains_key(&(cx, cz)) {
                    let chunk = self.generator.generate_chunk(cx, cz);
                    self.chunks.insert((cx, cz), chunk);
                }
            }
        }
    }

    /// Starts background generation of all chunks within `outer_radius` of
    /// `(center_cx, center_cz)`, skipping the inner square with radius
    /// `inner_radius`.
    ///
    /// Existing chunks are left untouched, so this can be called after a small
    /// synchronous preload without duplicating work.
    pub fn spawn_chunks_in_ring_async(
        world: Arc<RwLock<Self>>,
        center_cx: i32,
        center_cz: i32,
        inner_radius: i32,
        outer_radius: i32,
    ) {
        thread::spawn(move || {
            let seed = world.read().seed;
            let generator = ChunkGenerator::new(seed);

            for cx in (center_cx - outer_radius)..=(center_cx + outer_radius) {
                for cz in (center_cz - outer_radius)..=(center_cz + outer_radius) {
                    if (cx - center_cx).abs().max((cz - center_cz).abs()) <= inner_radius {
                        continue;
                    }
                    let chunk = generator.generate_chunk(cx, cz);
                    let mut world = world.write();
                    if !world.chunks.contains_key(&(cx, cz)) {
                        world.chunks.insert((cx, cz), chunk);
                    }
                }
            }
        });
    }

    /// Ensures chunk `(cx, cz)` is present in the world, generating it
    /// synchronously if it has not been loaded yet.
    ///
    /// Used for ad-hoc lookups (e.g., raycast, decoration queries) that must
    /// not return `Air` just because the chunk hasn't been scheduled yet.
    pub fn ensure_chunk_generated(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        self.generate_chunk(cx, cz);
    }

    /// Unloads chunks that have moved outside `CHUNK_UNLOAD_DISTANCE` of the
    /// player's current chunk column.
    ///
    /// The sweep is skipped when the player hasn't moved to a different chunk
    /// column since the last call (tracked via `last_cleanup_cx/cz`), avoiding
    /// the cost of iterating the full chunk map every frame.
    ///
    /// # Returns
    /// The list of `(cx, cz)` keys that were removed.  The caller uses this
    /// to invalidate GPU buffers for those chunk columns.
    pub fn update_chunks_around_player(&mut self, player_x: f32, player_z: f32) -> Vec<(i32, i32)> {
        let player_cx = (player_x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (player_z / CHUNK_SIZE as f32).floor() as i32;

        // Early exit: player is still in the same chunk column.
        if player_cx == self.last_cleanup_cx && player_cz == self.last_cleanup_cz {
            return Vec::new();
        }
        self.last_cleanup_cx = player_cx;
        self.last_cleanup_cz = player_cz;

        // Collect keys to remove; can't remove while iterating.
        let chunks_to_remove: Vec<(i32, i32)> = self
            .chunks
            .keys()
            .filter(|(cx, cz)| {
                let dx = (*cx - player_cx).abs();
                let dz = (*cz - player_cz).abs();
                dx > CHUNK_UNLOAD_DISTANCE || dz > CHUNK_UNLOAD_DISTANCE
            })
            .cloned()
            .collect();

        for key in &chunks_to_remove {
            self.chunks.remove(key);
        }

        chunks_to_remove
    }

    // ── Generator pass-throughs ───────────────────────────────────────────── //

    /// Returns the biome at world position `(x, z)`.
    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        self.generator.get_biome(x, z)
    }

    /// Returns the terrain surface height at world position `(x, z)`.
    pub fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        self.generator.get_terrain_height_pub(x, z)
    }

    /// Returns `true` if `(x, z)` at the given `surface_height` is a cave
    /// entrance column according to the generator's entrance heuristic.
    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.generator.is_cave_entrance_pub(x, z, surface_height)
    }

    // ── Chunk generation ──────────────────────────────────────────────────── //

    /// Generates chunk `(cx, cz)` and inserts it into the world map.
    ///
    /// This is the synchronous path used by `ensure_chunk_generated` and
    /// `new_with_seed`.  Background generation is handled by `ChunkLoader`.
    fn generate_chunk(&mut self, cx: i32, cz: i32) {
        let chunk = self.generator.generate_chunk(cx, cz);
        self.chunks.insert((cx, cz), chunk);
    }

    // ── Block access ──────────────────────────────────────────────────────── //

    /// Returns the block type at world position `(x, y, z)`.
    ///
    /// Returns `Air` for positions outside `[0, WORLD_HEIGHT)` or in
    /// unloaded chunks.
    ///
    /// # Coordinate conversion
    /// Chunk coordinates are computed with floor division (`div_euclid` for
    /// negatives) so that negative world coordinates map to negative chunk
    /// coordinates rather than chunk 0.  Local coordinates use `rem_euclid`
    /// to stay in `[0, CHUNK_SIZE)` regardless of sign.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= WORLD_HEIGHT {
            return BlockType::Air;
        }
        // Floor-division for correct negative-coordinate chunk lookup.
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
            chunk.get_block(lx, y, lz)
        } else {
            BlockType::Air // chunk not loaded
        }
    }

    /// Sets the block at world position `(x, y, z)` to `block`.
    ///
    /// Silently no-ops if `y` is out of range or the chunk is not loaded.
    /// Does **not** set `chunk.player_modified`; use [`set_block_player`] for
    /// player-initiated edits that should be preserved by the save system.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set_block(lx, y, lz, block);
        }
    }

    /// Sets the block at world position `(x, y, z)` and marks the chunk as
    /// player-modified.
    ///
    /// Player-modified chunks are included in the world save file (F5) so
    /// that edits persist across sessions.  Use this for all block changes
    /// initiated by player interaction (digging, placing).  Use [`set_block`]
    /// for programmatic changes (cave carving, world load restoration) that
    /// should not trigger save inclusion on their own.
    ///
    /// Placing any non-air block directly above grass immediately turns that
    /// grass block into dirt, matching the covered-grass behavior players
    /// expect from block placement.
    pub fn set_block_player(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let cx = if x >= 0 {
            x / CHUNK_SIZE
        } else {
            (x - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let cz = if z >= 0 {
            z / CHUNK_SIZE
        } else {
            (z - CHUNK_SIZE + 1) / CHUNK_SIZE
        };
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set_block(lx, y, lz, block);
            chunk.player_modified = true; // flag for save-on-F5
        }

        if block != BlockType::Air && y > 0 && self.get_block(x, y - 1, z) == BlockType::Grass {
            if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
                chunk.set_block(lx, y - 1, lz, BlockType::Dirt);
                chunk.player_modified = true;
            }
        }
    }

    /// Returns `true` if the block at `(x, y, z)` is solid (i.e., has
    /// non-zero collision volume).
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_block(x, y, z).is_solid()
    }

    /// Estimates how much sky light reaches a point from directly above.
    ///
    /// This is intentionally cheap and camera-centric: terrain lighting still
    /// runs per fragment on the GPU, while this value tells the shader whether
    /// the current area is broadly outdoors or under a solid ceiling.
    pub fn sky_visibility_at(&self, x: i32, y: i32, z: i32) -> f32 {
        if y < 0 {
            return 0.0;
        }
        if y >= WORLD_HEIGHT {
            return 1.0;
        }

        const SAMPLE_OFFSETS: [(i32, i32); 5] = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)];

        let start_y = (y + 1).clamp(0, WORLD_HEIGHT);
        let mut visibility_sum = 0.0;

        for (dx, dz) in SAMPLE_OFFSETS {
            let mut column_visibility = 1.0;

            for sample_y in start_y..WORLD_HEIGHT {
                let block = self.get_block(x + dx, sample_y, z + dz);
                if block.is_solid_opaque() {
                    column_visibility = 0.0;
                    break;
                }
            }

            visibility_sum += column_visibility;
        }

        visibility_sum / SAMPLE_OFFSETS.len() as f32
    }

    // ── Occlusion culling ─────────────────────────────────────────────────── //

    /// Returns `true` if sub-chunk `(cx, cz, sy)` is fully occluded and can
    /// be skipped by the renderer entirely.
    ///
    /// A sub-chunk is considered occluded when **all** of the following hold:
    ///
    /// 1. The sub-chunk itself is `is_fully_opaque` (no transparent gaps).
    /// 2. The sub-chunks directly above and below it are also `is_fully_opaque`.
    /// 3. The sub-chunk is not at the top or bottom of its chunk column
    ///    (boundary sub-chunks always face open air or unloaded space).
    /// 4. The sub-chunks at the same Y level in all four cardinal-direction
    ///    neighbors are `is_fully_opaque` **and** those neighbor chunks are
    ///    loaded.
    ///
    /// If any neighbor chunk is absent the function returns `false`
    /// conservatively (treat as visible) rather than incorrectly culling.
    pub fn is_subchunk_occluded(&self, cx: i32, cz: i32, sy: i32) -> bool {
        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
            // Rule 1: the sub-chunk itself must be fully opaque.
            if !chunk.subchunks[sy as usize].is_fully_opaque {
                return false;
            }
            // Rule 2: vertical neighbors must also be fully opaque.
            if sy > 0 && !chunk.subchunks[(sy - 1) as usize].is_fully_opaque {
                return false;
            }
            if sy < NUM_SUBCHUNKS - 1 && !chunk.subchunks[(sy + 1) as usize].is_fully_opaque {
                return false;
            }
            // Rule 3: boundary sub-chunks are never occluded (they always
            // border open air or unloaded terrain above/below).
            if sy == 0 || sy == NUM_SUBCHUNKS - 1 {
                return false;
            }

            // Rule 4: all four horizontal neighbors at the same Y level must
            // be fully opaque and present.
            let neighbors = [(cx - 1, cz), (cx + 1, cz), (cx, cz - 1), (cx, cz + 1)];
            for (ncx, ncz) in neighbors {
                if let Some(nchunk) = self.chunks.get(&(ncx, ncz)) {
                    if !nchunk.subchunks[sy as usize].is_fully_opaque {
                        return false;
                    }
                } else {
                    return false; // unloaded neighbour → assume visible
                }
            }

            return true;
        }
        false
    }

    // ── Spawn point search ────────────────────────────────────────────────── //

    /// Searches outward from the origin in a spiral of expanding radii to find
    /// a suitable player spawn position.
    ///
    /// A column is acceptable when its terrain height is at or above sea level
    /// and its biome is not Ocean, River, or Lake (the player would spawn
    /// underwater or on an unstable floor).
    ///
    /// The returned Y coordinate places the player one block above the surface
    /// with a small XZ offset so the player doesn't fall into a 1×1 crevice
    /// at exactly (0, y, 0).
    ///
    /// # Returns
    /// `(x, y, z)` in world space.  Falls back to `(0.5, 80.0, 0.5)` if no
    /// suitable column is found within radius 50 (which should never happen in
    /// practice for non-degenerate worlds).
    pub fn find_spawn_point(&self) -> (f32, f32, f32) {
        for radius in 0..50 {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let x = dx;
                    let z = dz;
                    let height = self.get_terrain_height(x, z);
                    let biome = self.get_biome(x, z);
                    let spawn_x = x as f32 + 0.3;
                    let spawn_y = (height + 1) as f32;
                    let spawn_z = z as f32 + 0.5;

                    if height >= SEA_LEVEL
                        && !matches!(biome, Biome::Ocean | Biome::River | Biome::Lake)
                        && !self.is_cave_entrance(x, z, height)
                        && self.spawn_space_is_clear(spawn_x, spawn_y, spawn_z)
                    {
                        // +0.3 / +0.5 offsets prevent the player from being
                        // centred on a block edge and avoid false collision
                        // positives at the moment of spawn.
                        return (spawn_x, spawn_y, spawn_z);
                    }
                }
            }
        }
        (0.5, 80.0, 0.5) // fallback
    }

    /// Returns whether the standing player's body AABB is free of solid blocks.
    ///
    /// The player is 1.8 blocks tall, so checking the two block layers crossed
    /// by the body prevents spawning inside trees, ceilings, or other terrain
    /// decorations. Unloaded chunks are deliberately treated as unsafe here:
    /// spawn selection is performed before gameplay starts and must not choose
    /// a position whose contents are unknown.
    fn spawn_space_is_clear(&self, x: f32, y: f32, z: f32) -> bool {
        let min_x = (x - PLAYER_WIDTH).floor() as i32;
        let max_x = (x + PLAYER_WIDTH).floor() as i32;
        let min_z = (z - PLAYER_WIDTH).floor() as i32;
        let max_z = (z + PLAYER_WIDTH).floor() as i32;
        let min_y = y.floor() as i32;
        let max_y = (y + PLAYER_HEIGHT).floor() as i32;

        for bx in min_x..=max_x {
            for by in min_y..=max_y {
                for bz in min_z..=max_z {
                    let chunk_coords = (
                        bx.div_euclid(CHUNK_SIZE),
                        bz.div_euclid(CHUNK_SIZE),
                    );
                    if !self.chunks.contains_key(&chunk_coords) {
                        return false;
                    }
                    if self.get_block(bx, by, bz).is_solid() {
                        return false;
                    }
                }
            }
        }
        true
    }

    // ── Mesh generation ───────────────────────────────────────────────────── //

    /// Builds the opaque and water vertex/index meshes for one sub-chunk.
    ///
    /// # Algorithm overview
    ///
    /// ## 1. Block cache (`block_cache`)
    ///
    /// A flat array of size `(CHUNK_SIZE + 2) × (SUBCHUNK_HEIGHT + 2) ×
    /// (CHUNK_SIZE + 2)` (18³ = 5832 entries) is filled with block types
    /// sampled from `self.chunks`.  The extra ±1 padding on all six faces
    /// means that neighbor lookups during face visibility tests never need a
    /// hash-map access — they always hit the cache.  Blocks in unloaded
    /// neighboring chunks default to `Water` below sea level and `Air` above.
    ///
    /// ## 2. WoodStairs (custom geometry, pre-pass)
    ///
    /// Stair blocks require non-axis-aligned geometry (two horizontally
    /// stacked half-blocks) that cannot be expressed as a single full-face
    /// quad. They are rendered as individual packed descriptors in a
    /// dedicated pre-pass loop and then **excluded** from the greedy meshing
    /// loop via an explicit `continue`.
    ///
    /// ## 3. Greedy meshing (main pass, 6 face directions)
    ///
    /// For each of the six axis-aligned face directions the algorithm:
    ///
    /// a. **Populates a `mask`** – a 2-D array of [`FaceAttrs`] for the
    ///    current slice.  A slot is active when the block on the near side
    ///    should render a face against its neighbor on the far side.
    ///    Water blocks are handled specially: they are emitted immediately as
    ///    individual quads rather than entering the mask (no greedy merging for
    ///    water, since water faces never share the same texture/color).
    ///    Stair blocks are also skipped here (already handled above).
    ///
    /// b. **Greedy merges** – scans the mask in row-major order.  Starting
    ///    from each active cell, extends a rectangle first along `d2` (width)
    ///    until the next cell differs, then along `d1` (height) checking
    ///    that every cell in the expanded row matches.  The merged rectangle
    ///    is emitted as a single packed descriptor and the covered
    ///    cells are marked inactive.
    ///
    /// The greedy approach dramatically reduces vertex count for large flat
    /// surfaces (e.g., a 16×16 grass top becomes one quad instead of 256).
    ///
    /// ## Face direction encoding
    ///
    /// | `face_dir` | Normal | Slice axis | d1 axis | d2 axis |
    /// |---|---|---|---|---|
    /// | 0 | −X | X | Y | Z |
    /// | 1 | +X | X | Y | Z |
    /// | 2 | −Y | Y | X | Z |
    /// | 3 | +Y | Y | X | Z |
    /// | 4 | −Z | Z | X | Y |
    /// | 5 | +Z | Z | X | Y |
    ///
    /// ## Lighting and texture selection
    ///
    /// The packed vertex color stores a coarse local GI tint rather than
    /// material albedo. It is derived from whether the exposed air cell can see
    /// the sky and how open the nearby voxel neighborhood is. Texture color
    /// still comes from the atlas.
    ///
    /// GI colors are quantized to the packed vertex color precision before
    /// storage in `FaceAttrs` so that hidden floating-point differences do not
    /// prevent adjacent faces from being merged.
    ///
    /// # Parameters
    /// - `chunk_x`   – Chunk column X coordinate.
    /// - `chunk_z`   – Chunk column Z coordinate.
    /// - `subchunk_y` – Sub-chunk vertical index within the column.
    ///
    /// # Returns
    /// A pair of packed terrain and water descriptor streams:
    /// - First tuple: opaque geometry.
    /// - Second tuple: water (translucent) geometry.
    pub fn snapshot_subchunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> Option<SubchunkMeshSnapshot> {
        let mesh_version = self
            .chunks
            .get(&(chunk_x, chunk_z))?
            .subchunks
            .get(subchunk_y as usize)?
            .mesh_version;

        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;

        let mut block_cache = [BlockType::Air; MESH_CACHE_LEN];
        let mut sky_height_cache = [-1i16; MESH_SKY_CACHE_LEN];
        let mut has_blocks = false;

        for px in 0..MESH_CACHE_SIZE as i32 {
            for py in 0..MESH_CACHE_HEIGHT as i32 {
                for pz in 0..MESH_CACHE_SIZE as i32 {
                    let wx = base_x + px - MESH_CACHE_PAD as i32;
                    let wy = base_y + py - MESH_CACHE_PAD as i32;
                    let wz = base_z + pz - MESH_CACHE_PAD as i32;
                    let block = if wy < 0 || wy >= WORLD_HEIGHT {
                        BlockType::Air
                    } else {
                        let cx = wx.div_euclid(CHUNK_SIZE);
                        let cz = wz.div_euclid(CHUNK_SIZE);
                        let lx = wx.rem_euclid(CHUNK_SIZE);
                        let lz = wz.rem_euclid(CHUNK_SIZE);
                        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                            chunk.get_block(lx, wy, lz)
                        } else if wy < SEA_LEVEL {
                            BlockType::Water
                        } else {
                            BlockType::Air
                        }
                    };

                    if px > 0
                        && px < (MESH_CACHE_SIZE - 1) as i32
                        && py > 0
                        && py < (MESH_CACHE_HEIGHT - 1) as i32
                        && pz > 0
                        && pz < (MESH_CACHE_SIZE - 1) as i32
                        && block != BlockType::Air
                    {
                        has_blocks = true;
                    }

                    block_cache[(px as usize) * MESH_CACHE_HEIGHT * MESH_CACHE_SIZE
                        + (py as usize) * MESH_CACHE_SIZE
                        + (pz as usize)] = block;
                }
            }
        }

        if has_blocks {
            for px in 0..MESH_CACHE_SIZE as i32 {
                for pz in 0..MESH_CACHE_SIZE as i32 {
                    let wx = base_x + px - MESH_CACHE_PAD as i32;
                    let wz = base_z + pz - MESH_CACHE_PAD as i32;
                    let cx = wx.div_euclid(CHUNK_SIZE);
                    let cz = wz.div_euclid(CHUNK_SIZE);
                    let lx = wx.rem_euclid(CHUNK_SIZE);
                    let lz = wz.rem_euclid(CHUNK_SIZE);
                    let highest_opaque = self
                        .chunks
                        .get(&(cx, cz))
                        .map(|chunk| chunk.highest_opaque_y(lx, lz))
                        .unwrap_or(-1);

                    sky_height_cache[(px as usize) * MESH_CACHE_SIZE + pz as usize] =
                        highest_opaque;
                }
            }
        }

        Some(SubchunkMeshSnapshot {
            chunk_x,
            chunk_z,
            subchunk_y,
            mesh_version,
            has_blocks,
            block_cache,
            sky_height_cache,
        })
    }

    pub fn build_subchunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> (
        Vec<crate::render::quad::PackedQuad>,
        Vec<crate::render::quad::PackedQuad>,
    ) {
        let snapshot = match self.snapshot_subchunk_mesh(chunk_x, chunk_z, subchunk_y) {
            Some(snapshot) => snapshot,
            None => return (Vec::new(), Vec::new()),
        };

        Self::build_subchunk_mesh_from_snapshot(&self.generator, &snapshot)
    }

    pub fn build_subchunk_mesh_from_snapshot(
        _generator: &ChunkGenerator,
        snapshot: &SubchunkMeshSnapshot,
    ) -> (
        Vec<crate::render::quad::PackedQuad>,
        Vec<crate::render::quad::PackedQuad>,
    ) {
        let mut terrain_quads = Vec::with_capacity(512);
        let mut water_quads = Vec::with_capacity(128);

        if !snapshot.has_blocks {
            return (Vec::new(), Vec::new());
        }

        let chunk_x = snapshot.chunk_x;
        let chunk_z = snapshot.chunk_z;
        let subchunk_y = snapshot.subchunk_y;
        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;
        let subchunk_origin = [base_x, base_y, base_z];

        // ── Block cache setup ─────────────────────────────────────────────── //
        // 1-block padding on all sides so neighbor lookups never need a
        // hash-map access during the face visibility and greedy merge tests.
        const PAD: usize = MESH_CACHE_PAD;
        const S: usize = MESH_CACHE_SIZE;
        const SH: usize = MESH_CACHE_HEIGHT;
        let block_cache = &snapshot.block_cache;

        // Fast cache lookup in sub-chunk-local coordinates.
        let get_block_fast = |lx: i32, ly: i32, lz: i32| -> BlockType {
            let px = (lx + PAD as i32) as usize;
            let py = (ly + PAD as i32) as usize;
            let pz = (lz + PAD as i32) as usize;
            block_cache[px * SH * S + py * S + pz]
        };

        // Fast cache lookup in world coordinates (converts to local first).
        let get_block_world = |wx: i32, wy: i32, wz: i32| -> BlockType {
            get_block_fast(wx - base_x, wy - base_y, wz - base_z)
        };

        let get_block_world_safe = |wx: i32, wy: i32, wz: i32| -> BlockType {
            if !(0..WORLD_HEIGHT).contains(&wy) {
                return BlockType::Air;
            }

            let px = wx - base_x + PAD as i32;
            let py = wy - base_y + PAD as i32;
            let pz = wz - base_z + PAD as i32;

            if px < 0 || px >= S as i32 || py < 0 || py >= SH as i32 || pz < 0 || pz >= S as i32 {
                return BlockType::Air;
            }

            block_cache[(px as usize) * SH * S + (py as usize) * S + pz as usize]
        };

        let sky_open_column = |wx: i32, wy: i32, wz: i32| -> f32 {
            let px = wx - base_x + PAD as i32;
            let pz = wz - base_z + PAD as i32;

            if px < 0 || px >= S as i32 || pz < 0 || pz >= S as i32 {
                return 1.0;
            }

            let highest_opaque = snapshot.sky_height_cache[(px as usize) * S + pz as usize];
            if highest_opaque < 0 || wy >= highest_opaque as i32 {
                1.0
            } else {
                0.0
            }
        };

        let sample_sky_aperture = |wx: i32, wy: i32, wz: i32| -> f32 {
            const SKY_OFFSETS: [(i32, i32, f32); 5] = [
                (0, 0, 0.52),
                (1, 0, 0.12),
                (-1, 0, 0.12),
                (0, 1, 0.12),
                (0, -1, 0.12),
            ];

            let mut sky = 0.0;
            let mut weight = 0.0;

            for (dx, dz, w) in SKY_OFFSETS {
                sky += sky_open_column(wx + dx, wy, wz + dz) * w;
                weight += w;
            }

            sky / weight
        };

        let sample_local_openness = |wx: i32, wy: i32, wz: i32| -> f32 {
            const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ];

            let mut open = 0.0;
            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                if get_block_world_safe(wx + dx, wy + dy, wz + dz).is_transparent() {
                    open += 1.0;
                }
            }

            open / NEIGHBOR_OFFSETS.len() as f32
        };

        let face_gi_tint = |face_dir: i32, wx: i32, wy: i32, wz: i32| -> [f32; 3] {
            let (sx, sy, sz) = match face_dir {
                0 => (wx - 1, wy, wz),
                1 => (wx + 1, wy, wz),
                2 => (wx, wy - 1, wz),
                3 => (wx, wy + 1, wz),
                4 => (wx, wy, wz - 1),
                5 => (wx, wy, wz + 1),
                _ => unreachable!(),
            };

            let sky = sample_sky_aperture(sx, sy, sz);
            let openness = sample_local_openness(sx, sy, sz);
            let facing_sky = match face_dir {
                3 => 0.95,
                0 | 1 | 4 | 5 => 0.68,
                2 => 0.24,
                _ => 0.0,
            };
            let bounce = sky * openness;
            let energy =
                (0.13 + sky * facing_sky * 0.86 + bounce * 0.20 + openness * 0.05).clamp(0.10, 1.0);

            [
                energy * (0.88 + sky * 0.08 + bounce * 0.04),
                energy * (0.91 + sky * 0.06 + bounce * 0.03),
                energy * (0.98 + sky * 0.02),
            ]
        };

        let occludes_ao = |wx: i32, wy: i32, wz: i32| -> bool {
            get_block_world_safe(wx, wy, wz).is_solid_opaque()
        };

        let vertex_ao = |side_a: bool, side_b: bool, diagonal: bool| -> u8 {
            if side_a && side_b {
                0
            } else {
                3u8 - side_a as u8 - side_b as u8 - diagonal as u8
            }
        };

        let face_corner_ao = |face_dir: i32, wx: i32, wy: i32, wz: i32| -> [u8; 4] {
            match face_dir {
                0 => {
                    let ox = wx - 1;
                    let sample = |sy: i32, sz: i32| {
                        vertex_ao(
                            occludes_ao(ox, wy + sy, wz),
                            occludes_ao(ox, wy, wz + sz),
                            occludes_ao(ox, wy + sy, wz + sz),
                        )
                    };
                    [sample(-1, -1), sample(-1, 1), sample(1, 1), sample(1, -1)]
                }
                1 => {
                    let ox = wx + 1;
                    let sample = |sy: i32, sz: i32| {
                        vertex_ao(
                            occludes_ao(ox, wy + sy, wz),
                            occludes_ao(ox, wy, wz + sz),
                            occludes_ao(ox, wy + sy, wz + sz),
                        )
                    };
                    [sample(-1, 1), sample(-1, -1), sample(1, -1), sample(1, 1)]
                }
                2 => {
                    let oy = wy - 1;
                    let sample = |sx: i32, sz: i32| {
                        vertex_ao(
                            occludes_ao(wx + sx, oy, wz),
                            occludes_ao(wx, oy, wz + sz),
                            occludes_ao(wx + sx, oy, wz + sz),
                        )
                    };
                    [sample(-1, 1), sample(-1, -1), sample(1, -1), sample(1, 1)]
                }
                3 => {
                    let oy = wy + 1;
                    let sample = |sx: i32, sz: i32| {
                        vertex_ao(
                            occludes_ao(wx + sx, oy, wz),
                            occludes_ao(wx, oy, wz + sz),
                            occludes_ao(wx + sx, oy, wz + sz),
                        )
                    };
                    [sample(-1, -1), sample(-1, 1), sample(1, 1), sample(1, -1)]
                }
                4 => {
                    let oz = wz - 1;
                    let sample = |sx: i32, sy: i32| {
                        vertex_ao(
                            occludes_ao(wx + sx, wy, oz),
                            occludes_ao(wx, wy + sy, oz),
                            occludes_ao(wx + sx, wy + sy, oz),
                        )
                    };
                    [sample(1, -1), sample(-1, -1), sample(-1, 1), sample(1, 1)]
                }
                5 => {
                    let oz = wz + 1;
                    let sample = |sx: i32, sy: i32| {
                        vertex_ao(
                            occludes_ao(wx + sx, wy, oz),
                            occludes_ao(wx, wy + sy, oz),
                            occludes_ao(wx + sx, wy + sy, oz),
                        )
                    };
                    [sample(-1, -1), sample(1, -1), sample(1, 1), sample(-1, 1)]
                }
                _ => unreachable!(),
            }
        };

        // ── FaceAttrs: per-cell data stored in the greedy mask ────────────── //
        // Two faces can be merged only when all fields are equal, so local GI
        // tints are pre-quantized (see `quantize_color`) to suppress
        // floating-point rounding noise that would otherwise prevent merging.
        #[derive(Clone, Copy, PartialEq)]
        struct FaceAttrs {
            block: BlockType,
            color: [u8; 3],
            ao: [u8; 4],
            tex_index: u8,
            is_active: bool,
        }

        impl Default for FaceAttrs {
            fn default() -> Self {
                FaceAttrs {
                    block: BlockType::Air,
                    color: [0, 0, 0],
                    ao: [3; 4],
                    tex_index: 0,
                    is_active: false,
                }
            }
        }

        // Quantize to the same 3-bit precision used by the packed vertex.
        // This keeps greedy merging aligned with what the shader can display.
        let quantize_color = |c: [f32; 3]| -> [u8; 3] {
            [
                (c[0].clamp(0.0, 1.0) * 7.0) as u8,
                (c[1].clamp(0.0, 1.0) * 7.0) as u8,
                (c[2].clamp(0.0, 1.0) * 7.0) as u8,
            ]
        };
        // Custom shapes (currently stairs) still describe their corners
        // explicitly, but write the compact descriptor immediately.
        let emit_corner_quad = |quads: &mut Vec<PackedQuad>,
                                v0: [f32; 3],
                                v1: [f32; 3],
                                v2: [f32; 3],
                                _v3: [f32; 3],
                                normal: [f32; 3],
                                color: [f32; 3],
                                material: f32,
                                _roughness: f32,
                                _metallic: f32| {
            let face = if normal[0] < -0.5 {
                0
            } else if normal[0] > 0.5 {
                1
            } else if normal[1] < -0.5 {
                2
            } else if normal[1] > 0.5 {
                3
            } else if normal[2] < -0.5 {
                4
            } else {
                5
            };
            let edge_len = |a: [f32; 3], b: [f32; 3]| {
                ((a[0] - b[0])
                    .abs()
                    .max((a[1] - b[1]).abs())
                    .max((a[2] - b[2]).abs())
                    * 2.0)
                    .round() as u32
            };
            emit_packed_quad(
                quads,
                subchunk_origin,
                v0,
                face,
                edge_len(v0, v1),
                edge_len(v1, v2),
                material as u8,
                quantize_color(color),
                [3; 4],
            );
        };
        let empty_face = FaceAttrs::default();
        let mut mask = [empty_face; (CHUNK_SIZE as usize) * (SUBCHUNK_HEIGHT as usize)];

        // ── Pass 1: WoodStairs custom geometry ────────────────────────────── //
        // Stair blocks are composed of two non-unit-height quads that cannot
        // be expressed as standard greedy-merged full faces.  They are emitted
        // here as explicit packed descriptors and excluded from pass 2.
        for lx in 0..CHUNK_SIZE {
            for ly in 0..SUBCHUNK_HEIGHT {
                for lz in 0..CHUNK_SIZE {
                    let y = base_y + ly;
                    let world_x = base_x + lx;
                    let world_z = base_z + lz;
                    let block = get_block_world(world_x, y, world_z);

                    if block == BlockType::Air {
                        continue;
                    }

                    if block == BlockType::WoodStairs {
                        let x = world_x as f32;
                        let y_f = y as f32;
                        let z = world_z as f32;
                        let light_neg_x = face_gi_tint(0, world_x, y, world_z);
                        let light_pos_x = face_gi_tint(1, world_x, y, world_z);
                        let light_neg_y = face_gi_tint(2, world_x, y, world_z);
                        let light_pos_y = face_gi_tint(3, world_x, y, world_z);
                        let light_neg_z = face_gi_tint(4, world_x, y, world_z);
                        let light_pos_z = face_gi_tint(5, world_x, y, world_z);
                        let tex_top = block.tex_top();
                        let tex_side = block.tex_side();
                        let r = block.roughness();
                        let m = block.metallic();

                        let neighbors = [
                            get_block_world(world_x - 1, y, world_z), // 0: −X
                            get_block_world(world_x + 1, y, world_z), // 1: +X
                            get_block_world(world_x, y - 1, world_z), // 2: −Y
                            get_block_world(world_x, y + 1, world_z), // 3: +Y
                            get_block_world(world_x, y, world_z - 1), // 4: −Z
                            get_block_world(world_x, y, world_z + 1), // 5: +Z
                        ];

                        // Bottom face (full, conditional on −Y neighbor).
                        if block.should_render_face_against(neighbors[2]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x, y_f, z + 1.0],
                                [x, y_f, z],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f, z + 1.0],
                                [0.0, -1.0, 0.0],
                                light_neg_y,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Lower half-top (always visible: the step tread at Y+0.5,
                        // front half Z=[0, 0.5]).
                        emit_corner_quad(
                            &mut terrain_quads,
                            [x, y_f + 0.5, z],
                            [x, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z],
                            [0.0, 1.0, 0.0],
                            light_pos_y,
                            tex_top,
                            r,
                            m,
                        );
                        // Upper full-top (conditional on +Y neighbor).
                        if block.should_render_face_against(neighbors[3]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x, y_f + 1.0, z + 0.5],
                                [x, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [0.0, 1.0, 0.0],
                                light_pos_y,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Front face (−Z, lower half only, conditional).
                        if block.should_render_face_against(neighbors[4]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x + 1.0, y_f, z],
                                [x, y_f, z],
                                [x, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z],
                                [0.0, 0.0, -1.0],
                                light_neg_z,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Step riser (always visible: the vertical face between
                        // the lower and upper treads at Z+0.5).
                        emit_corner_quad(
                            &mut terrain_quads,
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x, y_f + 0.5, z + 0.5],
                            [x, y_f + 1.0, z + 0.5],
                            [x + 1.0, y_f + 1.0, z + 0.5],
                            [0.0, 0.0, -1.0],
                            light_neg_z,
                            tex_side,
                            r,
                            m,
                        );
                        // Back face (+Z, full height, conditional).
                        if block.should_render_face_against(neighbors[5]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x, y_f, z + 1.0],
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [0.0, 0.0, 1.0],
                                light_pos_z,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Left face (−X): two quads – lower half and upper-back half.
                        if block.should_render_face_against(neighbors[0]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x, y_f, z],
                                [x, y_f, z + 1.0],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 0.5, z],
                                [-1.0, 0.0, 0.0],
                                light_neg_x,
                                tex_side,
                                r,
                                m,
                            );
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x, y_f + 0.5, z + 0.5],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 0.5],
                                [-1.0, 0.0, 0.0],
                                light_neg_x,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Right face (+X): two quads – lower half and upper-back half.
                        if block.should_render_face_against(neighbors[1]) {
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [1.0, 0.0, 0.0],
                                light_pos_x,
                                tex_side,
                                r,
                                m,
                            );
                            emit_corner_quad(
                                &mut terrain_quads,
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [x + 1.0, y_f + 0.5, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [1.0, 0.0, 0.0],
                                light_pos_x,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        continue; // skip greedy pass for this block
                    }
                }
            }
        }

        // ── Pass 2: greedy meshing for all standard blocks (6 face directions) //
        for face_dir in 0..6 {
            // Map face direction to (slice axis count, d1 axis size, d2 axis size).
            let (slice_count, dim1_size, dim2_size): (i32, i32, i32) = match face_dir {
                0 | 1 => (CHUNK_SIZE, SUBCHUNK_HEIGHT, CHUNK_SIZE), // X-slices
                2 | 3 => (SUBCHUNK_HEIGHT, CHUNK_SIZE, CHUNK_SIZE), // Y-slices
                4 | 5 => (CHUNK_SIZE, CHUNK_SIZE, SUBCHUNK_HEIGHT), // Z-slices
                _ => unreachable!(),
            };

            for slice in 0..slice_count {
                // The mask stores one FaceAttrs entry per (d1, d2) cell.
                let mask_len = (dim1_size * dim2_size) as usize;
                mask[..mask_len].fill(empty_face);

                // ── Populate mask for this slice ──────────────────────────── //
                for d1 in 0..dim1_size {
                    for d2 in 0..dim2_size {
                        // Convert (slice, d1, d2) to sub-chunk local coords.
                        let (lx, ly, lz): (i32, i32, i32) = match face_dir {
                            0 | 1 => (slice, d1, d2),
                            2 | 3 => (d1, slice, d2),
                            4 | 5 => (d1, d2, slice),
                            _ => unreachable!(),
                        };

                        let y = base_y + ly;
                        let world_x = base_x + lx;
                        let world_z = base_z + lz;
                        let block = get_block_world(world_x, y, world_z);

                        // Water top faces are vertex-displaced in the shader.
                        // Keep them as 1x1 quads so the wave deformation has
                        // enough tessellation and does not produce large planar
                        // facets across merged surfaces.
                        if block == BlockType::Water && face_dir == 3 {
                            let neighbor = get_block_world(world_x, y + 1, world_z);
                            if block.should_render_face_against(neighbor) {
                                let x = world_x as f32;
                                let y_f = y as f32;
                                let z = world_z as f32;
                                emit_corner_quad(
                                    &mut water_quads,
                                    [x, y_f + 1.0, z],
                                    [x, y_f + 1.0, z + 1.0],
                                    [x + 1.0, y_f + 1.0, z + 1.0],
                                    [x + 1.0, y_f + 1.0, z],
                                    [0.0, 1.0, 0.0],
                                    block.color(),
                                    block.tex_for_face(3),
                                    block.roughness(),
                                    block.metallic(),
                                );
                            }
                            continue;
                        }

                        // Skip Air and Stairs (handled in pass 1 or by transparency).
                        if block == BlockType::Air || block == BlockType::WoodStairs {
                            continue;
                        }

                        // Determine the world position of the neighbor in the
                        // face-normal direction.
                        let (nx, ny, nz) = match face_dir {
                            0 => (world_x - 1, y, world_z),
                            1 => (world_x + 1, y, world_z),
                            2 => (world_x, y - 1, world_z),
                            3 => (world_x, y + 1, world_z),
                            4 => (world_x, y, world_z - 1),
                            5 => (world_x, y, world_z + 1),
                            _ => unreachable!(),
                        };
                        let neighbor = get_block_world(nx, ny, nz);

                        // Face is only visible when the current block "should
                        // render" against its neighbor (transparent, different
                        // water status, etc.).
                        if !block.should_render_face_against(neighbor) {
                            continue;
                        }

                        let gi_tint = face_gi_tint(face_dir, world_x, y, world_z);
                        let ao = if block == BlockType::Water {
                            [3; 4]
                        } else {
                            face_corner_ao(face_dir, world_x, y, world_z)
                        };

                        // Select the atlas texture index by face direction.
                        let tex_index = block.tex_for_face(face_dir);

                        let idx = (d1 * dim2_size + d2) as usize;
                        mask[idx] = FaceAttrs {
                            block,
                            color: quantize_color(gi_tint),
                            ao,
                            tex_index: tex_index as u8,
                            is_active: true,
                        };
                    }
                }

                // ── Greedy merge and emit quads ───────────────────────────── //
                for d1 in 0..dim1_size {
                    let mut d2 = 0;
                    while d2 < dim2_size {
                        let idx = (d1 * dim2_size + d2) as usize;
                        let face = mask[idx];

                        if !face.is_active {
                            d2 += 1;
                            continue;
                        }

                        // AO is defined at the four corners of a unit face.
                        // Merging a partially occluded face would stretch those
                        // four values across many blocks and produces incorrect
                        // gradients. Only fully open faces are safe to merge.
                        let can_merge = face.ao == [3; 4];

                        // Extend width along d2 while faces match.
                        let mut width = 1i32;
                        while can_merge && d2 + width < dim2_size {
                            let next_idx = (d1 * dim2_size + d2 + width) as usize;
                            if mask[next_idx] == face {
                                width += 1;
                            } else {
                                break;
                            }
                        }

                        // Extend height along d1 while each row is fully
                        // covered by matching faces.
                        let mut height = 1i32;
                        'height_loop: while can_merge && d1 + height < dim1_size {
                            for w in 0..width {
                                let check_idx = ((d1 + height) * dim2_size + d2 + w) as usize;
                                if mask[check_idx] != face {
                                    break 'height_loop;
                                }
                            }
                            height += 1;
                        }

                        // Mark the merged rectangle as consumed.
                        for h in 0..height {
                            for w in 0..width {
                                let clear_idx = ((d1 + h) * dim2_size + d2 + w) as usize;
                                mask[clear_idx].is_active = false;
                            }
                        }

                        let target_quads = if face.block == BlockType::Water {
                            &mut water_quads
                        } else {
                            &mut terrain_quads
                        };
                        // The descriptor origin is the first corner in the old
                        // winding order, preserving shader UV orientation.
                        let (origin, quad_width, quad_height) = match face_dir {
                            0 => (
                                [base_x + slice, base_y + d1, base_z + d2],
                                width * 2,
                                height * 2,
                            ),
                            1 => (
                                [base_x + slice + 1, base_y + d1, base_z + d2 + width],
                                width * 2,
                                height * 2,
                            ),
                            2 => (
                                [base_x + d1, base_y + slice, base_z + d2 + width],
                                width * 2,
                                height * 2,
                            ),
                            3 => (
                                [base_x + d1, base_y + slice + 1, base_z + d2],
                                width * 2,
                                height * 2,
                            ),
                            4 => (
                                [base_x + d1 + height, base_y + d2, base_z + slice],
                                height * 2,
                                width * 2,
                            ),
                            5 => (
                                [base_x + d1, base_y + d2, base_z + slice + 1],
                                height * 2,
                                width * 2,
                            ),
                            _ => unreachable!(),
                        };
                        emit_packed_quad(
                            target_quads,
                            subchunk_origin,
                            [origin[0] as f32, origin[1] as f32, origin[2] as f32],
                            face_dir as u8,
                            quad_width as u32,
                            quad_height as u32,
                            face.tex_index,
                            face.color,
                            face.ao,
                        );

                        d2 += width; // advance past the merged run
                    }
                }
            }
        }

        (terrain_quads, water_quads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placing_player_block_on_grass_turns_support_to_dirt() {
        let mut world = World::new_empty_with_seed(1);
        world.chunks.insert((0, 0), Chunk::new(0, 0));

        world.set_block(1, 10, 1, BlockType::Grass);
        world.set_block_player(1, 11, 1, BlockType::Stone);

        assert_eq!(world.get_block(1, 10, 1), BlockType::Dirt);
        assert_eq!(world.get_block(1, 11, 1), BlockType::Stone);
    }

    #[test]
    fn air_player_write_does_not_turn_grass_to_dirt() {
        let mut world = World::new_empty_with_seed(1);
        world.chunks.insert((0, 0), Chunk::new(0, 0));

        world.set_block(1, 10, 1, BlockType::Grass);
        world.set_block_player(1, 11, 1, BlockType::Air);

        assert_eq!(world.get_block(1, 10, 1), BlockType::Grass);
        assert_eq!(world.get_block(1, 11, 1), BlockType::Air);
    }
}
