use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::render::quad::{PackedQuad, emit_packed_quad};
use crate::world::generator::ChunkGenerator;
use crate::world::skylight;
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

    /// Unique per `World` value, so GPU-side caches keyed by chunk
    /// coordinates (the sun-shadow grid) notice when a new world replaces
    /// this one.
    pub instance_id: u64,
}

/// Evicted column keys and the rendered geometry removed with them.
#[derive(Default)]
pub struct ChunkUnloadResult {
    pub chunks: Vec<(i32, i32)>,
    pub rendered_columns: u32,
    pub rendered_subchunks: u32,
}

const MESH_CACHE_PAD: usize = 2;
const MESH_CACHE_SIZE: usize = CHUNK_SIZE as usize + MESH_CACHE_PAD * 2;
const MESH_CACHE_HEIGHT: usize = SUBCHUNK_HEIGHT as usize + MESH_CACHE_PAD * 2;
const MESH_CACHE_LEN: usize = MESH_CACHE_SIZE * MESH_CACHE_HEIGHT * MESH_CACHE_SIZE;

#[derive(Clone)]
pub struct SubchunkMeshSnapshot {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub subchunk_y: i32,
    pub mesh_version: u64,
    pub has_blocks: bool,
    pub has_stairs: bool,
    /// Climate tint palette index of each column of the meshed chunk,
    /// indexed `x * CHUNK_SIZE + z`.
    pub tint_cache: [u8; (CHUNK_SIZE * CHUNK_SIZE) as usize],
    pub block_cache: [BlockType; MESH_CACHE_LEN],
    /// Light transport medium of every cell in the padded light region
    /// (see [`crate::world::skylight`]).
    pub light_media: Box<[u8]>,
    /// Highest opaque block (world Y) of every light-region column.
    pub light_sky_height: Box<[i16]>,
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
            instance_id: {
                static NEXT_WORLD_ID: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(1);
                NEXT_WORLD_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
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
        self.unload_chunks_around_player(player_x, player_z).chunks
    }

    /// Performs one unload sweep and counts rendered geometry only for removed
    /// columns. No map traversal is needed while the player stays in a column.
    pub fn unload_chunks_around_player(
        &mut self,
        player_x: f32,
        player_z: f32,
    ) -> ChunkUnloadResult {
        let player_cx = (player_x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (player_z / CHUNK_SIZE as f32).floor() as i32;

        // Early exit: player is still in the same chunk column.
        if player_cx == self.last_cleanup_cx && player_cz == self.last_cleanup_cz {
            return ChunkUnloadResult::default();
        }
        self.last_cleanup_cx = player_cx;
        self.last_cleanup_cz = player_cz;

        let mut removed = ChunkUnloadResult::default();
        self.chunks.retain(|&(cx, cz), chunk| {
            if (cx - player_cx).abs() <= CHUNK_UNLOAD_DISTANCE
                && (cz - player_cz).abs() <= CHUNK_UNLOAD_DISTANCE
            {
                return true;
            }
            let subchunks = chunk
                .subchunks
                .iter()
                .filter(|subchunk| subchunk.num_quads > 0 || subchunk.num_water_quads > 0)
                .count() as u32;
            removed.rendered_columns += u32::from(subchunks > 0);
            removed.rendered_subchunks += subchunks;
            removed.chunks.push((cx, cz));
            false
        });
        removed
    }

    // ── Generator pass-throughs ───────────────────────────────────────────── //

    /// Returns the biome at world position `(x, z)`.
    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        self.generator.get_biome(x, z)
    }

    /// Column near the origin chosen by the generator as a pleasant place to
    /// start (dry, flat land). Generate the chunks around it before calling
    /// [`Self::find_spawn_point`].
    pub fn spawn_column(&self) -> (i32, i32) {
        self.generator.find_spawn_column()
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

        let mut visibility_sum = 0.0;
        for (dx, dz) in SAMPLE_OFFSETS {
            let wx = x + dx;
            let wz = z + dz;
            let highest = self
                .chunks
                .get(&(wx.div_euclid(CHUNK_SIZE), wz.div_euclid(CHUNK_SIZE)))
                .map_or(-1, |chunk| {
                    chunk.highest_opaque_y(wx.rem_euclid(CHUNK_SIZE), wz.rem_euclid(CHUNK_SIZE))
                });
            if i32::from(highest) <= y {
                visibility_sum += 1.0;
            }
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
        let (origin_x, origin_z) = self.spawn_column();
        // Rings only: the inner square was already rejected.
        for radius in 0i32..50 {
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    if dx.abs() != radius && dz.abs() != radius {
                        continue;
                    }
                    let x = origin_x + dx;
                    let z = origin_z + dz;
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
                    let chunk_coords = (bx.div_euclid(CHUNK_SIZE), bz.div_euclid(CHUNK_SIZE));
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
    /// A flat array of size `(CHUNK_SIZE + 4) × (SUBCHUNK_HEIGHT + 4) ×
    /// (CHUNK_SIZE + 4)` is filled with block types sampled from `self.chunks`.
    /// The extra ±2 padding on all six faces
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
    /// Each face corner carries a smooth world-space sky light level (see
    /// [`crate::world::skylight`]), solved over a region padded by
    /// `LIGHT_PAD` blocks, together with the classic per-corner voxel AO.
    /// Texture color still comes from the atlas.
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

        let base_y = subchunk_y * SUBCHUNK_HEIGHT;

        let mut block_cache = [BlockType::Air; MESH_CACHE_LEN];
        let mut light_media = vec![skylight::MEDIUM_AIR; skylight::LIGHT_LEN].into_boxed_slice();
        let mut light_sky_height =
            vec![-1i16; skylight::LIGHT_SIZE * skylight::LIGHT_SIZE].into_boxed_slice();
        let mut has_blocks = false;
        let mut has_stairs = false;
        let mut tint_cache = [0u8; (CHUNK_SIZE * CHUNK_SIZE) as usize];

        // Resolve the nine columns once, rather than hashing for every voxel.
        let neighbors: [[Option<&Chunk>; 3]; 3] = std::array::from_fn(|nx| {
            std::array::from_fn(|nz| {
                self.chunks
                    .get(&(chunk_x + nx as i32 - 1, chunk_z + nz as i32 - 1))
            })
        });
        // One pass over the padded light region fills both the light media
        // and, for the inner one-block-padded box, the mesher's block cache.
        let pad = skylight::LIGHT_PAD;
        let inner = -(MESH_CACHE_PAD as i32)..(CHUNK_SIZE + MESH_CACHE_PAD as i32);
        let inner_y = -(MESH_CACHE_PAD as i32)..(SUBCHUNK_HEIGHT + MESH_CACHE_PAD as i32);
        for rx in 0..skylight::LIGHT_SIZE {
            let x = rx as i32 - pad;
            let nx = (x.div_euclid(CHUNK_SIZE) + 1) as usize;
            let lx = x.rem_euclid(CHUNK_SIZE);
            for rz in 0..skylight::LIGHT_SIZE {
                let z = rz as i32 - pad;
                let nz = (z.div_euclid(CHUNK_SIZE) + 1) as usize;
                let lz = z.rem_euclid(CHUNK_SIZE);
                let chunk = neighbors[nx][nz];
                light_sky_height[rx * skylight::LIGHT_SIZE + rz] =
                    chunk.map_or(-1, |chunk| chunk.highest_opaque_y(lx, lz));

                for ry in 0..skylight::LIGHT_HEIGHT {
                    let y = ry as i32 - pad;
                    let wy = base_y + y;
                    let block = if !(0..WORLD_HEIGHT).contains(&wy) {
                        BlockType::Air
                    } else if let Some(chunk) = chunk {
                        chunk.get_block(lx, wy, lz)
                    } else if wy < SEA_LEVEL {
                        BlockType::Water
                    } else {
                        BlockType::Air
                    };
                    light_media[skylight::light_index(rx, ry, rz)] = skylight::medium(block);
                    if !(inner.contains(&x) && inner_y.contains(&y) && inner.contains(&z)) {
                        continue;
                    }
                    let px = (x + MESH_CACHE_PAD as i32) as usize;
                    let py = (y + MESH_CACHE_PAD as i32) as usize;
                    let pz = (z + MESH_CACHE_PAD as i32) as usize;
                    if (0..CHUNK_SIZE).contains(&x)
                        && (0..SUBCHUNK_HEIGHT).contains(&y)
                        && (0..CHUNK_SIZE).contains(&z)
                        && block != BlockType::Air
                    {
                        has_blocks = true;
                    }
                    has_stairs |= block == BlockType::WoodStairs;
                    block_cache[(px * MESH_CACHE_HEIGHT + py) * MESH_CACHE_SIZE + pz] = block;
                }
            }
        }
        if let Some(chunk) = neighbors[1][1] {
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    tint_cache[(x * CHUNK_SIZE + z) as usize] = chunk.column_tint(x, z);
                }
            }
        }

        Some(SubchunkMeshSnapshot {
            chunk_x,
            chunk_z,
            subchunk_y,
            mesh_version,
            has_blocks,
            has_stairs,
            tint_cache,
            block_cache,
            light_media,
            light_sky_height,
        })
    }

    pub fn build_subchunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> SubchunkMeshes {
        let snapshot = match self.snapshot_subchunk_mesh(chunk_x, chunk_z, subchunk_y) {
            Some(snapshot) => snapshot,
            None => return SubchunkMeshes::default(),
        };

        Self::build_subchunk_mesh_from_snapshot(&snapshot)
    }

    pub fn build_subchunk_mesh_from_snapshot(snapshot: &SubchunkMeshSnapshot) -> SubchunkMeshes {
        if !snapshot.has_blocks {
            return SubchunkMeshes::default();
        }
        let mut terrain_quads = Vec::with_capacity(512);
        let mut water_quads = Vec::with_capacity(128);
        // Distant-water version of the surface: the same tops merged into
        // large sheets (see `SubchunkMeshes::water_far`).
        let mut water_far_quads = Vec::new();
        // Per (x, z) of a Y slice: water-top depth + 1 (0 = no water top).
        let mut water_far_mask = [0u8; (CHUNK_SIZE * CHUNK_SIZE) as usize];

        let chunk_x = snapshot.chunk_x;
        let chunk_z = snapshot.chunk_z;
        let subchunk_y = snapshot.subchunk_y;
        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;
        let subchunk_origin = [base_x, base_y, base_z];

        // ── Block cache setup ─────────────────────────────────────────────── //
        // 2-block padding on all sides supports face visibility and the 2×2×2
        // distant LOD without hash-map lookups.
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

        // ── Global sky light ──────────────────────────────────────────────── //
        // Solved over the padded light region; depends only on the world.
        let light = skylight::solve(
            &snapshot.light_media,
            &snapshot.light_sky_height,
            base_y - skylight::LIGHT_PAD,
        );
        let light_at = |wx: i32, wy: i32, wz: i32| -> Option<u8> {
            let rx = (wx - base_x + skylight::LIGHT_PAD) as usize;
            let ry = (wy - base_y + skylight::LIGHT_PAD) as usize;
            let rz = (wz - base_z + skylight::LIGHT_PAD) as usize;
            if rx >= skylight::LIGHT_SIZE || ry >= skylight::LIGHT_HEIGHT || rz >= skylight::LIGHT_SIZE {
                return Some(skylight::MAX_SKY_LIGHT);
            }
            let i = skylight::light_index(rx, ry, rz);
            (snapshot.light_media[i] != skylight::MEDIUM_OPAQUE).then_some(light[i])
        };
        // Water depth (blocks, capped at 15) below a water surface cell:
        // counted down through the light region, which reaches LIGHT_PAD
        // blocks below the subchunk; anything deeper counts as deep water.
        let water_depth = |wx: i32, wy: i32, wz: i32| -> u8 {
            let rx = (wx - base_x + skylight::LIGHT_PAD) as usize;
            let rz = (wz - base_z + skylight::LIGHT_PAD) as usize;
            let top = wy - base_y + skylight::LIGHT_PAD;
            if rx >= skylight::LIGHT_SIZE || rz >= skylight::LIGHT_SIZE || top < 0 {
                return 15;
            }
            let mut depth = 0u8;
            for ry in (0..=top.min(skylight::LIGHT_HEIGHT as i32 - 1)).rev() {
                let medium = snapshot.light_media[skylight::light_index(rx, ry as usize, rz)];
                if medium != skylight::MEDIUM_WATER {
                    return depth;
                }
                depth += 1;
                if depth >= 15 {
                    return 15;
                }
            }
            15
        };
        // Smooth light at one face corner: the average of the open cells
        // among the four in front of the face that touch the corner. The
        // diagonal cell is ignored when both side cells are opaque, so light
        // does not leak around solid corners.
        let corner_light = |front: (i32, i32, i32), a: (i32, i32, i32), b: (i32, i32, i32)| -> u8 {
            let (fx, fy, fz) = front;
            let centre = light_at(fx, fy, fz).unwrap_or(0);
            let side_a = light_at(fx + a.0, fy + a.1, fz + a.2);
            let side_b = light_at(fx + b.0, fy + b.1, fz + b.2);
            let diagonal = if side_a.is_none() && side_b.is_none() {
                None
            } else {
                light_at(fx + a.0 + b.0, fy + a.1 + b.1, fz + a.2 + b.2)
            };
            let (mut sum, mut count) = (u32::from(centre), 1u32);
            for value in [side_a, side_b, diagonal].into_iter().flatten() {
                sum += u32::from(value);
                count += 1;
            }
            ((sum + count / 2) / count) as u8
        };
        // Corner order matches `face_corner_ao` (and therefore the shader).
        let face_corner_light = |face_dir: i32, wx: i32, wy: i32, wz: i32| -> [u8; 4] {
            let (front, axis_a, axis_b, signs): ((i32, i32, i32), (i32, i32, i32), (i32, i32, i32), [(i32, i32); 4]) =
                match face_dir {
                    0 => ((wx - 1, wy, wz), (0, 1, 0), (0, 0, 1), [(-1, -1), (-1, 1), (1, 1), (1, -1)]),
                    1 => ((wx + 1, wy, wz), (0, 1, 0), (0, 0, 1), [(-1, 1), (-1, -1), (1, -1), (1, 1)]),
                    2 => ((wx, wy - 1, wz), (1, 0, 0), (0, 0, 1), [(-1, 1), (-1, -1), (1, -1), (1, 1)]),
                    3 => ((wx, wy + 1, wz), (1, 0, 0), (0, 0, 1), [(-1, -1), (-1, 1), (1, 1), (1, -1)]),
                    4 => ((wx, wy, wz - 1), (1, 0, 0), (0, 1, 0), [(1, -1), (-1, -1), (-1, 1), (1, 1)]),
                    5 => ((wx, wy, wz + 1), (1, 0, 0), (0, 1, 0), [(-1, -1), (1, -1), (1, 1), (-1, 1)]),
                    _ => unreachable!(),
                };
            signs.map(|(sa, sb)| {
                corner_light(
                    front,
                    (axis_a.0 * sa, axis_a.1 * sa, axis_a.2 * sa),
                    (axis_b.0 * sb, axis_b.1 * sb, axis_b.2 * sb),
                )
            })
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
        // Two faces can be merged only when all fields are equal.
        #[derive(Clone, Copy, PartialEq)]
        struct FaceAttrs {
            block: BlockType,
            color: [u8; 3],
            ao: [u8; 4],
            light: [u8; 4],
            tex_index: u8,
            tint: u8,
            is_active: bool,
        }

        impl Default for FaceAttrs {
            fn default() -> Self {
                FaceAttrs {
                    block: BlockType::Air,
                    color: [0, 0, 0],
                    ao: [3; 4],
                    light: [skylight::MAX_SKY_LIGHT; 4],
                    tex_index: 0,
                    tint: 0,
                    is_active: false,
                }
            }
        }

        // Custom shapes (currently stairs) still describe their corners
        // explicitly, but write the compact descriptor immediately.
        let emit_corner_quad = |quads: &mut Vec<PackedQuad>,
                                v0: [f32; 3],
                                v1: [f32; 3],
                                v2: [f32; 3],
                                _v3: [f32; 3],
                                normal: [f32; 3],
                                light: u8,
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
                [7, 7, 7],
                [3; 4],
                [light; 4],
                0,
            );
        };
        let empty_face = FaceAttrs::default();
        let mut mask = [empty_face; (CHUNK_SIZE as usize) * (SUBCHUNK_HEIGHT as usize)];

        // ── Pass 1: WoodStairs custom geometry ────────────────────────────── //
        // Stair blocks are composed of two non-unit-height quads that cannot
        // be expressed as standard greedy-merged full faces.  They are emitted
        // here as explicit packed descriptors and excluded from pass 2.
        // Most subchunks hold no stairs, so the scan is skipped for them.
        for lx in 0..(if snapshot.has_stairs { CHUNK_SIZE } else { 0 }) {
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
                        // Stairs are lit uniformly per face from the open
                        // cell in front of it (or the stair cell itself).
                        let own = light_at(world_x, y, world_z).unwrap_or(skylight::MAX_SKY_LIGHT);
                        let face_light =
                            |dx: i32, dy: i32, dz: i32| light_at(world_x + dx, y + dy, world_z + dz).unwrap_or(own);
                        let light_neg_x = face_light(-1, 0, 0);
                        let light_pos_x = face_light(1, 0, 0);
                        let light_neg_y = face_light(0, -1, 0);
                        let light_pos_y = face_light(0, 1, 0);
                        let light_neg_z = face_light(0, 0, -1);
                        let light_pos_z = face_light(0, 0, 1);
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

        // Distant variant (`SubchunkMeshes::terrain_far`): the same faces
        // merged with relaxed rules. Stair quads (pass 1) are shared.
        let stair_quad_count = terrain_quads.len();
        let mut terrain_far_quads: Vec<PackedQuad> = Vec::new();
        let mut far_mask = [empty_face; (CHUNK_SIZE as usize) * (SUBCHUNK_HEIGHT as usize)];

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
                if face_dir == 3 {
                    water_far_mask.fill(0);
                }

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
                                // Water quads carry the water depth below each
                                // corner (in the light slot), averaged over the
                                // four columns touching it. Shores average in
                                // dry land as depth 0, so the realistic water
                                // shader gets smooth shallows and foam lines.
                                let corner_depth = |cx: i32, cz: i32| -> u8 {
                                    let mut sum = 0u32;
                                    for (dx, dz) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
                                        sum += u32::from(water_depth(cx + dx, y, cz + dz));
                                    }
                                    ((sum + 2) / 4) as u8
                                };
                                let depths = [
                                    corner_depth(world_x, world_z),
                                    corner_depth(world_x, world_z + 1),
                                    corner_depth(world_x + 1, world_z + 1),
                                    corner_depth(world_x + 1, world_z),
                                ];
                                emit_packed_quad(
                                    &mut water_quads,
                                    subchunk_origin,
                                    [world_x as f32, (y + 1) as f32, world_z as f32],
                                    3,
                                    2,
                                    2,
                                    block.tex_for_face(3) as u8,
                                    [7, 7, 7],
                                    [3; 4],
                                    depths,
                                    0,
                                );
                                // Y slices: d1 = x, d2 = z.
                                water_far_mask[(d1 * dim2_size + d2) as usize] =
                                    water_depth(world_x, y, world_z) + 1;
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

                        let light = face_corner_light(face_dir, world_x, y, world_z);
                        let ao = if block == BlockType::Water {
                            [3; 4]
                        } else {
                            face_corner_ao(face_dir, world_x, y, world_z)
                        };

                        // Select the atlas texture index by face direction.
                        let tex_index = block.tex_for_face(face_dir);

                        let idx = (d1 * dim2_size + d2) as usize;
                        let tint = if crate::core::biome::tint_applies(tex_index as u8) {
                            snapshot.tint_cache[(lx * CHUNK_SIZE + lz) as usize]
                        } else {
                            0
                        };
                        mask[idx] = FaceAttrs {
                            block,
                            color: [7, 7, 7],
                            light,
                            ao,
                            tex_index: tex_index as u8,
                            tint,
                            is_active: true,
                        };
                    }
                }

                // The near merge below consumes the mask; keep a copy for
                // the distant variant.
                far_mask[..mask_len].copy_from_slice(&mask[..mask_len]);

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
                        // gradients. Only fully open faces are safe to merge;
                        // the same holds for the per-corner sky light.
                        let can_merge = face.ao == [3; 4]
                            && face.light.iter().all(|&l| l == face.light[0]);

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
                            face.light,
                            face.tint,
                        );

                        d2 += width; // advance past the merged run
                    }
                }

                // ── Distant variant: relaxed merging ──────────────────────── //
                // Beyond the far-mesh distance ambient occlusion and small
                // sky-light differences are invisible, so faces merge
                // whenever block, texture and tint match and their darkest
                // corner falls in the same light band (3 levels wide). This
                // removes most of the 1×1 faces AO leaves on hilly terrain.
                let band = |f: &FaceAttrs| f.light.iter().copied().min().unwrap_or(0) / 3;
                for d1 in 0..dim1_size {
                    let mut d2 = 0;
                    while d2 < dim2_size {
                        let face = far_mask[(d1 * dim2_size + d2) as usize];
                        if !face.is_active || face.block == BlockType::Water {
                            d2 += 1;
                            continue;
                        }
                        let key = (face.block, face.tex_index, face.tint, band(&face));
                        let same = |f: &FaceAttrs| {
                            f.is_active && (f.block, f.tex_index, f.tint, band(f)) == key
                        };
                        let mut width = 1i32;
                        while d2 + width < dim2_size
                            && same(&far_mask[(d1 * dim2_size + d2 + width) as usize])
                        {
                            width += 1;
                        }
                        let mut height = 1i32;
                        'far_height: while d1 + height < dim1_size {
                            for w in 0..width {
                                if !same(&far_mask[((d1 + height) * dim2_size + d2 + w) as usize]) {
                                    break 'far_height;
                                }
                            }
                            height += 1;
                        }
                        for h in 0..height {
                            for w in 0..width {
                                far_mask[((d1 + h) * dim2_size + d2 + w) as usize].is_active = false;
                            }
                        }
                        // Same corner conventions as the near merge above.
                        let (origin, quad_width, quad_height) = match face_dir {
                            0 => ([base_x + slice, base_y + d1, base_z + d2], width * 2, height * 2),
                            1 => ([base_x + slice + 1, base_y + d1, base_z + d2 + width], width * 2, height * 2),
                            2 => ([base_x + d1, base_y + slice, base_z + d2 + width], width * 2, height * 2),
                            3 => ([base_x + d1, base_y + slice + 1, base_z + d2], width * 2, height * 2),
                            4 => ([base_x + d1 + height, base_y + d2, base_z + slice], height * 2, width * 2),
                            _ => ([base_x + d1, base_y + d2, base_z + slice + 1], height * 2, width * 2),
                        };
                        // Preserve the actual AO and light at the rectangle's
                        // four outer corners instead of flattening them. Each
                        // entry is (d1 edge, d2 edge, source face corner).
                        let corner_map = match face_dir {
                            0 | 3 => [(false, false, 0), (false, true, 1), (true, true, 2), (true, false, 3)],
                            1 | 2 => [(false, true, 0), (false, false, 1), (true, false, 2), (true, true, 3)],
                            4 => [(true, false, 0), (false, false, 1), (false, true, 2), (true, true, 3)],
                            _ => [(false, true, 0), (true, true, 1), (true, false, 2), (false, false, 3)],
                        };
                        let mut far_ao = [3; 4];
                        let mut far_light = [skylight::MAX_SKY_LIGHT; 4];
                        for (corner, (high_d1, high_d2, source_corner)) in
                            corner_map.into_iter().enumerate()
                        {
                            let corner_d1 = d1 + if high_d1 { height - 1 } else { 0 };
                            let corner_d2 = d2 + if high_d2 { width - 1 } else { 0 };
                            let source = far_mask[(corner_d1 * dim2_size + corner_d2) as usize];
                            far_ao[corner] = source.ao[source_corner];
                            far_light[corner] = source.light[source_corner];
                        }
                        emit_packed_quad(
                            &mut terrain_far_quads,
                            subchunk_origin,
                            [origin[0] as f32, origin[1] as f32, origin[2] as f32],
                            face_dir as u8,
                            quad_width as u32,
                            quad_height as u32,
                            face.tex_index,
                            face.color,
                            far_ao,
                            far_light,
                            face.tint,
                        );
                        d2 += width;
                    }
                }

                // Distant water: merge this slice's tops into sheets of equal
                // depth (open water has one depth, so it merges into a
                // handful of quads per subchunk).
                if face_dir == 3 {
                    Self::merge_water_tops(
                        &mut water_far_mask,
                        &mut water_far_quads,
                        subchunk_origin,
                        base_y + slice + 1,
                    );
                }
            }
        }

        // The far sets replace the whole near sets, so they also carry the
        // shared stair quads and the (already merged) water side faces.
        water_far_quads.extend(water_quads.iter().filter(|quad| quad.face() != 3).copied());
        terrain_far_quads.extend_from_slice(&terrain_quads[..stair_quad_count]);
        // Not worth the memory when relaxed merging barely helps.
        if terrain_far_quads.len() * 10 > terrain_quads.len() * 9 {
            terrain_far_quads.clear();
        }
        let mut terrain_lod_quads = build_lod_terrain_quads(snapshot, subchunk_origin, &light_at);
        let terrain_far_count = if terrain_far_quads.is_empty() {
            terrain_quads.len()
        } else {
            terrain_far_quads.len()
        };
        if terrain_lod_quads.len() * 10 > terrain_far_count * 9 {
            terrain_lod_quads.clear();
        }
        SubchunkMeshes {
            terrain: terrain_quads,
            terrain_far: terrain_far_quads,
            terrain_lod: terrain_lod_quads,
            water: water_quads,
            water_far: water_far_quads,
        }
    }

    /// Greedy-merges a 16×16 mask of water tops (`depth + 1`, 0 = none,
    /// indexed `x * 16 + z`) at height `top_y` into large quads.
    fn merge_water_tops(
        mask: &mut [u8; (CHUNK_SIZE * CHUNK_SIZE) as usize],
        quads: &mut Vec<PackedQuad>,
        subchunk_origin: [i32; 3],
        top_y: i32,
    ) {
        let size = CHUNK_SIZE;
        for x in 0..size {
            let mut z = 0;
            while z < size {
                let value = mask[(x * size + z) as usize];
                if value == 0 {
                    z += 1;
                    continue;
                }
                let mut width = 1;
                while z + width < size && mask[(x * size + z + width) as usize] == value {
                    width += 1;
                }
                let mut height = 1;
                'grow: while x + height < size {
                    for w in 0..width {
                        if mask[((x + height) * size + z + w) as usize] != value {
                            break 'grow;
                        }
                    }
                    height += 1;
                }
                for h in 0..height {
                    for w in 0..width {
                        mask[((x + h) * size + z + w) as usize] = 0;
                    }
                }
                // Same corner convention as the +Y faces of the main mesher:
                // origin at (x, top, z), width along z, height along x.
                emit_packed_quad(
                    quads,
                    subchunk_origin,
                    [
                        (subchunk_origin[0] + x) as f32,
                        top_y as f32,
                        (subchunk_origin[2] + z) as f32,
                    ],
                    3,
                    (width * 2) as u32,
                    (height * 2) as u32,
                    BlockType::Water.tex_for_face(3) as u8,
                    [7, 7, 7],
                    [3; 4],
                    [value - 1; 4],
                    0,
                );
                z += width;
            }
        }
    }
}

fn build_lod_terrain_quads(
    snapshot: &SubchunkMeshSnapshot,
    subchunk_origin: [i32; 3],
    light_at: &impl Fn(i32, i32, i32) -> Option<u8>,
) -> Vec<PackedQuad> {
    const SCALE: i32 = 2;
    const SIZE: i32 = CHUNK_SIZE / SCALE;
    const PAD: i32 = MESH_CACHE_PAD as i32;
    const GRID_SIZE: usize = (SIZE + 2) as usize;

    #[derive(Clone, Copy, Default)]
    struct LodCell {
        block: BlockType,
        tint: u8,
    }

    #[derive(Clone, Copy, Default)]
    struct LodFace {
        block: BlockType,
        tex_index: u8,
        tint: u8,
        light: u8,
        active: bool,
    }

    let block_at = |lx: i32, ly: i32, lz: i32| {
        let world_y = subchunk_origin[1] + ly;
        if !(0..WORLD_HEIGHT).contains(&world_y) {
            return BlockType::Air;
        }
        let (px, py, pz) = (lx + PAD, ly + PAD, lz + PAD);
        if px < 0
            || px >= MESH_CACHE_SIZE as i32
            || py < 0
            || py >= MESH_CACHE_HEIGHT as i32
            || pz < 0
            || pz >= MESH_CACHE_SIZE as i32
        {
            return BlockType::Air;
        }
        snapshot.block_cache[px as usize * MESH_CACHE_HEIGHT * MESH_CACHE_SIZE
            + py as usize * MESH_CACHE_SIZE
            + pz as usize]
    };

    let cell_index = |x: i32, y: i32, z: i32| {
        ((x + 1) as usize * GRID_SIZE + (y + 1) as usize) * GRID_SIZE + (z + 1) as usize
    };
    let aggregate_cell = |x: i32, y: i32, z: i32| {
        let mut counts = [0u8; BlockType::ALL.len()];
        let mut sources = [[0i32; 3]; BlockType::ALL.len()];
        for dx in 0..SCALE {
            for dy in 0..SCALE {
                for dz in 0..SCALE {
                    let local = [x * SCALE + dx, y * SCALE + dy, z * SCALE + dz];
                    let block = block_at(local[0], local[1], local[2]);
                    if matches!(block, BlockType::Air | BlockType::Water) {
                        continue;
                    }
                    let index = block as usize;
                    if counts[index] == 0 {
                        sources[index] = local;
                    }
                    counts[index] += 1;
                }
            }
        }
        let mut selected = BlockType::Air;
        let mut selected_count = 0;
        for (index, &count) in counts.iter().enumerate() {
            if count > selected_count {
                selected = BlockType::ALL[index];
                selected_count = count;
            }
        }
        if selected_count < 2 && selected != BlockType::WoodStairs {
            return LodCell::default();
        }
        let source = sources[selected as usize];
        let tint_x = source[0].clamp(0, CHUNK_SIZE - 1) as usize;
        let tint_z = source[2].clamp(0, CHUNK_SIZE - 1) as usize;
        LodCell {
            block: selected,
            tint: snapshot.tint_cache[tint_x * CHUNK_SIZE as usize + tint_z],
        }
    };

    // Include one neighbouring 2×2×2 cell on every side. The two-block
    // snapshot padding makes decisions consistent across chunk/subchunk edges.
    let mut cells = vec![LodCell::default(); GRID_SIZE * GRID_SIZE * GRID_SIZE];
    for x in -1..=SIZE {
        for y in -1..=SIZE {
            for z in -1..=SIZE {
                cells[cell_index(x, y, z)] = aggregate_cell(x, y, z);
            }
        }
    }

    let mut quads = Vec::new();
    let mut mask = [LodFace::default(); (SIZE * SIZE) as usize];
    for face_dir in 0..6 {
        let (dx, dy, dz) = match face_dir {
            0 => (-1, 0, 0),
            1 => (1, 0, 0),
            2 => (0, -1, 0),
            3 => (0, 1, 0),
            4 => (0, 0, -1),
            _ => (0, 0, 1),
        };
        for slice in 0..SIZE {
            mask.fill(LodFace::default());
            for d1 in 0..SIZE {
                for d2 in 0..SIZE {
                    let (x, y, z) = match face_dir {
                        0 | 1 => (slice, d1, d2),
                        2 | 3 => (d1, slice, d2),
                        _ => (d1, d2, slice),
                    };
                    let cell = cells[cell_index(x, y, z)];
                    if cell.block == BlockType::Air {
                        continue;
                    }
                    let neighbor = cells[cell_index(x + dx, y + dy, z + dz)];
                    if !cell.block.should_render_face_against(neighbor.block) {
                        continue;
                    }
                    let tex_index = cell.block.tex_for_face(face_dir) as u8;
                    let tint = if crate::core::biome::tint_applies(tex_index) {
                        cell.tint
                    } else {
                        0
                    };
                    let front = [
                        subchunk_origin[0]
                            + x * SCALE
                            + if dx < 0 { -1 } else if dx > 0 { SCALE } else { 1 },
                        subchunk_origin[1]
                            + y * SCALE
                            + if dy < 0 { -1 } else if dy > 0 { SCALE } else { 1 },
                        subchunk_origin[2]
                            + z * SCALE
                            + if dz < 0 { -1 } else if dz > 0 { SCALE } else { 1 },
                    ];
                    let light =
                        light_at(front[0], front[1], front[2]).unwrap_or(skylight::MAX_SKY_LIGHT);
                    let index = (d1 * SIZE + d2) as usize;
                    mask[index] = LodFace {
                        block: cell.block,
                        tex_index,
                        tint,
                        light: light / 3,
                        active: true,
                    };
                }
            }

            for d1 in 0..SIZE {
                let mut d2 = 0;
                while d2 < SIZE {
                    let index = (d1 * SIZE + d2) as usize;
                    let face = mask[index];
                    if !face.active {
                        d2 += 1;
                        continue;
                    }
                    let same = |other: LodFace| {
                        other.active
                            && other.block == face.block
                            && other.tex_index == face.tex_index
                            && other.tint == face.tint
                            && other.light == face.light
                    };
                    let mut width = 1;
                    while d2 + width < SIZE && same(mask[(d1 * SIZE + d2 + width) as usize]) {
                        width += 1;
                    }
                    let mut height = 1;
                    'grow: while d1 + height < SIZE {
                        for w in 0..width {
                            if !same(mask[((d1 + height) * SIZE + d2 + w) as usize]) {
                                break 'grow;
                            }
                        }
                        height += 1;
                    }
                    for h in 0..height {
                        for w in 0..width {
                            mask[((d1 + h) * SIZE + d2 + w) as usize].active = false;
                        }
                    }
                    let (origin, quad_width, quad_height) = match face_dir {
                        0 => (
                            [
                                subchunk_origin[0] + slice * SCALE,
                                subchunk_origin[1] + d1 * SCALE,
                                subchunk_origin[2] + d2 * SCALE,
                            ],
                            width * SCALE * 2,
                            height * SCALE * 2,
                        ),
                        1 => (
                            [
                                subchunk_origin[0] + (slice + 1) * SCALE,
                                subchunk_origin[1] + d1 * SCALE,
                                subchunk_origin[2] + (d2 + width) * SCALE,
                            ],
                            width * SCALE * 2,
                            height * SCALE * 2,
                        ),
                        2 => (
                            [
                                subchunk_origin[0] + d1 * SCALE,
                                subchunk_origin[1] + slice * SCALE,
                                subchunk_origin[2] + (d2 + width) * SCALE,
                            ],
                            width * SCALE * 2,
                            height * SCALE * 2,
                        ),
                        3 => (
                            [
                                subchunk_origin[0] + d1 * SCALE,
                                subchunk_origin[1] + (slice + 1) * SCALE,
                                subchunk_origin[2] + d2 * SCALE,
                            ],
                            width * SCALE * 2,
                            height * SCALE * 2,
                        ),
                        4 => (
                            [
                                subchunk_origin[0] + (d1 + height) * SCALE,
                                subchunk_origin[1] + d2 * SCALE,
                                subchunk_origin[2] + slice * SCALE,
                            ],
                            height * SCALE * 2,
                            width * SCALE * 2,
                        ),
                        _ => (
                            [
                                subchunk_origin[0] + d1 * SCALE,
                                subchunk_origin[1] + d2 * SCALE,
                                subchunk_origin[2] + (slice + 1) * SCALE,
                            ],
                            height * SCALE * 2,
                            width * SCALE * 2,
                        ),
                    };
                    let light = (face.light * 3 + 2).min(skylight::MAX_SKY_LIGHT);
                    emit_packed_quad(
                        &mut quads,
                        subchunk_origin,
                        [origin[0] as f32, origin[1] as f32, origin[2] as f32],
                        face_dir as u8,
                        quad_width as u32,
                        quad_height as u32,
                        face.tex_index,
                        [7; 3],
                        [3; 4],
                        [light; 4],
                        face.tint,
                    );
                    d2 += width;
                }
            }
        }
    }
    quads
}

/// Packed geometry of one subchunk.
#[derive(Default)]
pub struct SubchunkMeshes {
    /// Opaque and alpha-tested terrain.
    pub terrain: Vec<PackedQuad>,
    /// The same terrain merged with relaxed rules (no ambient occlusion,
    /// coarse light bands): far fewer quads, drawn instead of `terrain`
    /// beyond the far-mesh distance. Empty when it would not save much.
    pub terrain_far: Vec<PackedQuad>,
    /// A 2×2×2 voxel aggregation for the furthest terrain ring.
    pub terrain_lod: Vec<PackedQuad>,
    /// Water near the camera: 1×1 surface quads (the wave displacement
    /// needs the vertices) plus any water side faces.
    pub water: Vec<PackedQuad>,
    /// The same water merged into large quads (plus the side faces), drawn
    /// instead of `water` for subchunks beyond the wave-displacement range.
    pub water_far: Vec<PackedQuad>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_snapshot(
        world: &World,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> Option<SubchunkMeshSnapshot> {
        let mesh_version = world
            .chunks
            .get(&(chunk_x, chunk_z))?
            .subchunks
            .get(subchunk_y as usize)?
            .mesh_version;

        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;

        let mut block_cache = [BlockType::Air; MESH_CACHE_LEN];
        let mut has_blocks = false;
        let block_at = |wx: i32, wy: i32, wz: i32| {
            if !(0..WORLD_HEIGHT).contains(&wy) {
                return BlockType::Air;
            }
            let key = (wx.div_euclid(CHUNK_SIZE), wz.div_euclid(CHUNK_SIZE));
            match world.chunks.get(&key) {
                Some(chunk) => chunk.get_block(wx.rem_euclid(CHUNK_SIZE), wy, wz.rem_euclid(CHUNK_SIZE)),
                None if wy < SEA_LEVEL => BlockType::Water,
                None => BlockType::Air,
            }
        };
        let pad = skylight::LIGHT_PAD;
        let mut light_media = vec![0u8; skylight::LIGHT_LEN];
        let mut light_sky_height = vec![-1i16; skylight::LIGHT_SIZE * skylight::LIGHT_SIZE];
        for rx in 0..skylight::LIGHT_SIZE {
            for rz in 0..skylight::LIGHT_SIZE {
                let (wx, wz) = (base_x + rx as i32 - pad, base_z + rz as i32 - pad);
                light_sky_height[rx * skylight::LIGHT_SIZE + rz] = world
                    .chunks
                    .get(&(wx.div_euclid(CHUNK_SIZE), wz.div_euclid(CHUNK_SIZE)))
                    .map_or(-1, |c| c.highest_opaque_y(wx.rem_euclid(CHUNK_SIZE), wz.rem_euclid(CHUNK_SIZE)));
                for ry in 0..skylight::LIGHT_HEIGHT {
                    light_media[skylight::light_index(rx, ry, rz)] =
                        skylight::medium(block_at(wx, base_y + ry as i32 - pad, wz));
                }
            }
        }

        for px in 0..MESH_CACHE_SIZE as i32 {
            for py in 0..MESH_CACHE_HEIGHT as i32 {
                for pz in 0..MESH_CACHE_SIZE as i32 {
                    let wx = base_x + px - MESH_CACHE_PAD as i32;
                    let wy = base_y + py - MESH_CACHE_PAD as i32;
                    let wz = base_z + pz - MESH_CACHE_PAD as i32;
                    let block = if !(0..WORLD_HEIGHT).contains(&wy) {
                        BlockType::Air
                    } else {
                        let cx = wx.div_euclid(CHUNK_SIZE);
                        let cz = wz.div_euclid(CHUNK_SIZE);
                        let lx = wx.rem_euclid(CHUNK_SIZE);
                        let lz = wz.rem_euclid(CHUNK_SIZE);
                        if let Some(chunk) = world.chunks.get(&(cx, cz)) {
                            chunk.get_block(lx, wy, lz)
                        } else if wy < SEA_LEVEL {
                            BlockType::Water
                        } else {
                            BlockType::Air
                        }
                    };

                    if px >= MESH_CACHE_PAD as i32
                        && px < (MESH_CACHE_PAD as i32 + CHUNK_SIZE)
                        && py >= MESH_CACHE_PAD as i32
                        && py < (MESH_CACHE_PAD as i32 + SUBCHUNK_HEIGHT)
                        && pz >= MESH_CACHE_PAD as i32
                        && pz < (MESH_CACHE_PAD as i32 + CHUNK_SIZE)
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

        Some(SubchunkMeshSnapshot {
            chunk_x,
            chunk_z,
            subchunk_y,
            mesh_version,
            has_blocks,
            has_stairs: true,
            tint_cache: [0; (CHUNK_SIZE * CHUNK_SIZE) as usize],
            block_cache,
            light_media: light_media.into_boxed_slice(),
            light_sky_height: light_sky_height.into_boxed_slice(),
        })
    }

    fn quad_ao(quad: &PackedQuad) -> [u8; 4] {
        std::array::from_fn(|corner| {
            ((quad.size_material_ao >> (20 + corner * 2)) & 0x3) as u8
        })
    }

    fn quad_light(quad: &PackedQuad) -> [u8; 4] {
        std::array::from_fn(|corner| ((quad.light >> (corner * 4)) & 0xf) as u8)
    }

    fn quad_corner_positions(quad: &PackedQuad) -> [[i32; 3]; 4] {
        let face = quad.face();
        if face >= 6 {
            return [[-1; 3]; 4];
        }
        let origin = [
            (quad.origin_and_face & 0x3f) as i32,
            ((quad.origin_and_face >> 6) & 0x3f) as i32,
            ((quad.origin_and_face >> 12) & 0x3f) as i32,
        ];
        let width = (quad.size_material_ao & 0x3f) as i32;
        let height = ((quad.size_material_ao >> 6) & 0x3f) as i32;
        let u = [
            [0, 0, 1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, 1],
            [-1, 0, 0],
            [1, 0, 0],
        ][face];
        let v = [
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ][face];
        let offsets = [(0, 0), (width, 0), (width, height), (0, height)];
        std::array::from_fn(|corner| {
            let (along_u, along_v) = offsets[corner];
            std::array::from_fn(|axis| {
                origin[axis] + u[axis] * along_u + v[axis] * along_v
            })
        })
    }

    fn far_corners_match_near(far: &PackedQuad, near: &[PackedQuad]) -> bool {
        let positions = quad_corner_positions(far);
        let ao = quad_ao(far);
        let light = quad_light(far);
        positions.iter().enumerate().all(|(corner, position)| {
            near.iter().any(|quad| {
                if quad.face() != far.face() || quad.material() != far.material() {
                    return false;
                }
                let near_positions = quad_corner_positions(quad);
                let near_ao = quad_ao(quad);
                let near_light = quad_light(quad);
                near_positions.iter().enumerate().any(|(near_corner, near_position)| {
                    near_position == position
                        && near_ao[near_corner] == ao[corner]
                        && near_light[near_corner] == light[corner]
                })
            })
        })
    }

    fn lod_quads_with_world_light(snapshot: &SubchunkMeshSnapshot) -> Vec<PackedQuad> {
        let origin = [
            snapshot.chunk_x * CHUNK_SIZE,
            snapshot.subchunk_y * SUBCHUNK_HEIGHT,
            snapshot.chunk_z * CHUNK_SIZE,
        ];
        let solved = skylight::solve(
            &snapshot.light_media,
            &snapshot.light_sky_height,
            origin[1] - skylight::LIGHT_PAD,
        );
        let light_at = |wx: i32, wy: i32, wz: i32| {
            let rx = wx - origin[0] + skylight::LIGHT_PAD;
            let ry = wy - origin[1] + skylight::LIGHT_PAD;
            let rz = wz - origin[2] + skylight::LIGHT_PAD;
            if rx < 0
                || rx >= skylight::LIGHT_SIZE as i32
                || ry < 0
                || ry >= skylight::LIGHT_HEIGHT as i32
                || rz < 0
                || rz >= skylight::LIGHT_SIZE as i32
            {
                return Some(skylight::MAX_SKY_LIGHT);
            }
            let index = skylight::light_index(rx as usize, ry as usize, rz as usize);
            (snapshot.light_media[index] != skylight::MEDIUM_OPAQUE).then_some(solved[index])
        };
        build_lod_terrain_quads(snapshot, origin, &light_at)
    }

    #[test]
    fn distant_water_is_merged_into_sheets() {
        // 3×3 chunks of stone below y = 40 and water up to sea level.
        let mut world = World::new_empty_with_seed(1);
        for cx in -1..=1 {
            for cz in -1..=1 {
                let mut chunk = Chunk::new(cx, cz);
                for x in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        for y in 0..SEA_LEVEL {
                            let block = if y < 40 { BlockType::Stone } else { BlockType::Water };
                            chunk.set_block_raw(x, y, z, block);
                        }
                    }
                }
                chunk.rebuild_metadata();
                world.chunks.insert((cx, cz), chunk);
            }
        }
        let meshes = world.build_subchunk_mesh(0, 0, (SEA_LEVEL - 1) / SUBCHUNK_HEIGHT);
        assert_eq!(meshes.water.len(), 256, "near water keeps 1×1 wave tiles");
        assert_eq!(meshes.water_far.len(), 1, "open water merges into one sheet");
        let sheet = meshes.water_far[0];
        assert_eq!(sheet.face(), 3);
        // Covers the whole 16×16 subchunk (sizes are in half blocks).
        assert_eq!(sheet.size_material_ao & 0x3f, 32);
        assert_eq!((sheet.size_material_ao >> 6) & 0x3f, 32);
        // Buried subchunks have no water at all.
        assert!(world.build_subchunk_mesh(0, 0, 1).water_far.is_empty());
    }

    #[test]
    fn far_terrain_variant_needs_fewer_quads() {
        let generator = crate::world::ChunkGenerator::new(2026);
        let mut world = World::new_empty_with_seed(2026);
        for cx in -3..=3 {
            for cz in -3..=3 {
                world.chunks.insert((cx, cz), generator.generate_chunk(cx, cz));
            }
        }
        let (mut near, mut far, mut lod) = (0usize, 0usize, 0usize);
        let (mut near_corner_variation, mut far_corner_variation, mut matched_far_corners) =
            (false, false, false);
        for cx in -2..=2 {
            for cz in -2..=2 {
                for sy in 0..NUM_SUBCHUNKS {
                    let meshes = world.build_subchunk_mesh(cx, cz, sy);
                    near += meshes.terrain.len();
                    near_corner_variation |= meshes.terrain.iter().any(|quad| {
                        let ao = quad_ao(quad);
                        let light = quad_light(quad);
                        ao.iter().any(|&value| value != ao[0])
                            || light.iter().any(|&value| value != light[0])
                    });
                    far_corner_variation |= meshes.terrain_far.iter().any(|quad| {
                        let ao = quad_ao(quad);
                        let light = quad_light(quad);
                        ao.iter().any(|&value| value != ao[0])
                            || light.iter().any(|&value| value != light[0])
                    });
                    if !matched_far_corners {
                        matched_far_corners = meshes.terrain_far.iter().any(|quad| {
                            let width = quad.size_material_ao & 0x3f;
                            let height = (quad.size_material_ao >> 6) & 0x3f;
                            (width > 2 || height > 2)
                                && far_corners_match_near(quad, &meshes.terrain)
                        });
                    }
                    // An empty far set means "draw the near one".
                    far += if meshes.terrain_far.is_empty() {
                        meshes.terrain.len()
                    } else {
                        meshes.terrain_far.len()
                    };
                    lod += if meshes.terrain_lod.is_empty() {
                        meshes.terrain_far.len().max(meshes.terrain.len())
                    } else {
                        meshes.terrain_lod.len()
                    };
                }
            }
        }
        println!("terrain quads: near {near}, far {far}, 2×2×2 {lod}");
        assert!(far * 4 < near * 3, "near {near}, far {far}");
        assert!(lod < far, "far {far}, 2×2×2 {lod}");
        assert!(near_corner_variation, "near mesh should retain varied corner shading");
        assert!(far_corner_variation, "far merges should retain varied outer-corner AO/light");
        assert!(matched_far_corners, "far outer-corner AO/light must match the near mesh");
    }

    #[test]
    fn lod_face_light_is_sampled_in_front_of_open_and_enclosed_surfaces() {
        let mut open_world = World::new_empty_with_seed(1);
        let mut open_chunk = Chunk::new(0, 0);
        for x in 4..6 {
            for y in 4..6 {
                for z in 4..6 {
                    open_chunk.set_block_raw(x, y, z, BlockType::Stone);
                }
            }
        }
        open_chunk.rebuild_metadata();
        open_world.chunks.insert((0, 0), open_chunk);

        let open_snapshot = reference_snapshot(&open_world, 0, 0, 0).unwrap();
        let open_mesh = lod_quads_with_world_light(&open_snapshot);
        let sky_face = open_mesh
            .iter()
            .find(|quad| quad.face() == 3 && quad.origin_and_face & 0x3f == 8)
            .expect("open supercell should have an LOD top face");
        assert_eq!(quad_light(sky_face), [15; 4]);

        let mut cave_world = World::new_empty_with_seed(1);
        let mut cave_chunk = Chunk::new(0, 0);
        for x in 2..12 {
            for y in 2..12 {
                for z in 2..12 {
                    let inside_cave = (4..10).contains(&x)
                        && (4..10).contains(&y)
                        && (4..10).contains(&z);
                    if !inside_cave {
                        cave_chunk.set_block_raw(x, y, z, BlockType::Stone);
                    }
                }
            }
        }
        cave_chunk.rebuild_metadata();
        cave_world.chunks.insert((0, 0), cave_chunk);

        let cave_snapshot = reference_snapshot(&cave_world, 0, 0, 0).unwrap();
        let cave_mesh = lod_quads_with_world_light(&cave_snapshot);
        let cave_wall = cave_mesh
            .iter()
            .find(|quad| quad.face() == 1 && quad.origin_and_face & 0x3f == 8)
            .expect("enclosed cave wall should have an LOD face");
        assert!(
            quad_light(cave_wall).iter().all(|&light| light < 15),
            "cave wall should use the dark light from the open cave side"
        );
    }

    #[test]
    fn terrain_lod_uses_two_block_faces_and_merges_rough_surfaces() {
        let mut world = World::new_empty_with_seed(1);
        let mut chunk = Chunk::new(0, 0);
        for x in 0..8 {
            for z in 0..8 {
                let height = 8 + (x + z) % 2;
                for y in 0..=height {
                    chunk.set_block_raw(x, y, z, BlockType::Stone);
                }
            }
        }
        chunk.rebuild_metadata();
        world.chunks.insert((0, 0), chunk);

        let meshes = world.build_subchunk_mesh(0, 0, 0);
        assert!(!meshes.terrain_lod.is_empty());
        assert!(meshes.terrain_lod.len() < meshes.terrain.len());
        for quad in meshes.terrain_lod {
            assert_eq!(
                (quad.size_material_ao & 0x3f) % 4,
                0,
                "LOD width must align to 2 blocks"
            );
            assert_eq!(
                ((quad.size_material_ao >> 6) & 0x3f) % 4,
                0,
                "LOD height must align to 2 blocks"
            );
        }
    }

    #[test]
    fn terrain_lod_uses_neighboring_supercells_at_chunk_boundaries() {
        let mut world = World::new_empty_with_seed(1);
        let mut left = Chunk::new(0, 0);
        let mut right = Chunk::new(1, 0);
        for x in 14..16 {
            for y in 4..8 {
                for z in 8..12 {
                    left.set_block_raw(x, y, z, BlockType::Stone);
                }
            }
        }
        for x in 0..2 {
            for y in 4..8 {
                for z in 8..12 {
                    right.set_block_raw(x, y, z, BlockType::Stone);
                }
            }
        }
        // Add a rough surface so the distant LOD is retained for comparison.
        for x in 0..8 {
            for z in 0..8 {
                let height = 8 + (x + z) % 2;
                for y in 0..=height {
                    left.set_block_raw(x, y, z, BlockType::Stone);
                }
            }
        }
        left.rebuild_metadata();
        right.rebuild_metadata();
        world.chunks.insert((0, 0), left);
        world.chunks.insert((1, 0), right);

        let lod = world.build_subchunk_mesh(0, 0, 0).terrain_lod;
        assert!(!lod.is_empty());
        let has_internal_boundary_face = lod.iter().any(|quad| {
            let face = (quad.origin_and_face >> 18) & 0x7;
            let x = quad.origin_and_face & 0x3f;
            let y = (quad.origin_and_face >> 6) & 0x3f;
            let z = (quad.origin_and_face >> 12) & 0x3f;
            face == 1 && x == 32 && (8..16).contains(&y) && (16..24).contains(&z)
        });
        assert!(
            !has_internal_boundary_face,
            "neighboring 2×2×2 cells must hide their shared face"
        );
    }

    fn snapshot_fixture() -> World {
        let mut world = World::new_empty_with_seed(1);
        // Include negative coordinates, diagonal padding, and an absent neighbor.
        for cx in -2i32..=1 {
            for cz in -2i32..=1 {
                if (cx, cz) == (0, -1) {
                    continue;
                }
                let mut chunk = Chunk::new(cx, cz);
                for x in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        for y in [0, 15, 16, 63, 64, 127, 255] {
                            let block = match (x + z + y + cx + cz).rem_euclid(4) {
                                0 => BlockType::Stone,
                                1 => BlockType::Water,
                                2 => BlockType::Leaves,
                                _ => BlockType::Air,
                            };
                            chunk.set_block_raw(x, y, z, block);
                        }
                    }
                }
                chunk.rebuild_metadata();
                world.chunks.insert((cx, cz), chunk);
            }
        }
        world
    }

    #[test]
    fn cached_snapshot_matches_voxel_reference_at_world_and_chunk_boundaries() {
        let world = snapshot_fixture();
        for (cx, cz) in [(-1, -1), (0, 0)] {
            for sy in 0..NUM_SUBCHUNKS {
                let expected = reference_snapshot(&world, cx, cz, sy).unwrap();
                let actual = world.snapshot_subchunk_mesh(cx, cz, sy).unwrap();
                assert_eq!(actual.block_cache, expected.block_cache, "{cx}, {cz}, {sy}");
                assert_eq!(actual.light_media, expected.light_media);
                assert_eq!(actual.light_sky_height, expected.light_sky_height);
                assert_eq!(actual.has_blocks, expected.has_blocks);
                assert_eq!(actual.mesh_version, expected.mesh_version);
            }
        }
        assert!(world.snapshot_subchunk_mesh(0, -1, 0).is_none());
        assert!(world.snapshot_subchunk_mesh(0, 0, -1).is_none());
        assert!(world.snapshot_subchunk_mesh(0, 0, NUM_SUBCHUNKS).is_none());
    }

    #[test]
    fn cached_sky_visibility_matches_vertical_scan_after_edits() {
        let mut world = snapshot_fixture();
        for (x, z) in [(-1, -1), (0, 0), (15, 15), (16, -16), (32, 32)] {
            // Removing the highest opaque block must expose the next one.
            world.set_block(x, WORLD_HEIGHT - 1, z, BlockType::Stone);
            world.set_block(x, WORLD_HEIGHT - 1, z, BlockType::Air);
            for y in [-1, 0, 15, 63, 64, 127, 254, 255, 256] {
                let expected = if y < 0 {
                    0.0
                } else {
                    [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
                        .into_iter()
                        .filter(|&(dx, dz)| {
                            !(y + 1..WORLD_HEIGHT)
                                .any(|sy| world.get_block(x + dx, sy, z + dz).is_solid_opaque())
                        })
                        .count() as f32
                        / 5.0
                };
                assert_eq!(world.sky_visibility_at(x, y, z), expected);
            }
        }
    }

    #[test]
    fn unload_counts_geometry_once_and_keeps_radius_boundary() {
        let mut world = World::new_empty_with_seed(1);
        for cx in [
            -CHUNK_UNLOAD_DISTANCE,
            CHUNK_UNLOAD_DISTANCE,
            CHUNK_UNLOAD_DISTANCE + 1,
        ] {
            let mut chunk = Chunk::new(cx, 0);
            chunk.subchunks[0].num_quads = 1;
            chunk.subchunks[1].num_water_quads = 1;
            world.chunks.insert((cx, 0), chunk);
        }
        let removed = world.unload_chunks_around_player(0.0, 0.0);
        assert_eq!(removed.chunks, vec![(CHUNK_UNLOAD_DISTANCE + 1, 0)]);
        assert_eq!(
            (removed.rendered_columns, removed.rendered_subchunks),
            (1, 2)
        );
        let repeated = world.unload_chunks_around_player(0.0, 0.0);
        assert!(repeated.chunks.is_empty());
        assert_eq!(repeated.rendered_subchunks, 0);
        assert_eq!(world.chunks.len(), 2);
        let moved = world.unload_chunks_around_player(-1.0, 0.0);
        assert_eq!(moved.chunks, vec![(CHUNK_UNLOAD_DISTANCE, 0)]);
    }

    #[test]
    #[ignore = "manual CPU snapshot benchmark; run with an optimized test profile"]
    fn benchmark_mesh_snapshot() {
        use std::{hint::black_box, time::Instant};
        let world = snapshot_fixture();
        let iterations = 20_000;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(reference_snapshot(black_box(&world), -1, -1, 3));
        }
        let reference = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(world.snapshot_subchunk_mesh(-1, -1, 3));
        }
        let cached = start.elapsed();
        eprintln!(
            "snapshot: {iterations} iterations, reference={reference:?}, cached={cached:?}, speedup={:.2}x",
            reference.as_secs_f64() / cached.as_secs_f64()
        );
    }

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
