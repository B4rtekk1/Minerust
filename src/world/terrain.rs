use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::core::vertex::Vertex;
use crate::render::mesh::{add_greedy_quad, add_quad};
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
    #[allow(dead_code)]
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
    }

    /// Returns `true` if the block at `(x, y, z)` is solid (i.e., has
    /// non-zero collision volume).
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_block(x, y, z).is_solid()
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

                    if height >= SEA_LEVEL
                        && !matches!(biome, Biome::Ocean | Biome::River | Biome::Lake)
                    {
                        // +0.3 / +0.5 offsets prevent the player from being
                        // centred on a block edge and avoid false collision
                        // positives at the moment of spawn.
                        return (x as f32 + 0.3, (height + 1) as f32, z as f32 + 0.5);
                    }
                }
            }
        }
        (0.5, 80.0, 0.5) // fallback
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
    /// quad.  They are rendered with individual `add_quad` calls in a
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
    ///    is emitted as a single `add_greedy_quad` call and the covered
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
    /// ## Color and texture selection
    ///
    /// Face color is chosen per-direction:
    /// - Bottom face (dir 2): `block.bottom_color()`.
    /// - Top face (dir 3): biome grass color for Grass blocks; `block.top_color()` otherwise.
    /// - Side faces (dirs 0, 1, 4, 5): biome leaves color for Leaves; `block.color()` otherwise.
    ///
    /// Biome lookups are cached in `biome_map` (a 16×16 grid of `Option<Biome>`)
    /// so each XZ column is queried at most once per sub-chunk mesh build.
    ///
    /// Colors are quantized to 6 bits per channel (`& 0xFC`) before storage
    /// in `FaceAttrs` so that floating-point rounding noise doesn't prevent
    /// adjacent faces from being merged.
    ///
    /// # Parameters
    /// - `chunk_x`   – Chunk column X coordinate.
    /// - `chunk_z`   – Chunk column Z coordinate.
    /// - `subchunk_y` – Sub-chunk vertical index within the column.
    ///
    /// # Returns
    /// A pair of `(vertices, indices)` tuples:
    /// - First tuple: opaque geometry.
    /// - Second tuple: water (translucent) geometry.
    pub fn build_subchunk_mesh(
        &self,
        chunk_x: i32,
        chunk_z: i32,
        subchunk_y: i32,
    ) -> ((Vec<Vertex>, Vec<u32>), (Vec<Vertex>, Vec<u32>)) {
        let mut vertices = Vec::with_capacity(4096);
        let mut indices = Vec::with_capacity(2048);
        let mut water_vertices = Vec::with_capacity(1024);
        let mut water_indices = Vec::with_capacity(512);

        let base_x = chunk_x * CHUNK_SIZE;
        let base_y = subchunk_y * SUBCHUNK_HEIGHT;
        let base_z = chunk_z * CHUNK_SIZE;

        // ── Block cache setup ─────────────────────────────────────────────── //
        // 1-block padding on all sides so neighbor lookups never need a
        // hash-map access during the face visibility and greedy merge tests.
        const PAD: usize = 1;
        const S: usize = CHUNK_SIZE as usize + PAD * 2; // 18
        const SH: usize = SUBCHUNK_HEIGHT as usize + PAD * 2; // 18

        let mut block_cache = [BlockType::Air; S * SH * S];

        // Fetch a block from the world, defaulting to Water/Air for unloaded
        // neighboring chunks (below/above sea level respectively).
        let fetch = |wx: i32, wy: i32, wz: i32| -> BlockType {
            if wy < 0 || wy >= WORLD_HEIGHT {
                return BlockType::Air;
            }
            let cx = wx.div_euclid(CHUNK_SIZE);
            let cz = wz.div_euclid(CHUNK_SIZE);
            let lx = wx.rem_euclid(CHUNK_SIZE);
            let lz = wz.rem_euclid(CHUNK_SIZE);
            if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                chunk.get_block(lx, wy, lz)
            } else if wy < SEA_LEVEL {
                BlockType::Water // fill unloaded ocean columns with water
            } else {
                BlockType::Air
            }
        };

        // Populate the cache: layout is [px][py][pz] linearised as
        // `px * SH * S + py * S + pz` with (px, py, pz) being padded coords.
        for px in 0..S as i32 {
            for py in 0..SH as i32 {
                for pz in 0..S as i32 {
                    let wx = base_x + px - PAD as i32;
                    let wy = base_y + py - PAD as i32;
                    let wz = base_z + pz - PAD as i32;
                    block_cache[(px as usize) * SH * S + (py as usize) * S + (pz as usize)] =
                        fetch(wx, wy, wz);
                }
            }
        }

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

        // Biome cache: queried lazily, at most once per XZ column.
        let mut biome_map: [[Option<Biome>; CHUNK_SIZE as usize]; CHUNK_SIZE as usize] =
            [[None; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];

        // ── FaceAttrs: per-cell data stored in the greedy mask ────────────── //
        // Two faces can be merged only when all fields are equal, so colors
        // are pre-quantized (see `quantize_color`) to suppress floating-point
        // rounding noise that would otherwise prevent merging.
        #[derive(Clone, Copy, PartialEq)]
        struct FaceAttrs {
            block: BlockType,
            color: [u8; 3],
            tex_index: u8,
            is_active: bool,
        }

        impl Default for FaceAttrs {
            fn default() -> Self {
                FaceAttrs {
                    block: BlockType::Air,
                    color: [0, 0, 0],
                    tex_index: 0,
                    is_active: false,
                }
            }
        }

        // Quantize a linear RGB float color to 6 bits per channel.
        // Masking with 0xFC rounds the low 2 bits to zero so that minor
        // floating-point differences between adjacent face color queries
        // don't prevent greedy merging.
        let quantize_color = |c: [f32; 3]| -> [u8; 3] {
            [
                ((c[0] * 255.0) as u8) & 0xFC,
                ((c[1] * 255.0) as u8) & 0xFC,
                ((c[2] * 255.0) as u8) & 0xFC,
            ]
        };

        // ── Pass 1: WoodStairs custom geometry ────────────────────────────── //
        // Stair blocks are composed of two non-unit-height quads that cannot
        // be expressed as standard greedy-merged full faces.  They are emitted
        // here with explicit `add_quad` calls and excluded from pass 2.
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

                    let is_water = block == BlockType::Water;
                    let (target_verts, target_inds) = if is_water {
                        (&mut water_vertices, &mut water_indices)
                    } else {
                        (&mut vertices, &mut indices)
                    };

                    if block == BlockType::WoodStairs {
                        let x = world_x as f32;
                        let y_f = y as f32;
                        let z = world_z as f32;
                        let color = block.color();
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
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z + 1.0],
                                [x, y_f, z],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f, z + 1.0],
                                [0.0, -1.0, 0.0],
                                color,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Lower half-top (always visible: the step tread at Y+0.5,
                        // front half Z=[0, 0.5]).
                        add_quad(
                            target_verts,
                            target_inds,
                            [x, y_f + 0.5, z],
                            [x, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x + 1.0, y_f + 0.5, z],
                            [0.0, 1.0, 0.0],
                            color,
                            tex_top,
                            r,
                            m,
                        );
                        // Upper full-top (conditional on +Y neighbor).
                        if block.should_render_face_against(neighbors[3]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f + 1.0, z + 0.5],
                                [x, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [0.0, 1.0, 0.0],
                                color,
                                tex_top,
                                r,
                                m,
                            );
                        }
                        // Front face (−Z, lower half only, conditional).
                        if block.should_render_face_against(neighbors[4]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f, z],
                                [x, y_f, z],
                                [x, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z],
                                [0.0, 0.0, -1.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Step riser (always visible: the vertical face between
                        // the lower and upper treads at Z+0.5).
                        add_quad(
                            target_verts,
                            target_inds,
                            [x + 1.0, y_f + 0.5, z + 0.5],
                            [x, y_f + 0.5, z + 0.5],
                            [x, y_f + 1.0, z + 0.5],
                            [x + 1.0, y_f + 1.0, z + 0.5],
                            [0.0, 0.0, -1.0],
                            color,
                            tex_side,
                            r,
                            m,
                        );
                        // Back face (+Z, full height, conditional).
                        if block.should_render_face_against(neighbors[5]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z + 1.0],
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [0.0, 0.0, 1.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Left face (−X): two quads – lower half and upper-back half.
                        if block.should_render_face_against(neighbors[0]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f, z],
                                [x, y_f, z + 1.0],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 0.5, z],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                            add_quad(
                                target_verts,
                                target_inds,
                                [x, y_f + 0.5, z + 0.5],
                                [x, y_f + 0.5, z + 1.0],
                                [x, y_f + 1.0, z + 1.0],
                                [x, y_f + 1.0, z + 0.5],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                        }
                        // Right face (+X): two quads – lower half and upper-back half.
                        if block.should_render_face_against(neighbors[1]) {
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f, z + 1.0],
                                [x + 1.0, y_f, z],
                                [x + 1.0, y_f + 0.5, z],
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [1.0, 0.0, 0.0],
                                color,
                                tex_side,
                                r,
                                m,
                            );
                            add_quad(
                                target_verts,
                                target_inds,
                                [x + 1.0, y_f + 0.5, z + 1.0],
                                [x + 1.0, y_f + 0.5, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 0.5],
                                [x + 1.0, y_f + 1.0, z + 1.0],
                                [1.0, 0.0, 0.0],
                                color,
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
                let mut mask: Vec<FaceAttrs> =
                    vec![FaceAttrs::default(); (dim1_size * dim2_size) as usize];

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
                                add_quad(
                                    &mut water_vertices,
                                    &mut water_indices,
                                    [x, y_f + 1.0, z],
                                    [x, y_f + 1.0, z + 1.0],
                                    [x + 1.0, y_f + 1.0, z + 1.0],
                                    [x + 1.0, y_f + 1.0, z],
                                    [0.0, 1.0, 0.0],
                                    block.color(),
                                    block.tex_top(),
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

                        // Biome lookup: only needed for Grass and Leaves.
                        let needs_biome = block == BlockType::Grass || block == BlockType::Leaves;
                        let biome = if needs_biome {
                            let lx_idx = lx as usize;
                            let lz_idx = lz as usize;
                            if biome_map[lx_idx][lz_idx].is_none() {
                                biome_map[lx_idx][lz_idx] = Some(self.get_biome(world_x, world_z));
                            }
                            biome_map[lx_idx][lz_idx]
                        } else {
                            None
                        };

                        // Select the face color based on direction and block type.
                        let color = match face_dir {
                            2 => block.bottom_color(), // bottom face
                            3 => {
                                // Top face: grass uses biome colour.
                                if block == BlockType::Grass {
                                    biome.map(|b| b.grass_color()).unwrap_or([0.4, 0.8, 0.2])
                                } else {
                                    block.top_color()
                                }
                            }
                            _ => {
                                // Side face: leaves use biome color.
                                if block == BlockType::Grass {
                                    block.color()
                                } else if block == BlockType::Leaves {
                                    biome.map(|b| b.leaves_color()).unwrap_or([0.2, 0.6, 0.2])
                                } else {
                                    block.color()
                                }
                            }
                        };

                        // Select the atlas texture index by face direction.
                        let tex_index = match face_dir {
                            2 => block.tex_bottom(),
                            3 => block.tex_top(),
                            _ => block.tex_side(),
                        };

                        let idx = (d1 * dim2_size + d2) as usize;
                        mask[idx] = FaceAttrs {
                            block,
                            color: quantize_color(color),
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

                        // Extend width along d2 while faces match.
                        let mut width = 1i32;
                        while d2 + width < dim2_size {
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
                        'height_loop: while d1 + height < dim1_size {
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

                        let color = [
                            face.color[0] as f32 / 255.0,
                            face.color[1] as f32 / 255.0,
                            face.color[2] as f32 / 255.0,
                        ];
                        let tex_index = face.tex_index as f32;
                        let roughness = face.block.roughness();
                        let metallic = face.block.metallic();
                        let (target_verts, target_inds) = if face.block == BlockType::Water {
                            (&mut water_vertices, &mut water_indices)
                        } else {
                            (&mut vertices, &mut indices)
                        };

                        // Convert (slice, d1, d2, width, height) back to world-
                        // space corner coordinates for the merged quad.
                        let (x0, y0, z0, x1, y1, z1) = match face_dir {
                            0 => {
                                let x = (base_x + slice) as f32;
                                let y0 = (base_y + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                (x, y0, z0, x, y0 + height as f32, z0 + width as f32)
                            }
                            1 => {
                                let x = (base_x + slice + 1) as f32;
                                let y0 = (base_y + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                (x, y0, z0, x, y0 + height as f32, z0 + width as f32)
                            }
                            2 => {
                                let y = (base_y + slice) as f32;
                                let x0 = (base_x + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                (x0, y, z0, x0 + height as f32, y, z0 + width as f32)
                            }
                            3 => {
                                let y = (base_y + slice + 1) as f32;
                                let x0 = (base_x + d1) as f32;
                                let z0 = (base_z + d2) as f32;
                                (x0, y, z0, x0 + height as f32, y, z0 + width as f32)
                            }
                            4 => {
                                let z = (base_z + slice) as f32;
                                let x0 = (base_x + d1) as f32;
                                let y0 = (base_y + d2) as f32;
                                (x0, y0, z, x0 + height as f32, y0 + width as f32, z)
                            }
                            5 => {
                                let z = (base_z + slice + 1) as f32;
                                let x0 = (base_x + d1) as f32;
                                let y0 = (base_y + d2) as f32;
                                (x0, y0, z, x0 + height as f32, y0 + width as f32, z)
                            }
                            _ => unreachable!(),
                        };

                        // Emit the merged quad with outward-facing winding.
                        // `add_greedy_quad` takes explicit width/height so the
                        // UV coordinates tile correctly across the merged surface.
                        match face_dir {
                            0 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z0],
                                [x0, y0, z1],
                                [x0, y1, z1],
                                [x0, y1, z0],
                                [-1.0, 0.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            1 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x1, y0, z1],
                                [x1, y0, z0],
                                [x1, y1, z0],
                                [x1, y1, z1],
                                [1.0, 0.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            2 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z1],
                                [x0, y0, z0],
                                [x1, y0, z0],
                                [x1, y0, z1],
                                [0.0, -1.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            3 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y1, z0],
                                [x0, y1, z1],
                                [x1, y1, z1],
                                [x1, y1, z0],
                                [0.0, 1.0, 0.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                width as f32,
                                height as f32,
                            ),
                            4 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x1, y0, z0],
                                [x0, y0, z0],
                                [x0, y1, z0],
                                [x1, y1, z0],
                                [0.0, 0.0, -1.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                height as f32,
                                width as f32,
                            ),
                            5 => add_greedy_quad(
                                target_verts,
                                target_inds,
                                [x0, y0, z1],
                                [x1, y0, z1],
                                [x1, y1, z1],
                                [x0, y1, z1],
                                [0.0, 0.0, 1.0],
                                color,
                                tex_index,
                                roughness,
                                metallic,
                                height as f32,
                                width as f32,
                            ),
                            _ => {}
                        }

                        d2 += width; // advance past the merged run
                    }
                }
            }
        }

        ((vertices, indices), (water_vertices, water_indices))
    }
}
