use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::world::spline::TerrainSpline;

// ─────────────────────────────────────────────────────────────────────────────
// ChunkGenerator
// ─────────────────────────────────────────────────────────────────────────────

/// Produces fully-detailed [`Chunk`] data from a deterministic seed.
///
/// One `ChunkGenerator` instance is shared across all chunk-worker threads
/// (via `Arc` or `Clone`); it is cheaply cloneable because cloning simply
/// re-runs `new` with the same seed.
///
/// # Noise budget
///
/// Each field is a separate `FastNoiseLite` sampler with an independent seed
/// offset and frequency.  Using distinct samplers (rather than one sampler
/// with different offsets) ensures there are no cross-correlation artifacts
/// between layers.
///
/// | Field | Role | Freq. | Type |
/// |---|---|---|---|
/// | `noise_continents` | Large-scale land/ocean shape | 0.0018 | FBm |
/// | `noise_terrain` | Mid-scale terrain undulation | 0.007 | FBm |
/// | `noise_detail` | Fine surface noise | 0.018 | FBm |
/// | `noise_temperature` | Biome temperature axis | 0.006 | Simplex |
/// | `noise_moisture` | Biome moisture axis | 0.008 | Simplex |
/// | `noise_river` | River channel carving | 0.055 | Simplex |
/// | `noise_lake` | Lake basin placement | 0.022 | Simplex |
/// | `noise_trees` | Tree/vegetation density | 0.12 | Simplex |
/// | `noise_island` | Ocean island elevation | 0.045 | Simplex |
/// | `noise_cave1/2/3` | 3-D cave volumes | 0.045/0.032/0.038 | 3-D Simplex |
/// | `noise_erosion` | Erosion multiplier for slopes | 0.004 | FBm |
/// | `noise_warp_x/z` | Domain warp for terrain/biomes | 0.005 | FBm |
/// | `noise_ridged` | Ridged mountain peaks | 0.009 | Ridged FBm |
/// | `noise_pv` | Peaks-and-valleys offset | 0.004 | FBm |
/// | `noise_decor` | Decoration placement (reserved) | 0.15 | Simplex |
/// | `noise_cave_warp_x/z` | Domain warp inside caves | 0.018 | FBm |
/// | `noise_surface_entrance` | Surface cave-entrance detection | 0.025 | FBm |
pub struct ChunkGenerator {
    /// Low-frequency FBm controlling continental land masses vs. ocean basins.
    noise_continents: FastNoiseLite,
    /// Mid-frequency FBm shaping hills and valleys within a biome.
    noise_terrain: FastNoiseLite,
    /// High-frequency FBm adding surface micro-variation.
    noise_detail: FastNoiseLite,
    /// Smooth simplex noise defining the temperature axis of the biome grid.
    noise_temperature: FastNoiseLite,
    /// Smooth simplex noise defining the moisture axis of the biome grid.
    noise_moisture: FastNoiseLite,
    /// Simplex noise used to carve river channels (peaks of `1 - |n|`).
    noise_river: FastNoiseLite,
    /// Simplex noise used to identify lake basin locations.
    noise_lake: FastNoiseLite,
    /// High-frequency simplex noise controlling tree/foliage spawn density.
    noise_trees: FastNoiseLite,
    /// Simplex noise used to raise island terrain above the ocean floor.
    noise_island: FastNoiseLite,
    /// 3-D simplex noise for the primary "cheese" cave volume.
    noise_cave1: FastNoiseLite,
    /// 3-D simplex noise for the secondary cave layer (also used for spaghetti caves).
    noise_cave2: FastNoiseLite,
    /// 3-D simplex noise for the tertiary cave layer (noodle + worm tunnel component).
    noise_cave3: FastNoiseLite,
    /// Low-frequency FBm whose value is converted by a spline into an erosion multiplier.
    noise_erosion: FastNoiseLite,
    /// FBm domain-warp X component applied before all terrain/biome sampling.
    noise_warp_x: FastNoiseLite,
    /// FBm domain-warp Z component applied before all terrain/biome sampling.
    noise_warp_z: FastNoiseLite,
    /// Ridged FBm producing sharp mountain ridges.
    noise_ridged: FastNoiseLite,
    /// FBm feeding the peaks-and-valleys spline to push mountains higher.
    noise_pv: FastNoiseLite,
    /// Reserved for future decoration placement; currently unused.
    #[allow(dead_code)]
    noise_decor: FastNoiseLite,
    /// FBm domain-warp X component applied inside the cave volume to make tunnels meander.
    noise_cave_warp_x: FastNoiseLite,
    /// FBm domain-warp Z component applied inside the cave volume.
    noise_cave_warp_z: FastNoiseLite,
    /// FBm used to detect candidate locations for vertical cave-entrance shafts.
    noise_surface_entrance: FastNoiseLite,
    /// The seed used to initialize all noise samplers.  Also stored so
    /// `Clone` can reproduce an identical generator without cloning each
    /// `FastNoiseLite` individually.
    pub seed: u32,
}

impl ChunkGenerator {
    /// Constructs a new generator from `seed`, creating all noise samplers.
    ///
    /// Each sampler receives `seed` incremented by a unique constant offset
    /// so that no two layers share the same random sequence, even at their
    /// lowest frequencies.  The offsets are intentionally non-consecutive
    /// (0–11 for the main layers, 20–31 for warp, 40 for surface entrances)
    /// to leave room for future additions without shifting existing seeds.
    pub fn new(seed: u32) -> Self {
        ChunkGenerator {
            noise_continents:       Self::create_fbm_noise(seed, 0.0018),
            noise_terrain:          Self::create_fbm_noise(seed.wrapping_add(1),  0.013),
            noise_detail:           Self::create_fbm_noise(seed.wrapping_add(2),  0.018),
            noise_temperature:      Self::create_noise(seed.wrapping_add(3),      0.009),
            noise_moisture:         Self::create_noise(seed.wrapping_add(4),      0.0012),
            noise_river:            Self::create_noise(seed.wrapping_add(5),      0.032),
            noise_lake:             Self::create_noise(seed.wrapping_add(6),      0.062),
            noise_trees:            Self::create_noise(seed.wrapping_add(7),      0.232),
            noise_island:           Self::create_noise(seed.wrapping_add(8),      0.035),
            noise_cave1:            Self::create_3d_noise(seed.wrapping_add(9),   0.025),
            noise_cave2:            Self::create_3d_noise(seed.wrapping_add(10),  0.0192),
            noise_cave3:            Self::create_3d_noise(seed.wrapping_add(11),  0.0128),
            noise_erosion:          Self::create_fbm_noise(seed.wrapping_add(12), 0.006),
            noise_warp_x:           Self::create_fbm_noise(seed.wrapping_add(20), 0.007),
            noise_warp_z:           Self::create_fbm_noise(seed.wrapping_add(21), 0.003),
            noise_ridged:           Self::create_ridged_noise(seed.wrapping_add(22), 0.006),
            noise_pv:               Self::create_fbm_noise(seed.wrapping_add(23), 0.005),
            noise_decor:            Self::create_noise(seed.wrapping_add(24),      0.13),
            noise_cave_warp_x:      Self::create_fbm_noise(seed.wrapping_add(30), 0.0218),
            noise_cave_warp_z:      Self::create_fbm_noise(seed.wrapping_add(31), 0.014),
            noise_surface_entrance: Self::create_fbm_noise(seed.wrapping_add(40), 0.015),
            seed,
        }
    }

    // ── Noise factory helpers ─────────────────────────────────────────────── //

    /// Creates a single-octave OpenSimplex2 sampler at the given frequency.
    ///
    /// Used for biome axis noise (temperature, moisture) and decoration
    /// placement where a smooth, featureless distribution is preferred.
    fn create_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(frequency));
        noise
    }

    /// Creates a 5-octave FBm (Fractional Brownian Motion) sampler.
    ///
    /// FBm stacks successive octaves at doubled frequency (`lacunarity = 2`)
    /// and halved amplitude (`gain = 0.5`), producing natural-looking
    /// self-similar terrain.  Used for continents, terrain shape, detail, and
    /// all domain-warp layers.
    fn create_fbm_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_fractal_type(Some(FractalType::FBm));
        noise.set_fractal_octaves(Some(5));
        noise.set_fractal_lacunarity(Some(2.0));
        noise.set_fractal_gain(Some(0.5));
        noise.set_frequency(Some(frequency));
        noise
    }

    /// Creates a 5-octave Ridged FBm sampler.
    ///
    /// Ridged noise folds negative values upward (`|n|`) and then inverts so
    /// that the highest values appear as sharp ridges.  A slightly higher
    /// lacunarity (`2.2`) adds extra crinkliness to mountain silhouettes.
    /// Used exclusively for the mountain-ridge layer.
    fn create_ridged_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_fractal_type(Some(FractalType::Ridged));
        noise.set_fractal_octaves(Some(5));
        noise.set_fractal_lacunarity(Some(2.2));
        noise.set_fractal_gain(Some(0.5));
        noise.set_frequency(Some(frequency));
        noise
    }

    /// Creates a single-octave 3-D OpenSimplex2 sampler.
    ///
    /// 3-D sampling is required for cave volumes so that the noise varies
    /// continuously both horizontally and vertically (2-D noise would produce
    /// columns of uniform cave/solid blocks at any given XZ position).
    fn create_3d_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(frequency));
        noise
    }

    // ── Public chunk generation ───────────────────────────────────────────── //

    /// Generates a complete [`Chunk`] for chunk column `(cx, cz)`.
    ///
    /// # Generation passes (in order)
    ///
    /// 1. **Biome & height pre-pass** – samples `get_biome` and
    ///    `get_terrain_height` for every column and caches results in
    ///    `biome_map` and `height_map`.  Pre-caching avoids redundant noise
    ///    evaluations in the block-fill loops below.
    ///
    /// 2. **Block fill** – for each column iterates Y from 0 to `max_y` and
    ///    places solid terrain, water, and ice.  Mountain and Island biomes use
    ///    an extra 3-D density query near the surface so overhangs and arches
    ///    are possible.
    ///
    /// 3. **Cave carving** – builds a `cave_entrance_map` first (used to relax
    ///    the surface-proximity guard near real openings), then calls
    ///    `is_cave` for every sub-surface block and sets matching blocks to Air.
    ///    Bedrock is never carved.
    ///
    /// 4. **Cave decoration** – scans cave-air columns for:
    ///    - Gravel patches on cave floors.
    ///    - Clay deposits at mid-depth (Y 35–55).
    ///    - Stalagmites growing upward from Stone floors.
    ///    - Stalactites hanging downward from Stone ceilings.
    ///
    /// 5. **Surface cave-entrance shafts** – places vertical cylindrical shafts
    ///    that break the surface and connect to the cave system below, giving
    ///    players visible entry points without special dungeon structures.
    ///    Shafts are skipped near ocean/river/lake/beach biomes and at low
    ///    elevations.
    ///
    /// 6. **Surface decorations** – delegates to `generate_decorations` for
    ///    trees, cacti, dead bushes, mountain gravel, and snow caps.
    ///
    /// 7. **Sub-chunk metadata** – calls `check_empty` and `check_fully_opaque`
    ///    on every sub-chunk so the renderer can skip invisible sections.
    ///
    /// # Parameters
    /// - `cx` – Chunk column X coordinate (in chunks, not blocks).
    /// - `cz` – Chunk column Z coordinate (in chunks, not blocks).
    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cz);
        let base_x = cx * CHUNK_SIZE;
        let base_z = cz * CHUNK_SIZE;

        // ── Pass 1: biome and height pre-computation ──────────────────────── //
        // Caching avoids evaluating the same multi-octave noise up to three
        // more times per column in the loops below.
        let mut biome_map  = [[Biome::Plains; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        let mut height_map = [[0i32;          CHUNK_SIZE as usize]; CHUNK_SIZE as usize];

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let lx_usize = lx as usize;
                let lz_usize = lz as usize;
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                biome_map[lx_usize][lz_usize]  = self.get_biome(world_x, world_z);
                height_map[lx_usize][lz_usize] = self.get_terrain_height(world_x, world_z);
            }
        }

        // ── Pass 2: block fill ────────────────────────────────────────────── //
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome         = biome_map[lx as usize][lz as usize];
                let surface_height = height_map[lx as usize][lz as usize];

                // Mountains and islands can have overhangs above the 2-D
                // surface height, so their column must extend higher to
                // accommodate 3-D density blocks.  All other biomes only
                // need to reach just above sea level or the surface.
                let max_y = if matches!(biome, Biome::Mountains | Biome::Island) {
                    WORLD_HEIGHT - 20
                } else {
                    (surface_height + 5).max(SEA_LEVEL)
                };

                for y in 0..max_y {
                    // Default: solid below the 2-D surface height.
                    let mut is_solid = y < surface_height;

                    // For biomes that support overhangs, blend in a 3-D
                    // density field within the top 8 blocks of the surface
                    // so arches and cliff faces emerge naturally.
                    if matches!(biome, Biome::Mountains | Biome::Island)
                        && y >= surface_height - 8
                    {
                        let density = self.get_3d_density(
                            world_x, y, world_z, biome, surface_height,
                        );
                        if density > 0.0 {
                            is_solid = true;
                        }
                    }

                    if is_solid {
                        let block = self.get_block_for_biome(
                            biome, y, surface_height, world_x, world_z,
                        );
                        if block != BlockType::Air {
                            chunk.set_block(lx, y, lz, block);
                        }
                    } else if y >= surface_height && y < SEA_LEVEL {
                        // Below sea level but above the terrain surface →
                        // water or ice depending on the biome.
                        if biome == Biome::Tundra && y == SEA_LEVEL - 1 {
                            chunk.set_block(lx, y, lz, BlockType::Ice);
                        } else {
                            chunk.set_block(lx, y, lz, BlockType::Water);
                        }
                    }
                }
            }
        }

        // ── Pass 3: cave carving ──────────────────────────────────────────── //
        // Build the entrance map first so `is_cave` can relax the surface-
        // proximity guard near columns that have a real cave opening.
        let mut cave_entrance_map = [[false; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let height = height_map[lx as usize][lz as usize];
                cave_entrance_map[lx as usize][lz as usize] =
                    self.is_cave_entrance(world_x, world_z, height);
            }
        }

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x   = base_x + lx;
                let world_z   = base_z + lz;
                let height    = height_map[lx as usize][lz as usize];
                let is_entrance = cave_entrance_map[lx as usize][lz as usize];

                // Start at Y = 1 to preserve bedrock at Y = 0.
                for y in 1..height.min(WORLD_HEIGHT - 1) {
                    if self.is_cave(world_x, y, world_z, height, is_entrance) {
                        let current = chunk.get_block(lx, y, lz);
                        // Never carve bedrock; skip air (already empty).
                        if current != BlockType::Bedrock && current != BlockType::Air {
                            chunk.set_block(lx, y, lz, BlockType::Air);
                        }
                    }
                }
            }
        }

        // ── Pass 4: cave decoration (floor/ceiling features) ──────────────── //
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let height  = height_map[lx as usize][lz as usize];
                let hash_xz = self.position_hash(world_x, world_z);

                // Y = 5 lower bound prevents gravel/stalagmites displacing
                // the probabilistic bedrock layer.
                for y in 5..height.min(WORLD_HEIGHT - 2) {
                    let current = chunk.get_block(lx, y, lz);
                    if current != BlockType::Air {
                        continue; // only decorate air cells
                    }

                    let below = chunk.get_block(lx, y - 1, lz);
                    let above = chunk.get_block(lx, y + 1, lz);

                    // ---- Floor features (solid below, air above) ------------
                    if below != BlockType::Air
                        && below != BlockType::Water
                        && above == BlockType::Air
                    {
                        let hash3 = self.position_hash_3d(world_x, y, world_z);

                        // Convert stone/dirt/gravel floors to pure gravel (15%)
                        // or clay at mid-depth (8% chance, Y 35–55).
                        if matches!(
                            below,
                            BlockType::Stone | BlockType::Dirt | BlockType::Gravel
                        ) {
                            if hash3 % 100 < 15 {
                                chunk.set_block(lx, y - 1, lz, BlockType::Gravel);
                            } else if y >= 35 && y <= 55 && hash3 % 100 < 8 {
                                chunk.set_block(lx, y - 1, lz, BlockType::Clay);
                            }
                        }

                        // Stalagmites grow 1–3 blocks upward from stone floors.
                        // Gated by `hash_xz` (8%) and a minimum Y of 8 so they
                        // don't appear in the bedrock transition zone.
                        if below == BlockType::Stone && hash_xz % 100 < 8 && y >= 8 {
                            let stalagmite_h = 1 + (hash3 % 3) as i32;
                            for dy in 0..stalagmite_h {
                                let ny = y + dy;
                                if ny < WORLD_HEIGHT
                                    && chunk.get_block(lx, ny, lz) == BlockType::Air
                                {
                                    chunk.set_block(lx, ny, lz, BlockType::Stone);
                                } else {
                                    break; // stop if we hit an existing block
                                }
                            }
                        }
                    }

                    // ---- Ceiling features (solid above, air below) ----------
                    if above != BlockType::Air
                        && above != BlockType::Water
                        && below == BlockType::Air
                    {
                        let hash3 = self.position_hash_3d(world_x, y, world_z);

                        // Stalactites hang 1–2 blocks downward from stone ceilings.
                        // Lower probability than stalagmites (6%) for visual balance.
                        if above == BlockType::Stone
                            && hash_xz.wrapping_add(7) % 100 < 6
                        {
                            let stalactite_h = 1 + (hash3 % 2) as i32;
                            for dy in 0..stalactite_h {
                                let ny = y - dy;
                                if ny > 4
                                    && chunk.get_block(lx, ny, lz) == BlockType::Air
                                {
                                    chunk.set_block(lx, ny, lz, BlockType::Stone);
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Pass 5: surface cave-entrance shafts ──────────────────────────── //
        // Vertical cylindrical shafts punch through the surface at positions
        // confirmed to have a cave below, giving the player a visible way in.
        // Inner border (1 block margin) is excluded to avoid clipping the
        // shaft circle against the chunk boundary.
        for lx in 1..(CHUNK_SIZE - 1) {
            for lz in 1..(CHUNK_SIZE - 1) {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome  = biome_map[lx as usize][lz as usize];
                let height = height_map[lx as usize][lz as usize];

                // Shafts are only dug in dry above-water biomes.
                if matches!(
                    biome,
                    Biome::Ocean | Biome::River | Biome::Lake | Biome::Beach
                ) || height <= SEA_LEVEL + 3
                {
                    continue;
                }

                if !self.is_surface_cave_entrance(world_x, world_z, height) {
                    continue;
                }

                let hash = self.position_hash(world_x, world_z);
                // Radius 2 for 1-in-3 locations, radius 1 otherwise.
                let shaft_radius: i32 = if hash % 3 == 0 { 2 } else { 1 };

                let max_shaft_depth = 24;
                let shaft_start = height - 1;
                let shaft_end   = (shaft_start - max_shaft_depth).max(SEA_LEVEL + 1);

                // Iterate downward so we can bail early (`break 'shaft`) as
                // soon as we hit an air block far enough below the surface,
                // which indicates we've connected to an existing cave.
                'shaft: for y in (shaft_end..=shaft_start).rev() {
                    for dx in -shaft_radius..=shaft_radius {
                        for dz in -shaft_radius..=shaft_radius {
                            // Circular cross-section via squared-distance check.
                            if dx * dx + dz * dz
                                > shaft_radius * shaft_radius + shaft_radius
                            {
                                continue;
                            }
                            let nx = lx + dx;
                            let nz = lz + dz;
                            if nx < 0 || nx >= CHUNK_SIZE || nz < 0 || nz >= CHUNK_SIZE {
                                continue;
                            }
                            let current = chunk.get_block(nx, y, nz);
                            // If we find an air cell more than 3 blocks below
                            // the shaft start we've joined a cave: stop digging.
                            if current == BlockType::Air && y < shaft_start - 3 {
                                break 'shaft;
                            }
                            if current != BlockType::Bedrock
                                && current != BlockType::Air
                            {
                                chunk.set_block(nx, y, nz, BlockType::Air);
                            }
                        }
                    }
                }
            }
        }

        // ── Pass 6: surface decorations (trees, cacti, snow, etc.) ───────── //
        self.generate_decorations(&mut chunk, cx, cz, &biome_map, &height_map);

        // ── Pass 7: sub-chunk metadata ────────────────────────────────────── //
        // `check_empty` marks sub-chunks that contain only Air so the mesh
        // builder can skip them.  `check_fully_opaque` marks sub-chunks that
        // are completely filled with opaque blocks so face-culling can skip
        // their interior faces.
        for subchunk in &mut chunk.subchunks {
            subchunk.check_empty();
            subchunk.check_fully_opaque();
        }

        chunk
    }

    // ── Public forwarding accessors ───────────────────────────────────────── //
    // These thin wrappers expose private methods to the `ChunkLoader` and
    // editor tools that need to query terrain metadata without generating a
    // full chunk.

    /// Returns the terrain surface height at world position `(x, z)`.
    ///
    /// Equivalent to calling `get_terrain_height` directly; exposed publicly
    /// because the loader needs it for spawn-point search and LOD decisions.
    pub fn get_terrain_height_pub(&self, x: i32, z: i32) -> i32 {
        self.get_terrain_height(x, z)
    }

    /// Returns `true` if `(x, z)` is a cave-entrance column at the given height.
    ///
    /// Exposed so the `ChunkLoader` can pre-query entrance positions when
    /// scheduling chunk generation order.
    pub fn is_cave_entrance_pub(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.is_cave_entrance(x, z, surface_height)
    }

    /// Returns the 2-D position hash for `(x, z)`.
    ///
    /// Exposed for editor and debug tooling that needs a cheap, deterministic
    /// pseudo-random value at a block position without constructing a noise
    /// sampler.
    pub fn position_hash_pub(&self, x: i32, z: i32) -> u32 {
        self.position_hash(x, z)
    }

    // ── Biome classification ──────────────────────────────────────────────── //

    /// Classifies the biome at world position `(x, z)`.
    ///
    /// # Algorithm
    ///
    /// 1. **Domain warp** – offset `(x, z)` by up to ±80 blocks using
    ///    `noise_warp_x/z` to break up biome boundaries from straight lines
    ///    into natural-looking coastlines.
    ///
    /// 2. **Water bodies** – Rivers and lakes are tested first because they
    ///    take priority over any land biome at the same location.
    ///
    /// 3. **Ocean / island** – Negative continental values indicate open ocean;
    ///    sparse island noise can override this in low-continent regions.
    ///
    /// 4. **Beach** – A thin transition band between −0.38 and −0.22 on the
    ///    continental axis.
    ///
    /// 5. **Mountains** – Triggered by high peaks-and-valleys value combined
    ///    with low erosion and positive continental value.
    ///
    /// 6. **Temperature / moisture grid** – The remaining land biomes
    ///    (Tundra, Desert, Swamp, Forest, Plains) are selected by querying
    ///    the 2-D temperature and moisture axes.
    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        let fx = x as f32;
        let fz = z as f32;

        // Domain warp: shift the sample point before evaluating any large-scale
        // noise, producing irregular, coast-like biome boundaries.
        let warp_scale = 80.0_f32;
        let wx = fx
            + self.noise_warp_x.get_noise_2d(fx * 0.004, fz * 0.004) * warp_scale;
        let wz = fz
            + self.noise_warp_z
            .get_noise_2d(fx * 0.004 + 100.0, fz * 0.004 + 100.0)
            * warp_scale;

        let continent   = self.noise_continents.get_noise_2d(wx * 0.0018, wz * 0.0018);
        let river_noise = self.noise_river.get_noise_2d(wx * 0.055, wz * 0.055);
        // `1 - |n| * 2` transforms the raw noise into a sharp peak at the
        // center of the river channel; values above 0.88 are inside the river.
        let river_value = 1.0 - river_noise.abs() * 2.0;
        let lake_noise  = self.noise_lake.get_noise_2d(wx * 0.022, wz * 0.022);

        // Rivers take precedence but only on land (continent > −0.25).
        if river_value > 0.88 && continent > -0.25 {
            return Biome::River;
        }

        // Lakes form in basins where lake noise is very low, again only on land.
        if lake_noise < -0.62 && continent > -0.15 {
            return Biome::Lake;
        }

        // Deep ocean, potentially dotted with islands.
        if continent < -0.38 {
            let island_noise = self.noise_island.get_noise_2d(wx * 0.045, wz * 0.045);
            if island_noise > 0.60 {
                return Biome::Island;
            }
            return Biome::Ocean;
        }

        // Transitional beach strip.
        if continent < -0.22 {
            return Biome::Beach;
        }

        // Above here we are on solid land.  Sample climate axes once.
        let temp    = self.noise_temperature.get_noise_2d(wx * 0.006, wz * 0.006);
        let moist   = self.noise_moisture.get_noise_2d(wx * 0.008, wz * 0.008);
        let erosion = self.noise_erosion.get_noise_2d(wx * 0.004, wz * 0.004);
        let pv      = self.noise_pv.get_noise_2d(wx * 0.004, wz * 0.004);

        // Mountains: high peaks-and-valleys value + low erosion on mainland.
        if pv > 0.3 && erosion < 0.25 && continent > 0.0 {
            return Biome::Mountains;
        }

        // Very cold → Tundra.
        if temp < -0.3 {
            return Biome::Tundra;
        }

        // Hot climate.
        if temp > 0.4 {
            if moist < -0.2 {
                return Biome::Desert; // hot & dry
            }
            if moist > 0.15 {
                return Biome::Swamp; // hot & wet
            }
        }

        // Wet mid-temperature → Swamp.
        if moist > 0.45 && temp > -0.1 {
            return Biome::Swamp;
        }

        // Moderate moisture → Forest.
        if moist > -0.05 {
            return Biome::Forest;
        }

        // Default: open grassland.
        Biome::Plains
    }

    // ── Terrain height ────────────────────────────────────────────────────── //

    /// Returns the surface terrain height at world position `(x, z)`.
    ///
    /// Rather than sampling the height at exactly `(x, z)`, this method
    /// blends heights from a 5×5 neighborhood (radius 2) using a Gaussian
    /// weight (`exp(-dist² / (radius × 1.5))`).  Blending softens the hard
    /// biome transitions that would otherwise produce vertical walls where,
    /// e.g., Plains meets Mountains.
    ///
    /// The blended result is clamped to `[1, WORLD_HEIGHT - 20]` so chunks
    /// always have at least one surface block and never reach the absolute
    /// ceiling.
    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        let blend_radius = 2i32;
        let center_biome = self.get_biome(x, z);
        let mut total_height = 0.0;
        let mut weights      = 0.0;

        for dx in -blend_radius..=blend_radius {
            for dz in -blend_radius..=blend_radius {
                let wx = x + dx;
                let wz = z + dz;
                let dist_sq = (dx * dx + dz * dz) as f64;
                // Gaussian falloff centred on (x, z); nearby samples dominate.
                let weight = (-dist_sq / (blend_radius as f64 * 1.5)).exp();

                let height = self.calculate_base_height_with_biome(wx, wz, center_biome);
                total_height += height * weight;
                weights      += weight;
            }
        }

        let base_height = total_height / weights;
        (base_height as i32).clamp(1, WORLD_HEIGHT - 20)
    }

    /// Computes the raw (unblended) height at `(x, z)` by first classifying
    /// the biome and then delegating to `calculate_base_height_with_biome`.
    ///
    /// Currently unused in favor of the blended `get_terrain_height`, but
    /// kept for debugging and future LOD purposes.
    #[allow(dead_code)]
    fn calculate_base_height(&self, x: i32, z: i32) -> f64 {
        let biome = self.get_biome(x, z);
        self.calculate_base_height_with_biome(x, z, biome)
    }

    /// Core height function for a single `(x, z)` sample given a pre-computed `biome`.
    ///
    /// # Algorithm
    ///
    /// 1. Apply domain warp (±60 blocks) to break up axis-aligned patterns.
    /// 2. Sample all relevant noise layers once.
    /// 3. Convert continental, erosion, and peaks-and-valleys noise through
    ///    hand-tuned splines (see [`TerrainSpline`]) that map raw `[-1, 1]`
    ///    values to world-space height contributions.
    /// 4. Combine contributions according to biome-specific formulas.
    ///
    /// The per-biome formulas use different base heights, noise weights, and
    /// erosion multipliers to produce distinct characteristic landscapes:
    /// - **Ocean/River/Lake** – shallow sub-sea-level floors.
    /// - **Beach/Island** – gentle low-elevation coasts.
    /// - **Plains/Forest** – rolling hills above sea level.
    /// - **Desert** – flat sandy plateau with dune ripples.
    /// - **Tundra** – moderate hills, slightly above sea level.
    /// - **Mountains** – ridged peaks powered by the ridged FBm and PV spline.
    /// - **Swamp** – nearly flat, hovering just above sea level.
    fn calculate_base_height_with_biome(&self, x: i32, z: i32, biome: Biome) -> f64 {
        let fx = x as f32;
        let fz = z as f32;

        // Domain warp (slightly smaller amplitude than biome warp to keep
        // the terrain shape coherent within a classified biome region).
        let warp_scale = 60.0_f32;
        let wx = fx
            + self.noise_warp_x.get_noise_2d(fx * 0.005, fz * 0.005) * warp_scale;
        let wz = fz
            + self.noise_warp_z
            .get_noise_2d(fx * 0.005 + 200.0, fz * 0.005 + 200.0)
            * warp_scale;

        let continental = self.noise_continents.get_noise_2d(wx, wz) as f64;
        let terrain     = self.noise_terrain.get_noise_2d(wx, wz) as f64;
        let detail      = self.noise_detail.get_noise_2d(wx, wz) as f64;
        let erosion     = self.noise_erosion.get_noise_2d(wx, wz) as f64;
        let ridged      = self.noise_ridged.get_noise_2d(wx, wz) as f64;
        let pv          = self.noise_pv.get_noise_2d(wx, wz) as f64;

        // Spline lookups convert raw noise into physically-motivated offsets.
        let cont_spline  = TerrainSpline::continental();
        let cont_height  = cont_spline.sample(continental);

        let erosion_spline = TerrainSpline::erosion();
        let erosion_mult   = erosion_spline.sample(erosion); // typically 0.3–1.0

        let pv_spline  = TerrainSpline::peaks_valleys();
        let pv_offset  = pv_spline.sample(pv);

        match biome {
            Biome::Ocean => {
                // Ocean floor: deepens as the continental value drops further
                // below zero; detail adds subtle seabed variation.
                let depth = 20.0 + (continental + 1.0) * 0.5 * 18.0;
                depth + detail * 2.5
            }
            Biome::River => {
                // Uśrednienie wysokości z sąsiadami dla łagodniejszych brzegów
                let mut sum = 0.0;
                let mut count = 0.0;
                for dx in -1..=1 {
                    for dz in -1..=1 {
                        let nx = x + dx;
                        let nz = z + dz;
                        // Pomijamy środek, by nie podwajać
                        if dx != 0 || dz != 0 {
                            sum += self.calculate_base_height_with_biome(nx, nz, Biome::Plains);
                            count += 1.0;
                        }
                    }
                }
                let avg = sum / count;
                // Dno rzeki to średnia z sąsiadów (plains) - 2, plus drobny detal
                (avg - 2.0).min((SEA_LEVEL - 2) as f64) + detail * 1.5
            },
            Biome::Lake  => (SEA_LEVEL - 5) as f64 + detail * 2.0,
            Biome::Beach => SEA_LEVEL as f64 + terrain * 3.5 * erosion_mult + detail * 1.5,
            Biome::Island => {
                let island_noise =
                    self.noise_island.get_noise_2d(wx * 0.045, wz * 0.045) as f64;
                // Island elevation: remapped island noise (0–28 blocks above
                // sea level) plus terrain variation.
                let island_h = (island_noise + 1.0) * 0.5 * 28.0;
                (SEA_LEVEL as f64
                    + island_h
                    + terrain * 4.0 * erosion_mult
                    + detail * 3.0)
                    .max(SEA_LEVEL as f64 - 3.0)
            }
            Biome::Plains => {
                // Gentle rolling hills: a second terrain sample at a slightly
                // higher frequency adds medium-scale undulation.
                let rolling = self.noise_terrain.get_noise_2d(wx * 0.012, wz * 0.012) as f64;
                cont_height.max(66.0)
                    + terrain * 5.0 * erosion_mult
                    + rolling * 3.5
                    + detail * 2.0
            }
            Biome::Forest => {
                let hills = self.noise_terrain.get_noise_2d(wx * 0.010, wz * 0.010) as f64;
                cont_height.max(67.0)
                    + terrain * 9.0 * erosion_mult
                    + hills * 7.0
                    + detail * 4.0
            }
            Biome::Desert => {
                // Dune layer: high-frequency detail remapped to [0, 12] blocks.
                let dune   = self.noise_detail.get_noise_2d(wx * 0.022, wz * 0.022) as f64;
                let dune_h = (dune + 1.0) * 0.5 * 12.0;
                62.0 + terrain * 7.0 * erosion_mult + dune_h + detail * 3.0
            }
            Biome::Tundra => {
                let frozen = self.noise_terrain.get_noise_2d(wx * 0.009, wz * 0.009) as f64;
                66.0 + terrain * 9.0 * erosion_mult + frozen * 6.0 + detail * 3.5
            }
            Biome::Mountains => {
                // Ridge strength: normalize ridged noise to [0, 1] and apply
                // a power curve to sharpen peaks, then scale to 80 blocks.
                let ridge_strength = ((ridged + 1.0) * 0.5).powf(1.8) * 80.0;
                let base = cont_height.max(80.0);
                base + ridge_strength
                    + pv_offset.max(0.0) * 0.6
                    + terrain * 12.0 * erosion_mult
                    + detail * 5.0
            }
            Biome::Swamp => {
                let lumps = self.noise_detail.get_noise_2d(wx * 0.035, wz * 0.035) as f64;
                SEA_LEVEL as f64
                    + 1.5
                    + terrain * 2.5 * erosion_mult
                    + lumps * 2.5
                    + detail * 1.0
            }
        }
    }

    // ── Cave system ───────────────────────────────────────────────────────── //

    /// Returns `true` if the block at `(x, y, z)` should be hollow (cave air).
    ///
    /// Three cave algorithms run in a depth-dependent stack:
    ///
    /// | Name | Depth | Mechanism |
    /// |---|---|---|
    /// | Cheese | Y < 54 | Product of two clamped noise values; produces large rounded voids. |
    /// | Spaghetti | all depths | 2-D distance in the `(s1, s2)` plane; radius shrinks with altitude. |
    /// | Noodle | Y > 20 | Thinner spaghetti variant using higher-frequency noise. |
    /// | Worm | Y < 30 | Widest of the thin tunnels, exclusive to the deep zone. |
    ///
    /// All layers are evaluated in warped space: `(wx, wy, wz)` are shifted
    /// by `noise_cave_warp_x/z` (amplitude 12 blocks XZ, 1.8 blocks Y) so
    /// tunnels meander instead of running straight.
    ///
    /// A **surface proximity guard** prevents caves from breaking through the
    /// surface.  Near a cave entrance the guard relaxes gracefully (8 → 4 block
    /// minimum distance) so the entrance shaft can connect to the cave naturally.
    ///
    /// # Parameters
    /// - `x, y, z`        – World-space block position.
    /// - `surface_height` – Terrain surface Y at `(x, z)`.
    /// - `is_entrance`    – Whether this column has been flagged as a cave entrance.
    fn is_cave(
        &self,
        x: i32, y: i32, z: i32,
        surface_height: i32,
        is_entrance: bool,
    ) -> bool {
        // Bedrock zone is never carved.
        if y <= 4 {
            return false;
        }

        // Surface proximity guard: keep caves from punching through the roof.
        // At entrance columns the guard shrinks linearly over the top 10 blocks
        // so there is a smooth neck connecting to the shaft above.
        let min_surface_dist = if is_entrance {
            let t = ((surface_height - y) as f32 / 10.0).clamp(0.0, 1.0);
            (4.0 + t * 4.0) as i32
        } else {
            8
        };
        if y >= surface_height - min_surface_dist {
            return false;
        }

        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        // Domain warp: meanders tunnels both horizontally (±12 blocks) and
        // vertically (±12 × 0.15 ≈ ±1.8 blocks) using two FBm samplers.
        let warp_amp = 12.0_f32;
        let wx = fx
            + self.noise_cave_warp_x
            .get_noise_3d(fx * 0.018, fy * 0.010, fz * 0.018)
            * warp_amp;
        let wy = fy
            + self.noise_cave_warp_z.get_noise_3d(
            fx * 0.018 + 100.0,
            fy * 0.010,
            fz * 0.018 + 100.0,
        ) * warp_amp * 0.15;
        let wz = fz
            + self.noise_cave_warp_x.get_noise_3d(
            fx * 0.018 + 200.0,
            fy * 0.010,
            fz * 0.018 + 200.0,
        ) * warp_amp;

        let in_lower  = y < 54;
        let in_middle = y >= 54 && y < 90;

        // ---- Cheese caves (deep zone only) ---------------------------------
        // Two noise values are clamped to [0, ∞) and multiplied; the product
        // is positive only where both are simultaneously above zero, creating
        // isolated rounded voids rather than connected tunnels.
        if in_lower {
            let c1 = self.noise_cave1.get_noise_3d(wx * 0.030, wy * 0.010, wz * 0.030);
            let c2 = self.noise_cave2.get_noise_3d(
                wx * 0.022 + 400.0,
                wy * 0.008 + 400.0,
                wz * 0.022 + 400.0,
            );
            let cheese_product = c1.max(0.0) * c2.max(0.0);
            if cheese_product > 0.20 {
                return true;
            }
        }

        // ---- Spaghetti caves (all depths) ----------------------------------
        // Two noise values are treated as a 2-D point; tunnels exist where the
        // Euclidean distance from the origin is below `spag_radius`.  The
        // radius shrinks with altitude to produce thinner tunnels near the surface.
        let s1 = self.noise_cave1.get_noise_3d(wx * 0.060 + 500.0, wy * 0.025, wz * 0.060);
        let s2 = self.noise_cave3.get_noise_3d(wx * 0.060 + 900.0, wy * 0.025, wz * 0.060);
        let spag_dist   = (s1 * s1 + s2 * s2).sqrt();
        let spag_radius = if in_lower {
            0.13
        } else if in_middle {
            0.10
        } else {
            0.07
        };
        if spag_dist < spag_radius {
            return true;
        }

        // ---- Noodle caves (mid + upper zone) --------------------------------
        // Higher-frequency, smaller-radius variant of spaghetti: narrow winding
        // tubes that appear above Y = 20 and are particularly visible in the
        // upper 90–150 block range.
        if y > 20 {
            let n1 = self.noise_cave2.get_noise_3d(wx * 0.090 + 800.0,  wy * 0.040, wz * 0.090);
            let n2 = self.noise_cave3.get_noise_3d(wx * 0.090 + 1200.0, wy * 0.040, wz * 0.090);
            let noodle_dist   = (n1 * n1 + n2 * n2).sqrt();
            let noodle_radius = if in_lower { 0.075 } else { 0.055 };
            if noodle_dist < noodle_radius {
                return true;
            }
        }

        // ---- Worm tunnels (deep zone only) ----------------------------------
        // Widest of the thin tunnels; lower frequency than noodles gives them
        // a more sinuous, winding character.
        if y < 30 {
            let w1 = self.noise_cave2.get_noise_3d(wx * 0.042 + 800.0,  wy * 0.015, wz * 0.042);
            let w2 = self.noise_cave3.get_noise_3d(wx * 0.042 + 1200.0, wy * 0.015, wz * 0.042);
            let worm_dist = (w1 * w1 + w2 * w2).sqrt();
            if worm_dist < 0.11 {
                return true;
            }
        }

        false
    }

    /// Returns `true` if `(x, z)` at the given surface height is a location
    /// where the cave system reaches close enough to the surface to be
    /// considered a natural hillside or cliff entrance.
    ///
    /// # Algorithm
    /// 1. Reject underwater columns (`surface_height ≤ SEA_LEVEL + 2`).
    /// 2. Filter by a high-threshold 2-D noise value to limit entrance density.
    /// 3. Prefer hillside locations (high `terrain` slope) by using a lower
    ///    threshold when `terrain_slope > 0.18`.
    /// 4. Apply a hash-based random gate (4% on hillsides, 10% elsewhere) so
    ///    not every candidate column actually becomes an entrance.
    /// 5. Verify that at least one block in the column (surface − 40 to
    ///    surface − 6) matches the cheese-cave condition, confirming that the
    ///    cave truly exists below.
    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 2 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        // 2-D noise pre-filter: eliminates the vast majority of columns cheaply.
        let entrance_noise = self
            .noise_cave1
            .get_noise_2d(fx * 0.014 + 1000.0, fz * 0.014 + 1000.0);

        // Slope proxy: high absolute terrain noise indicates a hillside, which
        // is a more plausible cliff-face entrance location.
        let terrain_slope = self
            .noise_terrain
            .get_noise_2d(fx * 0.018, fz * 0.018)
            .abs();
        let is_hillside = terrain_slope > 0.18;

        let threshold = if is_hillside { 0.68 } else { 0.82 };
        if entrance_noise < threshold {
            return false;
        }

        // Hash gate: reduces spatial density of entrances.
        let hash = self.position_hash(x, z);
        let entrance_chance = if is_hillside { 4 } else { 10 };
        if hash % entrance_chance != 0 {
            return false;
        }

        // Confirm that a cave volume actually exists below this column.
        for check_y in (surface_height - 40).max(8)..=(surface_height - 6) {
            let fy = check_y as f32;
            let c1 = self.noise_cave1.get_noise_3d(fx * 0.045, fy * 0.022, fz * 0.045);
            let c2 = self.noise_cave2.get_noise_3d(fx * 0.032, fy * 0.018, fz * 0.032);
            if c1 > 0.62 && c2 > 0.62 {
                return true;
            }
        }

        false
    }

    /// Returns `true` if `(x, z)` is a candidate for a **vertical shaft**
    /// entrance that breaks through the surface from above.
    ///
    /// Unlike `is_cave_entrance` (which tests for a hillside opening), this
    /// method identifies locations where a straight-down shaft should be
    /// carved to create a visible sinkhole or pit.  The shaft is then dug
    /// in pass 5 of `generate_chunk`.
    ///
    /// # Algorithm
    /// 1. Reject underwater columns.
    /// 2. Apply a high-threshold noise filter (`ent_noise > 0.72`).
    /// 3. Hash gate: 1-in-8 surviving columns proceed.
    /// 4. Confirm that either a cheese-cave or a spaghetti-cave passes through
    ///    the column at some depth between (surface − 22) and (surface − 5).
    fn is_surface_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 3 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        let ent_noise = self
            .noise_surface_entrance
            .get_noise_2d(fx * 0.025, fz * 0.025);
        if ent_noise < 0.72 {
            return false;
        }

        let hash = self.position_hash(x, z);
        if hash % 8 != 0 {
            return false;
        }

        // Scan the column for either a cheese or spaghetti cave confirmation.
        for check_y in (surface_height - 22).max(8)..=(surface_height - 5) {
            let fy = check_y as f32;

            // Cheese-cave check (same thresholds as in `is_cave`).
            let c1 = self.noise_cave1.get_noise_3d(fx * 0.045, fy * 0.022, fz * 0.045);
            let c2 = self.noise_cave2.get_noise_3d(fx * 0.032, fy * 0.018, fz * 0.032);
            if c1 > 0.55 && c2 > 0.55 {
                return true;
            }

            // Spaghetti-cave check.
            let s1 = self
                .noise_cave1
                .get_noise_3d(fx * 0.065 + 500.0, fy * 0.055, fz * 0.065);
            let s2 = self
                .noise_cave3
                .get_noise_3d(fx * 0.065 + 900.0, fy * 0.055, fz * 0.065);
            if (s1 * s1 + s2 * s2).sqrt() < 0.11 {
                return true;
            }
        }

        false
    }

    // ── 3-D density (overhangs) ───────────────────────────────────────────── //

    /// Computes a signed density value for 3-D terrain overhangs in mountain
    /// and island biomes.
    ///
    /// Positive values indicate solid rock; zero or negative values indicate
    /// air (an overhang or arch).  The value is evaluated only within 8 blocks
    /// of the 2-D surface height.
    ///
    /// # Density formula
    ///
    /// `density = vertical_gradient + density_noise`
    ///
    /// - **`vertical_gradient`** – `(surface_height − y) / 8.0`: positive below the
    ///   surface, transitions through zero at the surface.  This provides a
    ///   natural bias toward solid rock deep inside and air above.
    /// - **`density_noise`** – biome-specific noise blend:
    ///   - *Mountains*: combination of a 2-D terrain layer (0.55 weight) and a
    ///     full 3-D detail layer (0.45 weight) for complex cliff faces.
    ///   - *Island*: purely 3-D noise for rounded, bumpy island peaks.
    ///   - *Other biomes*: 0.0 (function should not be called for other biomes).
    fn get_3d_density(
        &self,
        x: i32, y: i32, z: i32,
        biome: Biome,
        surface_height: i32,
    ) -> f64 {
        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        // Linear gradient: +1 at the surface, −∞ deep inside.
        let vertical_gradient = (surface_height as f64 - y as f64) / 8.0;

        let density_noise = match biome {
            Biome::Mountains => {
                // 2-D terrain sampled at higher frequency for cliff-face detail,
                // blended with a full 3-D noise layer.
                let terrain = self.noise_terrain.get_noise_2d(fx * 0.018, fz * 0.018) as f64
                    * 0.55;
                let detail  = self.noise_detail
                    .get_noise_3d(fx * 0.038, fy * 0.038, fz * 0.038) as f64
                    * 0.45;
                terrain + detail
            }
            Biome::Island => {
                self.noise_terrain
                    .get_noise_3d(fx * 0.028, fy * 0.028, fz * 0.028) as f64
                    * 0.45
            }
            _ => 0.0,
        };

        vertical_gradient + density_noise
    }

    // ── Block type assignment ─────────────────────────────────────────────── //

    /// Returns the [`BlockType`] that should be placed at `(world_x, y, world_z)`
    /// given the pre-classified `biome` and `surface_height`.
    ///
    /// # Shared rules (applied before biome-specific logic)
    ///
    /// - Y = 0 is always `Bedrock`.
    /// - Y 1–4: probabilistic bedrock (`chance = (5 − y) × 20%`) to create a
    ///   rough, uneven bedrock floor rather than a flat slab.
    /// - Y < 8: 30% chance of Stone mixed into the bedrock transition zone.
    ///
    /// # Per-biome surface layer
    ///
    /// Each biome has a `dirt_depth` in the range [3, 5] (seeded by position
    /// hash) that controls how deep the topsoil layer extends before giving
    /// way to stone.
    ///
    /// | Biome | Surface | Subsurface | Substrate |
    /// |---|---|---|---|
    /// | Ocean/River/Lake | Sand (top 2) | Gravel (3–5) | Stone |
    /// | Beach/Island | Sand | Sand | Stone |
    /// | Desert | Sand (top 12) | Sand | Stone |
    /// | Tundra | Snow | Dirt | Stone |
    /// | Mountains (high) | Snow / Gravel / Stone | Stone | Stone |
    /// | Mountains (low) | Grass | Dirt | Stone |
    /// | Swamp | Clay (≤ sea level) or Grass | Dirt | Stone |
    /// | Plains/Forest | Grass | Dirt | Stone |
    fn get_block_for_biome(
        &self,
        biome: Biome,
        y: i32,
        surface_height: i32,
        world_x: i32,
        world_z: i32,
    ) -> BlockType {
        // ---- Bedrock floor --------------------------------------------------
        if y == 0 {
            return BlockType::Bedrock;
        }
        // Probabilistic bedrock: 80% at Y=1, 60% at Y=2, 40% at Y=3, 20% at Y=4.
        if y <= 4 {
            let bedrock_chance = (5 - y) as u32 * 20;
            let hash = self.position_hash_3d(world_x, y, world_z);
            if (hash % 100) < bedrock_chance {
                return BlockType::Bedrock;
            }
        }

        // Stone/bedrock transition zone.
        if y < 8 {
            let deep_hash = self.position_hash_3d(world_x, y, world_z);
            if deep_hash % 10 < 3 {
                return BlockType::Stone;
            }
        }

        let depth_from_surface = surface_height - y;
        // Randomize dirt depth per-column in [3, 5] to avoid a perfectly flat
        // grass-to-stone transition that would look artificial.
        let dirt_depth = 3 + (self.position_hash(world_x, world_z) % 3) as i32;

        match biome {
            Biome::Ocean | Biome::River | Biome::Lake => {
                if depth_from_surface > 5 {
                    BlockType::Stone
                } else if depth_from_surface > 2 {
                    BlockType::Gravel
                } else {
                    BlockType::Sand
                }
            }
            Biome::Beach | Biome::Island => {
                if depth_from_surface > 7 {
                    BlockType::Stone
                } else if depth_from_surface > 0 {
                    BlockType::Sand
                } else if y == surface_height - 1 {
                    // Islands above sea level grow grass; beaches stay sandy.
                    if biome == Biome::Island && y > SEA_LEVEL + 2 {
                        BlockType::Grass
                    } else {
                        BlockType::Sand
                    }
                } else {
                    BlockType::Air
                }
            }
            Biome::Desert => {
                if depth_from_surface > 12 {
                    BlockType::Stone
                } else {
                    BlockType::Sand
                }
            }
            Biome::Tundra => {
                if depth_from_surface > dirt_depth + 2 {
                    BlockType::Stone
                } else if depth_from_surface > 1 {
                    BlockType::Dirt
                } else if y == surface_height - 1 {
                    BlockType::Snow
                } else {
                    BlockType::Air
                }
            }
            Biome::Mountains => {
                if y > 150 {
                    // High alpine zone: everything is snow-capped or bare stone.
                    if y == surface_height - 1 {
                        BlockType::Snow
                    } else {
                        BlockType::Stone
                    }
                } else if y > 115 {
                    // Sub-alpine zone: surface occasionally has gravel scree.
                    let hash = self.position_hash_3d(world_x, y, world_z);
                    if depth_from_surface <= 1 {
                        if hash % 4 == 0 {
                            BlockType::Gravel
                        } else {
                            BlockType::Stone
                        }
                    } else {
                        BlockType::Stone
                    }
                } else if depth_from_surface > dirt_depth {
                    BlockType::Stone
                } else if depth_from_surface > 1 {
                    BlockType::Dirt
                } else if y == surface_height - 1 {
                    BlockType::Grass
                } else {
                    BlockType::Air
                }
            }
            Biome::Swamp => {
                if depth_from_surface > dirt_depth {
                    BlockType::Stone
                } else if depth_from_surface > 1 {
                    BlockType::Dirt
                } else if y == surface_height - 1 {
                    // Waterlogged swamp margins use clay; raised swamp uses grass.
                    if y <= SEA_LEVEL + 1 {
                        BlockType::Clay
                    } else {
                        BlockType::Grass
                    }
                } else {
                    BlockType::Air
                }
            }
            Biome::Plains | Biome::Forest => {
                if depth_from_surface > dirt_depth {
                    BlockType::Stone
                } else if depth_from_surface > 1 {
                    BlockType::Dirt
                } else if y == surface_height - 1 {
                    BlockType::Grass
                } else {
                    BlockType::Air
                }
            }
        }
    }

    // ── Surface decorations ───────────────────────────────────────────────── //

    /// Places biome-appropriate surface decorations (trees, cacti, snow, gravel)
    /// into `chunk`.
    ///
    /// A 4-block inward margin is maintained on all four sides of the chunk so
    /// that tree canopy geometry never writes outside the chunk boundary.
    /// Columns at or below sea level are skipped to prevent decorations
    /// appearing underwater.
    ///
    /// # Per-biome decoration rules
    ///
    /// | Condition | Decoration |
    /// |---|---|
    /// | `biome.has_trees()` && tree noise > density | Tree (18% gate via hash). |
    /// | Forest or Swamp, 1-in-7 trees | Large tree variant. |
    /// | Desert, hash < 3% | Cactus on sand. |
    /// | Desert, hash 3–10% | Dead bush on sand. |
    /// | Mountains Y > 110, hash < 8% | Surface Gravel scree. |
    /// | Mountains Y > 145, stone surface | Snow cap block. |
    fn generate_decorations(
        &self,
        chunk: &mut Chunk,
        cx: i32,
        cz: i32,
        biome_map: &[[Biome; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
        height_map: &[[i32; CHUNK_SIZE as usize]; CHUNK_SIZE as usize],
    ) {
        let base_x = cx * CHUNK_SIZE;
        let base_z = cz * CHUNK_SIZE;
        // 4-block margin keeps tree canopy within the chunk.
        let margin = 4;

        for lx in margin..(CHUNK_SIZE - margin) {
            for lz in margin..(CHUNK_SIZE - margin) {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome  = biome_map[lx as usize][lz as usize];
                let height = height_map[lx as usize][lz as usize];
                let hash   = self.position_hash(world_x, world_z);

                // No decorations on underwater surfaces.
                if height <= SEA_LEVEL {
                    continue;
                }

                // ---- Trees --------------------------------------------------
                if biome.has_trees() {
                    let tree_noise = self
                        .noise_trees
                        .get_noise_2d(world_x as f32, world_z as f32);
                    // `tree_density()` returns a threshold in [-1, 1]; higher
                    // thresholds produce sparser forests.
                    let density_threshold = biome.tree_density() as f32;

                    if tree_noise > density_threshold {
                        // Secondary 18% hash gate adds column-level variation
                        // even within high-density noise regions.
                        if hash % 100 < 18 {
                            let ground = chunk.get_block(lx, height - 1, lz);
                            if matches!(ground, BlockType::Grass | BlockType::Dirt) {
                                // Large trees (taller trunk, wider canopy) appear
                                // in Forest and Swamp at a 1-in-7 rate.
                                let is_large = hash % 7 == 0
                                    && matches!(biome, Biome::Forest | Biome::Swamp);
                                if self.can_place_tree(chunk, lx, height, lz, is_large) {
                                    self.place_tree(chunk, lx, height, lz, biome, is_large);
                                }
                            }
                        }
                    }
                }

                // ---- Desert flora -------------------------------------------
                if biome == Biome::Desert {
                    if hash % 100 < 3 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand {
                            self.place_cactus(chunk, lx, height, lz);
                        }
                    } else if hash % 100 < 10 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand && height < WORLD_HEIGHT - 1 {
                            chunk.set_block(lx, height, lz, BlockType::DeadBush);
                        }
                    }
                }

                // ---- Mountain scree -----------------------------------------
                // Occasional gravel patches on high mountain slopes give them
                // a more natural, rocky appearance.
                if biome == Biome::Mountains && height > 110 {
                    if hash % 100 < 8 {
                        let top = chunk.get_block(lx, height - 1, lz);
                        if matches!(top, BlockType::Stone | BlockType::Grass) {
                            chunk.set_block(lx, height - 1, lz, BlockType::Gravel);
                        }
                    }
                }

                // ---- High-altitude snow cap ----------------------------------
                if biome == Biome::Mountains && height > 145 {
                    if chunk.get_block(lx, height - 1, lz) == BlockType::Stone {
                        chunk.set_block(lx, height - 1, lz, BlockType::Snow);
                    }
                }
            }
        }
    }

    // ── Tree placement ────────────────────────────────────────────────────── //

    /// Returns `true` if a tree can be placed with its base at `(lx, y, lz)`.
    ///
    /// A tree is rejected if:
    /// - The ground block is not Grass or Dirt.
    /// - Any of the 3×3 ground neighbors contains a hard block (Stone, Gravel,
    ///   Sand, Water, Ice) — prevents trees on cliff edges or shorelines.
    /// - Another tree's trunk (`Wood`) is within `min_distance` blocks
    ///   (3 for normal trees, 5 for large trees) in the XZ plane and within
    ///   Y ±8 of the base — prevents overlapping canopies.
    fn can_place_tree(
        &self,
        chunk: &Chunk,
        lx: i32, y: i32, lz: i32,
        is_large: bool,
    ) -> bool {
        let ground_block = chunk.get_block(lx, y - 1, lz);
        if !matches!(ground_block, BlockType::Grass | BlockType::Dirt) {
            return false;
        }

        // Reject if any adjacent ground cell is a "hard" surface type.
        for dx in -1..=1 {
            for dz in -1..=1 {
                let nx = lx + dx;
                let nz = lz + dz;
                if nx >= 0 && nx < CHUNK_SIZE && nz >= 0 && nz < CHUNK_SIZE {
                    let neighbor = chunk.get_block(nx, y - 1, nz);
                    if matches!(
                        neighbor,
                        BlockType::Stone
                            | BlockType::Gravel
                            | BlockType::Sand
                            | BlockType::Water
                            | BlockType::Ice
                    ) {
                        return false;
                    }
                }
            }
        }

        // Minimum spacing between tree trunks.
        let min_distance = if is_large { 5 } else { 3 };
        for dx in -min_distance..=min_distance {
            for dz in -min_distance..=min_distance {
                let check_x = lx + dx;
                let check_z = lz + dz;

                if check_x < 0 || check_x >= CHUNK_SIZE || check_z < 0 || check_z >= CHUNK_SIZE {
                    continue;
                }

                // Check Y range that covers a full tree height above the base.
                for dy in -1..=8 {
                    let check_y = y + dy;
                    if check_y < 0 || check_y >= WORLD_HEIGHT {
                        continue;
                    }
                    if chunk.get_block(check_x, check_y, check_z) == BlockType::Wood {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Places a tree with its base at `(lx, y, lz)`.
    ///
    /// # Trunk
    /// The trunk is `trunk_height` blocks tall:
    /// - Small trees: 5 or 6 blocks (seeded by position hash).
    /// - Large trees: always 8 blocks.
    ///
    /// The grass block directly beneath the trunk is replaced with Dirt to
    /// prevent a floating grass block when the trunk is later removed by mining.
    ///
    /// # Canopy
    /// Leaves fill a square slab at each level from `leaf_start` to
    /// `trunk_height` (inclusive), with radius `leaf_radius` (2 for small,
    /// 3 for large).  The top two levels use `radius − 1` to taper the crown.
    ///
    /// Corner blocks are stochastically pruned to round the canopy:
    /// - Standard biomes: corners skipped 50% of the time.
    /// - Swamp biomes: corners skipped 67% of the time for a wispier look.
    ///
    /// A single leaf cap is placed one block above the topmost trunk block.
    fn place_tree(
        &self,
        chunk: &mut Chunk,
        lx: i32, y: i32, lz: i32,
        biome: Biome,
        is_large: bool,
    ) {
        let trunk_height = if is_large {
            8
        } else {
            5 + (self.position_hash(lx, lz) % 2) as i32 // 5 or 6
        };

        // Replace the grass immediately below the trunk with dirt so the
        // surface block doesn't float when the trunk is later removed.
        if chunk.get_block(lx, y - 1, lz) == BlockType::Grass {
            chunk.set_block(lx, y - 1, lz, BlockType::Dirt);
        }

        // Place trunk.
        for dy in 0..trunk_height {
            chunk.set_block(lx, y + dy, lz, BlockType::Wood);
        }

        let leaf_start  = if is_large { 4 } else { 3 };
        let leaf_radius = if is_large { 3 } else { 2 };

        // Place canopy layers.
        for dy in leaf_start..=trunk_height {
            // Taper the top two levels.
            let radius = if dy >= trunk_height - 1 {
                leaf_radius - 1
            } else {
                leaf_radius
            };
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let nx = lx + dx;
                    let nz = lz + dz;
                    if nx >= 0 && nx < CHUNK_SIZE && nz >= 0 && nz < CHUNK_SIZE {
                        let ny = y + dy;
                        if ny < WORLD_HEIGHT {
                            let existing = chunk.get_block(nx, ny, nz);
                            if existing == BlockType::Air || existing == BlockType::Leaves {
                                // Corner pruning: keeps canopy roughly circular.
                                let corner_skip = match biome {
                                    Biome::Swamp => {
                                        // Swamp trees are wispier; skip 2-in-3 corners.
                                        dx.abs() == radius
                                            && dz.abs() == radius
                                            && self.position_hash(nx, nz) % 3 != 0
                                    }
                                    _ => {
                                        // Standard: skip 1-in-2 corners.
                                        dx.abs() == radius
                                            && dz.abs() == radius
                                            && self.position_hash(nx, nz) % 2 == 0
                                    }
                                };
                                if !corner_skip {
                                    chunk.set_block(nx, ny, nz, BlockType::Leaves);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Single leaf cap above the trunk tip.
        let top_y = y + trunk_height;
        if top_y < WORLD_HEIGHT {
            let existing = chunk.get_block(lx, top_y, lz);
            if existing == BlockType::Air || existing == BlockType::Leaves {
                chunk.set_block(lx, top_y, lz, BlockType::Leaves);
            }
        }
    }

    /// Places a cactus of height 2–4 blocks at `(lx, y, lz)`.
    ///
    /// Height is randomized per-column in `[2, 4]` using the position hash.
    /// Each block is placed only if `y + dy < WORLD_HEIGHT` to prevent
    /// out-of-bounds writes near the world ceiling.
    fn place_cactus(&self, chunk: &mut Chunk, lx: i32, y: i32, lz: i32) {
        let height = 2 + (self.position_hash(lx, lz) % 3) as i32; // 2, 3, or 4
        for dy in 0..height {
            if y + dy < WORLD_HEIGHT {
                chunk.set_block(lx, y + dy, lz, BlockType::Cactus);
            }
        }
    }

    // ── Hash functions ────────────────────────────────────────────────────── //

    /// Computes a deterministic pseudo-random `u32` from a 2-D block position.
    ///
    /// Uses the FNV-style multiplicative hash with large prime constants
    /// (`73856093` and `19349663`) chosen to spread bits well across the
    /// XZ plane.  The seed is incorporated so different worlds produce
    /// different decoration patterns at the same coordinates.
    ///
    /// # Usage
    /// Gating decoration placement (trees, cacti, stalagmites) and randomizing
    /// block variants (gravel vs. clay, dirt depth, cactus height).
    fn position_hash(&self, x: i32, z: i32) -> u32 {
        let mut hash = self.seed;
        hash = hash.wrapping_add(x as u32).wrapping_mul(73856093);
        hash = hash.wrapping_add(z as u32).wrapping_mul(19349663);
        hash ^ (hash >> 16)
    }

    /// Computes a deterministic pseudo-random `u32` from a 3-D block position.
    ///
    /// Extends `position_hash` with a Y component using a third prime
    /// (`83492791`) so that the result varies vertically (required for
    /// probabilistic bedrock and per-block cave-decoration decisions).
    fn position_hash_3d(&self, x: i32, y: i32, z: i32) -> u32 {
        let mut hash = self.seed;
        hash = hash.wrapping_add(x as u32).wrapping_mul(73856093);
        hash = hash.wrapping_add(y as u32).wrapping_mul(19349663);
        hash = hash.wrapping_add(z as u32).wrapping_mul(83492791);
        hash ^ (hash >> 16)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Clone
// ─────────────────────────────────────────────────────────────────────────────

impl Clone for ChunkGenerator {
    /// Clones the generator by reconstructing it from the original seed.
    ///
    /// `FastNoiseLite` does not implement `Clone`, so deriving it is not
    /// possible.  Recreating via `new` is equivalent and cheap — each
    /// sampler is initialized with trivial integer state.
    fn clone(&self) -> Self {
        ChunkGenerator::new(self.seed)
    }
}

