use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use once_cell::sync::Lazy;

use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::world::spline::TerrainSpline;

/// Extra columns retained around the chunk. Surface material selection needs
/// a real neighbour on every side to calculate a slope without chunk seams.
const HEIGHT_MAP_BORDER: i32 = 1;
const HEIGHT_MAP_SIZE: usize = CHUNK_SIZE as usize + HEIGHT_MAP_BORDER as usize * 2;
/// Density is evaluated at this coarse resolution and trilinearly interpolated
/// for individual voxels.  Keeping it aligned to the chunk edge makes it
/// deterministic across neighbouring chunks.
const DENSITY_CELL_SIZE: i32 = 4;
const DENSITY_XZ_SAMPLES: usize = (CHUNK_SIZE / DENSITY_CELL_SIZE + 1) as usize;
const DENSITY_Y_SAMPLES: usize = (WORLD_HEIGHT / DENSITY_CELL_SIZE + 1) as usize;
const DENSITY_LATTICE_LEN: usize = DENSITY_XZ_SAMPLES * DENSITY_Y_SAMPLES * DENSITY_XZ_SAMPLES;
const TERRAIN_WARP_Z_OFFSET: f32 = 200.0;
const TERRAIN_WARP_AMPLITUDE: f32 = 72.0;
const RIVER_THRESHOLD_MIN: f32 = 0.875;
const RIVER_THRESHOLD_MAX: f32 = 0.935;
const LAKE_THRESHOLD: f32 = -0.69;

static CONTINENTAL_SPLINE: Lazy<TerrainSpline> = Lazy::new(TerrainSpline::continental);
static PV_OFFSET_ERODED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, -0.18),
        (-0.65, -0.12),
        (-0.20, -0.06),
        (0.0, 0.0),
        (0.25, 0.04),
        (0.65, 0.12),
        (1.0, 0.22),
    ])
});
static PV_OFFSET_RUGGED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, -0.30),
        (-0.65, -0.24),
        (-0.20, -0.14),
        (0.0, 0.04),
        (0.25, 0.16),
        (0.65, 0.32),
        (1.0, 0.45),
    ])
});
static PV_FACTOR_ERODED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, 0.76),
        (-0.65, 0.80),
        (-0.20, 0.86),
        (0.0, 0.90),
        (0.25, 0.94),
        (0.65, 1.02),
        (1.0, 1.10),
    ])
});
static PV_FACTOR_RUGGED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, 0.96),
        (-0.65, 1.02),
        (-0.20, 1.10),
        (0.0, 1.18),
        (0.25, 1.30),
        (0.65, 1.50),
        (1.0, 1.62),
    ])
});
#[derive(Clone, Copy)]
struct TerrainShape {
    /// Vertical translation of the density field in normalized Y space.
    offset: f64,
    /// Terrain response to Y/depth; larger values form steeper relief.
    factor: f64,
    jaggedness: f64,
}

#[derive(Clone, Copy)]
struct Climate {
    temperature: f64,
    humidity: f64,
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
}

#[derive(Clone, Copy)]
struct BiomePoint {
    biome: Biome,
    climate: Climate,
}

#[derive(Clone, Copy)]
struct TerrainNoiseSample {
    continental: f64,
    biome_continental: f32,
    terrain: f64,
    temperature: f32,
    moisture: f32,
    river_value: f32,
    river_threshold: f32,
    lake: f32,
    island: f32,
    erosion: f64,
    biome_erosion: f32,
    ridged: f64,
    biome_peaks_valleys: f32,
    weirdness: f64,
    plateau: f64,
    valley: f64,
}

#[derive(Clone, Copy)]
struct ColumnSample {
    biome: Biome,
    height: f64,
}

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
/// | `noise_river` | Major river centerline carving | 0.004 | Simplex |
/// | `noise_lake` | Lake basin placement | 0.022 | Simplex |
/// | `noise_trees` | Tree/vegetation density | 0.12 | Simplex |
/// | `noise_island` | Ocean island elevation | 0.045 | Simplex |
/// | `noise_cave1/2/3` | 3-D cave volumes | 0.025/0.019/0.013 | 3-D Simplex |
/// | `noise_erosion` | Erosion multiplier for slopes | 0.004 | FBm |
/// | `noise_warp_x/z` | Domain warp for terrain/biomes | 0.005 | FBm |
/// | `noise_ridged` | Ridged mountain peaks | 0.009 | Ridged FBm |
/// | `noise_weirdness` | Rare regional variation | 0.0035 | FBm |
/// | `noise_plateau` | Plateaus and terraces | 0.004 | FBm |
/// | `noise_valley` | Broad valley carving | 0.003 | FBm |
/// | `noise_river_width` | River width modulation | 0.0015 | FBm |
/// | `noise_vegetation` | Tree clusters and shrubs | 0.045 | Simplex |
/// | `noise_decor` | Clearings and small structure placement | 0.13 | Simplex |
/// | `noise_cave_warp_x/z` | Domain warp inside caves | 0.018 | FBm |
pub struct ChunkGenerator {
    noise_continents: FastNoiseLite,
    noise_terrain: FastNoiseLite,
    noise_detail: FastNoiseLite,
    noise_temperature: FastNoiseLite,
    noise_moisture: FastNoiseLite,
    noise_river: FastNoiseLite,
    noise_lake: FastNoiseLite,
    noise_trees: FastNoiseLite,
    noise_island: FastNoiseLite,
    noise_cave1: FastNoiseLite,
    noise_cave2: FastNoiseLite,
    noise_cave3: FastNoiseLite,
    noise_erosion: FastNoiseLite,
    noise_warp_x: FastNoiseLite,
    noise_warp_z: FastNoiseLite,
    noise_ridged: FastNoiseLite,
    noise_weirdness: FastNoiseLite,
    noise_plateau: FastNoiseLite,
    noise_valley: FastNoiseLite,
    noise_river_width: FastNoiseLite,
    noise_vegetation: FastNoiseLite,
    #[allow(dead_code)]
    noise_decor: FastNoiseLite,
    noise_ore: FastNoiseLite,
    noise_cave_warp_x: FastNoiseLite,
    noise_cave_warp_z: FastNoiseLite,
    noise_aquifer: FastNoiseLite,
    pub seed: u32,
}

impl ChunkGenerator {
    pub fn new(seed: u32) -> Self {
        ChunkGenerator {
            noise_continents: Self::create_fbm_noise(seed, 0.0018),
            // Tuned for larger, calmer landmasses (more Minecraft-like):
            // - lower mid/high frequencies => flatter plains and smoother biome edges
            // - temperature/moisture frequencies kept similar => less striping
            noise_terrain: Self::create_fbm_noise(seed.wrapping_add(1), 0.007),
            noise_detail: Self::create_fbm_noise(seed.wrapping_add(2), 0.015),
            noise_temperature: Self::create_noise(seed.wrapping_add(3), 0.006),
            noise_moisture: Self::create_noise(seed.wrapping_add(4), 0.008),
            noise_river: Self::create_noise(seed.wrapping_add(5), 0.004),
            noise_lake: Self::create_noise(seed.wrapping_add(6), 0.022),
            noise_trees: Self::create_noise(seed.wrapping_add(7), 0.12),
            noise_island: Self::create_noise(seed.wrapping_add(8), 0.045),
            // Cave samplers intentionally use unit frequency: their explicit
            // X/Y/Z scales below are the one coordinate convention for caves.
            noise_cave1: Self::create_unit_3d_noise(seed.wrapping_add(9)),
            noise_cave2: Self::create_unit_3d_noise(seed.wrapping_add(10)),
            noise_cave3: Self::create_unit_3d_noise(seed.wrapping_add(11)),
            noise_erosion: Self::create_fbm_noise(seed.wrapping_add(12), 0.006),
            noise_warp_x: Self::create_fbm_noise(seed.wrapping_add(20), 0.007),
            noise_warp_z: Self::create_fbm_noise(seed.wrapping_add(21), 0.005),
            noise_ridged: Self::create_ridged_noise(seed.wrapping_add(22), 0.006),
            noise_weirdness: Self::create_fbm_noise(seed.wrapping_add(26), 0.0035),
            noise_plateau: Self::create_fbm_noise(seed.wrapping_add(27), 0.004),
            noise_valley: Self::create_fbm_noise(seed.wrapping_add(28), 0.003),
            noise_river_width: Self::create_fbm_noise(seed.wrapping_add(29), 0.0015),
            noise_vegetation: Self::create_noise(seed.wrapping_add(32), 0.045),
            noise_decor: Self::create_noise(seed.wrapping_add(24), 0.13),
            noise_ore: Self::create_3d_noise(seed.wrapping_add(25), 0.065),
            noise_cave_warp_x: Self::create_unit_fbm_noise(seed.wrapping_add(30)),
            noise_cave_warp_z: Self::create_unit_fbm_noise(seed.wrapping_add(31)),
            noise_aquifer: Self::create_3d_noise(seed.wrapping_add(41), 0.008),
            seed,
        }
    }

    // ── Noise factory helpers ─────────────────────────────────────────────── //

    fn create_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(frequency));
        noise
    }

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

    fn create_3d_noise(seed: u32, frequency: f32) -> FastNoiseLite {
        let mut noise = FastNoiseLite::with_seed(seed as i32);
        noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        noise.set_frequency(Some(frequency));
        noise
    }

    fn create_unit_3d_noise(seed: u32) -> FastNoiseLite {
        Self::create_3d_noise(seed, 1.0)
    }

    fn create_unit_fbm_noise(seed: u32) -> FastNoiseLite {
        Self::create_fbm_noise(seed, 1.0)
    }

    // ── Public chunk generation ───────────────────────────────────────────── //

    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cz);
        let base_x = cx * CHUNK_SIZE;
        let base_z = cz * CHUNK_SIZE;

        // ── Pass 1: independent climate and terrain parameters ──────────── //
        // Terrain parameters are already smooth noise fields.  Deliberately do
        // not blur final heights: that erased cliffs and made every transition
        // look like a generic hill.
        let mut biome_map = [[Biome::Plains; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        let mut height_map = [[0i32; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        let mut bordered_heights = [[0i32; HEIGHT_MAP_SIZE]; HEIGHT_MAP_SIZE];
        for sample_x in -HEIGHT_MAP_BORDER..=(CHUNK_SIZE - 1 + HEIGHT_MAP_BORDER) {
            for sample_z in -HEIGHT_MAP_BORDER..=(CHUNK_SIZE - 1 + HEIGHT_MAP_BORDER) {
                let sample = self.sample_column(base_x + sample_x, base_z + sample_z);
                bordered_heights[(sample_x + HEIGHT_MAP_BORDER) as usize]
                    [(sample_z + HEIGHT_MAP_BORDER) as usize] =
                    (sample.height as i32).clamp(1, WORLD_HEIGHT - 20);
                if sample_x >= 0 && sample_x < CHUNK_SIZE && sample_z >= 0 && sample_z < CHUNK_SIZE
                {
                    biome_map[sample_x as usize][sample_z as usize] = sample.biome;
                    height_map[sample_x as usize][sample_z as usize] = sample.height as i32;
                }
            }
        }

        let mut slope_map = [[0i32; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let center_height = height_map[lx as usize][lz as usize];
                let mut max_delta = 0;

                for dx in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dz == 0 {
                            continue;
                        }

                        let nx = (lx + HEIGHT_MAP_BORDER + dx) as usize;
                        let nz = (lz + HEIGHT_MAP_BORDER + dz) as usize;
                        max_delta = max_delta.max((center_height - bordered_heights[nx][nz]).abs());
                    }
                }

                slope_map[lx as usize][lz as usize] = max_delta;
            }
        }

        // ── Pass 2: coarse 3-D terrain-density lattice ──────────────────── //
        let density_lattice = self.build_density_lattice(base_x, base_z);
        // Surface rules and vegetation need the visible density surface, not
        // merely the macro-height estimate used to construct the field.
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let macro_height = height_map[lx as usize][lz as usize];
                height_map[lx as usize][lz as usize] =
                    self.density_surface_height(&density_lattice, lx, lz, macro_height);
            }
        }
        // Surface rules must see the final density surface. At chunk edges the
        // preliminary border is a conservative fallback; interior gradients
        // are exact and prevent grass from covering density cliffs.
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let center = height_map[lx as usize][lz as usize];
                let mut max_delta = 0;
                for dx in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dz == 0 {
                            continue;
                        }
                        let nx = lx + dx;
                        let nz = lz + dz;
                        let neighbour =
                            if (0..CHUNK_SIZE).contains(&nx) && (0..CHUNK_SIZE).contains(&nz) {
                                height_map[nx as usize][nz as usize]
                            } else {
                                bordered_heights[(nx + HEIGHT_MAP_BORDER) as usize]
                                    [(nz + HEIGHT_MAP_BORDER) as usize]
                            };
                        max_delta = max_delta.max((center - neighbour).abs());
                    }
                }
                slope_map[lx as usize][lz as usize] = max_delta;
            }
        }

        // ── Pass 3: final density, caves and aquifers ───────────────────── //
        let mut cave_entrance_map = [[false; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                cave_entrance_map[lx as usize][lz as usize] =
                    self.is_cave_entrance(world_x, world_z, height_map[lx as usize][lz as usize]);
            }
        }
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome = biome_map[lx as usize][lz as usize];
                let surface_height = height_map[lx as usize][lz as usize];
                let slope = slope_map[lx as usize][lz as usize];

                let is_entrance = cave_entrance_map[lx as usize][lz as usize];
                for y in 0..WORLD_HEIGHT {
                    let terrain_density = self.sample_density_lattice(&density_lattice, lx, y, lz);
                    let cave_density =
                        self.cave_density(world_x, y, world_z, surface_height, is_entrance);
                    let is_solid = terrain_density.min(cave_density) > 0.0;

                    if is_solid {
                        let block = self.get_block_for_biome(
                            biome,
                            y,
                            surface_height,
                            world_x,
                            world_z,
                            slope,
                        );
                        let block = self.apply_stone_variants(world_x, y, world_z, block);
                        if block != BlockType::Air {
                            chunk.set_block_raw(lx, y, lz, block);
                        }
                    } else if self.has_fluid(world_x, y, world_z, surface_height) {
                        if biome == Biome::Tundra && y == SEA_LEVEL - 1 {
                            chunk.set_block_raw(lx, y, lz, BlockType::Ice);
                        } else {
                            chunk.set_block_raw(lx, y, lz, BlockType::Water);
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
                let height = height_map[lx as usize][lz as usize];
                let hash_xz = self.position_hash(world_x, world_z);

                for y in 5..height.min(WORLD_HEIGHT - 2) {
                    let current = chunk.get_block(lx, y, lz);
                    if current != BlockType::Air {
                        continue;
                    }

                    let below = chunk.get_block(lx, y - 1, lz);
                    let above = chunk.get_block(lx, y + 1, lz);

                    if below != BlockType::Air
                        && below != BlockType::Water
                        && above == BlockType::Air
                    {
                        let hash3 = self.position_hash_3d(world_x, y, world_z);

                        if matches!(
                            below,
                            BlockType::Stone | BlockType::Dirt | BlockType::Gravel
                        ) {
                            if hash3 % 100 < 15 {
                                chunk.set_block_raw(lx, y - 1, lz, BlockType::Gravel);
                            } else if y >= 35 && y <= 55 && hash3 % 100 < 8 {
                                chunk.set_block_raw(lx, y - 1, lz, BlockType::Clay);
                            }
                        }

                        if below == BlockType::Stone && hash_xz % 100 < 8 && y >= 8 {
                            let stalagmite_h = 1 + (hash3 % 3) as i32;
                            for dy in 0..stalagmite_h {
                                let ny = y + dy;
                                if ny < WORLD_HEIGHT
                                    && chunk.get_block(lx, ny, lz) == BlockType::Air
                                {
                                    chunk.set_block_raw(lx, ny, lz, BlockType::Stone);
                                } else {
                                    break;
                                }
                            }
                        }
                    }

                    if above != BlockType::Air
                        && above != BlockType::Water
                        && below == BlockType::Air
                    {
                        let hash3 = self.position_hash_3d(world_x, y, world_z);

                        if above == BlockType::Stone && hash_xz.wrapping_add(7) % 100 < 6 {
                            let stalactite_h = 1 + (hash3 % 2) as i32;
                            for dy in 0..stalactite_h {
                                let ny = y - dy;
                                if ny > 4 && chunk.get_block(lx, ny, lz) == BlockType::Air {
                                    chunk.set_block_raw(lx, ny, lz, BlockType::Stone);
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Pass 5: surface decorations ───────────────────────────────────── //
        self.generate_decorations(&mut chunk, cx, cz, &biome_map, &height_map);

        // ── Pass 6: chunk/sub-chunk metadata ──────────────────────────────── //
        chunk.rebuild_metadata();

        chunk
    }

    // ── Public forwarding accessors ───────────────────────────────────────── //

    /// Returns the raw (unblended) terrain height at `(x, z)`.
    ///
    /// Used by the `ChunkLoader` for spawn-point search and LOD decisions.
    /// Returns the single-sample height rather than the blended value because
    /// blending requires a full noise buffer and is only meaningful at chunk
    /// granularity (done inside `generate_chunk`).
    pub fn get_terrain_height_pub(&self, x: i32, z: i32) -> i32 {
        (self.sample_column(x, z).height as i32).clamp(1, WORLD_HEIGHT - 20)
    }

    pub fn is_cave_entrance_pub(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.is_cave_entrance(x, z, surface_height)
    }

    pub fn position_hash_pub(&self, x: i32, z: i32) -> u32 {
        self.position_hash(x, z)
    }

    // ── Biome classification ──────────────────────────────────────────────── //

    fn sample_column(&self, x: i32, z: i32) -> ColumnSample {
        let noise = self.sample_terrain_noise(x, z);
        let biome = self.classify_biome(&noise);
        let shape = self.terrain_shape(&noise);
        let height = self.preliminary_surface_level(&shape, &noise);
        ColumnSample { biome, height }
    }

    fn terrain_warp(&self, x: i32, z: i32) -> (f32, f32) {
        let fx = x as f32;
        let fz = z as f32;
        let wx = fx + self.noise_warp_x.get_noise_2d(fx, fz) * TERRAIN_WARP_AMPLITUDE;
        let wz = fz
            + self
                .noise_warp_z
                .get_noise_2d(fx + TERRAIN_WARP_Z_OFFSET, fz + TERRAIN_WARP_Z_OFFSET)
                * TERRAIN_WARP_AMPLITUDE;

        (wx, wz)
    }

    fn sample_terrain_noise(&self, x: i32, z: i32) -> TerrainNoiseSample {
        let (wx, wz) = self.terrain_warp(x, z);
        let biome_continental = self.noise_continents.get_noise_2d(wx, wz);
        let river_noise = self.noise_river.get_noise_2d(wx, wz);
        let river_width = self.noise_river_width.get_noise_2d(wx + 250.0, wz - 250.0);
        let river_threshold = lerp(
            RIVER_THRESHOLD_MIN as f64,
            RIVER_THRESHOLD_MAX as f64,
            ((river_width as f64 + 1.0) * 0.5).clamp(0.0, 1.0),
        ) as f32;

        let weirdness = self.noise_weirdness.get_noise_2d(wx, wz) as f64;
        TerrainNoiseSample {
            continental: self.noise_continents.get_noise_2d(wx, wz) as f64,
            biome_continental,
            terrain: self.noise_terrain.get_noise_2d(wx, wz) as f64,
            temperature: self.noise_temperature.get_noise_2d(wx, wz),
            moisture: self.noise_moisture.get_noise_2d(wx, wz),
            river_value: 1.0 - river_noise.abs() * 2.0,
            river_threshold,
            lake: self.noise_lake.get_noise_2d(wx, wz),
            island: self.noise_island.get_noise_2d(wx, wz),
            erosion: self.noise_erosion.get_noise_2d(wx, wz) as f64,
            biome_erosion: self.noise_erosion.get_noise_2d(wx, wz),
            ridged: self.noise_ridged.get_noise_2d(wx, wz) as f64,
            biome_peaks_valleys: peaks_and_valleys(weirdness) as f32,
            weirdness,
            plateau: self.noise_plateau.get_noise_2d(wx, wz) as f64,
            valley: self.noise_valley.get_noise_2d(wx, wz) as f64,
        }
    }

    fn classify_biome(&self, noise: &TerrainNoiseSample) -> Biome {
        if noise.river_value > noise.river_threshold && noise.biome_continental > -0.28 {
            return Biome::River;
        }

        if noise.lake < LAKE_THRESHOLD
            && noise.biome_continental > -0.12
            && noise.biome_erosion > -0.35
        {
            return Biome::Lake;
        }

        if noise.biome_continental < -0.42 {
            if noise.island > 0.58 && noise.weirdness > -0.45 {
                return Biome::Island;
            }
            return Biome::Ocean;
        }

        if noise.biome_continental < -0.18
            || (noise.biome_continental < -0.08 && noise.weirdness < -0.55)
        {
            return Biome::Beach;
        }

        self.select_land_biome(Climate {
            temperature: noise.temperature as f64,
            humidity: noise.moisture as f64,
            continentalness: noise.biome_continental as f64,
            erosion: noise.biome_erosion as f64,
            weirdness: noise.weirdness,
        })
    }

    /// Nearest-point selection makes climate extension data-driven: adding a
    /// biome no longer means growing a fragile chain of terrain conditionals.
    fn select_land_biome(&self, climate: Climate) -> Biome {
        const POINTS: [BiomePoint; 7] = [
            BiomePoint {
                biome: Biome::Plains,
                climate: Climate {
                    temperature: 0.10,
                    humidity: -0.10,
                    continentalness: 0.18,
                    erosion: 0.45,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Forest,
                climate: Climate {
                    temperature: 0.18,
                    humidity: 0.62,
                    continentalness: 0.22,
                    erosion: 0.10,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Desert,
                climate: Climate {
                    temperature: 0.78,
                    humidity: -0.72,
                    continentalness: 0.20,
                    erosion: 0.38,
                    weirdness: 0.05,
                },
            },
            BiomePoint {
                biome: Biome::Tundra,
                climate: Climate {
                    temperature: -0.72,
                    humidity: 0.05,
                    continentalness: 0.22,
                    erosion: 0.30,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Swamp,
                climate: Climate {
                    temperature: 0.52,
                    humidity: 0.82,
                    continentalness: -0.02,
                    erosion: 0.65,
                    weirdness: -0.12,
                },
            },
            BiomePoint {
                biome: Biome::Forest,
                climate: Climate {
                    temperature: -0.08,
                    humidity: 0.40,
                    continentalness: 0.55,
                    erosion: -0.30,
                    weirdness: 0.34,
                },
            },
            BiomePoint {
                biome: Biome::Plains,
                climate: Climate {
                    temperature: 0.35,
                    humidity: 0.12,
                    continentalness: 0.62,
                    erosion: -0.20,
                    weirdness: 0.38,
                },
            },
        ];
        let mut best = (Biome::Plains, f64::INFINITY);
        for point in POINTS {
            let p = point.climate;
            let distance = (climate.temperature - p.temperature).powi(2) * 1.5
                + (climate.humidity - p.humidity).powi(2) * 1.5
                + (climate.continentalness - p.continentalness).powi(2) * 0.35
                + (climate.erosion - p.erosion).powi(2) * 0.25
                + (climate.weirdness - p.weirdness).powi(2) * 0.15;
            if distance < best.1 {
                best = (point.biome, distance);
            }
        }
        best.0
    }

    /// Classifies the biome at world position `(x, z)`.
    ///
    /// Domain warp offsets match exactly those used by the height sampler
    /// (scale 0.005, Z offset +200) so biome boundaries and height boundaries
    /// are always coherent — no more mismatched warp between the two systems.
    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        let noise = self.sample_terrain_noise(x, z);
        self.classify_biome(&noise)
    }

    // ── Terrain height ────────────────────────────────────────────────────── //

    /// Core height function for a single pre-sampled `(x, z)` column.
    ///
    /// Domain warp uses the **same scale (0.005) and Z-offset (+200)** as
    /// `get_biome`, guaranteeing that the biome boundary and the height
    /// boundary stay in sync regardless of world position.
    fn terrain_shape(&self, noise: &TerrainNoiseSample) -> TerrainShape {
        let continentalness = noise.continental;
        let erosion = noise.erosion;
        let pv = peaks_and_valleys(noise.weirdness);
        let inland = ((continentalness + 0.12) / 0.72).clamp(0.0, 1.0);
        let low_erosion = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        let mountain_strength = inland * low_erosion * pv.max(0.0);
        let mut offset = self.shape_offset(continentalness, erosion, pv);
        let mut factor = self.shape_factor(continentalness, erosion, pv);
        let mut jaggedness = self.shape_jaggedness(continentalness, erosion, pv, noise.ridged);

        // Every macro landform participates in the final density shape.
        let river = self.river_channel_strength(noise) * (1.0 - mountain_strength * 0.78);
        let lake = self.lake_strength(noise);
        // PV supplies the primary valleys; the independent valley field is
        // only a subtle local accent rather than a second terrain system.
        let valley = self.valley_strength(noise) * 0.25;
        let plateau = self.plateau_strength(noise);
        let ocean_factor = ((-continentalness - 0.42) / 0.58).clamp(0.0, 1.0);
        let island = self.island_strength(noise) * ocean_factor;
        offset -= river * 0.12 + lake * 0.17 + valley * 0.15;
        offset += island * 0.60 + plateau * 0.05;
        factor *= 1.0 - river * 0.18 - lake * 0.10 + plateau * 0.08;
        jaggedness *= 1.0 - river * 0.70 - lake * 0.65;
        offset += noise.terrain * (0.025 + mountain_strength * 0.045);
        TerrainShape {
            offset,
            factor,
            jaggedness,
        }
    }

    fn shape_offset(&self, continentalness: f64, erosion: f64, pv: f64) -> f64 {
        let continental = (CONTINENTAL_SPLINE.sample(continentalness) - SEA_LEVEL as f64) / 64.0;
        let inland = ((continentalness + 0.12) / 0.72).clamp(0.0, 1.0);
        let roughness = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        continental + inland * self.pv_offset(pv, roughness)
    }

    fn shape_factor(&self, continentalness: f64, erosion: f64, pv: f64) -> f64 {
        let inland = ((continentalness + 0.12) / 0.72).clamp(0.0, 1.0);
        let roughness = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        lerp(0.72, self.pv_factor(pv, roughness), inland)
    }

    fn shape_jaggedness(&self, continentalness: f64, erosion: f64, pv: f64, ridged: f64) -> f64 {
        let inland = ((continentalness + 0.12) / 0.72).clamp(0.0, 1.0);
        let roughness = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        inland * roughness * pv.max(0.0) * (0.10 + (ridged + 1.0) * 0.12)
    }

    /// Piecewise PV spline: negative bands are valleys and lowlands, while
    /// positive bands progress through hills into sharp peaks.
    fn pv_offset(&self, pv: f64, roughness: f64) -> f64 {
        lerp(
            PV_OFFSET_ERODED.sample(pv),
            PV_OFFSET_RUGGED.sample(pv),
            roughness,
        )
    }

    fn pv_factor(&self, pv: f64, roughness: f64) -> f64 {
        lerp(
            PV_FACTOR_ERODED.sample(pv),
            PV_FACTOR_RUGGED.sample(pv),
            roughness,
        )
    }

    /// Terrain geometry is intentionally independent of the selected land
    /// biome.  A forest, desert or tundra may therefore occur on the same
    /// plains, valley, plateau or mountain terrain family.
    fn preliminary_surface_level(&self, shape: &TerrainShape, _noise: &TerrainNoiseSample) -> f64 {
        // A prediction of the zero crossing of the same terrain shape, not a
        // second geometry system with its own river/lake/island rules.
        (SEA_LEVEL as f64 + shape.offset * 64.0).clamp(2.0, (WORLD_HEIGHT - 2) as f64)
    }

    fn river_channel_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        let threshold = noise.river_threshold as f64;
        ((noise.river_value as f64 - threshold) / (1.0 - threshold).max(0.001)).clamp(0.0, 1.0)
    }

    fn lake_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        if noise.biome_continental <= -0.12 || noise.biome_erosion <= -0.35 {
            return 0.0;
        }
        smoothstep(((LAKE_THRESHOLD as f64 - noise.lake as f64) / 0.22).clamp(0.0, 1.0))
    }

    fn island_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        smoothstep(((noise.island as f64 + 1.0) * 0.5 - 0.76) / 0.24)
    }

    fn valley_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        let raw_valley = ((noise.valley + 1.0) * 0.5).clamp(0.0, 1.0);
        let land_factor = ((noise.biome_continental as f64 + 0.08) / 0.48).clamp(0.0, 1.0);
        let mountain_factor =
            smoothstep((noise.biome_peaks_valleys as f64 - 0.24) / 0.42).clamp(0.0, 1.0);

        smoothstep((raw_valley - 0.54) / 0.30) * land_factor * (1.0 - mountain_factor * 0.45)
    }

    fn plateau_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        let raw_plateau = ((noise.plateau + 1.0) * 0.5).clamp(0.0, 1.0);
        let land_factor = ((noise.biome_continental as f64 + 0.02) / 0.42).clamp(0.0, 1.0);
        let eroded_flatness = ((noise.biome_erosion as f64 + 1.0) * 0.5).clamp(0.0, 1.0);

        smoothstep((raw_plateau - 0.58) / 0.32)
            * smoothstep((eroded_flatness - 0.35) / 0.45)
            * land_factor
    }

    // ── Cave system ───────────────────────────────────────────────────────── //

    fn is_cave(&self, x: i32, y: i32, z: i32, surface_height: i32, is_entrance: bool) -> bool {
        if y <= 4 {
            return false;
        }

        let min_surface_dist = if is_entrance {
            let t = ((surface_height - y) as f32 / 18.0).clamp(0.0, 1.0);
            // Entrances use the same cave fields as underground cavities. No
            // post-process shaft is carved; a connected cave can simply reach
            // a hillside naturally, while ordinary caves remain masked.
            (2.0 + t * 10.0) as i32
        } else {
            18
        };
        if y >= surface_height - min_surface_dist {
            return false;
        }

        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        let warp_amp = 12.0_f32;
        let wx = fx
            + self
                .noise_cave_warp_x
                .get_noise_3d(fx * 0.018, fy * 0.010, fz * 0.018)
                * warp_amp;
        let wy = fy
            + self.noise_cave_warp_z.get_noise_3d(
                fx * 0.018 + 100.0,
                fy * 0.010,
                fz * 0.018 + 100.0,
            ) * warp_amp
                * 0.15;
        let wz = fz
            + self.noise_cave_warp_x.get_noise_3d(
                fx * 0.018 + 200.0,
                fy * 0.010,
                fz * 0.018 + 200.0,
            ) * warp_amp;

        let in_lower = y < 54;
        let in_middle = y >= 54 && y < 90;

        if in_lower {
            let c1 = self
                .noise_cave1
                .get_noise_3d(wx * 0.030, wy * 0.010, wz * 0.030);
            let c2 = self.noise_cave2.get_noise_3d(
                wx * 0.022 + 400.0,
                wy * 0.008 + 400.0,
                wz * 0.022 + 400.0,
            );
            let cheese_product = c1.max(0.0) * c2.max(0.0);
            if cheese_product > 0.30 {
                return true;
            }
        }

        let s1 = self
            .noise_cave1
            .get_noise_3d(wx * 0.060 + 500.0, wy * 0.025, wz * 0.060);
        let s2 = self
            .noise_cave3
            .get_noise_3d(wx * 0.060 + 900.0, wy * 0.025, wz * 0.060);
        let spag_dist = (s1 * s1 + s2 * s2).sqrt();
        let spag_radius = if in_lower {
            0.095
        } else if in_middle {
            0.075
        } else {
            0.05
        };
        if spag_dist < spag_radius {
            return true;
        }

        if y > 20 {
            let n1 = self
                .noise_cave2
                .get_noise_3d(wx * 0.090 + 800.0, wy * 0.040, wz * 0.090);
            let n2 = self
                .noise_cave3
                .get_noise_3d(wx * 0.090 + 1200.0, wy * 0.040, wz * 0.090);
            let noodle_dist = (n1 * n1 + n2 * n2).sqrt();
            let noodle_radius = if in_lower { 0.055 } else { 0.038 };
            if noodle_dist < noodle_radius {
                return true;
            }
        }

        if y < 30 {
            let w1 = self
                .noise_cave2
                .get_noise_3d(wx * 0.042 + 800.0, wy * 0.015, wz * 0.042);
            let w2 = self
                .noise_cave3
                .get_noise_3d(wx * 0.042 + 1200.0, wy * 0.015, wz * 0.042);
            let worm_dist = (w1 * w1 + w2 * w2).sqrt();
            if worm_dist < 0.085 {
                return true;
            }
        }

        false
    }

    /// Cave families are composed with terrain before blocks are emitted.
    /// Positive values preserve material; negative values represent cavity.
    fn cave_density(
        &self,
        x: i32,
        y: i32,
        z: i32,
        preliminary_surface: i32,
        is_entrance: bool,
    ) -> f64 {
        if self.is_cave(x, y, z, preliminary_surface, is_entrance) {
            -1.0
        } else {
            1.0
        }
    }

    fn has_fluid(&self, x: i32, y: i32, z: i32, preliminary_surface: i32) -> bool {
        // Open ocean and lakes retain a shared sea level. Underground cavities
        // use a local aquifer level, so adjacent caves can be dry or flooded.
        if preliminary_surface < SEA_LEVEL && y < SEA_LEVEL {
            return true;
        }
        let n = self
            .noise_aquifer
            .get_noise_3d(x as f32, y as f32, z as f32) as f64;
        let cell = self
            .noise_aquifer
            .get_noise_3d(x as f32 + 4000.0, 0.0, z as f32 - 4000.0) as f64;
        let fluid_level = 24 + ((cell + 1.0) * 0.5 * 34.0) as i32;
        y <= fluid_level && n > 0.10
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 6 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        let entrance_noise = self
            .noise_cave1
            .get_noise_2d(fx * 0.014 + 1000.0, fz * 0.014 + 1000.0);

        let terrain_slope = self
            .noise_terrain
            .get_noise_2d(fx * 0.018, fz * 0.018)
            .abs();
        let is_hillside = terrain_slope > 0.18;

        let threshold = if is_hillside { 0.78 } else { 0.90 };
        if entrance_noise < threshold {
            return false;
        }

        let hash = self.position_hash(x, z);
        let entrance_chance = if is_hillside { 12 } else { 24 };
        if hash % entrance_chance != 0 {
            return false;
        }

        for check_y in (surface_height - 40).max(8)..=(surface_height - 6) {
            let fy = check_y as f32;
            let c1 = self
                .noise_cave1
                .get_noise_3d(fx * 0.045, fy * 0.022, fz * 0.045);
            let c2 = self
                .noise_cave2
                .get_noise_3d(fx * 0.032, fy * 0.018, fz * 0.032);
            if c1 > 0.70 && c2 > 0.70 {
                return true;
            }
        }

        false
    }

    // ── 3-D density (overhangs) ───────────────────────────────────────────── //

    fn density_index(ix: usize, iy: usize, iz: usize) -> usize {
        (ix * DENSITY_Y_SAMPLES + iy) * DENSITY_XZ_SAMPLES + iz
    }

    fn build_density_lattice(&self, base_x: i32, base_z: i32) -> [f64; DENSITY_LATTICE_LEN] {
        let mut lattice = [0.0; DENSITY_LATTICE_LEN];
        for ix in 0..DENSITY_XZ_SAMPLES {
            for iz in 0..DENSITY_XZ_SAMPLES {
                let x = base_x + ix as i32 * DENSITY_CELL_SIZE;
                let z = base_z + iz as i32 * DENSITY_CELL_SIZE;
                let noise = self.sample_terrain_noise(x, z);
                let shape = self.terrain_shape(&noise);
                for iy in 0..DENSITY_Y_SAMPLES {
                    let y = iy as i32 * DENSITY_CELL_SIZE;
                    lattice[Self::density_index(ix, iy, iz)] =
                        self.terrain_density(x, y, z, &shape);
                }
            }
        }
        lattice
    }

    fn terrain_density(&self, x: i32, y: i32, z: i32, shape: &TerrainShape) -> f64 {
        let depth = self.y_gradient(y) + shape.offset;
        let jagged = shape.jaggedness
            * half_negative(self.noise_detail.get_noise_3d(x as f32, y as f32, z as f32) as f64);
        let shaped = (depth + jagged) * shape.factor;
        let base_3d =
            self.noise_terrain
                .get_noise_3d(x as f32 + 1900.0, y as f32 + 700.0, z as f32 - 1900.0)
                as f64;
        4.0 * quarter_negative(shaped) + base_3d * 0.34
    }

    fn y_gradient(&self, y: i32) -> f64 {
        (SEA_LEVEL as f64 - y as f64) / 64.0
    }

    fn sample_density_lattice(
        &self,
        lattice: &[f64; DENSITY_LATTICE_LEN],
        lx: i32,
        y: i32,
        lz: i32,
    ) -> f64 {
        let ix = (lx / DENSITY_CELL_SIZE) as usize;
        let iy = (y / DENSITY_CELL_SIZE) as usize;
        let iz = (lz / DENSITY_CELL_SIZE) as usize;
        let tx = (lx.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let ty = (y.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let tz = (lz.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let x1 = (ix + 1).min(DENSITY_XZ_SAMPLES - 1);
        let y1 = (iy + 1).min(DENSITY_Y_SAMPLES - 1);
        let z1 = (iz + 1).min(DENSITY_XZ_SAMPLES - 1);
        let at = |x, yy, z| lattice[Self::density_index(x, yy, z)];
        let a = lerp(
            lerp(at(ix, iy, iz), at(x1, iy, iz), tx),
            lerp(at(ix, iy, z1), at(x1, iy, z1), tx),
            tz,
        );
        let b = lerp(
            lerp(at(ix, y1, iz), at(x1, y1, iz), tx),
            lerp(at(ix, y1, z1), at(x1, y1, z1), tx),
            tz,
        );
        lerp(a, b, ty)
    }

    fn density_surface_height(
        &self,
        lattice: &[f64; DENSITY_LATTICE_LEN],
        lx: i32,
        lz: i32,
        macro_height: i32,
    ) -> i32 {
        // The preliminary surface is merely a fallback. The final top comes
        // from density and may be anywhere in the world vertical range.
        for y in (1..WORLD_HEIGHT).rev() {
            if self.sample_density_lattice(lattice, lx, y, lz) > 0.0 {
                return y + 1;
            }
        }
        macro_height
    }

    // ── Block type assignment ─────────────────────────────────────────────── //

    fn snow_line(&self, world_x: i32, world_z: i32, biome: Biome) -> i32 {
        let local_variation = self
            .noise_detail
            .get_noise_2d(world_x as f32 + 7000.0, world_z as f32 - 7000.0);
        let offset = (local_variation * 9.0) as i32;

        match biome {
            Biome::Tundra => SEA_LEVEL + 12 + offset,
            Biome::Mountains => 132 + offset,
            _ => 154 + offset,
        }
    }

    fn get_block_for_biome(
        &self,
        biome: Biome,
        y: i32,
        surface_height: i32,
        world_x: i32,
        world_z: i32,
        slope: i32,
    ) -> BlockType {
        if y == 0 {
            return BlockType::Bedrock;
        }
        if y <= 4 {
            let bedrock_chance = (5 - y) as u32 * 20;
            let hash = self.position_hash_3d(world_x, y, world_z);
            if (hash % 100) < bedrock_chance {
                return BlockType::Bedrock;
            }
        }

        if y < 8 {
            let deep_hash = self.position_hash_3d(world_x, y, world_z);
            if deep_hash % 10 < 3 {
                return BlockType::Stone;
            }
        }

        // Density terrain can intentionally extend beyond the column's macro
        // surface (overhangs, shelves and cliff lips). Those voxels are bare
        // rock; normal surface rules remain attached to the macro surface.
        if y >= surface_height {
            return BlockType::Stone;
        }

        let depth_from_surface = surface_height - y;
        let dirt_depth = 3 + (self.position_hash(world_x, world_z) % 3) as i32;
        let underwater_surface = y == surface_height - 1 && y < SEA_LEVEL;
        let is_surface = y == surface_height - 1;
        let high_snow =
            is_surface && !underwater_surface && y >= self.snow_line(world_x, world_z, biome);
        let steep_surface = is_surface && slope >= 7 && !underwater_surface;
        let surface_hash = self.position_hash(world_x, world_z);

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
                    if underwater_surface {
                        BlockType::Sand
                    } else if biome == Biome::Island && y > SEA_LEVEL + 2 {
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
                    if steep_surface && surface_hash % 4 == 0 {
                        BlockType::Stone
                    } else {
                        BlockType::Snow
                    }
                } else {
                    BlockType::Air
                }
            }
            Biome::Mountains => {
                if high_snow || y > 150 {
                    if y == surface_height - 1 {
                        BlockType::Snow
                    } else {
                        BlockType::Stone
                    }
                } else if y > 115 {
                    let hash = self.position_hash_3d(world_x, y, world_z);
                    if depth_from_surface <= 1 {
                        if slope >= 9 || hash % 4 == 0 {
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
                    if underwater_surface {
                        BlockType::Gravel
                    } else if steep_surface {
                        if surface_hash % 3 == 0 {
                            BlockType::Gravel
                        } else {
                            BlockType::Stone
                        }
                    } else {
                        BlockType::Grass
                    }
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
                    if underwater_surface || y <= SEA_LEVEL + 1 {
                        BlockType::Clay
                    } else if steep_surface && surface_hash % 4 == 0 {
                        BlockType::Dirt
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
                    if underwater_surface {
                        BlockType::Sand
                    } else if high_snow {
                        BlockType::Snow
                    } else if steep_surface {
                        if surface_hash % 5 == 0 {
                            BlockType::Gravel
                        } else {
                            BlockType::Stone
                        }
                    } else {
                        BlockType::Grass
                    }
                } else {
                    BlockType::Air
                }
            }
        }
    }

    fn apply_stone_variants(
        &self,
        world_x: i32,
        y: i32,
        world_z: i32,
        block: BlockType,
    ) -> BlockType {
        if block != BlockType::Stone {
            return block;
        }
        if y <= 6 || y >= WORLD_HEIGHT - 2 {
            return block;
        }

        // Simple “veins” using existing block set:
        // - Gravel pockets: common-ish mid-depth (adds variety to mining)
        // - Clay pockets: rarer, a bit higher (keeps it special vs gravel)
        let fx = world_x as f32;
        let fy = y as f32;
        let fz = world_z as f32;
        let n = self.noise_ore.get_noise_3d(fx, fy, fz);
        let n2 = self
            .noise_ore
            .get_noise_3d(fx + 200.0, fy + 200.0, fz + 200.0);

        if (18..=78).contains(&y) && n > 0.62 {
            return BlockType::Gravel;
        }
        if (38..=66).contains(&y) && n2 > 0.70 {
            return BlockType::Clay;
        }

        block
    }

    // ── Surface decorations ───────────────────────────────────────────────── //

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
        let margin = 3;

        for lx in margin..(CHUNK_SIZE - margin) {
            for lz in margin..(CHUNK_SIZE - margin) {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome = biome_map[lx as usize][lz as usize];
                let height = height_map[lx as usize][lz as usize];
                let hash = self.position_hash(world_x, world_z);
                let vegetation = self
                    .noise_vegetation
                    .get_noise_2d(world_x as f32 + 1000.0, world_z as f32 - 1000.0);
                let clearing = self
                    .noise_decor
                    .get_noise_2d(world_x as f32 * 0.025 + 2000.0, world_z as f32 * 0.025);

                if height <= SEA_LEVEL || height >= WORLD_HEIGHT - 12 {
                    continue;
                }

                if matches!(biome, Biome::Mountains | Biome::Tundra)
                    && height > SEA_LEVEL + 12
                    && hash % 100 < 5
                {
                    let ground = chunk.get_block(lx, height - 1, lz);
                    if matches!(
                        ground,
                        BlockType::Grass | BlockType::Stone | BlockType::Snow
                    ) {
                        self.place_boulder(chunk, lx, height, lz, world_x, world_z);
                    }
                }

                if matches!(biome, Biome::Forest | Biome::Swamp)
                    && vegetation > 0.30
                    && hash % 100 < 3
                {
                    let ground = chunk.get_block(lx, height - 1, lz);
                    if matches!(ground, BlockType::Grass | BlockType::Dirt | BlockType::Clay) {
                        self.place_fallen_log(chunk, lx, height, lz, world_x, world_z);
                    }
                }

                if biome.has_trees() {
                    let tree_noise = self
                        .noise_trees
                        .get_noise_2d(world_x as f32, world_z as f32);
                    let clearing_penalty = clearing.abs() * 0.16;
                    let density_threshold = (biome.tree_density() as f32 - vegetation * 0.12
                        + clearing_penalty)
                        .clamp(0.35, 0.95);
                    let tree_roll = tree_noise + vegetation * 0.18 - clearing_penalty;

                    if tree_roll > density_threshold {
                        let tree_spawn_chance = if biome == Biome::Forest { 28 } else { 18 };
                        if hash % 100 < tree_spawn_chance {
                            let ground = chunk.get_block(lx, height - 1, lz);
                            if matches!(ground, BlockType::Grass | BlockType::Dirt) {
                                let is_large =
                                    hash % 7 == 0 && matches!(biome, Biome::Forest | Biome::Swamp);
                                if self.can_place_tree(chunk, lx, height, lz, is_large) {
                                    self.place_tree(
                                        chunk, lx, height, lz, world_x, world_z, biome, is_large,
                                    );
                                }
                            }
                        }
                    }
                }

                if biome == Biome::Beach && hash % 100 < 4 {
                    let ground = chunk.get_block(lx, height - 1, lz);
                    if ground == BlockType::Sand {
                        if hash % 3 == 0 {
                            self.place_fallen_log(chunk, lx, height, lz, world_x, world_z);
                        }
                    }
                }

                if biome == Biome::Desert {
                    if hash % 100 < 3 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand {
                            self.place_cactus(chunk, lx, height, lz, world_x, world_z);
                        }
                    }
                }

                if biome == Biome::Mountains && height > 110 {
                    if hash % 100 < 8 {
                        let top = chunk.get_block(lx, height - 1, lz);
                        if matches!(top, BlockType::Stone | BlockType::Grass) {
                            chunk.set_block_raw(lx, height - 1, lz, BlockType::Gravel);
                        }
                    }
                }

                if biome == Biome::Mountains && height > 145 {
                    if chunk.get_block(lx, height - 1, lz) == BlockType::Stone {
                        chunk.set_block_raw(lx, height - 1, lz, BlockType::Snow);
                    }
                }
            }
        }
    }

    // ── Tree placement ────────────────────────────────────────────────────── //

    fn can_place_tree(&self, chunk: &Chunk, lx: i32, y: i32, lz: i32, is_large: bool) -> bool {
        let ground_block = chunk.get_block(lx, y - 1, lz);
        if !matches!(ground_block, BlockType::Grass | BlockType::Dirt) {
            return false;
        }

        let required_height = if is_large { 9 } else { 7 };
        for dy in 0..=required_height {
            let check_y = y + dy;
            if check_y >= WORLD_HEIGHT {
                return false;
            }
            let block = chunk.get_block(lx, check_y, lz);
            if !matches!(block, BlockType::Air | BlockType::Leaves) {
                return false;
            }
        }

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

        let min_distance = if is_large { 5 } else { 3 };
        for dx in -min_distance..=min_distance {
            for dz in -min_distance..=min_distance {
                let check_x = lx + dx;
                let check_z = lz + dz;

                if check_x < 0 || check_x >= CHUNK_SIZE || check_z < 0 || check_z >= CHUNK_SIZE {
                    continue;
                }

                for dy in -1..=8 {
                    let check_y = y + dy;
                    if check_y < 0 || check_y >= WORLD_HEIGHT {
                        continue;
                    }
                    if matches!(
                        chunk.get_block(check_x, check_y, check_z),
                        BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ
                    ) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn place_tree(
        &self,
        chunk: &mut Chunk,
        lx: i32,
        y: i32,
        lz: i32,
        world_x: i32,
        world_z: i32,
        biome: Biome,
        is_large: bool,
    ) {
        let trunk_height = if is_large {
            8
        } else {
            5 + (self.position_hash(world_x, world_z) % 2) as i32
        };

        if chunk.get_block(lx, y - 1, lz) == BlockType::Grass {
            chunk.set_block_raw(lx, y - 1, lz, BlockType::Dirt);
        }

        for dy in 0..trunk_height {
            chunk.set_block_raw(lx, y + dy, lz, BlockType::Wood);
        }

        let leaf_start = if is_large { 4 } else { 3 };
        let leaf_radius = if is_large { 3 } else { 2 };

        for dy in leaf_start..=trunk_height {
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
                                let corner_skip = match biome {
                                    Biome::Swamp => {
                                        dx.abs() == radius
                                            && dz.abs() == radius
                                            && self.position_hash(world_x + dx, world_z + dz) % 3
                                                != 0
                                    }
                                    _ => {
                                        dx.abs() == radius
                                            && dz.abs() == radius
                                            && self.position_hash(world_x + dx, world_z + dz) % 2
                                                == 0
                                    }
                                };
                                if !corner_skip {
                                    chunk.set_block_raw(nx, ny, nz, BlockType::Leaves);
                                }
                            }
                        }
                    }
                }
            }
        }

        let top_y = y + trunk_height;
        if top_y < WORLD_HEIGHT {
            let existing = chunk.get_block(lx, top_y, lz);
            if existing == BlockType::Air || existing == BlockType::Leaves {
                chunk.set_block_raw(lx, top_y, lz, BlockType::Leaves);
            }
        }
    }

    fn place_cactus(
        &self,
        chunk: &mut Chunk,
        lx: i32,
        y: i32,
        lz: i32,
        world_x: i32,
        world_z: i32,
    ) {
        let height = 2 + (self.position_hash(world_x, world_z) % 3) as i32;
        for dy in 0..height {
            if y + dy < WORLD_HEIGHT {
                chunk.set_block_raw(lx, y + dy, lz, BlockType::Cactus);
            }
        }
    }

    fn place_boulder(
        &self,
        chunk: &mut Chunk,
        lx: i32,
        y: i32,
        lz: i32,
        world_x: i32,
        world_z: i32,
    ) {
        let hash = self.position_hash(world_x, world_z);
        let radius = if hash % 5 == 0 { 2 } else { 1 };
        let radius_sq = (radius * radius) as f32 + 0.35;

        for dx in -radius..=radius {
            for dz in -radius..=radius {
                for dy in 0..=radius {
                    let nx = lx + dx;
                    let ny = y + dy;
                    let nz = lz + dz;
                    if nx < 0 || nx >= CHUNK_SIZE || nz < 0 || nz >= CHUNK_SIZE {
                        continue;
                    }
                    if ny <= 0 || ny >= WORLD_HEIGHT {
                        continue;
                    }

                    let dist = (dx * dx + dz * dz) as f32 + (dy as f32 * 1.35).powi(2);
                    if dist > radius_sq {
                        continue;
                    }

                    let existing = chunk.get_block(nx, ny, nz);
                    if matches!(existing, BlockType::Air | BlockType::Leaves) {
                        let hash3 = self.position_hash_3d(world_x + dx, ny, world_z + dz);
                        let block = if hash3 % 7 == 0 {
                            BlockType::Gravel
                        } else {
                            BlockType::Stone
                        };
                        chunk.set_block_raw(nx, ny, nz, block);
                    }
                }
            }
        }
    }

    fn place_fallen_log(
        &self,
        chunk: &mut Chunk,
        lx: i32,
        y: i32,
        lz: i32,
        world_x: i32,
        world_z: i32,
    ) {
        let hash = self.position_hash(world_x, world_z);
        let length = 4 + (hash % 4) as i32;
        let (step_x, step_z) = if hash & 1 == 0 { (1, 0) } else { (0, 1) };
        let side_x = if step_x == 0 { 1 } else { 0 };
        let side_z = if step_z == 0 { 1 } else { 0 };
        let start = -(length / 2);

        if self.fallen_log_collides_with_tree(chunk, lx, y, lz, length, step_x, step_z) {
            return;
        }

        for i in 0..length {
            let offset = start + i;
            let nx = lx + step_x * offset;
            let nz = lz + step_z * offset;
            if nx <= 0 || nx >= CHUNK_SIZE - 1 || nz <= 0 || nz >= CHUNK_SIZE - 1 {
                return;
            }

            let ground = chunk.get_block(nx, y - 1, nz);
            let target = chunk.get_block(nx, y, nz);
            if !matches!(
                ground,
                BlockType::Grass | BlockType::Dirt | BlockType::Clay | BlockType::Sand
            ) || target != BlockType::Air
            {
                return;
            }
        }

        let stump_side = if hash & 2 == 0 { -1 } else { 1 };
        let stump_x = lx + side_x * stump_side;
        let stump_z = lz + side_z * stump_side;
        let placed_stump =
            self.try_place_fallen_log_detail(chunk, stump_x, y, stump_z, BlockType::Wood);
        if placed_stump
            && y + 1 < WORLD_HEIGHT
            && hash % 3 == 0
            && chunk.get_block(stump_x, y + 1, stump_z) == BlockType::Air
        {
            chunk.set_block_raw(stump_x, y + 1, stump_z, BlockType::Wood);
        }

        for i in 0..length {
            let offset = start + i;
            let nx = lx + step_x * offset;
            let nz = lz + step_z * offset;
            let log_block = if step_x != 0 {
                BlockType::WoodLogX
            } else {
                BlockType::WoodLogZ
            };
            chunk.set_block_raw(nx, y, nz, log_block);

            let ground = chunk.get_block(nx, y - 1, nz);
            if matches!(ground, BlockType::Grass | BlockType::Dirt | BlockType::Clay)
                && self.position_hash_3d(world_x + step_x * offset, y, world_z + step_z * offset)
                    % 3
                    == 0
            {
                for side in [-1, 1] {
                    let sx = nx + side_x * side;
                    let sz = nz + side_z * side;
                    if sx >= 0
                        && sx < CHUNK_SIZE
                        && sz >= 0
                        && sz < CHUNK_SIZE
                        && chunk.get_block(sx, y, sz) == BlockType::Air
                    {
                        chunk.set_block_raw(sx, y, sz, BlockType::Leaves);
                    }
                }
            }

            let segment_hash =
                self.position_hash_3d(world_x + step_x * offset, y + i, world_z + step_z * offset);
            if i > 0 && i < length - 1 && segment_hash % 5 == 0 {
                let branch_side = if segment_hash & 1 == 0 { -1 } else { 1 };
                let branch_x = nx + side_x * branch_side;
                let branch_z = nz + side_z * branch_side;
                let branch_block = if step_x != 0 {
                    BlockType::WoodLogZ
                } else {
                    BlockType::WoodLogX
                };
                self.try_place_fallen_log_detail(chunk, branch_x, y, branch_z, branch_block);
            }
        }

        for end_offset in [start - 1, start + length] {
            let end_x = lx + step_x * end_offset;
            let end_z = lz + step_z * end_offset;
            self.try_place_fallen_log_detail(chunk, end_x, y, end_z, BlockType::Leaves);

            for side in [-1, 1] {
                let leaf_hash = self.position_hash_3d(
                    world_x + step_x * end_offset + side_x * side,
                    y,
                    world_z + step_z * end_offset + side_z * side,
                );
                if leaf_hash % 2 == 0 {
                    self.try_place_fallen_log_detail(
                        chunk,
                        end_x + side_x * side,
                        y,
                        end_z + side_z * side,
                        BlockType::Leaves,
                    );
                }
            }
        }
    }

    fn try_place_fallen_log_detail(
        &self,
        chunk: &mut Chunk,
        x: i32,
        y: i32,
        z: i32,
        block: BlockType,
    ) -> bool {
        if x <= 0 || x >= CHUNK_SIZE - 1 || z <= 0 || z >= CHUNK_SIZE - 1 || y <= 0 {
            return false;
        }

        let ground = chunk.get_block(x, y - 1, z);
        if !matches!(
            ground,
            BlockType::Grass | BlockType::Dirt | BlockType::Clay | BlockType::Sand
        ) || chunk.get_block(x, y, z) != BlockType::Air
        {
            return false;
        }

        chunk.set_block_raw(x, y, z, block);
        true
    }

    fn fallen_log_collides_with_tree(
        &self,
        chunk: &Chunk,
        lx: i32,
        y: i32,
        lz: i32,
        length: i32,
        step_x: i32,
        step_z: i32,
    ) -> bool {
        let start = -(length / 2);
        let side_x = if step_x == 0 { 1 } else { 0 };
        let side_z = if step_z == 0 { 1 } else { 0 };

        for i in 0..length {
            let offset = start + i;
            let nx = lx + step_x * offset;
            let nz = lz + step_z * offset;

            for side in -2..=2 {
                for dy in 0..=7 {
                    let check_x = nx + side_x * side;
                    let check_y = y + dy;
                    let check_z = nz + side_z * side;

                    if check_x < 0
                        || check_x >= CHUNK_SIZE
                        || check_y < 0
                        || check_y >= WORLD_HEIGHT
                        || check_z < 0
                        || check_z >= CHUNK_SIZE
                    {
                        continue;
                    }

                    if matches!(
                        chunk.get_block(check_x, check_y, check_z),
                        BlockType::Wood
                            | BlockType::WoodLogX
                            | BlockType::WoodLogZ
                            | BlockType::Leaves
                    ) {
                        return true;
                    }
                }
            }
        }

        false
    }

    // ── Hash functions ────────────────────────────────────────────────────── //

    fn position_hash(&self, x: i32, z: i32) -> u32 {
        let mut hash = self.seed;
        hash = hash.wrapping_add(x as u32).wrapping_mul(73856093);
        hash = hash.wrapping_add(z as u32).wrapping_mul(19349663);
        hash ^ (hash >> 16)
    }

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
    fn clone(&self) -> Self {
        ChunkGenerator::new(self.seed)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Minecraft-style ridge transform. A single smooth weirdness field becomes
/// alternating broad valleys, lowlands and peak bands instead of isolated
/// random mountain blobs.
fn peaks_and_valleys(weirdness: f64) -> f64 {
    (-((weirdness.abs() - 2.0 / 3.0).abs()) + 1.0 / 3.0) * 3.0
}

fn half_negative(value: f64) -> f64 {
    if value < 0.0 { value * 0.5 } else { value }
}

fn quarter_negative(value: f64) -> f64 {
    if value < 0.0 { value * 0.25 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_and_valleys_has_peak_and_valley_bands() {
        assert!((peaks_and_valleys(0.0) + 1.0).abs() < f64::EPSILON);
        assert!(peaks_and_valleys(1.0 / 3.0).abs() < f64::EPSILON);
        assert!((peaks_and_valleys(2.0 / 3.0) - 1.0).abs() < f64::EPSILON);
        assert!(peaks_and_valleys(1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn same_seed_generates_identical_chunk_blocks() {
        let first = ChunkGenerator::new(42).generate_chunk(2, -3);
        let second = ChunkGenerator::new(42).generate_chunk(2, -3);

        for x in 0..CHUNK_SIZE {
            for y in 0..WORLD_HEIGHT {
                for z in 0..CHUNK_SIZE {
                    assert_eq!(first.get_block(x, y, z), second.get_block(x, y, z));
                }
            }
        }
    }

    #[test]
    fn terrain_height_queries_stay_inside_world_bounds() {
        let generator = ChunkGenerator::new(2026);
        let points = [(0, 0), (31, -17), (-128, 256), (1024, -1024), (-4096, 2048)];

        for (x, z) in points {
            let height = generator.get_terrain_height_pub(x, z);
            assert!((1..=WORLD_HEIGHT - 20).contains(&height));
        }
    }

    #[test]
    fn generated_chunk_metadata_matches_opaque_columns() {
        let chunk = ChunkGenerator::new(1337).generate_chunk(0, 0);

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let expected = (0..WORLD_HEIGHT)
                    .rev()
                    .find(|&y| chunk.get_block(x, y, z).is_solid_opaque())
                    .map(|y| y as i16)
                    .unwrap_or(-1);

                assert_eq!(chunk.highest_opaque_y(x, z), expected);
            }
        }
    }

    #[test]
    fn sandy_surfaces_do_not_get_single_gravel_or_clay_speckles() {
        let generator = ChunkGenerator::new(90_210);

        for cx in -1..=1 {
            for cz in -1..=1 {
                let chunk = generator.generate_chunk(cx, cz);

                for x in 0..CHUNK_SIZE {
                    for y in 1..(WORLD_HEIGHT - 1) {
                        for z in 0..CHUNK_SIZE {
                            let block = chunk.get_block(x, y, z);
                            if !matches!(block, BlockType::Gravel | BlockType::Clay) {
                                continue;
                            }

                            let below = chunk.get_block(x, y - 1, z);
                            let above = chunk.get_block(x, y + 1, z);
                            assert!(
                                !(below == BlockType::Sand
                                    && matches!(above, BlockType::Air | BlockType::Water)),
                                "visible {:?} speckle on sand at chunk ({}, {}), local ({}, {}, {})",
                                block,
                                cx,
                                cz,
                                x,
                                y,
                                z
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decorations_do_not_generate_shrubs() {
        let generator = ChunkGenerator::new(12);

        for cx in -1..=1 {
            for cz in -1..=1 {
                let chunk = generator.generate_chunk(cx, cz);

                for x in 0..CHUNK_SIZE {
                    for y in 0..WORLD_HEIGHT {
                        for z in 0..CHUNK_SIZE {
                            assert_ne!(chunk.get_block(x, y, z), BlockType::DeadBush);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fallen_logs_use_axis_specific_blocks_and_textures() {
        assert_eq!(BlockType::WoodLogX.tex_for_face(0), TEX_WOOD_TOP);
        assert_eq!(BlockType::WoodLogX.tex_for_face(1), TEX_WOOD_TOP);
        assert_eq!(BlockType::WoodLogX.tex_for_face(4), TEX_WOOD_SIDE);
        assert_eq!(BlockType::WoodLogZ.tex_for_face(4), TEX_WOOD_TOP);
        assert_eq!(BlockType::WoodLogZ.tex_for_face(5), TEX_WOOD_TOP);
        assert_eq!(BlockType::WoodLogZ.tex_for_face(0), TEX_WOOD_SIDE);

        let generator = ChunkGenerator::new(77);
        let mut chunk = Chunk::new(0, 0);
        let y = 70;

        for x in 1..(CHUNK_SIZE - 1) {
            for z in 1..(CHUNK_SIZE - 1) {
                chunk.set_block_raw(x, y - 1, z, BlockType::Dirt);
            }
        }

        generator.place_fallen_log(&mut chunk, 8, y, 8, 8, 8);

        let expected_log = if generator.position_hash(8, 8) & 1 == 0 {
            BlockType::WoodLogX
        } else {
            BlockType::WoodLogZ
        };
        let mut oriented_logs = 0;
        let mut stump_blocks = 0;

        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if chunk.get_block(x, y, z) == expected_log {
                    oriented_logs += 1;
                }
                if chunk.get_block(x, y, z) == BlockType::Wood {
                    stump_blocks += 1;
                }
            }
        }

        assert!(oriented_logs >= 4);
        assert!(stump_blocks >= 1);
    }

    #[test]
    fn fallen_logs_do_not_cut_through_existing_trees() {
        let generator = ChunkGenerator::new(77);
        let mut chunk = Chunk::new(0, 0);
        let y = 70;

        for x in 1..(CHUNK_SIZE - 1) {
            for z in 1..(CHUNK_SIZE - 1) {
                chunk.set_block_raw(x, y - 1, z, BlockType::Dirt);
            }
        }

        for dy in 0..5 {
            chunk.set_block_raw(8, y + dy, 8, BlockType::Wood);
        }
        chunk.set_block_raw(9, y + 3, 8, BlockType::Leaves);

        generator.place_fallen_log(&mut chunk, 8, y, 8, 8, 8);

        let mut fallen_logs = 0;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if matches!(
                    chunk.get_block(x, y, z),
                    BlockType::WoodLogX | BlockType::WoodLogZ
                ) {
                    fallen_logs += 1;
                }
            }
        }

        assert_eq!(fallen_logs, 0);
        assert_eq!(chunk.get_block(8, y, 8), BlockType::Wood);
    }
}
