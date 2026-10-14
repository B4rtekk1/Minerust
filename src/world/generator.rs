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
/// Cave fields contain the most expensive 3-D noise in chunk generation.
/// Keep their lattice independently tunable from terrain-density sampling.
const CAVE_XZ_SAMPLES: usize = (CHUNK_SIZE / CAVE_CELL_SIZE + 1) as usize;
const CAVE_Y_SAMPLES: usize = (WORLD_HEIGHT / CAVE_CELL_SIZE + 1) as usize;
const CAVE_LATTICE_LEN: usize = CAVE_XZ_SAMPLES * CAVE_Y_SAMPLES * CAVE_XZ_SAMPLES;
const TERRAIN_WARP_Z_OFFSET: f32 = 200.0;
// Softer domain warp keeps large continental cells readable while still
// breaking axis-aligned noise striping.
const MACRO_WARP_AMPLITUDE: f32 = 160.0;
const REGIONAL_WARP_AMPLITUDE: f32 = 32.0;
const RIVER_THRESHOLD_MIN: f32 = 0.855;
const RIVER_THRESHOLD_MAX: f32 = 0.920;
const LAKE_THRESHOLD: f32 = -0.66;

static CONTINENTAL_SPLINE: Lazy<TerrainSpline> = Lazy::new(TerrainSpline::continental);
/// High-erosion lands stay near the continental base: shallow valleys, gentle
/// hills, almost no peak response. This is what makes Minecraft plains flat.
static PV_OFFSET_ERODED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, -0.10),
        (-0.70, -0.06),
        (-0.35, -0.02),
        (0.0, 0.0),
        (0.35, 0.02),
        (0.70, 0.05),
        (1.0, 0.08),
    ])
});
/// Low-erosion bands form valleys and peaks. Valley depth is intentionally
/// moderate — extreme negative offsets produced knife-cut pits between
/// neighbouring density lattice samples.
static PV_OFFSET_RUGGED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, -0.22),
        (-0.70, -0.16),
        (-0.35, -0.08),
        (0.0, 0.02),
        (0.25, 0.12),
        (0.50, 0.24),
        (0.75, 0.40),
        (1.0, 0.55),
    ])
});
static PV_FACTOR_ERODED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, 0.84),
        (-0.50, 0.88),
        (0.0, 0.92),
        (0.50, 0.95),
        (1.0, 0.98),
    ])
});
/// Keep factor close to 1 so density responds gently to Y. High factors turn
/// small offset changes into vertical cliffs / sudden hollows.
static PV_FACTOR_RUGGED: Lazy<TerrainSpline> = Lazy::new(|| {
    TerrainSpline::new(&[
        (-1.0, 0.94),
        (-0.50, 1.02),
        (0.0, 1.10),
        (0.35, 1.20),
        (0.65, 1.32),
        (1.0, 1.42),
    ])
});
#[derive(Clone, Copy)]
struct TerrainShape {
    /// Vertical translation of the density field in normalized Y space.
    offset: f64,
    /// Terrain response to Y/depth; larger values form steeper relief.
    factor: f64,
    jaggedness: f64,
    density3d_strength: f64,
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
struct BiomeNoiseSample {
    continental: f64,
    erosion: f64,
    weirdness: f64,
    temperature: f32,
    moisture: f32,
    river_value: f32,
    river_threshold: f32,
    lake: f32,
    island: f32,
}

#[derive(Clone, Copy)]
struct TerrainNoiseSample {
    continental: f64,
    biome_continental: f32,
    terrain: f64,
    river_value: f32,
    river_threshold: f32,
    lake: f32,
    island: f32,
    erosion: f64,
    biome_erosion: f32,
    ridged: f64,
    mountain_chain: f64,
    biome_peaks_valleys: f32,
    weirdness: f64,
    plateau: f64,
    valley: f64,
}

#[derive(Clone, Copy)]
struct ColumnSample {
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
            // Larger continental cells and slower climate axes → readable
            // Minecraft-scale landmasses and biome regions.
            noise_continents: Self::create_fbm_noise(seed, 0.0012),
            noise_terrain: Self::create_fbm_noise(seed.wrapping_add(1), 0.0045),
            noise_detail: Self::create_fbm_noise(seed.wrapping_add(2), 0.012),
            noise_temperature: Self::create_noise(seed.wrapping_add(3), 0.0011),
            noise_moisture: Self::create_noise(seed.wrapping_add(4), 0.0015),
            noise_river: Self::create_noise(seed.wrapping_add(5), 0.0032),
            noise_lake: Self::create_noise(seed.wrapping_add(6), 0.018),
            noise_trees: Self::create_noise(seed.wrapping_add(7), 0.12),
            noise_island: Self::create_noise(seed.wrapping_add(8), 0.040),
            // Cave samplers intentionally use unit frequency: their explicit
            // X/Y/Z scales below are the one coordinate convention for caves.
            noise_cave1: Self::create_unit_3d_noise(seed.wrapping_add(9)),
            noise_cave2: Self::create_unit_3d_noise(seed.wrapping_add(10)),
            noise_cave3: Self::create_unit_3d_noise(seed.wrapping_add(11)),
            noise_erosion: Self::create_fbm_noise(seed.wrapping_add(12), 0.0045),
            noise_warp_x: Self::create_fbm_noise(seed.wrapping_add(20), 0.0055),
            noise_warp_z: Self::create_fbm_noise(seed.wrapping_add(21), 0.0040),
            noise_ridged: Self::create_ridged_noise(seed.wrapping_add(22), 0.0048),
            noise_weirdness: Self::create_fbm_noise(seed.wrapping_add(26), 0.0026),
            noise_plateau: Self::create_fbm_noise(seed.wrapping_add(27), 0.0032),
            noise_valley: Self::create_fbm_noise(seed.wrapping_add(28), 0.0024),
            noise_river_width: Self::create_fbm_noise(seed.wrapping_add(29), 0.0012),
            noise_vegetation: Self::create_noise(seed.wrapping_add(32), 0.045),
            noise_decor: Self::create_noise(seed.wrapping_add(24), 0.13),
            noise_ore: Self::create_3d_noise(seed.wrapping_add(25), 0.065),
            noise_cave_warp_x: Self::create_unit_fbm_noise(seed.wrapping_add(30)),
            noise_cave_warp_z: Self::create_unit_fbm_noise(seed.wrapping_add(31)),
            noise_aquifer: Self::create_3d_noise(seed.wrapping_add(41), 0.007),
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
                let x = base_x + sample_x;
                let z = base_z + sample_z;
                if (0..CHUNK_SIZE).contains(&sample_x) && (0..CHUNK_SIZE).contains(&sample_z) {
                    // Interior heights come from the density field in pass 2.
                    // Biome selection does not need the five terrain-only fields.
                    biome_map[sample_x as usize][sample_z as usize] = self.get_biome(x, z);
                } else {
                    bordered_heights[(sample_x + HEIGHT_MAP_BORDER) as usize]
                        [(sample_z + HEIGHT_MAP_BORDER) as usize] =
                        (self.sample_column(x, z).height as i32).clamp(1, WORLD_HEIGHT - 20);
                }
            }
        }
        let mut slope_map = [[0i32; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];

        // ── Pass 2: coarse 3-D terrain-density lattice ──────────────────── //
        // Terrain-only parameters are needed at coarse lattice columns and
        // the border, rather than at every interior voxel column.
        let density_shapes = std::array::from_fn(|ix| {
            std::array::from_fn(|iz| {
                let x = base_x + ix as i32 * DENSITY_CELL_SIZE;
                let z = base_z + iz as i32 * DENSITY_CELL_SIZE;
                self.terrain_shape(&self.sample_terrain_noise(x, z))
            })
        });
        let density_lattice = self.build_density_lattice(base_x, base_z, &density_shapes);
        // Interpolate the horizontal dimensions once per voxel column. The
        // resulting values are vertical lattice endpoints, so individual
        // voxels only need one vertical lerp below.
        let mut density_columns =
            [[[0.0; DENSITY_Y_SAMPLES]; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        // Surface rules and vegetation need the visible density surface, not
        // merely the macro-height estimate used to construct the field.
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let density_column = self.density_vertical_column(&density_lattice, lx, lz);
                height_map[lx as usize][lz as usize] =
                    Self::density_surface_height(&density_column);
                density_columns[lx as usize][lz as usize] = density_column;
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
        // The expensive domain-warped cave field is sampled once per coarse
        // cell corner. Endpoints lie on chunk boundaries, preventing seams.
        let cave_lattice = self.build_cave_lattice(base_x, base_z, &density_lattice);
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome = biome_map[lx as usize][lz as usize];
                let surface_height = height_map[lx as usize][lz as usize];
                let slope = slope_map[lx as usize][lz as usize];

                let is_entrance = cave_entrance_map[lx as usize][lz as usize];
                // `surface_height` is the highest positive terrain-density
                // crossing in this column. Above it only empty space can
                // exist; retaining sea level covers ocean and lake water.
                let max_y = surface_height.max(SEA_LEVEL).min(WORLD_HEIGHT - 1);
                let density_column = &density_columns[lx as usize][lz as usize];
                let cave_column = self.cave_vertical_column(&cave_lattice, lx, lz, surface_height);
                let mut fluid_level = None;
                for y in 0..=max_y {
                    let terrain_density = Self::sample_density_column(density_column, y);
                    let is_solid = terrain_density > 0.0
                        && Self::sample_cave_column(&cave_column, y, surface_height, is_entrance)
                            > 0.0;

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
                    } else if self.has_fluid(world_x, y, world_z, surface_height, &mut fluid_level)
                    {
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
        let shape = self.terrain_shape(&noise);
        let height = self.preliminary_surface_level(&shape, &noise);
        ColumnSample { height }
    }

    fn terrain_warp(&self, x: i32, z: i32) -> (f32, f32) {
        let fx = x as f32;
        let fz = z as f32;
        let macro_x =
            fx + self.noise_warp_x.get_noise_2d(fx * 0.25, fz * 0.25) * MACRO_WARP_AMPLITUDE;
        let macro_z = fz
            + self.noise_warp_z.get_noise_2d(
                (fx + TERRAIN_WARP_Z_OFFSET) * 0.25,
                (fz + TERRAIN_WARP_Z_OFFSET) * 0.25,
            ) * MACRO_WARP_AMPLITUDE;

        let wx =
            macro_x + self.noise_warp_x.get_noise_2d(macro_x, macro_z) * REGIONAL_WARP_AMPLITUDE;
        let wz = macro_z
            + self.noise_warp_z.get_noise_2d(
                macro_x + TERRAIN_WARP_Z_OFFSET,
                macro_z + TERRAIN_WARP_Z_OFFSET,
            ) * REGIONAL_WARP_AMPLITUDE;

        (wx, wz)
    }

    /// Samples only fields shared by biome selection and terrain shaping.
    fn sample_biome_noise(&self, wx: f32, wz: f32) -> BiomeNoiseSample {
        let continental = self.noise_continents.get_noise_2d(wx, wz) as f64;
        let erosion = self.noise_erosion.get_noise_2d(wx, wz) as f64;
        let river_noise = self.noise_river.get_noise_2d(wx, wz);
        let river_width = self.noise_river_width.get_noise_2d(wx + 250.0, wz - 250.0);
        let river_threshold = lerp(
            RIVER_THRESHOLD_MIN as f64,
            RIVER_THRESHOLD_MAX as f64,
            ((river_width as f64 + 1.0) * 0.5).clamp(0.0, 1.0),
        ) as f32;

        let weirdness = self.noise_weirdness.get_noise_2d(wx, wz) as f64;
        BiomeNoiseSample {
            continental,
            erosion,
            weirdness,
            temperature: self.noise_temperature.get_noise_2d(wx, wz),
            moisture: self.noise_moisture.get_noise_2d(wx, wz),
            river_value: 1.0 - river_noise.abs() * 2.0,
            river_threshold,
            lake: self.noise_lake.get_noise_2d(wx, wz),
            island: self.noise_island.get_noise_2d(wx, wz),
        }
    }

    fn sample_terrain_noise(&self, x: i32, z: i32) -> TerrainNoiseSample {
        let (wx, wz) = self.terrain_warp(x, z);
        let biome = self.sample_biome_noise(wx, wz);
        TerrainNoiseSample {
            continental: biome.continental,
            biome_continental: biome.continental as f32,
            terrain: self.noise_terrain.get_noise_2d(wx, wz) as f64,
            river_value: biome.river_value,
            river_threshold: biome.river_threshold,
            lake: biome.lake,
            island: biome.island,
            erosion: biome.erosion,
            biome_erosion: biome.erosion as f32,
            ridged: self.noise_ridged.get_noise_2d(wx, wz) as f64,
            mountain_chain: self.noise_ridged.get_noise_2d(wx * 0.18, wz * 0.18) as f64,
            biome_peaks_valleys: peaks_and_valleys(biome.weirdness) as f32,
            weirdness: biome.weirdness,
            plateau: self.noise_plateau.get_noise_2d(wx, wz) as f64,
            valley: self.noise_valley.get_noise_2d(wx, wz) as f64,
        }
    }

    fn classify_biome(&self, noise: &BiomeNoiseSample) -> Biome {
        // Rivers only on land that can actually reach the water table.
        if noise.river_value > noise.river_threshold
            && (noise.continental as f32) > -0.22
            && (noise.continental as f32) < 0.72
            && (noise.erosion as f32) > -0.55
        {
            return Biome::River;
        }

        if noise.lake < LAKE_THRESHOLD
            && (noise.continental as f32) > -0.08
            && (noise.erosion as f32) > -0.25
        {
            return Biome::Lake;
        }

        if (noise.continental as f32) < -0.38 {
            if noise.island > 0.55 && noise.weirdness > -0.40 {
                return Biome::Island;
            }
            return Biome::Ocean;
        }

        // Wider beach band for Minecraft-like sandy coasts.
        if (noise.continental as f32) < -0.14
            || ((noise.continental as f32) < -0.05 && noise.weirdness < -0.50)
        {
            return Biome::Beach;
        }

        self.select_land_biome(Climate {
            temperature: noise.temperature as f64,
            humidity: noise.moisture as f64,
            continentalness: (noise.continental as f32) as f64,
            erosion: (noise.erosion as f32) as f64,
            weirdness: noise.weirdness,
        })
    }

    /// Nearest-point selection makes climate extension data-driven: adding a
    /// biome no longer means growing a fragile chain of terrain conditionals.
    /// Temperature/humidity dominate so regions stay large and readable;
    /// erosion/continentalness only tip mountain vs plains contests.
    fn select_land_biome(&self, climate: Climate) -> Biome {
        const POINTS: [BiomePoint; 8] = [
            BiomePoint {
                biome: Biome::Plains,
                climate: Climate {
                    temperature: 0.15,
                    humidity: -0.05,
                    continentalness: 0.22,
                    erosion: 0.55,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Forest,
                climate: Climate {
                    temperature: 0.20,
                    humidity: 0.70,
                    continentalness: 0.25,
                    erosion: 0.20,
                    weirdness: 0.05,
                },
            },
            BiomePoint {
                biome: Biome::Mountains,
                climate: Climate {
                    temperature: -0.05,
                    humidity: 0.05,
                    continentalness: 0.68,
                    erosion: -0.78,
                    weirdness: 0.55,
                },
            },
            BiomePoint {
                biome: Biome::Desert,
                climate: Climate {
                    temperature: 0.82,
                    humidity: -0.78,
                    continentalness: 0.18,
                    erosion: 0.42,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Tundra,
                climate: Climate {
                    temperature: -0.78,
                    humidity: 0.0,
                    continentalness: 0.20,
                    erosion: 0.35,
                    weirdness: 0.0,
                },
            },
            BiomePoint {
                biome: Biome::Swamp,
                climate: Climate {
                    temperature: 0.55,
                    humidity: 0.88,
                    continentalness: -0.05,
                    erosion: 0.70,
                    weirdness: -0.15,
                },
            },
            // Cool highland forest (taiga-like within existing Forest biome).
            BiomePoint {
                biome: Biome::Forest,
                climate: Climate {
                    temperature: -0.25,
                    humidity: 0.45,
                    continentalness: 0.50,
                    erosion: -0.25,
                    weirdness: 0.30,
                },
            },
            // Warm elevated plains / meadow-like.
            BiomePoint {
                biome: Biome::Plains,
                climate: Climate {
                    temperature: 0.40,
                    humidity: 0.15,
                    continentalness: 0.55,
                    erosion: -0.15,
                    weirdness: 0.35,
                },
            },
        ];
        let mut best = (Biome::Plains, f64::INFINITY);
        for point in POINTS {
            let p = point.climate;
            let distance = (climate.temperature - p.temperature).powi(2) * 2.0
                + (climate.humidity - p.humidity).powi(2) * 2.0
                + (climate.continentalness - p.continentalness).powi(2) * 0.25
                + (climate.erosion - p.erosion).powi(2) * 0.45
                + (climate.weirdness - p.weirdness).powi(2) * 0.20;
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
        let (wx, wz) = self.terrain_warp(x, z);
        self.classify_biome(&self.sample_biome_noise(wx, wz))
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
        let land = smoothstep(((continentalness + 0.22) / 0.38).clamp(0.0, 1.0));
        let inland = ((continentalness + 0.10) / 0.70).clamp(0.0, 1.0);
        let roughness = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        // Mountain chains follow PV peaks on low-erosion inland; ridged noise
        // only accents the crest instead of spawning a second mountain system.
        let peak_mask = smoothstep(((pv - 0.18) / 0.55).clamp(0.0, 1.0));
        let chain = smoothstep(((noise.mountain_chain - 0.22) / 0.48).clamp(0.0, 1.0));
        let mountain_strength = land * inland * roughness * peak_mask * (0.45 + 0.55 * chain);

        // Nested Minecraft 1.18 response: continentalness = base shelf,
        // erosion = relief amplitude, PV = valley↔peak profile.
        let base = (CONTINENTAL_SPLINE.sample(continentalness) - SEA_LEVEL as f64) / 64.0;
        let pv_profile = self.pv_offset(pv, roughness);
        // High erosion nearly kills amplitude → flat plains.
        let amplitude = land * (0.06 + inland * 0.94) * (0.10 + roughness * 0.80);
        let ridge = noise.ridged.max(0.0).powi(2) * mountain_strength;
        let mut offset = base + pv_profile * amplitude + ridge * 0.22;
        // Soften factor toward 1.0 so neighbouring columns don't form pits.
        let mut factor = lerp(0.88, self.pv_factor(pv, roughness), inland * roughness.max(0.15));
        let mut jaggedness = self.shape_jaggedness(continentalness, erosion, pv, noise.ridged)
            * (0.15 + mountain_strength * 0.55);

        let river = self.river_channel_strength(noise) * (1.0 - mountain_strength * 0.85);
        let lake = self.lake_strength(noise);
        // Secondary valley/plateau fields stay tiny accents so PV owns shape.
        let valley = self.valley_strength(noise) * 0.05;
        let plateau = self.plateau_strength(noise) * 0.45;
        let ocean_factor = ((-continentalness - 0.38) / 0.55).clamp(0.0, 1.0);
        let island = self.island_strength(noise) * ocean_factor;

        // Rivers/lakes still cut to the water table, but with a gentler bowl
        // so lattice interpolation does not produce vertical shafts.
        offset -= river * (0.14 + land * 0.08) + lake * 0.14 + valley * 0.06;
        offset += island * 0.50 + plateau * 0.04;
        factor *= 1.0 - river * 0.12 - lake * 0.08 + plateau * 0.04;
        jaggedness *= 1.0 - river * 0.80 - lake * 0.75;
        // Mid-scale relief gives plains gentle rolling ground and makes
        // eroded uplands visibly uneven. Keep it weaker than the PV profile
        // so it adds hills without turning every biome into generic noise.
        let local_relief = 0.018 + roughness * 0.075 + mountain_strength * 0.045;
        offset += noise.terrain * local_relief;
        TerrainShape {
            offset,
            factor,
            jaggedness,
            // Overhangs stay mountain-local; capping avoids sudden hollows.
            density3d_strength: (0.02 + mountain_strength * 0.42 + peak_mask * 0.06)
                .clamp(0.02, 0.48),
        }
    }

    fn shape_jaggedness(&self, continentalness: f64, erosion: f64, pv: f64, ridged: f64) -> f64 {
        let inland = ((continentalness + 0.10) / 0.70).clamp(0.0, 1.0);
        let roughness = (1.0 - (erosion + 1.0) * 0.5).clamp(0.0, 1.0);
        inland * roughness * pv.max(0.0).powi(2) * (0.08 + (ridged + 1.0) * 0.08)
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
        // Smoothstep the bank so rivers are U-shaped bowls, not V-cut trenches.
        let raw =
            ((noise.river_value as f64 - threshold) / (1.0 - threshold).max(0.001)).clamp(0.0, 1.0);
        let centerline = smoothstep(raw);
        let land = smoothstep(((noise.continental + 0.20) / 0.32).clamp(0.0, 1.0));
        let height = CONTINENTAL_SPLINE.sample(noise.continental);
        let low_terrain =
            (1.0 - ((height - SEA_LEVEL as f64) / 36.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        // Prefer PV valley bands so rivers follow Minecraft-like troughs.
        let valley =
            smoothstep(((-noise.biome_peaks_valleys as f64 + 0.15) / 0.65).clamp(0.0, 1.0));
        centerline * land * (0.40 + 0.60 * valley) * (0.40 + 0.60 * low_terrain)
    }

    fn lake_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        if noise.biome_continental <= -0.08 || noise.biome_erosion <= -0.25 {
            return 0.0;
        }
        let basin =
            smoothstep(((LAKE_THRESHOLD as f64 - noise.lake as f64) / 0.20).clamp(0.0, 1.0));
        let height = CONTINENTAL_SPLINE.sample(noise.continental);
        let lowland = (1.0 - ((height - SEA_LEVEL as f64) / 26.0).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        let flatness = ((noise.biome_erosion as f64 + 1.0) * 0.5).clamp(0.0, 1.0);
        basin * lowland * smoothstep((flatness - 0.30) / 0.50)
    }

    fn island_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        smoothstep(((noise.island as f64 + 1.0) * 0.5 - 0.74) / 0.26)
    }

    fn valley_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        let raw_valley = ((noise.valley + 1.0) * 0.5).clamp(0.0, 1.0);
        let land_factor = ((noise.biome_continental as f64 + 0.08) / 0.48).clamp(0.0, 1.0);
        let mountain_factor =
            smoothstep((noise.biome_peaks_valleys as f64 - 0.24) / 0.42).clamp(0.0, 1.0);

        smoothstep((raw_valley - 0.58) / 0.28) * land_factor * (1.0 - mountain_factor * 0.55)
    }

    fn plateau_strength(&self, noise: &TerrainNoiseSample) -> f64 {
        let raw_plateau = ((noise.plateau + 1.0) * 0.5).clamp(0.0, 1.0);
        let land_factor = ((noise.biome_continental as f64 + 0.02) / 0.42).clamp(0.0, 1.0);
        let eroded_flatness = ((noise.biome_erosion as f64 + 1.0) * 0.5).clamp(0.0, 1.0);

        smoothstep((raw_plateau - 0.62) / 0.30)
            * smoothstep((eroded_flatness - 0.40) / 0.40)
            * land_factor
    }

    // ── Cave system ───────────────────────────────────────────────────────── //

    fn is_cave(&self, x: i32, y: i32, z: i32, surface_height: i32, is_entrance: bool) -> bool {
        if y <= 4 {
            return false;
        }

        let min_surface_dist = if is_entrance {
            let t = ((surface_height - y) as f32 / 14.0).clamp(0.0, 1.0);
            // Entrances thin the surface mask so cheese/spaghetti can daylight
            // on hillsides without post-process shafts.
            (1.0 + t * 7.0) as i32
        } else {
            // Thinner default roof than before → more natural openings.
            11
        };
        if y >= surface_height - min_surface_dist {
            return false;
        }

        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        let warp_amp = 14.0_f32;
        let wx = fx
            + self
                .noise_cave_warp_x
                .get_noise_3d(fx * 0.016, fy * 0.009, fz * 0.016)
                * warp_amp;
        let wy = fy
            + self.noise_cave_warp_z.get_noise_3d(
                fx * 0.016 + 100.0,
                fy * 0.009,
                fz * 0.016 + 100.0,
            ) * warp_amp
                * 0.18;
        let wz = fz
            + self.noise_cave_warp_x.get_noise_3d(
                fx * 0.016 + 200.0,
                fy * 0.009,
                fz * 0.016 + 200.0,
            ) * warp_amp;

        let in_lower = y < 58;
        let in_middle = y >= 58 && y < 100;

        // Cheese caves: larger, lower-frequency chambers in the deep/mid band.
        if in_lower || (in_middle && y < 78) {
            let c1 = self
                .noise_cave1
                .get_noise_3d(wx * 0.026, wy * 0.009, wz * 0.026);
            if c1 > 0.0 {
                let c2 = self.noise_cave2.get_noise_3d(
                    wx * 0.018 + 400.0,
                    wy * 0.007 + 400.0,
                    wz * 0.018 + 400.0,
                );
                // Lower product threshold → bigger cheese rooms.
                if c1 * c2.max(0.0) > 0.22 {
                    return true;
                }
            }
        }

        let s1 = self
            .noise_cave1
            .get_noise_3d(wx * 0.052 + 500.0, wy * 0.022, wz * 0.052);
        let s2 = self
            .noise_cave3
            .get_noise_3d(wx * 0.052 + 900.0, wy * 0.022, wz * 0.052);
        let spag_dist = (s1 * s1 + s2 * s2).sqrt();
        let spag_radius = if in_lower {
            0.115
        } else if in_middle {
            0.090
        } else {
            0.060
        };
        if spag_dist < spag_radius {
            return true;
        }

        if y > 16 {
            let n1 = self
                .noise_cave2
                .get_noise_3d(wx * 0.080 + 800.0, wy * 0.035, wz * 0.080);
            let n2 = self
                .noise_cave3
                .get_noise_3d(wx * 0.080 + 1200.0, wy * 0.035, wz * 0.080);
            let noodle_dist = (n1 * n1 + n2 * n2).sqrt();
            let noodle_radius = if in_lower { 0.065 } else { 0.045 };
            if noodle_dist < noodle_radius {
                return true;
            }
        }

        if y < 36 {
            let w1 = self
                .noise_cave2
                .get_noise_3d(wx * 0.038 + 800.0, wy * 0.013, wz * 0.038);
            let w2 = self
                .noise_cave3
                .get_noise_3d(wx * 0.038 + 1200.0, wy * 0.013, wz * 0.038);
            let worm_dist = (w1 * w1 + w2 * w2).sqrt();
            if worm_dist < 0.100 {
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
        if y <= 4 || y >= preliminary_surface {
            return 1.0;
        }

        // Carved cave families override the continuous cheese field entirely.
        if self.is_cave(x, y, z, preliminary_surface, is_entrance) {
            return -1.0;
        }

        // Continuous cheese field around the binary carve threshold softens
        // chamber edges; spaghetti/noodle/worm still use exact openings above.
        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;
        let c1 = self
            .noise_cave1
            .get_noise_3d(fx * 0.026, fy * 0.009, fz * 0.026);
        if c1 <= 0.0 {
            return 1.0;
        }
        let c2 = self.noise_cave2.get_noise_3d(
            fx * 0.018 + 400.0,
            fy * 0.007 + 400.0,
            fz * 0.018 + 400.0,
        );
        let cheese_signed = (0.22 - c1.max(0.0) * c2.max(0.0)) * 5.0;
        cheese_signed.clamp(0.06, 1.0) as f64
    }

    fn has_fluid(
        &self,
        x: i32,
        y: i32,
        z: i32,
        preliminary_surface: i32,
        fluid_level: &mut Option<i32>,
    ) -> bool {
        if preliminary_surface < SEA_LEVEL && y < SEA_LEVEL {
            return true;
        }
        // Aquifer water table varies by column; sample once, then skip above.
        let level = *fluid_level.get_or_insert_with(|| {
            let cell =
                self.noise_aquifer
                    .get_noise_3d(x as f32 + 4000.0, 0.0, z as f32 - 4000.0) as f64;
            // Slightly higher water table so flooded caves feel Minecraft-like.
            28 + ((cell + 1.0) * 0.5 * 36.0) as i32
        });
        y <= level
            && self
                .noise_aquifer
                .get_noise_3d(x as f32, y as f32, z as f32) as f64
                > 0.06
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 4 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        let entrance_noise = self
            .noise_cave1
            .get_noise_2d(fx * 0.012 + 1000.0, fz * 0.012 + 1000.0);

        let terrain_slope = self
            .noise_terrain
            .get_noise_2d(fx * 0.016, fz * 0.016)
            .abs();
        let is_hillside = terrain_slope > 0.12;

        let threshold = if is_hillside { 0.68 } else { 0.84 };
        if entrance_noise < threshold {
            return false;
        }

        let hash = self.position_hash(x, z);
        let entrance_chance = if is_hillside { 7 } else { 16 };
        if hash % entrance_chance != 0 {
            return false;
        }

        for check_y in (surface_height - 48).max(6)..=(surface_height - 4) {
            let fy = check_y as f32;
            let c1 = self
                .noise_cave1
                .get_noise_3d(fx * 0.040, fy * 0.020, fz * 0.040);
            let c2 = self
                .noise_cave2
                .get_noise_3d(fx * 0.028, fy * 0.016, fz * 0.028);
            if c1 > 0.58 && c2 > 0.58 {
                return true;
            }
        }

        false
    }

    // ── 3-D density (overhangs) ───────────────────────────────────────────── //

    fn density_index(ix: usize, iy: usize, iz: usize) -> usize {
        (ix * DENSITY_Y_SAMPLES + iy) * DENSITY_XZ_SAMPLES + iz
    }

    fn cave_index(ix: usize, iy: usize, iz: usize) -> usize {
        (ix * CAVE_Y_SAMPLES + iy) * CAVE_XZ_SAMPLES + iz
    }

    fn build_cave_lattice(
        &self,
        base_x: i32,
        base_z: i32,
        density_lattice: &[f64; DENSITY_LATTICE_LEN],
    ) -> [f64; CAVE_LATTICE_LEN] {
        let mut lattice = [1.0; CAVE_LATTICE_LEN];

        for ix in 0..CAVE_XZ_SAMPLES {
            for iz in 0..CAVE_XZ_SAMPLES {
                let lx = ix as i32 * CAVE_CELL_SIZE;
                let lz = iz as i32 * CAVE_CELL_SIZE;
                let x = base_x + lx;
                let z = base_z + lz;
                // The final lattice row/column is outside this chunk, but is
                // required as its shared interpolation endpoint.
                let density_column = self.density_vertical_column(density_lattice, lx, lz);
                let surface_height = Self::density_surface_height(&density_column);
                let is_entrance = self.is_cave_entrance(x, z, surface_height);

                for iy in 0..CAVE_Y_SAMPLES {
                    let y = iy as i32 * CAVE_CELL_SIZE;
                    lattice[Self::cave_index(ix, iy, iz)] =
                        self.cave_density(x, y, z, surface_height, is_entrance);
                }
            }
        }

        lattice
    }

    fn build_density_lattice(
        &self,
        base_x: i32,
        base_z: i32,
        shapes: &[[TerrainShape; DENSITY_XZ_SAMPLES]; DENSITY_XZ_SAMPLES],
    ) -> [f64; DENSITY_LATTICE_LEN] {
        // FastNoiseLite's fields are bounded by [-1, 1]. Use twice that
        // amplitude as a conservative margin. Above the first plane whose
        // upper bound is negative for every shape, all horizontal/vertical
        // interpolants remain negative. Keep that first plane's exact values
        // because they can still interpolate with a positive plane below.
        let first_air_plane = (0..DENSITY_Y_SAMPLES).find(|&iy| {
            let y = iy as i32 * DENSITY_CELL_SIZE;
            shapes.iter().flatten().all(|shape| {
                let depth = self.y_gradient(y) + shape.offset;
                let shaped = (depth + shape.jaggedness.abs() * 2.0) * shape.factor;
                debug_assert!(shape.factor > 0.0);
                4.0 * quarter_negative(shaped) + shape.density3d_strength.abs() * 2.0 < -0.001
            })
        });
        // The symmetric lower bound skips guaranteed solid deep terrain.
        // Retain the last all-solid plane exactly for the transition above it.
        let last_solid_plane = (0..DENSITY_Y_SAMPLES).rev().find(|&iy| {
            let y = iy as i32 * DENSITY_CELL_SIZE;
            shapes.iter().flatten().all(|shape| {
                let depth = self.y_gradient(y) + shape.offset;
                let shaped = (depth - shape.jaggedness.abs() * 2.0) * shape.factor;
                4.0 * quarter_negative(shaped) - shape.density3d_strength.abs() * 2.0 > 0.001
            })
        });
        let sample_start = last_solid_plane.unwrap_or(0);
        let sample_end = first_air_plane.map_or(DENSITY_Y_SAMPLES, |iy| iy + 1);
        let mut lattice = [-1.0; DENSITY_LATTICE_LEN];
        for ix in 0..DENSITY_XZ_SAMPLES {
            for iz in 0..DENSITY_XZ_SAMPLES {
                let x = base_x + ix as i32 * DENSITY_CELL_SIZE;
                let z = base_z + iz as i32 * DENSITY_CELL_SIZE;
                let shape = shapes[ix][iz];
                for iy in 0..sample_start {
                    lattice[Self::density_index(ix, iy, iz)] = 1.0;
                }
                for iy in sample_start..sample_end {
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
        4.0 * quarter_negative(shaped) + base_3d * shape.density3d_strength
    }

    fn y_gradient(&self, y: i32) -> f64 {
        (SEA_LEVEL as f64 - y as f64) / 64.0
    }

    /// Horizontally interpolate all vertical endpoints for one `(lx, lz)`
    /// column. This turns voxel density sampling into a single vertical lerp.
    fn density_vertical_column(
        &self,
        lattice: &[f64; DENSITY_LATTICE_LEN],
        lx: i32,
        lz: i32,
    ) -> [f64; DENSITY_Y_SAMPLES] {
        let ix = (lx / DENSITY_CELL_SIZE) as usize;
        let iz = (lz / DENSITY_CELL_SIZE) as usize;
        let tx = (lx.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let tz = (lz.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let x1 = (ix + 1).min(DENSITY_XZ_SAMPLES - 1);
        let z1 = (iz + 1).min(DENSITY_XZ_SAMPLES - 1);
        let mut column = [0.0; DENSITY_Y_SAMPLES];

        for iy in 0..DENSITY_Y_SAMPLES {
            let at = |x, z| lattice[Self::density_index(x, iy, z)];
            column[iy] = lerp(
                lerp(at(ix, iz), at(x1, iz), tx),
                lerp(at(ix, z1), at(x1, z1), tx),
                tz,
            );
        }

        column
    }

    fn sample_density_column(column: &[f64; DENSITY_Y_SAMPLES], y: i32) -> f64 {
        let iy = (y / DENSITY_CELL_SIZE) as usize;
        let ty = (y.rem_euclid(DENSITY_CELL_SIZE) as f64) / DENSITY_CELL_SIZE as f64;
        let y1 = (iy + 1).min(DENSITY_Y_SAMPLES - 1);
        lerp(column[iy], column[y1], ty)
    }

    /// Keep the original X-then-Z-then-Y lerp order so thresholds are identical.
    fn cave_vertical_column(
        &self,
        lattice: &[f64; CAVE_LATTICE_LEN],
        lx: i32,
        lz: i32,
        surface_height: i32,
    ) -> [f64; CAVE_Y_SAMPLES] {
        let ix = (lx / CAVE_CELL_SIZE) as usize;
        let iz = (lz / CAVE_CELL_SIZE) as usize;
        let tx = lx.rem_euclid(CAVE_CELL_SIZE) as f64 / CAVE_CELL_SIZE as f64;
        let tz = lz.rem_euclid(CAVE_CELL_SIZE) as f64 / CAVE_CELL_SIZE as f64;
        let x1 = (ix + 1).min(CAVE_XZ_SAMPLES - 1);
        let z1 = (iz + 1).min(CAVE_XZ_SAMPLES - 1);
        let mut column = [1.0; CAVE_Y_SAMPLES];
        let last = (surface_height as usize / CAVE_CELL_SIZE as usize).min(CAVE_Y_SAMPLES - 1);
        for (iy, value) in column.iter_mut().enumerate().take(last + 1) {
            let at = |x, z| lattice[Self::cave_index(x, iy, z)];
            *value = lerp(
                lerp(at(ix, iz), at(x1, iz), tx),
                lerp(at(ix, z1), at(x1, z1), tx),
                tz,
            );
        }
        column
    }

    fn sample_cave_column(
        column: &[f64; CAVE_Y_SAMPLES],
        y: i32,
        surface_height: i32,
        is_entrance: bool,
    ) -> f64 {
        if y <= 4 || y >= surface_height {
            return 1.0;
        }
        let min_surface_dist = if is_entrance {
            let t = ((surface_height - y) as f32 / 14.0).clamp(0.0, 1.0);
            (1.0 + t * 7.0) as i32
        } else {
            11
        };
        if y >= surface_height - min_surface_dist {
            return 1.0;
        }
        let iy = (y / CAVE_CELL_SIZE) as usize;
        let ty = y.rem_euclid(CAVE_CELL_SIZE) as f64 / CAVE_CELL_SIZE as f64;
        let y1 = (iy + 1).min(CAVE_Y_SAMPLES - 1);
        lerp(column[iy], column[y1], ty)
    }

    #[cfg(test)]
    fn sample_cave_lattice(
        &self,
        lattice: &[f64; CAVE_LATTICE_LEN],
        lx: i32,
        y: i32,
        lz: i32,
        surface_height: i32,
        is_entrance: bool,
    ) -> f64 {
        // A local guard prevents interpolation from a lower cell opening the
        // surface, and avoids all interpolation for sky voxels.
        if y <= 4 || y >= surface_height {
            return 1.0;
        }

        let ix = (lx / CAVE_CELL_SIZE) as usize;
        let iy = (y / CAVE_CELL_SIZE) as usize;
        let iz = (lz / CAVE_CELL_SIZE) as usize;
        let tx = (lx.rem_euclid(CAVE_CELL_SIZE) as f64) / CAVE_CELL_SIZE as f64;
        let ty = (y.rem_euclid(CAVE_CELL_SIZE) as f64) / CAVE_CELL_SIZE as f64;
        let tz = (lz.rem_euclid(CAVE_CELL_SIZE) as f64) / CAVE_CELL_SIZE as f64;
        let x1 = (ix + 1).min(CAVE_XZ_SAMPLES - 1);
        let y1 = (iy + 1).min(CAVE_Y_SAMPLES - 1);
        let z1 = (iz + 1).min(CAVE_XZ_SAMPLES - 1);
        let at = |x, yy, z| lattice[Self::cave_index(x, yy, z)];
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
        let interpolated = lerp(a, b, ty);

        let min_surface_dist = if is_entrance {
            let t = ((surface_height - y) as f32 / 14.0).clamp(0.0, 1.0);
            (1.0 + t * 7.0) as i32
        } else {
            11
        };
        if y >= surface_height - min_surface_dist {
            1.0
        } else {
            interpolated
        }
    }

    fn density_surface_height(column: &[f64; DENSITY_Y_SAMPLES]) -> i32 {
        // First search exact lattice endpoints. Once the highest positive
        // endpoint is found, only its following four-voxel cell can contain
        // the final positive density, because vertical interpolation is
        // linear. This preserves the old result without scanning every voxel.
        for iy in (0..DENSITY_Y_SAMPLES).rev() {
            if column[iy] <= 0.0 {
                continue;
            }

            let cell_bottom = iy as i32 * DENSITY_CELL_SIZE;
            if cell_bottom >= WORLD_HEIGHT {
                continue;
            }
            let cell_top = (cell_bottom + DENSITY_CELL_SIZE - 1).min(WORLD_HEIGHT - 1);
            for y in (cell_bottom.max(1)..=cell_top).rev() {
                if Self::sample_density_column(column, y) > 0.0 {
                    return y + 1;
                }
            }
        }
        1
    }

    // ── Block type assignment ─────────────────────────────────────────────── //

    fn snow_line(&self, world_x: i32, world_z: i32, biome: Biome) -> i32 {
        let local_variation = self
            .noise_detail
            .get_noise_2d(world_x as f32 + 7000.0, world_z as f32 - 7000.0);
        let offset = (local_variation * 8.0) as i32;

        match biome {
            Biome::Tundra => SEA_LEVEL + 8 + offset,
            Biome::Mountains => 118 + offset,
            Biome::Forest => 148 + offset,
            _ => 160 + offset,
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
        // The deepest surface layer is desert sand (12 blocks). Below all
        // biome layers the base material is always stone; bedrock was handled
        // above. Ore variants are still applied by the caller.
        if depth_from_surface > 12 {
            return BlockType::Stone;
        }
        let dirt_depth = 3 + (self.position_hash(world_x, world_z) % 3) as i32;
        let underwater_surface = y == surface_height - 1 && y < SEA_LEVEL;
        let is_surface = y == surface_height - 1;
        let high_snow =
            is_surface && !underwater_surface && y >= self.snow_line(world_x, world_z, biome);
        // Steeper cliff threshold keeps grass on rolling hills, stone on cliffs.
        let steep_surface = is_surface && slope >= 6 && !underwater_surface;
        let surface_hash = self.position_hash(world_x, world_z);

        match biome {
            Biome::Ocean | Biome::River | Biome::Lake => {
                if depth_from_surface > 6 {
                    BlockType::Stone
                } else if depth_from_surface > 3 {
                    BlockType::Gravel
                } else if biome == Biome::River && depth_from_surface <= 2 {
                    // Riverbeds mix sand and gravel like Minecraft.
                    if surface_hash % 3 == 0 {
                        BlockType::Gravel
                    } else {
                        BlockType::Sand
                    }
                } else {
                    BlockType::Sand
                }
            }
            Biome::Beach | Biome::Island => {
                if depth_from_surface > 8 {
                    BlockType::Stone
                } else if depth_from_surface > 0 {
                    BlockType::Sand
                } else if y == surface_height - 1 {
                    if underwater_surface {
                        BlockType::Sand
                    } else if biome == Biome::Island && y > SEA_LEVEL + 3 {
                        BlockType::Grass
                    } else if biome == Biome::Beach && y > SEA_LEVEL + 3 && surface_hash % 5 == 0 {
                        // Sparse grass just inland of the wet beach.
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
                    if underwater_surface {
                        BlockType::Gravel
                    } else if steep_surface && surface_hash % 3 == 0 {
                        BlockType::Stone
                    } else {
                        BlockType::Snow
                    }
                } else {
                    BlockType::Air
                }
            }
            Biome::Mountains => {
                if high_snow || y > 140 {
                    if y == surface_height - 1 {
                        BlockType::Snow
                    } else {
                        BlockType::Stone
                    }
                } else if y > 105 {
                    let hash = self.position_hash_3d(world_x, y, world_z);
                    if depth_from_surface <= 1 {
                        if slope >= 7 || hash % 3 == 0 {
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
                    } else if y <= SEA_LEVEL + 2 && surface_hash % 3 != 0 {
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
                    } else if steep_surface && slope >= 8 {
                        if surface_hash % 4 == 0 {
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
        if !(18..=78).contains(&y) {
            return block;
        }

        // Simple “veins” using existing block set:
        // - Gravel pockets: common-ish mid-depth (adds variety to mining)
        // - Clay pockets: rarer, a bit higher (keeps it special vs gravel)
        let n = self
            .noise_ore
            .get_noise_3d(world_x as f32, y as f32, world_z as f32);
        if n > 0.62 {
            return BlockType::Gravel;
        }

        if (38..=66).contains(&y) {
            let n2 = self.noise_ore.get_noise_3d(
                world_x as f32 + 200.0,
                y as f32 + 200.0,
                world_z as f32 + 200.0,
            );
            if n2 > 0.70 {
                return BlockType::Clay;
            }
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
        // Small trees have a two-block crown.  Reserving only that much space
        // removes the conspicuous six-block-wide vegetation-free bands where
        // adjacent chunks meet, while large trees are handled separately
        // below because their crowns need the original three-block margin.
        let margin = 2;

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
                                let is_large = hash % 7 == 0
                                    && matches!(biome, Biome::Forest | Biome::Swamp)
                                    && lx >= 3
                                    && lx < CHUNK_SIZE - 3
                                    && lz >= 3
                                    && lz < CHUNK_SIZE - 3;
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

                if biome == Biome::Mountains && height > 100 {
                    if hash % 100 < 10 {
                        let top = chunk.get_block(lx, height - 1, lz);
                        if matches!(top, BlockType::Stone | BlockType::Grass) {
                            chunk.set_block_raw(lx, height - 1, lz, BlockType::Gravel);
                        }
                    }
                }

                if biome == Biome::Mountains && height > 130 {
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

        let leaf_start = if is_large { 3 } else { 2 };
        let leaf_radius = if is_large { 3 } else { 2 };

        // A rounded, seed-varied crown reads much more like a living tree
        // than the former stack of identical square leaf layers.  The trunk
        // remains deterministic, so a column always generates identically
        // regardless of chunk-loading order.
        for dy in leaf_start..=trunk_height {
            let from_base = dy - leaf_start;
            let from_top = trunk_height - dy;
            let radius = if from_base == 0 || from_top == 0 {
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
                                let distance = dx.abs() + dz.abs();
                                let edge_hash =
                                    self.position_hash_3d(world_x + dx, ny, world_z + dz);
                                // Preserve a full inner crown; only its edge
                                // is broken up. Swamp trees retain a slightly
                                // wider, more ragged canopy.
                                let edge_limit = radius + if biome == Biome::Swamp { 1 } else { 0 };
                                let keep_leaf = distance <= edge_limit
                                    && (distance < edge_limit
                                        || edge_hash % if biome == Biome::Swamp { 4 } else { 3 }
                                            != 0);
                                if keep_leaf {
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
    fn lazy_cheese_density_matches_eager_noise_samples() {
        for seed in [0, 42, 2026, u32::MAX] {
            let generator = ChunkGenerator::new(seed);
            for (x, z) in [(0, 0), (-17, 29), (1024, -2048), (-4096, 256)] {
                for surface in [32, 64, 120, 200] {
                    for entrance in [false, true] {
                        for y in 0..=WORLD_HEIGHT {
                            let expected = if y <= 4 || y >= surface {
                                1.0
                            } else {
                                let c1 = generator.noise_cave1.get_noise_3d(
                                    x as f32 * 0.026,
                                    y as f32 * 0.009,
                                    z as f32 * 0.026,
                                );
                                let c2 = generator.noise_cave2.get_noise_3d(
                                    x as f32 * 0.018 + 400.0,
                                    y as f32 * 0.007 + 400.0,
                                    z as f32 * 0.018 + 400.0,
                                );
                                if generator.is_cave(x, y, z, surface, entrance) {
                                    -1.0
                                } else if c1 <= 0.0 {
                                    1.0
                                } else {
                                    ((0.22 - c1.max(0.0) * c2.max(0.0)) * 5.0).clamp(0.06, 1.0)
                                        as f64
                                }
                            };
                            assert_eq!(
                                generator.cave_density(x, y, z, surface, entrance).to_bits(),
                                expected.to_bits(),
                                "seed={seed}, ({x},{y},{z}), surface={surface}, entrance={entrance}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn generation_matches_minecraft_like_fixtures() {
        // FNV-1a hashes of every block, highest opaque Y, empty flags and
        // opaque flags after the Minecraft-like landform/cave retune.
        // Regenerate with: cargo run --release --example regen_worldgen_hashes
        let fixtures = [
            (0, 0x1d319d9bf7489c48u64),
            (12, 0x5d7d311b6d6db69f),
            (42, 0xe23b1f417b218a95),
            (77, 0x43f93746890344be),
            (1337, 0x5e2f0bf930023299),
            (2026, 0xfb5a9ba68438ff2d),
            (90_210, 0xf35fae6801a37cbd),
            (u32::MAX, 0xa1df56680b1a0b79),
        ];
        for (seed, expected) in fixtures {
            let generator = ChunkGenerator::new(seed);
            let mut hash = 0xcbf29ce484222325u64;
            let mut feed = |byte: u8| {
                hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
            };
            for (cx, cz) in [(0, 0), (2, -3), (-256, 128), (17, -91)] {
                let chunk = generator.generate_chunk(cx, cz);
                for x in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        for byte in chunk.highest_opaque_y(x, z).to_le_bytes() {
                            feed(byte);
                        }
                        for y in 0..WORLD_HEIGHT {
                            feed(chunk.get_block(x, y, z) as u8);
                        }
                    }
                }
                for subchunk in &chunk.subchunks {
                    feed(subchunk.is_empty as u8);
                    feed(subchunk.is_fully_opaque as u8);
                }
            }
            assert_eq!(hash, expected, "seed {seed}");
        }
    }

    #[test]
    fn density_bounds_preserve_all_voxel_signs_and_heights() {
        for seed in [0, 42, 2026, u32::MAX] {
            let generator = ChunkGenerator::new(seed);
            for (cx, cz) in [(0, 0), (2, -3), (-256, 128), (17, -91)] {
                let base_x = cx * CHUNK_SIZE;
                let base_z = cz * CHUNK_SIZE;
                let shapes = std::array::from_fn(|ix| {
                    std::array::from_fn(|iz| {
                        generator.terrain_shape(&generator.sample_terrain_noise(
                            base_x + ix as i32 * DENSITY_CELL_SIZE,
                            base_z + iz as i32 * DENSITY_CELL_SIZE,
                        ))
                    })
                });
                let actual = generator.build_density_lattice(base_x, base_z, &shapes);
                let mut reference = [0.0; DENSITY_LATTICE_LEN];
                for ix in 0..DENSITY_XZ_SAMPLES {
                    for iz in 0..DENSITY_XZ_SAMPLES {
                        for iy in 0..DENSITY_Y_SAMPLES {
                            reference[ChunkGenerator::density_index(ix, iy, iz)] = generator
                                .terrain_density(
                                    base_x + ix as i32 * DENSITY_CELL_SIZE,
                                    iy as i32 * DENSITY_CELL_SIZE,
                                    base_z + iz as i32 * DENSITY_CELL_SIZE,
                                    &shapes[ix][iz],
                                );
                        }
                    }
                }
                // Include boundary columns used by the cave lattice as well.
                for lx in 0..=CHUNK_SIZE {
                    for lz in 0..=CHUNK_SIZE {
                        let a = generator.density_vertical_column(&actual, lx, lz);
                        let b = generator.density_vertical_column(&reference, lx, lz);
                        assert_eq!(
                            ChunkGenerator::density_surface_height(&a),
                            ChunkGenerator::density_surface_height(&b)
                        );
                        for y in 0..WORLD_HEIGHT {
                            assert_eq!(
                                ChunkGenerator::sample_density_column(&a, y) > 0.0,
                                ChunkGenerator::sample_density_column(&b, y) > 0.0,
                                "seed {seed}, chunk ({cx}, {cz}), voxel ({lx}, {y}, {lz})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cave_columns_match_trilinear_reference_exactly() {
        let generator = ChunkGenerator::new(42);
        // Mixed signs and a nontrivial field exercise all interpolation weights.
        let lattice = std::array::from_fn(|i| ((i * 17 % 101) as f64 - 50.0) / 50.0);
        for surface in [1, 5, 18, 64, 97, WORLD_HEIGHT - 1, WORLD_HEIGHT] {
            for lx in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    let column = generator.cave_vertical_column(&lattice, lx, lz, surface);
                    for is_entrance in [false, true] {
                        for y in 0..WORLD_HEIGHT {
                            let actual = ChunkGenerator::sample_cave_column(
                                &column,
                                y,
                                surface,
                                is_entrance,
                            );
                            let expected = generator.sample_cave_lattice(
                                &lattice,
                                lx,
                                y,
                                lz,
                                surface,
                                is_entrance,
                            );
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_aquifer_matches_per_voxel_reference() {
        for seed in [42, 2026, u32::MAX] {
            let generator = ChunkGenerator::new(seed);
            for (x, z) in [(0, 0), (31, -17), (-4096, 2048)] {
                for surface in [32, SEA_LEVEL, 88, WORLD_HEIGHT - 1] {
                    let mut level = None;
                    for y in 0..WORLD_HEIGHT {
                        let n = generator
                            .noise_aquifer
                            .get_noise_3d(x as f32, y as f32, z as f32)
                            as f64;
                        let cell = generator.noise_aquifer.get_noise_3d(
                            x as f32 + 4000.0,
                            0.0,
                            z as f32 - 4000.0,
                        ) as f64;
                        let old_level = 28 + ((cell + 1.0) * 0.5 * 36.0) as i32;
                        let expected =
                            (surface < SEA_LEVEL && y < SEA_LEVEL) || (y <= old_level && n > 0.06);
                        assert_eq!(generator.has_fluid(x, y, z, surface, &mut level), expected);
                    }
                }
            }
        }
    }

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
