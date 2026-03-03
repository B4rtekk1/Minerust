//! Thread-safe chunk generation using FastNoiseLite
//!
//! This module provides fast, parallel terrain generation that can run
//! on background threads without blocking the main render loop.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;

/// Thread-safe chunk generator with pre-configured FastNoiseLite instances
pub struct ChunkGenerator {
    // Core terrain noises
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
    noise_erosion: FastNoiseLite,
    pub seed: u32,
}

impl ChunkGenerator {
    /// Create a new ChunkGenerator with the specified seed
    pub fn new(seed: u32) -> Self {
        ChunkGenerator {
            noise_continents: Self::create_noise(seed, 0.002),
            noise_terrain: Self::create_fbm_noise(seed.wrapping_add(1), 0.008),
            noise_detail: Self::create_fbm_noise(seed.wrapping_add(2), 0.015),
            noise_temperature: Self::create_noise(seed.wrapping_add(3), 0.008),
            noise_moisture: Self::create_noise(seed.wrapping_add(4), 0.01),
            noise_river: Self::create_noise(seed.wrapping_add(5), 0.06),
            noise_lake: Self::create_noise(seed.wrapping_add(6), 0.025),
            noise_trees: Self::create_noise(seed.wrapping_add(7), 0.1),
            noise_island: Self::create_noise(seed.wrapping_add(8), 0.05),
            noise_cave1: Self::create_3d_noise(seed.wrapping_add(9), 0.05),
            noise_cave2: Self::create_3d_noise(seed.wrapping_add(10), 0.035),
            noise_erosion: Self::create_fbm_noise(seed.wrapping_add(12), 0.005),
            seed,
        }
    }

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
        noise.set_fractal_octaves(Some(4));
        noise.set_fractal_lacunarity(Some(2.0));
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

    /// Generate a complete chunk at the given coordinates
    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cz);
        let base_x = cx * CHUNK_SIZE;
        let base_z = cz * CHUNK_SIZE;

        // Pre-compute biome and height maps using FastNoiseLite
        // Fused loop for better cache locality: compute both in single pass
        let mut biome_map = [[Biome::Plains; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        let mut height_map = [[0i32; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let lx_usize = lx as usize;
                let lz_usize = lz as usize;
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                // Compute both biome and height in same cache line iteration
                biome_map[lx_usize][lz_usize] = self.get_biome(world_x, world_z);
                height_map[lx_usize][lz_usize] = self.get_terrain_height(world_x, world_z);
            }
        }

        // Terrain generation pass
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome = biome_map[lx as usize][lz as usize];
                let surface_height = height_map[lx as usize][lz as usize];

                let max_y = if matches!(biome, Biome::Mountains | Biome::Island) {
                    WORLD_HEIGHT - 20
                } else {
                    (surface_height + 5).max(SEA_LEVEL)
                };

                for y in 0..max_y {
                    let mut is_solid = y < surface_height;

                    if matches!(biome, Biome::Mountains | Biome::Island) && y >= surface_height - 8
                    {
                        let density =
                            self.get_3d_density(world_x, y, world_z, biome, surface_height);
                        if density > 0.0 {
                            is_solid = true;
                        }
                    }

                    if is_solid {
                        let block =
                            self.get_block_for_biome(biome, y, surface_height, world_x, world_z);
                        if block != BlockType::Air {
                            chunk.set_block(lx, y, lz, block);
                        }
                    } else if y >= surface_height && y < SEA_LEVEL {
                        if biome == Biome::Tundra && y == SEA_LEVEL - 1 {
                            chunk.set_block(lx, y, lz, BlockType::Ice);
                        } else {
                            chunk.set_block(lx, y, lz, BlockType::Water);
                        }
                    }
                }
            }
        }

        // Pre-compute cave entrance map to avoid redundant noise evaluation.
        // is_cave_entrance() executes up to 30 3D noise calls per column;
        // computing it once per (lx,lz) column instead of for every Y-block
        // gives ~256x speedup for the cave carving pass.
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

        // Cave carving pass
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let height = height_map[lx as usize][lz as usize];
                // Reuse pre-computed entrance result – no extra noise calls per Y
                let is_entrance = cave_entrance_map[lx as usize][lz as usize];

                for y in 1..height.min(WORLD_HEIGHT - 1) {
                    if self.is_cave(world_x, y, world_z, height, is_entrance) {
                        let current = chunk.get_block(lx, y, lz);
                        if current != BlockType::Water
                            && current != BlockType::Bedrock
                            && current != BlockType::Air
                        {
                            if y < SEA_LEVEL {
                                chunk.set_block(lx, y, lz, BlockType::Water);
                            } else {
                                chunk.set_block(lx, y, lz, BlockType::Air);
                            }
                        }
                    }
                }
            }
        }

        // Decorations pass
        self.generate_decorations(&mut chunk, cx, cz, &biome_map, &height_map);

        // Mark empty and fully opaque subchunks
        for subchunk in &mut chunk.subchunks {
            subchunk.check_empty();
            subchunk.check_fully_opaque();
        }

        chunk
    }

    /// Public proxy for World::get_terrain_height delegation.
    pub fn get_terrain_height_pub(&self, x: i32, z: i32) -> i32 {
        self.get_terrain_height(x, z)
    }

    /// Public proxy for World::is_cave_entrance delegation.
    pub fn is_cave_entrance_pub(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.is_cave_entrance(x, z, surface_height)
    }

    /// Public proxy for World::position_hash delegation.
    pub fn position_hash_pub(&self, x: i32, z: i32) -> u32 {
        self.position_hash(x, z)
    }

    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        let fx = x as f32;
        let fz = z as f32;

        let continent = self.noise_continents.get_noise_2d(fx * 0.002, fz * 0.002);
        let river_noise = self.noise_river.get_noise_2d(fx * 0.06, fz * 0.06);
        let river_value = 1.0 - river_noise.abs() * 1.5;

        let lake_noise = self.noise_lake.get_noise_2d(fx * 0.025, fz * 0.025);

        if river_value > 0.85 && continent > -0.3 {
            return Biome::River;
        }

        if lake_noise < -0.6 && continent > -0.2 {
            return Biome::Lake;
        }

        if continent < -0.35 {
            let island_noise = self.noise_island.get_noise_2d(fx * 0.05, fz * 0.05);
            if island_noise > 0.65 {
                return Biome::Island;
            }
            return Biome::Ocean;
        }

        if continent < -0.2 {
            return Biome::Beach;
        }

        let temp = self.noise_temperature.get_noise_2d(fx * 0.008, fz * 0.008);
        let moist = self.noise_moisture.get_noise_2d(fx * 0.01, fz * 0.01);
        let terrain_var = self.noise_terrain.get_noise_2d(fx * 0.005, fz * 0.005);
        let erosion = self.noise_erosion.get_noise_2d(fx * 0.004, fz * 0.004);

        // Mountains check - more common with varied terrain
        if terrain_var > 0.25 && erosion < 0.2 {
            return Biome::Mountains;
        }

        // Temperature-based biomes with more variety
        if temp < -0.25 {
            return Biome::Tundra;
        }

        if temp > 0.45 {
            if moist < -0.15 {
                return Biome::Desert;
            }
            // Hot + moist = fewer plains, more swamp
            if moist > 0.1 {
                return Biome::Swamp;
            }
        }

        // Forest is now more common (larger moisture range)
        if moist > -0.1 {
            return Biome::Forest;
        }

        // Swamp in wet areas
        if moist > 0.35 {
            return Biome::Swamp;
        }

        // Hills/varied terrain instead of flat plains
        if terrain_var > 0.1 || erosion < -0.1 {
            return Biome::Forest; // More interesting than plains
        }

        Biome::Plains
    }

    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        // blend_radius=1 means a 3×3 neighbourhood — compute the central biome once
        // and reuse it for all 9 sample points to avoid 9 redundant get_biome() calls
        // (each of which does 6+ noise evaluations).
        let blend_radius = 1;
        let center_biome = self.get_biome(x, z);
        let mut total_height = 0.0;
        let mut weights = 0.0;

        for dx in -blend_radius..=blend_radius {
            for dz in -blend_radius..=blend_radius {
                let wx = x + dx;
                let wz = z + dz;
                let dist_sq = (dx * dx + dz * dz) as f64;
                let weight = 1.0 / (1.0 + dist_sq);

                let height = self.calculate_base_height_with_biome(wx, wz, center_biome);
                total_height += height * weight;
                weights += weight;
            }
        }

        let base_height = total_height / weights;
        (base_height as i32).clamp(1, WORLD_HEIGHT - 20)
    }

    #[allow(dead_code)]
    fn calculate_base_height(&self, x: i32, z: i32) -> f64 {
        let biome = self.get_biome(x, z);
        self.calculate_base_height_with_biome(x, z, biome)
    }

    fn calculate_base_height_with_biome(&self, x: i32, z: i32, biome: Biome) -> f64 {
        let fx = x as f32;
        let fz = z as f32;

        // Use FastNoiseLite's built-in FBm
        let continental = self.noise_continents.get_noise_2d(fx, fz) as f64;
        let terrain = self.noise_terrain.get_noise_2d(fx, fz) as f64;
        let detail = self.noise_detail.get_noise_2d(fx, fz) as f64;
        let erosion = self.noise_erosion.get_noise_2d(fx, fz) as f64;

        // Additional micro-detail for more natural look
        let micro_detail = self.noise_detail.get_noise_2d(fx * 0.05, fz * 0.05) as f64 * 1.5;

        match biome {
            Biome::Ocean => {
                let depth = (continental + 1.0) * 0.5 * 15.0 + 35.0;
                depth + detail * 3.0 + micro_detail
            }
            Biome::River => (SEA_LEVEL - 3) as f64 + detail * 2.0 + micro_detail * 0.5,
            Biome::Lake => (SEA_LEVEL - 4) as f64 + detail * 2.0 + micro_detail * 0.5,
            Biome::Beach => SEA_LEVEL as f64 + terrain * 3.0 + detail * 1.5 + micro_detail,
            Biome::Island => {
                let island_noise = self.noise_island.get_noise_2d(fx * 0.05, fz * 0.05) as f64;
                let island_height = (island_noise + 1.0) * 0.5 * 25.0;
                (SEA_LEVEL as f64 + island_height + detail * 3.0 + micro_detail)
                    .max(SEA_LEVEL as f64 - 5.0)
            }
            Biome::Plains => {
                // More varied plains with gentle rolling hills
                let rolling = self.noise_terrain.get_noise_2d(fx * 0.015, fz * 0.015) as f64;
                let flatness = 1.0 - erosion.abs() * 0.3;
                66.0 + terrain * 6.0 * flatness + rolling * 4.0 + detail * 2.5 + micro_detail
            }
            Biome::Forest => {
                // Hillier forest terrain
                let hills = self.noise_terrain.get_noise_2d(fx * 0.012, fz * 0.012) as f64;
                68.0 + terrain * 10.0 + hills * 6.0 + detail * 4.0 + micro_detail
            }
            Biome::Desert => {
                let dune = self.noise_detail.get_noise_2d(fx * 0.02, fz * 0.02) as f64;
                let dune_height = (dune + 1.0) * 0.5 * 10.0;
                65.0 + terrain * 6.0 + dune_height + detail * 3.0 + micro_detail
            }
            Biome::Tundra => {
                // More varied tundra with frozen hills
                let frozen_hills = self.noise_terrain.get_noise_2d(fx * 0.01, fz * 0.01) as f64;
                68.0 + terrain * 8.0 + frozen_hills * 5.0 + detail * 3.0 + micro_detail
            }
            Biome::Mountains => {
                let peaks = self.noise_terrain.get_noise_2d(fx + 1000.0, fz + 1000.0) as f64;
                let mountain_height = (terrain + 1.0) * 0.5 * 40.0;
                let peak_factor = (peaks + 1.0) * 0.5;
                let cliff_noise = self.noise_detail.get_noise_2d(fx * 0.03, fz * 0.03) as f64;
                85.0 + mountain_height * (0.5 + peak_factor * 0.5)
                    + cliff_noise * 4.0
                    + detail * 4.0
            }
            Biome::Swamp => {
                // Lumpy swamp terrain
                let lumps = self.noise_detail.get_noise_2d(fx * 0.04, fz * 0.04) as f64;
                SEA_LEVEL as f64 + 1.0 + terrain * 3.0 + lumps * 2.0 + detail + micro_detail * 0.5
            }
        }
    }

    /// Cave check with pre-computed entrance flag (avoids repeated is_cave_entrance noise calls).
    /// The `is_entrance` parameter should be pre-computed once per (x,z) column.
    fn is_cave(&self, x: i32, y: i32, z: i32, surface_height: i32, is_entrance: bool) -> bool {
        if y <= 5 {
            return false;
        }

        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        // Gradual entrance transition - caves can get closer to surface near entrances
        let entrance_gradient = if is_entrance {
            // Smooth gradient from surface into cave
            let dist_from_surface = (surface_height - y) as f32;
            (dist_from_surface / 5.0).min(1.0).max(0.0)
        } else {
            0.0
        };

        let min_surface_distance = if is_entrance {
            (2.0 - entrance_gradient * 2.0) as i32
        } else {
            6
        };

        if y >= surface_height - min_surface_distance {
            return false;
        }

        // 3D cave noise using FastNoiseLite
        let cave1 = self
            .noise_cave1
            .get_noise_3d(fx * 0.05, fy * 0.025, fz * 0.05);
        let cave2 = self
            .noise_cave2
            .get_noise_3d(fx * 0.035, fy * 0.02, fz * 0.035);

        // Lower threshold near entrances for smoother transitions
        let cheese_threshold = if is_entrance { 0.6 } else { 0.68 };
        let is_cheese_cave = cave1 > cheese_threshold && cave2 > cheese_threshold;

        let spag1 = self
            .noise_cave1
            .get_noise_3d(fx * 0.08 + 500.0, fy * 0.08, fz * 0.08);
        let spag2 = self
            .noise_cave2
            .get_noise_3d(fx * 0.08 + 500.0, fy * 0.08, fz * 0.08);
        let spaghetti_threshold = 0.88;
        let is_spaghetti_cave =
            spag1.abs() < (1.0 - spaghetti_threshold) && spag2.abs() < (1.0 - spaghetti_threshold);

        let depth_factor = if y < 30 {
            1.0
        } else if y < 50 {
            0.85
        } else {
            0.6
        };

        (is_cheese_cave || is_spaghetti_cave)
            && (self.position_hash_3d(x, y, z) % 100) as f32 / 100.0 < depth_factor
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 3 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        // More natural entrance detection using terrain slope
        let entrance_noise = self
            .noise_cave1
            .get_noise_2d(fx * 0.015 + 1000.0, fz * 0.015 + 1000.0);

        // Check for natural slope/hillside (good for cave openings)
        let terrain_slope = self.noise_terrain.get_noise_2d(fx * 0.02, fz * 0.02).abs();
        let is_hillside = terrain_slope > 0.2;

        // Lower threshold for hillsides (natural cave entrance locations)
        let threshold = if is_hillside { 0.7 } else { 0.85 };
        if entrance_noise < threshold {
            return false;
        }

        let hash = self.position_hash(x, z);
        // More entrances on hillsides
        let entrance_chance = if is_hillside { 5 } else { 12 };
        if hash % entrance_chance != 0 {
            return false;
        }

        // Check deeper for cave connection
        for check_y in (surface_height - 35).max(10)..=(surface_height - 5) {
            let fy = check_y as f32;
            let cave1 = self
                .noise_cave1
                .get_noise_3d(fx * 0.05, fy * 0.025, fz * 0.05);
            let cave2 = self
                .noise_cave2
                .get_noise_3d(fx * 0.035, fy * 0.02, fz * 0.035);
            if cave1 > 0.65 && cave2 > 0.65 {
                return true;
            }
        }

        false
    }

    fn get_3d_density(&self, x: i32, y: i32, z: i32, biome: Biome, surface_height: i32) -> f64 {
        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        let vertical_gradient = (surface_height as f64 - y as f64) / 8.0;

        let density_noise = match biome {
            Biome::Mountains => {
                let terrain = self.noise_terrain.get_noise_2d(fx * 0.02, fz * 0.02) as f64 * 0.5;
                let detail =
                    self.noise_detail
                        .get_noise_3d(fx * 0.04, fy * 0.04, fz * 0.04) as f64
                        * 0.5;
                terrain + detail
            }
            Biome::Island => {
                self.noise_terrain
                    .get_noise_3d(fx * 0.03, fy * 0.03, fz * 0.03) as f64
                    * 0.4
            }
            _ => 0.0,
        };

        vertical_gradient + density_noise
    }

    fn get_block_for_biome(
        &self,
        biome: Biome,
        y: i32,
        surface_height: i32,
        world_x: i32,
        world_z: i32,
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

        let depth_from_surface = surface_height - y;
        let dirt_depth = 3 + (self.position_hash(world_x, world_z) % 3) as i32;

        match biome {
            Biome::Ocean | Biome::River | Biome::Lake => {
                if depth_from_surface > 4 {
                    BlockType::Stone
                } else if depth_from_surface > 1 {
                    BlockType::Gravel
                } else if y < surface_height {
                    BlockType::Sand
                } else {
                    BlockType::Air
                }
            }
            Biome::Beach | Biome::Island => {
                if depth_from_surface > 6 {
                    BlockType::Stone
                } else if depth_from_surface > 0 {
                    BlockType::Sand
                } else if y == surface_height - 1 {
                    if biome == Biome::Island && y > SEA_LEVEL {
                        BlockType::Grass
                    } else {
                        BlockType::Sand
                    }
                } else {
                    BlockType::Air
                }
            }
            Biome::Desert => {
                if depth_from_surface > 10 {
                    BlockType::Stone
                } else if depth_from_surface > 0 {
                    BlockType::Sand
                } else if y == surface_height - 1 {
                    BlockType::Sand
                } else {
                    BlockType::Air
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
                if y > 140 {
                    if y == surface_height - 1 {
                        BlockType::Snow
                    } else if depth_from_surface > 0 {
                        BlockType::Stone
                    } else {
                        BlockType::Air
                    }
                } else if y > 110 {
                    if depth_from_surface > 2 {
                        BlockType::Stone
                    } else if y == surface_height - 1 {
                        BlockType::Grass
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
                    if y <= SEA_LEVEL {
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
        let margin = 4;

        for lx in margin..(CHUNK_SIZE - margin) {
            for lz in margin..(CHUNK_SIZE - margin) {
                let world_x = base_x + lx;
                let world_z = base_z + lz;
                let biome = biome_map[lx as usize][lz as usize];
                let height = height_map[lx as usize][lz as usize];

                if height <= SEA_LEVEL {
                    continue;
                }

                // Tree generation
                if biome.has_trees() {
                    let tree_noise = self
                        .noise_trees
                        .get_noise_2d(world_x as f32, world_z as f32);
                    let density_threshold = biome.tree_density() as f32;

                    if tree_noise > density_threshold {
                        let hash = self.position_hash(world_x, world_z);
                        if hash % 100 < 15 {
                            let ground = chunk.get_block(lx, height - 1, lz);
                            if matches!(ground, BlockType::Grass | BlockType::Dirt) {
                                let is_large =
                                    hash % 7 == 0 && matches!(biome, Biome::Forest | Biome::Swamp);
                                if self.can_place_tree(chunk, lx, height, lz, is_large) {
                                    self.place_tree(chunk, lx, height, lz, biome, is_large);
                                }
                            }
                        }
                    }
                }

                // Cactus in desert
                if biome == Biome::Desert {
                    let hash = self.position_hash(world_x, world_z);
                    if hash % 100 < 2 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand {
                            self.place_cactus(chunk, lx, height, lz);
                        }
                    }
                }
            }
        }
    }

    fn can_place_tree(&self, chunk: &Chunk, lx: i32, y: i32, lz: i32, is_large: bool) -> bool {
        // Check ground is valid (Grass or Dirt only)
        let ground_block = chunk.get_block(lx, y - 1, lz);
        if !matches!(ground_block, BlockType::Grass | BlockType::Dirt) {
            return false;
        }

        // Check neighbors don't have invalid blocks (Stone, Gravel, Sand, Water, Ice)
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

        // Check no existing trees nearby
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
                    if chunk.get_block(check_x, check_y, check_z) == BlockType::Wood {
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
        _biome: Biome,
        is_large: bool,
    ) {
        let trunk_height = if is_large { 8 } else { 5 };

        // Convert grass to dirt under the tree trunk
        if chunk.get_block(lx, y - 1, lz) == BlockType::Grass {
            chunk.set_block(lx, y - 1, lz, BlockType::Dirt);
        }

        // Trunk
        for dy in 0..trunk_height {
            chunk.set_block(lx, y + dy, lz, BlockType::Wood);
        }

        // Leaves
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
                            // Only place leaves on air or other leaves
                            if existing == BlockType::Air || existing == BlockType::Leaves {
                                // Skip corners for more natural shape
                                if dx.abs() != radius
                                    || dz.abs() != radius
                                    || (self.position_hash(nx, nz) % 2 == 0)
                                {
                                    chunk.set_block(nx, ny, nz, BlockType::Leaves);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Top leaves
        let top_y = y + trunk_height;
        if top_y < WORLD_HEIGHT {
            let existing = chunk.get_block(lx, top_y, lz);
            if existing == BlockType::Air || existing == BlockType::Leaves {
                chunk.set_block(lx, top_y, lz, BlockType::Leaves);
            }
        }
    }

    fn place_cactus(&self, chunk: &mut Chunk, lx: i32, y: i32, lz: i32) {
        let height = 2 + (self.position_hash(lx, lz) % 2) as i32;
        for dy in 0..height {
            if y + dy < WORLD_HEIGHT {
                chunk.set_block(lx, y + dy, lz, BlockType::Cactus);
            }
        }
    }

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

// Allow cloning for worker threads
impl Clone for ChunkGenerator {
    fn clone(&self) -> Self {
        ChunkGenerator::new(self.seed)
    }
}
