//! Thread-safe chunk generation using FastNoiseLite
//!
//! This module provides fast, parallel terrain generation that can run
//! on background threads without blocking the main render loop.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

use crate::constants::*;
use crate::core::biome::Biome;
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;
use crate::world::spline::TerrainSpline;

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
    /// Third cave noise — spaghetti tunnel axis B
    noise_cave3: FastNoiseLite,
    noise_erosion: FastNoiseLite,
    /// Domain warp noises for organic terrain deformation
    noise_warp_x: FastNoiseLite,
    noise_warp_z: FastNoiseLite,
    /// Ridged noise for mountain peaks
    noise_ridged: FastNoiseLite,
    /// Peaks & valleys noise
    noise_pv: FastNoiseLite,
    /// Extra decoration noise (flowers, rocks, dead bushes)
    noise_decor: FastNoiseLite,
    /// Cave domain-warp noises (organic, non-repeating shapes)
    noise_cave_warp_x: FastNoiseLite,
    noise_cave_warp_z: FastNoiseLite,
    pub seed: u32,
}

impl ChunkGenerator {
    /// Create a new ChunkGenerator with the specified seed
    pub fn new(seed: u32) -> Self {
        ChunkGenerator {
            noise_continents: Self::create_fbm_noise(seed, 0.0018),
            noise_terrain: Self::create_fbm_noise(seed.wrapping_add(1), 0.007),
            noise_detail: Self::create_fbm_noise(seed.wrapping_add(2), 0.018),
            noise_temperature: Self::create_noise(seed.wrapping_add(3), 0.006),
            noise_moisture: Self::create_noise(seed.wrapping_add(4), 0.008),
            noise_river: Self::create_noise(seed.wrapping_add(5), 0.055),
            noise_lake: Self::create_noise(seed.wrapping_add(6), 0.022),
            noise_trees: Self::create_noise(seed.wrapping_add(7), 0.12),
            noise_island: Self::create_noise(seed.wrapping_add(8), 0.045),
            noise_cave1: Self::create_3d_noise(seed.wrapping_add(9), 0.045),
            noise_cave2: Self::create_3d_noise(seed.wrapping_add(10), 0.032),
            noise_cave3: Self::create_3d_noise(seed.wrapping_add(11), 0.038),
            noise_erosion: Self::create_fbm_noise(seed.wrapping_add(12), 0.004),
            noise_warp_x: Self::create_fbm_noise(seed.wrapping_add(20), 0.005),
            noise_warp_z: Self::create_fbm_noise(seed.wrapping_add(21), 0.005),
            noise_ridged: Self::create_ridged_noise(seed.wrapping_add(22), 0.009),
            noise_pv: Self::create_fbm_noise(seed.wrapping_add(23), 0.004),
            noise_decor: Self::create_noise(seed.wrapping_add(24), 0.15),
            // Cave domain-warp: low-frequency FBm for organic cave shapes
            noise_cave_warp_x: Self::create_fbm_noise(seed.wrapping_add(30), 0.018),
            noise_cave_warp_z: Self::create_fbm_noise(seed.wrapping_add(31), 0.018),
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

        // Pre-compute cave entrance map
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

        // Domain warp to make biome borders more organic
        let warp_scale = 80.0_f32;
        let wx = fx + self.noise_warp_x.get_noise_2d(fx * 0.004, fz * 0.004) * warp_scale;
        let wz = fz + self.noise_warp_z.get_noise_2d(fx * 0.004 + 100.0, fz * 0.004 + 100.0) * warp_scale;

        let continent = self.noise_continents.get_noise_2d(wx * 0.0018, wz * 0.0018);
        let river_noise = self.noise_river.get_noise_2d(wx * 0.055, wz * 0.055);
        let river_value = 1.0 - river_noise.abs() * 2.0;

        let lake_noise = self.noise_lake.get_noise_2d(wx * 0.022, wz * 0.022);

        if river_value > 0.88 && continent > -0.25 {
            return Biome::River;
        }

        if lake_noise < -0.62 && continent > -0.15 {
            return Biome::Lake;
        }

        if continent < -0.38 {
            let island_noise = self.noise_island.get_noise_2d(wx * 0.045, wz * 0.045);
            if island_noise > 0.60 {
                return Biome::Island;
            }
            return Biome::Ocean;
        }

        if continent < -0.22 {
            return Biome::Beach;
        }

        let temp = self.noise_temperature.get_noise_2d(wx * 0.006, wz * 0.006);
        let moist = self.noise_moisture.get_noise_2d(wx * 0.008, wz * 0.008);
        let erosion = self.noise_erosion.get_noise_2d(wx * 0.004, wz * 0.004);
        let pv = self.noise_pv.get_noise_2d(wx * 0.004, wz * 0.004);

        // Mountains: high peaks & valleys value + low erosion
        if pv > 0.3 && erosion < 0.25 && continent > 0.0 {
            return Biome::Mountains;
        }

        // Very cold → Tundra
        if temp < -0.3 {
            return Biome::Tundra;
        }

        // Hot
        if temp > 0.4 {
            if moist < -0.2 {
                return Biome::Desert;
            }
            if moist > 0.15 {
                return Biome::Swamp;
            }
        }

        // Wet mid-temp → Swamp
        if moist > 0.45 && temp > -0.1 {
            return Biome::Swamp;
        }

        // Moderate moisture → Forest
        if moist > -0.05 {
            return Biome::Forest;
        }

        Biome::Plains
    }

    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        // Wider 5×5 biome blending for smoother transitions
        let blend_radius = 2i32;
        let center_biome = self.get_biome(x, z);
        let mut total_height = 0.0;
        let mut weights = 0.0;

        for dx in -blend_radius..=blend_radius {
            for dz in -blend_radius..=blend_radius {
                let wx = x + dx;
                let wz = z + dz;
                let dist_sq = (dx * dx + dz * dz) as f64;
                // Gaussian-like weighting
                let weight = (-dist_sq / (blend_radius as f64 * 1.5)).exp();

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

        // --- Domain warp for organic terrain ---
        let warp_scale = 60.0_f32;
        let wx = fx + self.noise_warp_x.get_noise_2d(fx * 0.005, fz * 0.005) * warp_scale;
        let wz = fz + self.noise_warp_z.get_noise_2d(fx * 0.005 + 200.0, fz * 0.005 + 200.0) * warp_scale;

        let continental = self.noise_continents.get_noise_2d(wx, wz) as f64;
        let terrain = self.noise_terrain.get_noise_2d(wx, wz) as f64;
        let detail = self.noise_detail.get_noise_2d(wx, wz) as f64;
        let erosion = self.noise_erosion.get_noise_2d(wx, wz) as f64;
        let ridged = self.noise_ridged.get_noise_2d(wx, wz) as f64;
        let pv = self.noise_pv.get_noise_2d(wx, wz) as f64;

        // Spline-mapped continentalness for realistic ocean-to-mountain transition
        let cont_spline = TerrainSpline::continental();
        let cont_height = cont_spline.sample(continental);

        // Erosion modulates terrain roughness
        let erosion_spline = TerrainSpline::erosion();
        let erosion_mult = erosion_spline.sample(erosion);

        // Peaks & valleys offset
        let pv_spline = TerrainSpline::peaks_valleys();
        let pv_offset = pv_spline.sample(pv);

        match biome {
            Biome::Ocean => {
                let depth = 20.0 + (continental + 1.0) * 0.5 * 18.0;
                depth + detail * 2.5
            }
            Biome::River => (SEA_LEVEL - 2) as f64 + detail * 1.5,
            Biome::Lake => (SEA_LEVEL - 5) as f64 + detail * 2.0,
            Biome::Beach => {
                SEA_LEVEL as f64 + terrain * 3.5 * erosion_mult + detail * 1.5
            }
            Biome::Island => {
                let island_noise = self.noise_island.get_noise_2d(wx * 0.045, wz * 0.045) as f64;
                let island_h = (island_noise + 1.0) * 0.5 * 28.0;
                (SEA_LEVEL as f64 + island_h + terrain * 4.0 * erosion_mult + detail * 3.0)
                    .max(SEA_LEVEL as f64 - 3.0)
            }
            Biome::Plains => {
                // Gentle rolling hills using spline base + small terrain noise
                let rolling = self.noise_terrain.get_noise_2d(wx * 0.012, wz * 0.012) as f64;
                cont_height.max(66.0) + terrain * 5.0 * erosion_mult + rolling * 3.5 + detail * 2.0
            }
            Biome::Forest => {
                let hills = self.noise_terrain.get_noise_2d(wx * 0.010, wz * 0.010) as f64;
                cont_height.max(67.0) + terrain * 9.0 * erosion_mult + hills * 7.0 + detail * 4.0
            }
            Biome::Desert => {
                let dune = self.noise_detail.get_noise_2d(wx * 0.022, wz * 0.022) as f64;
                let dune_h = (dune + 1.0) * 0.5 * 12.0;
                62.0 + terrain * 7.0 * erosion_mult + dune_h + detail * 3.0
            }
            Biome::Tundra => {
                let frozen = self.noise_terrain.get_noise_2d(wx * 0.009, wz * 0.009) as f64;
                66.0 + terrain * 9.0 * erosion_mult + frozen * 6.0 + detail * 3.5
            }
            Biome::Mountains => {
                // Ridged noise for sharp peaks; pv_offset for dramatic valleys
                let ridge_strength = ((ridged + 1.0) * 0.5).powf(1.8) * 80.0;
                let base = cont_height.max(80.0);
                base + ridge_strength + pv_offset.max(0.0) * 0.6
                    + terrain * 12.0 * erosion_mult
                    + detail * 5.0
            }
            Biome::Swamp => {
                let lumps = self.noise_detail.get_noise_2d(wx * 0.035, wz * 0.035) as f64;
                SEA_LEVEL as f64 + 1.5 + terrain * 2.5 * erosion_mult + lumps * 2.5 + detail * 1.0
            }
        }
    }

    /// Cave check with pre-computed entrance flag.
    ///
    /// Uses smooth distance-based metrics instead of hard thresholds to produce
    /// rounded, organic caverns without jagged edges:
    ///
    ///  - **Cheese caves**: treat two noise fields as XZ/XY pseudodistances and
    ///    carve when their *product* exceeds a threshold — gives smooth ellipsoidal
    ///    chambers.
    ///  - **Spaghetti tunnels**: treat (n1, n2) as a 2-D displacement from a
    ///    virtual tunnel axis; carve when `sqrt(n1²+n2²) < radius` — produces
    ///    circular cross-sections.
    ///  - **Domain warping**: offset the sample coordinates with a separate FBm
    ///    noise before querying cave noises — prevents grid-aligned repetition and
    ///    adds natural meanders.
    ///  - **Depth gradient**: deterministic smooth falloff near the surface instead
    ///    of per-voxel random jitter.
    fn is_cave(&self, x: i32, y: i32, z: i32, surface_height: i32, is_entrance: bool) -> bool {
        if y <= 5 {
            return false;
        }

        // --- Surface proximity guard ---
        let min_surface_dist = if is_entrance {
            // Blend from 3 (at surface) to 7 (deep) so the opening looks natural
            let t = ((surface_height - y) as f32 / 8.0).clamp(0.0, 1.0);
            (3.0 + t * 4.0) as i32
        } else {
            7
        };
        if y >= surface_height - min_surface_dist {
            return false;
        }

        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        // --- Domain warp: shift (fx, fz) with low-frequency FBm ---
        // Amplitude ~20 blocks gives organic, non-repeating cave shapes.
        let warp_amp = 20.0_f32;
        let wx = fx + self.noise_cave_warp_x.get_noise_3d(fx * 0.022, fy * 0.014, fz * 0.022) * warp_amp;
        let wy = fy + self.noise_cave_warp_z.get_noise_3d(fx * 0.022 + 100.0, fy * 0.014, fz * 0.022 + 100.0) * warp_amp * 0.4;
        let wz = fz + self.noise_cave_warp_x.get_noise_3d(fx * 0.022 + 200.0, fy * 0.014, fz * 0.022 + 200.0) * warp_amp;

        // --- Depth factor: smooth gradient, fully deterministic ---
        // More caves deep down, fewer near sea-level.
        let depth_factor = if y < 15 {
            1.0_f32
        } else if y < 40 {
            1.0 - (y as f32 - 15.0) / 60.0   // ~0.58 at y=40
        } else {
            0.58 - (y as f32 - 40.0) / 180.0  // ~0.36 at y=100
        }.max(0.25);

        // ── CHEESE CAVES ─────────────────────────────────────────────────────
        // Sample two independent noise fields with the warped coordinates.
        // Using low-frequency noise (longer horizontal scale than vertical)
        // gives squashed, room-like chambers.
        let c1 = self.noise_cave1.get_noise_3d(wx * 0.038, wy * 0.016, wz * 0.038);
        let c2 = self.noise_cave2.get_noise_3d(wx * 0.028 + 400.0, wy * 0.013 + 400.0, wz * 0.028 + 400.0);

        // Carve when *both* fields are positive and their product is large.
        // This is equivalent to carving the intersection of two smooth blobs —
        // the result is rounded chambers without jagged noise spikes.
        let cheese_product = c1.max(0.0) * c2.max(0.0);
        // Scale threshold by depth: deeper → bigger chambers allowed.
        let cheese_threshold = 0.18 * (1.0 - depth_factor * 0.25);
        let is_cheese = cheese_product > cheese_threshold;

        // ── SPAGHETTI TUNNELS ─────────────────────────────────────────────────
        // Model a tunnel as a 1-D curve through space.  We sample two *different*
        // noise fields (s1, s2) and interpret them as offsets in a 2-D plane
        // perpendicular to the tunnel direction.  Distance from the axis is
        // sqrt(s1² + s2²); carve when that distance is below the tunnel radius.
        //
        // Using *different* frequencies for the two components gives oval rather
        // than perfectly circular tunnels, which looks more natural.
        let s1 = self.noise_cave1.get_noise_3d(wx * 0.065 + 500.0, wy * 0.055, wz * 0.065);
        let s2 = self.noise_cave3.get_noise_3d(wx * 0.065 + 900.0, wy * 0.055, wz * 0.065);
        let tunnel_dist = (s1 * s1 + s2 * s2).sqrt();
        // Base radius: 0.10 gives ~2-block-wide tunnels; scale with depth.
        let spag_radius = (0.09 + depth_factor * 0.04).min(0.14);
        let is_spaghetti = tunnel_dist < spag_radius;

        // ── LARGE WORM TUNNELS ────────────────────────────────────────────────
        // A second, wider tunnel type that snakes more slowly through the world.
        // Gives Minecraft-style "ravine-like" deep passages.
        let w1 = self.noise_cave2.get_noise_3d(wx * 0.040 + 800.0, wy * 0.032, wz * 0.040);
        let w2 = self.noise_cave3.get_noise_3d(wx * 0.040 + 1200.0, wy * 0.032, wz * 0.040);
        let worm_dist = (w1 * w1 + w2 * w2).sqrt();
        let worm_radius = (0.06 + depth_factor * 0.03).min(0.10);
        let is_worm = worm_dist < worm_radius && y < 55;

        is_cheese || is_spaghetti || is_worm
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface_height: i32) -> bool {
        if surface_height <= SEA_LEVEL + 2 {
            return false;
        }

        let fx = x as f32;
        let fz = z as f32;

        let entrance_noise = self
            .noise_cave1
            .get_noise_2d(fx * 0.014 + 1000.0, fz * 0.014 + 1000.0);

        let terrain_slope = self.noise_terrain.get_noise_2d(fx * 0.018, fz * 0.018).abs();
        let is_hillside = terrain_slope > 0.18;

        let threshold = if is_hillside { 0.68 } else { 0.82 };
        if entrance_noise < threshold {
            return false;
        }

        let hash = self.position_hash(x, z);
        let entrance_chance = if is_hillside { 4 } else { 10 };
        if hash % entrance_chance != 0 {
            return false;
        }

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

    fn get_3d_density(&self, x: i32, y: i32, z: i32, biome: Biome, surface_height: i32) -> f64 {
        let fx = x as f32;
        let fy = y as f32;
        let fz = z as f32;

        let vertical_gradient = (surface_height as f64 - y as f64) / 8.0;

        let density_noise = match biome {
            Biome::Mountains => {
                let terrain = self.noise_terrain.get_noise_2d(fx * 0.018, fz * 0.018) as f64 * 0.55;
                let detail =
                    self.noise_detail
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

        // Ore-like stone variation (deepslate feel below y=8)
        if y < 8 {
            let deep_hash = self.position_hash_3d(world_x, y, world_z);
            if deep_hash % 10 < 3 {
                return BlockType::Stone; // Could be DeepSlate if block type added
            }
        }

        let depth_from_surface = surface_height - y;
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
                    if y == surface_height - 1 {
                        BlockType::Snow
                    } else {
                        BlockType::Stone
                    }
                } else if y > 115 {
                    // Gravel scree on high slopes
                    let hash = self.position_hash_3d(world_x, y, world_z);
                    if depth_from_surface <= 1 {
                        if hash % 4 == 0 { BlockType::Gravel } else { BlockType::Stone }
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
                let hash = self.position_hash(world_x, world_z);

                if height <= SEA_LEVEL {
                    continue;
                }

                // --- Trees ---
                if biome.has_trees() {
                    let tree_noise = self
                        .noise_trees
                        .get_noise_2d(world_x as f32, world_z as f32);
                    let density_threshold = biome.tree_density() as f32;

                    if tree_noise > density_threshold {
                        if hash % 100 < 18 {
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

                // --- Cactus in desert ---
                if biome == Biome::Desert {
                    if hash % 100 < 3 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand {
                            self.place_cactus(chunk, lx, height, lz);
                        }
                    }
                    // Dead bush on sand surface
                    else if hash % 100 < 10 {
                        let ground = chunk.get_block(lx, height - 1, lz);
                        if ground == BlockType::Sand && height < WORLD_HEIGHT - 1 {
                            chunk.set_block(lx, height, lz, BlockType::DeadBush);
                        }
                    }
                }

                // --- Gravel patches in mountains / beaches ---
                if biome == Biome::Mountains && height > 110 {
                    if hash % 100 < 8 {
                        let top = chunk.get_block(lx, height - 1, lz);
                        if matches!(top, BlockType::Stone | BlockType::Grass) {
                            chunk.set_block(lx, height - 1, lz, BlockType::Gravel);
                        }
                    }
                }

                // --- Snow on mountain tops > 145 ---
                if biome == Biome::Mountains && height > 145 {
                    if chunk.get_block(lx, height - 1, lz) == BlockType::Stone {
                        chunk.set_block(lx, height - 1, lz, BlockType::Snow);
                    }
                }
            }
        }
    }

    fn can_place_tree(&self, chunk: &Chunk, lx: i32, y: i32, lz: i32, is_large: bool) -> bool {
        let ground_block = chunk.get_block(lx, y - 1, lz);
        if !matches!(ground_block, BlockType::Grass | BlockType::Dirt) {
            return false;
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
        biome: Biome,
        is_large: bool,
    ) {
        let trunk_height = if is_large { 8 } else { 5 + (self.position_hash(lx, lz) % 2) as i32 };

        if chunk.get_block(lx, y - 1, lz) == BlockType::Grass {
            chunk.set_block(lx, y - 1, lz, BlockType::Dirt);
        }

        for dy in 0..trunk_height {
            chunk.set_block(lx, y + dy, lz, BlockType::Wood);
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
                                // Slightly rounder canopy based on biome
                                let corner_skip = match biome {
                                    Biome::Swamp => dx.abs() == radius && dz.abs() == radius && self.position_hash(nx, nz) % 3 != 0,
                                    _ => dx.abs() == radius && dz.abs() == radius && self.position_hash(nx, nz) % 2 == 0,
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

        let top_y = y + trunk_height;
        if top_y < WORLD_HEIGHT {
            let existing = chunk.get_block(lx, top_y, lz);
            if existing == BlockType::Air || existing == BlockType::Leaves {
                chunk.set_block(lx, top_y, lz, BlockType::Leaves);
            }
        }
    }

    fn place_cactus(&self, chunk: &mut Chunk, lx: i32, y: i32, lz: i32) {
        let height = 2 + (self.position_hash(lx, lz) % 3) as i32;
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
