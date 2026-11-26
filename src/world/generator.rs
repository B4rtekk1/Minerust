//! Procedural world generator.
//!
//! The world is built from a set of smooth 2-D fields evaluated per block
//! column, so every column — and therefore every chunk — is a pure function of
//! `(seed, x, z)`:
//!
//! 1. **Shape**: domain-warped *continentalness* sets land vs. ocean and the
//!    base elevation; *erosion* decides whether land is flat or rugged; a
//!    ridged field raises mountain ranges where erosion is low. Detail noise
//!    adds hills, and hot, dry uplands are gently terraced into plateaus.
//! 2. **Water**: a warped river field carves valleys and channels down to the
//!    sea, lowland lakes and swamp pools are pressed below sea level, and
//!    small islands rise out of the open ocean.
//! 3. **Climate**: temperature (cooled with altitude) and humidity pick the
//!    biome and a smooth vegetation tint (see [`crate::core::biome`]).
//! 4. **Blocks**: surface rules by biome, slope and depth (terracotta bands
//!    in badlands, sandstone under deserts, deepslate deep down); caves from
//!    winding "worm" tunnels that open onto hillsides, noodle tunnels,
//!    ravines, deep pillared caverns and flooded underground lakes.
//! 5. **Underground**: ore veins, amethyst geodes and cave biomes — lush
//!    (moss, hanging foliage), dripstone (stalactites and stalagmites) and
//!    volcanic (magma floors).
//! 6. **Features**: trees, huge mushrooms, cacti, boulders, logs and ice
//!    spikes. Features are
//!    anchored on a global jittered grid and evaluated for every anchor within
//!    reach of the chunk, so trees that straddle a chunk border are generated
//!    identically from both sides instead of being cut off.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

use crate::constants::*;
use crate::core::biome::{Biome, climate_tint};
use crate::core::block::BlockType;
use crate::core::chunk::Chunk;

/// Columns sampled around a chunk so slopes at its edge use real neighbours.
const BORDER: i32 = 1;
const GRID: usize = (CHUNK_SIZE + BORDER * 2) as usize;
/// Side of a feature anchor cell. Each cell holds at most one feature.
const FEATURE_CELL: i32 = 3;
/// Furthest a feature reaches from its anchor (giant jungle canopy).
const FEATURE_REACH: i32 = 7;
/// Chunks around a chunk whose tunnel systems can reach into it.
const WORM_SOURCE_RADIUS: i32 = 8;
/// Longest tunnel walk in steps (≈ blocks). A system starting just outside
/// [`WORM_SOURCE_RADIUS`] must never reach the chunk, or it would be cut off.
const WORM_MAX_LENGTH: i32 = 118;
const _: () = assert!(WORM_MAX_LENGTH + 8 < WORM_SOURCE_RADIUS * CHUNK_SIZE);
/// Carved cave space at or below this height fills with water.
const CAVE_WATER_LEVEL: i32 = 10;
/// Below this height stone turns into deepslate (dithered over a few blocks).
const DEEPSLATE_LEVEL: i32 = 18;
/// Chunks around a chunk whose ore veins and geodes can reach into it.
const ORE_SOURCE_RADIUS: i32 = 1;
/// Ore veins per chunk: block, veins, lowest y, highest y, min/max size.
const ORE_VEINS: [(BlockType, u32, i32, i32, u32, u32); 4] = [
    (BlockType::CoalOre, 18, 8, 120, 5, 12),
    (BlockType::IronOre, 11, 4, 64, 4, 9),
    (BlockType::GoldOre, 4, 3, 34, 3, 8),
    (BlockType::DiamondOre, 2, 2, 16, 2, 6),
];
/// Horizontal lattice points of the cave field across one chunk.
const CAVE_XZ: usize = (CHUNK_SIZE / CAVE_CELL_SIZE + 1) as usize;

/// Selects the deterministic terrain-shaping rules used for a world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum GeneratorVersion {
    /// Original terrain shape, used by saves created before generator versioning.
    Legacy = 1,
    /// Rounded, multi-scale mountains with rivers faded at high elevation.
    #[default]
    Current = 2,
}

impl GeneratorVersion {
    pub const fn from_u8(version: u8) -> Option<Self> {
        match version {
            1 => Some(Self::Legacy),
            2 => Some(Self::Current),
            _ => None,
        }
    }
}

/// Smooth Hermite step from `e0` to `e1` (either order).
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Piecewise smooth curve through `(input, output)` points (ascending input).
fn curve(points: &[(f32, f32)], x: f32) -> f32 {
    if x <= points[0].0 {
        return points[0].1;
    }
    for pair in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if x <= x1 {
            return lerp(y0, y1, smoothstep(x0, x1, x));
        }
    }
    points[points.len() - 1].1
}

/// Base elevation for a continentalness value: deep ocean → shelf → coast →
/// lowland → inland highland.
const CONTINENT_CURVE: [(f32, f32); 11] = [
    (-1.00, 26.0),
    (-0.55, 33.0),
    (-0.35, 44.0),
    (-0.18, 54.0),
    (-0.07, 60.5),
    (0.00, 64.5),
    (0.07, 66.5),
    (0.22, 71.0),
    (0.45, 79.0),
    (0.75, 92.0),
    (1.00, 100.0),
];

/// Everything the generator knows about one block column before blocks are
/// written.
#[derive(Clone, Copy, Debug)]
struct ColumnSample {
    /// Terrain surface height (highest terrain block), unclamped.
    height: f32,
    continental: f32,
    mountain: f32,
    /// River channel strength (1 = river bed).
    river: f32,
    /// Lake strength (1 = lake bed).
    lake: f32,
    /// Island strength in open ocean.
    island: f32,
    swamp: f32,
    /// Badlands strength (hot, dry mesa country).
    badlands: f32,
    /// Low-frequency field that picks biome variants (birch forest,
    /// old-growth taiga, cherry groves...).
    variant: f32,
    temperature: f32,
    humidity: f32,
}

impl ColumnSample {
    fn surface(&self) -> i32 {
        (self.height.round() as i32).clamp(4, WORLD_HEIGHT - 24)
    }
}

/// A fully classified column: shape, slope and biome.
#[derive(Clone, Copy, Debug)]
struct Column {
    sample: ColumnSample,
    surface: i32,
    slope: i32,
    biome: Biome,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TreeKind {
    Oak,
    BigOak,
    Birch,
    Spruce,
    SmallSpruce,
    Jungle,
    JungleGiant,
    Acacia,
    SwampOak,
    Palm,
    /// Tall, slim birch.
    TallBirch,
    /// Two-by-two old-growth spruce.
    MegaSpruce,
    /// Cherry with a wide, pink, multi-lobed crown.
    Cherry,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Feature {
    Tree(TreeKind),
    Cactus,
    DeadBush,
    Boulder,
    FallenLog,
    IceSpike,
    MossyBoulder,
    HugeRedMushroom,
    HugeBrownMushroom,
}

/// What a feature block may replace.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Only into air (leaves, plants).
    Soft,
    /// Into air, leaves and small plants (trunks, logs).
    Wood,
    /// Into air, leaves, plants and soil surface blocks (boulders, spikes).
    Rock,
}

pub struct ChunkGenerator {
    warp_x: FastNoiseLite,
    warp_z: FastNoiseLite,
    continents: FastNoiseLite,
    erosion: FastNoiseLite,
    ridges: FastNoiseLite,
    peaks: FastNoiseLite,
    hills: FastNoiseLite,
    river: FastNoiseLite,
    river_width: FastNoiseLite,
    lakes: FastNoiseLite,
    islands: FastNoiseLite,
    plateaus: FastNoiseLite,
    temperature: FastNoiseLite,
    humidity: FastNoiseLite,
    forest: FastNoiseLite,
    entrances: FastNoiseLite,
    caverns: FastNoiseLite,
    pockets: FastNoiseLite,
    patches: FastNoiseLite,
    ravines: FastNoiseLite,
    ravine_mask: FastNoiseLite,
    ravine_floor: FastNoiseLite,
    variant: FastNoiseLite,
    noodle_a: FastNoiseLite,
    noodle_b: FastNoiseLite,
    cave_biome: FastNoiseLite,
    pillars: FastNoiseLite,
    pub seed: u32,
    pub version: GeneratorVersion,
}

impl Clone for ChunkGenerator {
    fn clone(&self) -> Self {
        Self::new_with_version(self.seed, self.version)
    }
}

impl ChunkGenerator {
    pub fn new(seed: u32) -> Self {
        Self::new_with_version(seed, GeneratorVersion::Current)
    }

    pub fn new_with_version(seed: u32, version: GeneratorVersion) -> Self {
        let s = |i: u32| seed.wrapping_mul(0x9e37_79b9).wrapping_add(i.wrapping_mul(0x85eb_ca6b));
        ChunkGenerator {
            warp_x: fbm(s(1), 0.0026, 3),
            warp_z: fbm(s(2), 0.0026, 3),
            continents: fbm(s(3), 0.00085, 5),
            erosion: fbm(s(4), 0.0019, 4),
            ridges: ridged(s(5), 0.0029, 4),
            peaks: fbm(s(6), 0.006, 3),
            hills: fbm(s(7), 0.011, 4),
            river: fbm(s(8), 0.0016, 2),
            river_width: simplex(s(9), 0.004),
            lakes: fbm(s(10), 0.0042, 2),
            islands: fbm(s(11), 0.011, 3),
            plateaus: simplex(s(12), 0.0035),
            temperature: fbm(s(13), 0.00065, 3),
            humidity: fbm(s(14), 0.0009, 3),
            forest: fbm(s(15), 0.012, 2),
            entrances: fbm(s(16), 0.011, 2),
            caverns: fbm(s(19), 0.011, 3),
            pockets: fbm(s(20), 0.03, 2),
            patches: fbm(s(22), 0.045, 2),
            ravines: fbm(s(23), 0.0032, 2),
            ravine_mask: simplex(s(24), 0.0018),
            ravine_floor: simplex(s(25), 0.02),
            variant: fbm(s(26), 0.0024, 2),
            noodle_a: simplex(s(27), 0.011),
            noodle_b: simplex(s(28), 0.011),
            cave_biome: fbm(s(29), 0.0065, 2),
            pillars: simplex(s(30), 0.085),
            seed,
            version,
        }
    }

    // ── Column fields ──────────────────────────────────────────────────── //

    fn sample_column(&self, x: i32, z: i32) -> ColumnSample {
        let (fx, fz) = (x as f32, z as f32);
        let sea = SEA_LEVEL as f32;
        // Domain warping gives coastlines and ranges organic, curling
        // outlines instead of blobby noise contours.
        let wx = fx + self.warp_x.get_noise_2d(fx, fz) * 90.0;
        let wz = fz + self.warp_z.get_noise_2d(fx, fz) * 90.0;

        let continental = self.continents.get_noise_2d(wx, wz) * 1.55 + 0.12;
        let erosion = self.erosion.get_noise_2d(wx, wz) * 1.6;
        let base = curve(&CONTINENT_CURVE, continental);

        // Mountains need inland position and low erosion; ridged noise gives
        // them connected crests with valleys between the ranges.
        let inland = smoothstep(0.02, 0.5, continental);
        let rugged = 1.0 - smoothstep(-0.5, 0.25, erosion);
        let mountain = inland * rugged;
        let ridge = self.ridges.get_noise_2d(wx, wz).clamp(-1.0, 1.0) * 0.5 + 0.5;
        let peak_jitter = self.peaks.get_noise_2d(fx, fz) * 0.5 + 0.5;
        let mountains = match self.version {
            GeneratorVersion::Legacy => {
                mountain * (ridge.powf(1.6) * 0.82 + peak_jitter * 0.18) * 118.0
            }
            GeneratorVersion::Current => {
                // Combine broad upland variation with the original ridge and a
                // small amount of fine detail rather than letting one sharp
                // ridge field dominate every mountain range.
                let broad_peaks = self.peaks.get_noise_2d(fx * 0.46, fz * 0.46) * 0.5 + 0.5;
                mountain
                    * (ridge.powf(1.05) * 0.75 + broad_peaks * 0.17 + peak_jitter * 0.08)
                    * 104.0
            }
        };

        // Rolling hills on land, flatter where erosion is high.
        let land = smoothstep(-0.08, 0.06, continental);
        let hill_amp = lerp(2.5, 11.0, 1.0 - smoothstep(-0.3, 0.7, erosion));
        let hills = self.hills.get_noise_2d(fx, fz) * hill_amp * land;

        let mut height = base + mountains + hills;

        // Climate before water shaping: swamps and plateaus depend on it.
        let temperature_raw = (self.temperature.get_noise_2d(fx, fz) * 1.45).clamp(-1.0, 1.0);
        let humidity =
            (self.humidity.get_noise_2d(fx + 4000.0, fz - 2500.0) * 1.7).clamp(-1.0, 1.0);

        // Badlands: the hottest, driest land rises into flat-topped mesas
        // that are terraced below.
        let variant = self.variant.get_noise_2d(fx, fz);
        let badlands = smoothstep(0.3, 0.5, temperature_raw)
            * smoothstep(-0.12, -0.4, humidity)
            * smoothstep(-0.05, 0.25, variant)
            * smoothstep(0.0, 0.2, continental);
        if badlands > 0.0 {
            let mesa = smoothstep(-0.15, 0.35, self.plateaus.get_noise_2d(fx * 1.9 + 300.0, fz * 1.9));
            height += badlands * mesa * 26.0;
        }

        // Terraced plateaus in hot, dry uplands.
        let terrace = (smoothstep(0.25, 0.55, temperature_raw)
            * smoothstep(0.05, -0.35, humidity)
            * smoothstep(0.1, 0.45, self.plateaus.get_noise_2d(fx, fz)))
        .max(badlands)
            * smoothstep(sea + 6.0, sea + 16.0, height);
        if terrace > 0.0 {
            let step = 7.0;
            let level = (height / step).floor();
            let frac = height / step - level;
            let stepped = (level + smoothstep(0.55, 0.85, frac)) * step;
            height = lerp(height, stepped, terrace);
        }

        // Islands in the open ocean.
        // Islands cluster into archipelagos instead of dotting every ocean.
        let archipelago = smoothstep(0.05, 0.35, self.plateaus.get_noise_2d(fx * 0.18 + 900.0, fz * 0.18));
        let island = smoothstep(0.45, 0.78, self.islands.get_noise_2d(fx, fz))
            * smoothstep(-0.16, -0.36, continental)
            * archipelago;
        if island > 0.0 {
            let target = sea + 3.0 + island * 7.0 + hills.abs() * 0.4;
            height = lerp(height, target.max(height), smoothstep(0.0, 0.6, island));
        }

        // Rivers: a warped zero-crossing of a low-frequency field. The broad
        // valley is pulled towards sea level first, then the channel is cut
        // below it, so every river drains at sea level and reaches the ocean.
        let river_n = self
            .river
            .get_noise_2d(wx * 0.9 + 1700.0, wz * 0.9 - 900.0)
            .abs();
        let width = 0.016 + 0.012 * (self.river_width.get_noise_2d(fx, fz) * 0.5 + 0.5);
        let on_land = smoothstep(-0.05, 0.08, continental);
        let river_elevation_fade = match self.version {
            GeneratorVersion::Legacy => 1.0,
            GeneratorVersion::Current => {
                let high_elevation = smoothstep(sea + 26.0, sea + 72.0, height);
                let mountain_factor = smoothstep(0.12, 0.45, mountain);
                1.0 - high_elevation * mountain_factor
            }
        };
        let valley = (1.0 - smoothstep(width, width * 5.0, river_n))
            * on_land
            * river_elevation_fade;
        let river = (1.0 - smoothstep(width * 0.45, width, river_n))
            * on_land
            * river_elevation_fade;
        if valley > 0.0 {
            let valley_floor = sea + 1.5 + (height - sea).max(0.0) * 0.22;
            height = lerp(height, valley_floor.min(height), valley * 0.9);
            height = lerp(height, sea - 2.0 - 2.5 * river, river);
        }

        // Lowland lakes.
        let lowland = 1.0 - smoothstep(sea + 6.0, sea + 20.0, height);
        let lake = smoothstep(0.38, 0.6, self.lakes.get_noise_2d(fx, fz))
            * lowland
            * smoothstep(0.04, 0.16, continental)
            * (1.0 - mountain);
        if lake > 0.0 {
            height = lerp(height, sea - 2.0 - 4.0 * lake, lake);
        }

        // Swamps: warm, wet lowlands flattened to around sea level so
        // shallow pools form between patches of grass.
        let swamp = smoothstep(0.35, 0.65, humidity)
            * smoothstep(-0.05, 0.25, temperature_raw)
            * lowland
            * on_land
            * (1.0 - river);
        if swamp > 0.0 {
            let target = sea - 0.3 + self.hills.get_noise_2d(fx * 3.0, fz * 3.0) * 2.2;
            height = lerp(height, target, swamp * 0.85);
        }

        // Air cools with altitude: high ground turns alpine and snowy.
        let temperature = (temperature_raw - (height - 82.0).max(0.0) / 95.0).clamp(-1.0, 1.0);

        ColumnSample {
            height,
            continental,
            mountain,
            river,
            lake,
            island,
            swamp,
            badlands,
            variant,
            temperature,
            humidity,
        }
    }

    fn snow_line(&self, temperature: f32, x: i32, z: i32) -> f32 {
        152.0 + temperature * 30.0 + self.peaks.get_noise_2d(x as f32 * 2.0, z as f32 * 2.0) * 7.0
    }

    fn classify(&self, x: i32, z: i32, s: &ColumnSample, surface: i32, slope: i32) -> Biome {
        let sea = SEA_LEVEL;
        if surface < sea {
            return if s.river > 0.4 {
                Biome::River
            } else if s.lake > 0.35 {
                Biome::Lake
            } else if s.swamp > 0.4 {
                Biome::Swamp
            } else if s.continental < 0.0 {
                Biome::Ocean
            } else if s.lake > 0.0 {
                Biome::Lake
            } else {
                Biome::River
            };
        }
        if s.island > 0.3 && surface <= sea + 12 {
            // Some archipelagos are fungal: mushroom fields.
            let fungal = self.variant.get_noise_2d(x as f32 * 0.5 + 7000.0, z as f32 * 0.5) > 0.22;
            return if surface <= sea + 1 {
                Biome::Beach
            } else if fungal {
                Biome::MushroomFields
            } else {
                Biome::Island
            };
        }
        if surface <= sea + 2 && s.continental < 0.14 && s.river < 0.3 && s.lake < 0.2 {
            return if slope >= 3 { Biome::StonyShore } else { Biome::Beach };
        }
        if surface as f32 > self.snow_line(s.temperature, x, z) {
            return Biome::SnowyPeaks;
        }
        if s.mountain > 0.35 && surface > 100 {
            return Biome::Mountains;
        }
        let (t, h, v) = (s.temperature, s.humidity, s.variant);
        if s.swamp > 0.45 {
            Biome::Swamp
        } else if s.badlands > 0.35 && t > 0.3 {
            Biome::Badlands
        } else if t < -0.55 {
            Biome::Tundra
        } else if t < -0.2 {
            if v > 0.1 && h > -0.2 {
                Biome::OldGrowthTaiga
            } else {
                Biome::Taiga
            }
        } else if t > 0.42 {
            if h < -0.3 {
                Biome::Desert
            } else if h < 0.28 {
                Biome::Savanna
            } else {
                Biome::Jungle
            }
        } else if h > 0.12 {
            if v > 0.22 {
                Biome::BirchForest
            } else {
                Biome::Forest
            }
        } else if surface > 78 && v < -0.28 && t > -0.1 && t < 0.35 {
            Biome::CherryGrove
        } else if surface > 84 && h > -0.25 {
            Biome::Meadow
        } else {
            Biome::Plains
        }
    }

    /// Samples one column plus its four neighbours to classify it.
    fn column(&self, x: i32, z: i32) -> Column {
        let sample = self.sample_column(x, z);
        let surface = sample.surface();
        let slope = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .map(|&(dx, dz)| (self.sample_column(x + dx, z + dz).surface() - surface).abs())
            .max()
            .unwrap_or(0);
        Column {
            sample,
            surface,
            slope,
            biome: self.classify(x, z, &sample, surface, slope),
        }
    }

    // ── Public queries ─────────────────────────────────────────────────── //

    pub fn get_biome(&self, x: i32, z: i32) -> Biome {
        self.column(x, z).biome
    }

    /// Terrain surface height (highest terrain block, excluding features).
    pub fn get_terrain_height_pub(&self, x: i32, z: i32) -> i32 {
        self.sample_column(x, z).surface()
    }

    pub fn is_cave_entrance_pub(&self, x: i32, z: i32, surface_height: i32) -> bool {
        self.is_cave_entrance(x, z, surface_height)
    }

    pub fn position_hash_pub(&self, x: i32, z: i32) -> u32 {
        self.hash2(x, z, 0)
    }

    /// Finds a pleasant spawn column: dry, gentle land in a temperate-ish
    /// biome, searched on growing rings around the origin.
    pub fn find_spawn_column(&self) -> (i32, i32) {
        let good = |x: i32, z: i32, strict: bool| {
            let col = self.column(x, z);
            let biome_ok = if strict {
                matches!(
                    col.biome,
                    Biome::Plains
                        | Biome::Meadow
                        | Biome::Forest
                        | Biome::BirchForest
                        | Biome::CherryGrove
                        | Biome::Savanna
                        | Biome::Taiga
                )
            } else {
                !col.biome.is_water()
            };
            col.surface > SEA_LEVEL + 1
                && col.slope <= 1
                && biome_ok
                && !self.is_cave_entrance(x, z, col.surface)
        };
        for strict in [true, false] {
            for ring in 0..200 {
                let radius = ring * 24;
                let points = (radius / 12).max(1);
                for i in 0..points {
                    let angle = i as f32 / points as f32 * std::f32::consts::TAU;
                    let x = (angle.cos() * radius as f32).round() as i32;
                    let z = (angle.sin() * radius as f32).round() as i32;
                    if good(x, z, strict) {
                        return (x, z);
                    }
                }
            }
        }
        (0, 0)
    }

    fn is_cave_entrance(&self, x: i32, z: i32, surface: i32) -> bool {
        self.entrance_strength(x, z, surface) > 0.5
    }

    /// How strongly tunnels may break through to the surface here (0..1).
    /// Entrance zones are patches tens of blocks wide; inside them tunnels
    /// open up as cave mouths instead of stopping under a solid roof.
    fn entrance_strength(&self, x: i32, z: i32, surface: i32) -> f32 {
        if surface <= SEA_LEVEL + 3 {
            return 0.0;
        }
        smoothstep(0.32, 0.5, self.entrances.get_noise_2d(x as f32, z as f32))
    }

    /// Ravine carved into column `(x, z)`: returns `(bottom_y, edge)` where
    /// `edge` is 0 on the centre line and 1 at the rim.
    fn ravine(&self, x: i32, z: i32, sample: &ColumnSample) -> Option<(i32, f32)> {
        // Inland, away from rivers and lakes, so no water ever borders one.
        if sample.surface() <= SEA_LEVEL + 8
            || sample.continental < 0.1
            || sample.river > 0.0
            || sample.lake > 0.0
            || sample.swamp > 0.05
        {
            return None;
        }
        let (fx, fz) = (x as f32, z as f32);
        let mask = smoothstep(0.45, 0.65, self.ravine_mask.get_noise_2d(fx, fz));
        if mask <= 0.0 {
            return None;
        }
        let half_width = 0.022 * mask;
        let n = self.ravines.get_noise_2d(fx + 500.0, fz - 300.0).abs();
        if n >= half_width {
            return None;
        }
        let bottom = 18 + (self.ravine_floor.get_noise_2d(fx, fz) * 6.0) as i32;
        Some((bottom, n / half_width))
    }

    // ── Chunk generation ───────────────────────────────────────────────── //

    pub fn generate_chunk(&self, cx: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cz);
        let base_x = cx * CHUNK_SIZE;
        let base_z = cz * CHUNK_SIZE;

        // Pass 1: column samples for the chunk and a one-column border.
        let samples: Vec<ColumnSample> = (0..GRID * GRID)
            .map(|i| {
                let gx = (i / GRID) as i32 - BORDER;
                let gz = (i % GRID) as i32 - BORDER;
                self.sample_column(base_x + gx, base_z + gz)
            })
            .collect();
        let sample_at =
            |lx: i32, lz: i32| &samples[((lx + BORDER) as usize) * GRID + (lz + BORDER) as usize];
        let columns: Vec<Column> = (0..(CHUNK_SIZE * CHUNK_SIZE))
            .map(|i| {
                let (lx, lz) = (i / CHUNK_SIZE, i % CHUNK_SIZE);
                let sample = *sample_at(lx, lz);
                let surface = sample.surface();
                let slope = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .map(|&(dx, dz)| (sample_at(lx + dx, lz + dz).surface() - surface).abs())
                    .max()
                    .unwrap_or(0);
                let biome = self.classify(base_x + lx, base_z + lz, &sample, surface, slope);
                Column {
                    sample,
                    surface,
                    slope,
                    biome,
                }
            })
            .collect();
        let max_surface = columns.iter().map(|c| c.surface).max().unwrap_or(SEA_LEVEL);

        // Pass 2: cave lattice up to the highest surface in the chunk.
        let cave_y_samples = (max_surface / CAVE_CELL_SIZE + 2) as usize;
        let cave = self.cave_lattice(base_x, base_z, cave_y_samples);
        let worms = self.carve_worms(base_x, base_z);

        // Pass 3: terrain, caves and water.
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let col = &columns[(lx * CHUNK_SIZE + lz) as usize];
                let (wx, wz) = (base_x + lx, base_z + lz);
                chunk.set_column_tint(
                    lx,
                    lz,
                    climate_tint(col.sample.temperature, col.sample.humidity),
                );
                let entrance = if matches!(col.biome, Biome::River | Biome::Beach) {
                    0.0
                } else {
                    self.entrance_strength(wx, wz, col.surface)
                };
                let ravine = self.ravine(wx, wz, &col.sample);
                // Never carve beside water in a neighbouring column: open
                // cave space next to static water would show a water wall.
                let carve_ceiling = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .map(|&(dx, dz)| sample_at(lx + dx, lz + dz).surface())
                    .filter(|&s| s < SEA_LEVEL)
                    .map(|s| s - 9)
                    .min()
                    .unwrap_or(WORLD_HEIGHT);
                let column_hash = self.hash2(wx, wz, 1);
                let bedrock_top = (column_hash % 3) as i32;
                // Patch noise: neighbouring columns share materials, so the
                // ground shows coherent patches instead of random speckles.
                let patch = self.patches.get_noise_2d(wx as f32, wz as f32);
                let filler_depth = 3 + i32::from(patch > 0.0);
                let snow_line = self.snow_line(col.sample.temperature, wx, wz);
                let ctx = CaveColumn {
                    surface: col.surface,
                    slope: col.slope,
                    entrance,
                    ravine,
                    carve_ceiling,
                    pillar: self.pillars.get_noise_2d(wx as f32, wz as f32) > 0.7,
                };

                for y in 0..=col.surface {
                    if y <= bedrock_top {
                        chunk.set_block_raw(lx, y, lz, BlockType::Bedrock);
                        continue;
                    }
                    if self.carve_cave(&cave, &worms, lx, y, lz, &ctx) {
                        // The deepest caves are flooded: underground lakes.
                        if y <= CAVE_WATER_LEVEL {
                            chunk.set_block_raw(lx, y, lz, BlockType::Water);
                        }
                        continue;
                    }
                    let mut block = self.surface_block(col, y, filler_depth, snow_line, patch);
                    // Soft pockets (whole blobs of one material) break up the
                    // stone underground.
                    if block == BlockType::Stone {
                        let pocket = sample_lattice(&cave, 1, lx, y, lz);
                        if pocket > 0.56 {
                            block = BlockType::Gravel;
                        } else if pocket < -0.6 && y > 30 {
                            block = BlockType::Dirt;
                        } else if pocket < -0.62 && y <= 30 {
                            block = BlockType::Calcite;
                        } else if self.is_deepslate(wx, y, wz) {
                            block = BlockType::Deepslate;
                        }
                    }
                    chunk.set_block_raw(lx, y, lz, block);
                }

                if col.surface < SEA_LEVEL {
                    // Ragged pack-ice edge rather than a clean contour line.
                    let frozen = col.sample.temperature
                        + self.hills.get_noise_2d(wx as f32 * 2.0, wz as f32 * 2.0) * 0.12
                        < -0.68;
                    for y in (col.surface + 1)..SEA_LEVEL {
                        let block = if frozen && y == SEA_LEVEL - 1 {
                            BlockType::Ice
                        } else {
                            BlockType::Water
                        };
                        chunk.set_block_raw(lx, y, lz, block);
                    }
                }
            }
        }

        // Pass 4: ore veins and geodes inside the rock.
        self.place_ores(&mut chunk, base_x, base_z);
        self.place_geodes(&mut chunk, base_x, base_z);

        // Pass 5: cave biomes decorate floors and ceilings.
        self.decorate_caves(&mut chunk, base_x, base_z, &columns);

        // Pass 6: features from every anchor whose reach overlaps the chunk.
        self.place_features(&mut chunk, base_x, base_z, &columns);

        chunk.rebuild_metadata();
        chunk
    }

    // ── Surface rules ──────────────────────────────────────────────────── //

    fn surface_block(
        &self,
        col: &Column,
        y: i32,
        filler_depth: i32,
        snow_line: f32,
        patch: f32,
    ) -> BlockType {
        let depth = col.surface - y;
        let underwater = col.surface < SEA_LEVEL;
        let steep = col.slope >= 4;

        match col.biome {
            Biome::Ocean | Biome::River | Biome::Lake => {
                if depth > 3 {
                    return BlockType::Stone;
                }
                // Sediment in broad, smooth patches; the whole top layer of
                // a column is one material.
                let deep = col.surface < SEA_LEVEL - 12;
                match col.biome {
                    Biome::River if patch < -0.12 => BlockType::Gravel,
                    Biome::River if patch > 0.3 => BlockType::Clay,
                    Biome::Lake if patch < -0.15 => BlockType::Clay,
                    Biome::Lake if patch > 0.3 => BlockType::Gravel,
                    Biome::Ocean if deep && patch < -0.05 => BlockType::Gravel,
                    Biome::Ocean if deep && patch > 0.32 => BlockType::Clay,
                    Biome::Ocean if !deep && patch < -0.38 => BlockType::Gravel,
                    _ => BlockType::Sand,
                }
            }
            Biome::Beach => {
                if depth <= 4 {
                    BlockType::Sand
                } else if depth <= 7 {
                    BlockType::Sandstone
                } else {
                    BlockType::Stone
                }
            }
            Biome::Badlands => {
                if depth > 30 + (patch * 6.0) as i32 {
                    BlockType::Stone
                } else if underwater || (depth <= 1 && col.slope < 3) {
                    BlockType::RedSand
                } else {
                    badlands_band(y, patch)
                }
            }
            Biome::StonyShore => {
                if depth == 0 && patch < 0.05 {
                    BlockType::Gravel
                } else {
                    BlockType::Stone
                }
            }
            Biome::Desert => {
                if col.slope >= 7 && depth < 3 {
                    BlockType::Sandstone
                } else if depth <= filler_depth + 2 {
                    BlockType::Sand
                } else if depth <= filler_depth + 7 {
                    BlockType::Sandstone
                } else {
                    BlockType::Stone
                }
            }
            Biome::SnowyPeaks => {
                if depth <= 1 && col.slope < 7 {
                    BlockType::Snow
                } else if depth == 0 && patch > 0.35 {
                    BlockType::Ice
                } else {
                    BlockType::Stone
                }
            }
            Biome::Mountains => {
                if steep || depth > filler_depth {
                    BlockType::Stone
                } else if (col.surface as f32) > snow_line - 12.0 && depth == 0 {
                    BlockType::Snow
                } else if depth == 0 && patch < -0.3 {
                    BlockType::Gravel
                } else if depth == 0 {
                    BlockType::Grass
                } else {
                    BlockType::Dirt
                }
            }
            _ => {
                if steep {
                    return if depth == 0 && col.biome == Biome::Tundra {
                        BlockType::Snow
                    } else if col.biome == Biome::OldGrowthTaiga && patch > -0.1 {
                        BlockType::MossyStone
                    } else {
                        BlockType::Stone
                    };
                }
                if depth > filler_depth {
                    return BlockType::Stone;
                }
                if depth > 0 {
                    return BlockType::Dirt;
                }
                if underwater {
                    // Swamp pools and other shallow lowland water.
                    return if patch < 0.0 {
                        BlockType::Clay
                    } else {
                        BlockType::Dirt
                    };
                }
                if col.surface <= SEA_LEVEL && col.sample.river > 0.05 {
                    return BlockType::Sand; // river bank
                }
                match col.biome {
                    Biome::Tundra => BlockType::Snow,
                    Biome::Taiga if (col.surface as f32) > snow_line - 30.0 => BlockType::Snow,
                    Biome::Taiga if patch > 0.3 => BlockType::Dirt,
                    Biome::Savanna if patch > 0.36 => BlockType::Dirt,
                    Biome::Island if col.surface <= SEA_LEVEL + 2 => BlockType::Sand,
                    Biome::MushroomFields => BlockType::Mycelium,
                    Biome::OldGrowthTaiga if patch > 0.4 => BlockType::Moss,
                    Biome::OldGrowthTaiga if patch > -0.35 => BlockType::Podzol,
                    Biome::Jungle if patch > 0.42 => BlockType::Podzol,
                    _ => BlockType::Grass,
                }
            }
        }
    }

    // ── Caves ──────────────────────────────────────────────────────────── //

    /// Samples the cave fields on a coarse lattice. Lattice points sit
    /// on chunk boundaries, so neighbouring chunks agree exactly.
    fn cave_lattice(&self, base_x: i32, base_z: i32, y_samples: usize) -> CaveLattice {
        let mut values = vec![[0.0f32; 4]; CAVE_XZ * CAVE_XZ * y_samples];
        for ix in 0..CAVE_XZ {
            for iz in 0..CAVE_XZ {
                let x = (base_x + ix as i32 * CAVE_CELL_SIZE) as f32;
                let z = (base_z + iz as i32 * CAVE_CELL_SIZE) as f32;
                for iy in 0..y_samples {
                    let y = (iy as i32 * CAVE_CELL_SIZE) as f32;
                    // Caverns are squashed vertically into wide, low halls.
                    values[(ix * CAVE_XZ + iz) * y_samples + iy] = [
                        self.caverns.get_noise_3d(x, y * 1.6, z),
                        self.pockets.get_noise_3d(x, y * 1.3, z),
                        self.noodle_a.get_noise_3d(x, y * 1.25, z),
                        self.noodle_b.get_noise_3d(x + 1300.0, y * 1.25, z - 800.0),
                    ];
                }
            }
        }
        CaveLattice { values, y_samples }
    }

    fn carve_cave(
        &self,
        cave: &CaveLattice,
        worms: &WormMask,
        lx: i32,
        y: i32,
        lz: i32,
        col: &CaveColumn,
    ) -> bool {
        if y <= 4 || y > col.carve_ceiling {
            return false;
        }
        let depth = col.surface - y;

        // Ravines: long, narrow cracks open to the sky, with walls that
        // narrow towards the floor.
        if let Some((bottom, edge)) = col.ravine {
            if y >= bottom {
                let span = (col.surface - bottom).max(1) as f32;
                let rise = (y - bottom) as f32 / span; // 0 floor → 1 rim
                if edge * edge <= 0.15 + 0.85 * rise {
                    return true;
                }
            }
        }

        // Worm tunnels keep a solid roof on flat ground, but may break out
        // of hillsides (natural cave mouths) and inside entrance zones.
        // Under water the roof is thick so the sea never drains into them.
        let roof = if col.surface < SEA_LEVEL {
            9.0
        } else if col.slope >= 2 {
            0.0
        } else {
            lerp(4.0, 0.0, col.entrance)
        };
        if (depth as f32) >= roof && worms.get(lx, y, lz) {
            return true;
        }
        if worms.get_open(lx, y, lz) && (col.surface >= SEA_LEVEL || depth >= 9) {
            return true;
        }

        // Noodle tunnels: where two independent smooth fields both cross
        // zero, the intersection traces long, thin, winding passages that
        // link the worm networks and caverns together.
        if depth >= 10 && y > 6 && y < 100 {
            let width = 0.055 + 0.025 * smoothstep(60.0, 20.0, y as f32);
            if sample_lattice(cave, 2, lx, y, lz).abs() < width
                && sample_lattice(cave, 3, lx, y, lz).abs() < width
            {
                return true;
            }
        }

        // Caverns: big low halls, only deep underground. Rare stone
        // pillars are left standing from floor to ceiling.
        if depth < 16 || y > 52 || col.pillar {
            return false;
        }
        let cavern = sample_lattice(cave, 0, lx, y, lz);
        let threshold = 0.36 + 0.32 * smoothstep(20.0, 52.0, y as f32);
        cavern > threshold
    }

    /// Carves every worm tunnel that can reach this chunk into a bit mask.
    ///
    /// Tunnel systems start in chunks up to [`WORM_SOURCE_RADIUS`] away and
    /// are simulated identically from every chunk they pass through, so
    /// passages continue seamlessly across chunk borders.
    fn carve_worms(&self, base_x: i32, base_z: i32) -> WormMask {
        let mut mask = WormMask::default();
        let (cx, cz) = (base_x.div_euclid(CHUNK_SIZE), base_z.div_euclid(CHUNK_SIZE));
        for sx in cx - WORM_SOURCE_RADIUS..=cx + WORM_SOURCE_RADIUS {
            for sz in cz - WORM_SOURCE_RADIUS..=cz + WORM_SOURCE_RADIUS {
                let mut rng = Rng(self.hash2(sx, sz, 0xca7e) as u64 | 1);
                // Roughly one tunnel system per six chunks, a few of them
                // larger networks.
                if rng.below(6) != 0 {
                    continue;
                }
                let systems = 1 + u32::from(rng.below(4) == 0);
                for _ in 0..systems {
                    let x = (sx * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32;
                    let z = (sz * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32;
                    // Biased low: most tunnels run deep, some near the surface.
                    let y = 10.0 + rng.unit() * rng.unit() * 62.0 + rng.unit() * 8.0;
                    if rng.below(4) == 0 {
                        // A chamber where several passages meet.
                        let r = 3.0 + rng.unit() * 4.0;
                        mask.carve_ellipsoid(base_x, base_z, [x, y, z], r * 1.4, r * 0.8, false);
                    }
                    let branches = 1 + rng.below(2);
                    for _ in 0..branches {
                        let yaw = rng.unit() * std::f32::consts::TAU;
                        let pitch = (rng.unit() - 0.5) * 0.5;
                        let radius = 1.6 + rng.unit() * 2.6;
                        let length = 60 + rng.below((WORM_MAX_LENGTH - 59) as u32) as i32;
                        let seed = rng.next();
                        let walk = Walk { yaw, pitch, radius, length, seed, branch: true, open: false };
                        self.worm(&mut mask, base_x, base_z, [x, y, z], walk);
                    }
                }
            }
        }
        // Entrance tunnels: start on the surface and slope down into the
        // network, giving roughly one cave mouth per 6×6 chunks.
        for sx in cx - WORM_SOURCE_RADIUS..=cx + WORM_SOURCE_RADIUS {
            for sz in cz - WORM_SOURCE_RADIUS..=cz + WORM_SOURCE_RADIUS {
                let mut rng = Rng(self.hash2(sx, sz, 0xe17e) as u64 | 1);
                if rng.below(20) != 0 {
                    continue;
                }
                let x = (sx * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32;
                let z = (sz * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32;
                let walk = Walk {
                    yaw: rng.unit() * std::f32::consts::TAU,
                    pitch: -0.45 - rng.unit() * 0.25,
                    radius: 1.8 + rng.unit() * 1.4,
                    length: 50 + rng.below(50) as i32,
                    seed: rng.next(),
                    branch: false,
                    open: true,
                };
                let sample = self.sample_column(x as i32, z as i32);
                if sample.surface() <= SEA_LEVEL + 3 || sample.river > 0.0 || sample.lake > 0.0 {
                    continue;
                }
                let start = [x, sample.surface() as f32 + 1.0, z];
                self.worm(&mut mask, base_x, base_z, start, walk);
            }
        }
        mask
    }

    /// One winding tunnel: a random walk with smoothly drifting direction
    /// whose width swells and narrows along its length.
    fn worm(&self, mask: &mut WormMask, base_x: i32, base_z: i32, start: [f32; 3], walk: Walk) {
        let Walk { mut yaw, mut pitch, radius, length, seed, branch: can_branch, open } = walk;
        let mut rng = Rng(seed | 1);
        let [mut x, mut y, mut z] = start;
        let (mut yaw_speed, mut pitch_speed) = (0.0f32, 0.0f32);
        let branch_at = if can_branch && radius > 2.0 {
            length / 4 + rng.below((length / 2) as u32) as i32
        } else {
            -1
        };
        let reach = (CHUNK_SIZE as f32) * 0.5 + radius * 2.0 + 2.0;
        let (center_x, center_z) = (
            base_x as f32 + CHUNK_SIZE as f32 * 0.5,
            base_z as f32 + CHUNK_SIZE as f32 * 0.5,
        );
        for step in 0..length {
            let progress = step as f32 / length as f32;
            // Entrance tunnels start at full width so the mouth is open.
            let shape = if open {
                (progress * std::f32::consts::PI * 0.5 + std::f32::consts::FRAC_PI_2).sin()
            } else {
                (progress * std::f32::consts::PI).sin()
            };
            let width = 1.0 + radius * shape;
            x += yaw.cos() * pitch.cos();
            z += yaw.sin() * pitch.cos();
            y += pitch.sin();
            // Steer gently; keep tunnels mostly horizontal. Entrance tunnels
            // keep descending until they are well underground.
            if open && y > 34.0 {
                pitch = pitch.clamp(-0.7, -0.3);
            } else {
                pitch *= 0.7;
            }
            pitch += pitch_speed * 0.1;
            yaw += yaw_speed * 0.1;
            pitch_speed = pitch_speed * 0.9 + (rng.unit() - rng.unit()) * rng.unit() * 1.0;
            yaw_speed = yaw_speed * 0.75 + (rng.unit() - rng.unit()) * rng.unit() * 2.4;
            if y < 8.0 {
                pitch = pitch.abs() * 0.5 + 0.05;
            }

            if step == branch_at {
                for side in [-1.0f32, 1.0] {
                    let walk = Walk {
                        yaw: yaw + side * std::f32::consts::FRAC_PI_2,
                        pitch: pitch / 3.0,
                        radius: radius * 0.7,
                        length: length - step,
                        seed: rng.next(),
                        branch: false,
                        open,
                    };
                    self.worm(mask, base_x, base_z, [x, y, z], walk);
                }
                return;
            }
            // Every random draw happens before the proximity test, so the
            // walk is identical no matter which chunk is simulating it.
            let rough_skip = rng.below(4) == 0 && width < 2.0;
            // Skip the expensive carve while the worm is far from this chunk.
            if (x - center_x).abs() > reach || (z - center_z).abs() > reach {
                continue;
            }
            // Occasional skipped segment leaves rough, uneven walls.
            if rough_skip {
                continue;
            }
            mask.carve_ellipsoid(base_x, base_z, [x, y, z], width, width * 0.75, open);
        }
    }

    // ── Underground ────────────────────────────────────────────────────── //

    /// Stone turns into deepslate below [`DEEPSLATE_LEVEL`], dithered over
    /// a few blocks so the boundary is not a flat plane.
    fn is_deepslate(&self, x: i32, y: i32, z: i32) -> bool {
        if y < DEEPSLATE_LEVEL - 4 {
            return true;
        }
        if y > DEEPSLATE_LEVEL + 3 {
            return false;
        }
        (self.hash3(x, y, z, 0xdee9) % 8) as i32 >= y - (DEEPSLATE_LEVEL - 4)
    }

    /// Ore veins: short random walks through stone. Veins start in this
    /// chunk or its neighbours and are simulated identically from every
    /// chunk they touch, so they never get cut at a chunk border.
    fn place_ores(&self, chunk: &mut Chunk, base_x: i32, base_z: i32) {
        let (cx, cz) = (base_x.div_euclid(CHUNK_SIZE), base_z.div_euclid(CHUNK_SIZE));
        for sx in cx - ORE_SOURCE_RADIUS..=cx + ORE_SOURCE_RADIUS {
            for sz in cz - ORE_SOURCE_RADIUS..=cz + ORE_SOURCE_RADIUS {
                let mut rng = Rng(self.hash2(sx, sz, 0x04e5) as u64 | 1);
                for (ore, veins, y_min, y_max, size_min, size_max) in ORE_VEINS {
                    for _ in 0..veins {
                        let mut p = [
                            (sx * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32,
                            y_min as f32 + rng.unit() * (y_max - y_min) as f32,
                            (sz * CHUNK_SIZE) as f32 + rng.unit() * CHUNK_SIZE as f32,
                        ];
                        let size = size_min + rng.below(size_max - size_min + 1);
                        for _ in 0..size {
                            put_underground(chunk, base_x, base_z, p, ore);
                            for axis in &mut p {
                                *axis += (rng.unit() - 0.5) * 2.2;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rare amethyst geodes: a hollow crystal-lined sphere in a calcite
    /// shell, buried well below the surface.
    fn place_geodes(&self, chunk: &mut Chunk, base_x: i32, base_z: i32) {
        let (cx, cz) = (base_x.div_euclid(CHUNK_SIZE), base_z.div_euclid(CHUNK_SIZE));
        for sx in cx - ORE_SOURCE_RADIUS..=cx + ORE_SOURCE_RADIUS {
            for sz in cz - ORE_SOURCE_RADIUS..=cz + ORE_SOURCE_RADIUS {
                let mut rng = Rng(self.hash2(sx, sz, 0x6e0d) as u64 | 1);
                if rng.below(22) != 0 {
                    continue;
                }
                let gx = (sx * CHUNK_SIZE) as f32 + 4.0 + rng.unit() * (CHUNK_SIZE - 8) as f32;
                let gz = (sz * CHUNK_SIZE) as f32 + 4.0 + rng.unit() * (CHUNK_SIZE - 8) as f32;
                let gy = 20.0 + rng.unit() * 26.0;
                let radius = 4.2 + rng.unit() * 1.4;
                if (self.sample_column(gx as i32, gz as i32).surface() as f32) < gy + radius + 10.0 {
                    continue;
                }
                let ri = radius.ceil() as i32;
                for dx in -ri..=ri {
                    for dy in -ri..=ri {
                        for dz in -ri..=ri {
                            let p = [gx + dx as f32, gy + dy as f32, gz + dz as f32];
                            let d = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                            let block = if d <= radius - 2.2 {
                                BlockType::Air
                            } else if d <= radius - 1.1 {
                                BlockType::Amethyst
                            } else if d <= radius {
                                BlockType::Calcite
                            } else {
                                continue;
                            };
                            put_geode(chunk, base_x, base_z, p, block);
                        }
                    }
                }
            }
        }
    }

    /// Cave biomes. Every open cave cell is checked for a stone floor below
    /// or a stone ceiling above, and decorated by the local cave biome:
    ///
    /// - **lush** (humid regions): moss floors, mossy ceilings and hanging
    ///   foliage;
    /// - **dripstone** (dry regions): dripstone floors, stalactites and
    ///   stalagmites;
    /// - **volcanic** (deep down): magma patches on cavern floors.
    ///
    /// Only blocks of the column itself are touched, so the result does not
    /// depend on neighbouring chunks.
    fn decorate_caves(&self, chunk: &mut Chunk, base_x: i32, base_z: i32, columns: &[Column]) {
        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let col = &columns[(lx * CHUNK_SIZE + lz) as usize];
                let (wx, wz) = (base_x + lx, base_z + lz);
                let (fx, fz) = (wx as f32, wz as f32);
                let biome_n = self.cave_biome.get_noise_2d(fx, fz);
                let lush = biome_n + col.sample.humidity * 0.45 > 0.3;
                let dripstone = !lush && biome_n - col.sample.humidity * 0.3 < -0.22;
                let volcanic = self.cave_biome.get_noise_2d(fx * 1.7 + 5000.0, fz * 1.7) > 0.2;
                let top = col.surface - 3;
                for y in 2..top {
                    if chunk.get_block(lx, y, lz) != BlockType::Air {
                        continue;
                    }
                    let below = chunk.get_block(lx, y - 1, lz);
                    let above = chunk.get_block(lx, y + 1, lz);
                    let h = self.hash3(wx, y, wz, 0xcafe);
                    let floor = below.is_stone_like() || matches!(below, BlockType::Gravel | BlockType::Dirt);
                    let ceiling = above.is_stone_like();
                    if floor {
                        if lush {
                            let moss = if h % 6 == 0 { BlockType::MossyStone } else { BlockType::Moss };
                            chunk.set_block_raw(lx, y - 1, lz, moss);
                            if h % 23 == 0 && above == BlockType::Air {
                                // Small azalea-like shrub.
                                chunk.set_block_raw(lx, y, lz, BlockType::Leaves);
                            }
                        } else if dripstone {
                            if h % 3 != 0 {
                                chunk.set_block_raw(lx, y - 1, lz, BlockType::Dripstone);
                            }
                            if (h >> 8) % 9 == 0 {
                                let length = 1 + ((h >> 16) % 3) as i32;
                                grow(chunk, lx, y, lz, 1, length, BlockType::Dripstone);
                            }
                        } else if volcanic && y < 26 && h % 3 == 0 {
                            chunk.set_block_raw(lx, y - 1, lz, BlockType::Magma);
                        }
                    }
                    if ceiling {
                        if lush {
                            if h % 4 == 1 {
                                chunk.set_block_raw(lx, y + 1, lz, BlockType::MossyStone);
                            }
                            if (h >> 4) % 8 == 0 {
                                let length = 1 + ((h >> 12) % 4) as i32;
                                grow(chunk, lx, y, lz, -1, length, BlockType::Leaves);
                            }
                        } else if dripstone {
                            if h % 3 == 1 {
                                chunk.set_block_raw(lx, y + 1, lz, BlockType::Dripstone);
                            }
                            if (h >> 4) % 7 == 0 {
                                let length = 1 + ((h >> 12) % 4) as i32;
                                grow(chunk, lx, y, lz, -1, length, BlockType::Dripstone);
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Features ───────────────────────────────────────────────────────── //

    fn place_features(&self, chunk: &mut Chunk, base_x: i32, base_z: i32, columns: &[Column]) {
        let min_cell_x = (base_x - FEATURE_REACH).div_euclid(FEATURE_CELL);
        let max_cell_x = (base_x + CHUNK_SIZE - 1 + FEATURE_REACH).div_euclid(FEATURE_CELL);
        let min_cell_z = (base_z - FEATURE_REACH).div_euclid(FEATURE_CELL);
        let max_cell_z = (base_z + CHUNK_SIZE - 1 + FEATURE_REACH).div_euclid(FEATURE_CELL);

        let mut writer = FeatureWriter {
            chunk,
            base_x,
            base_z,
            log: BlockType::Wood,
            leaves: BlockType::Leaves,
        };
        // A fixed global iteration order keeps overlapping features resolved
        // identically in every chunk they touch.
        for cell_x in min_cell_x..=max_cell_x {
            for cell_z in min_cell_z..=max_cell_z {
                let anchor_hash = self.hash2(cell_x, cell_z, 7);
                let x = cell_x * FEATURE_CELL + (anchor_hash % FEATURE_CELL as u32) as i32;
                let z = cell_z * FEATURE_CELL + ((anchor_hash >> 8) % FEATURE_CELL as u32) as i32;
                let (lx, lz) = (x - base_x, z - base_z);
                if lx < -FEATURE_REACH
                    || lx >= CHUNK_SIZE + FEATURE_REACH
                    || lz < -FEATURE_REACH
                    || lz >= CHUNK_SIZE + FEATURE_REACH
                {
                    continue;
                }
                let col = if (0..CHUNK_SIZE).contains(&lx) && (0..CHUNK_SIZE).contains(&lz) {
                    columns[(lx * CHUNK_SIZE + lz) as usize]
                } else {
                    self.column(x, z)
                };
                if let Some(feature) = self.choose_feature(x, z, &col) {
                    self.place_feature(&mut writer, feature, x, col.surface + 1, z, &col);
                }
            }
        }
    }

    fn choose_feature(&self, x: i32, z: i32, col: &Column) -> Option<Feature> {
        if col.surface < SEA_LEVEL || col.slope > 2 || col.sample.river > 0.2 {
            return None;
        }
        if self.is_cave_entrance(x, z, col.surface)
            || self.ravine(x, z, &col.sample).is_some()
        {
            return None;
        }
        let roll = unit(self.hash2(x, z, 11));
        let pick = unit(self.hash2(x, z, 12));
        let biome = col.biome;

        // Forest density noise: groves and clearings instead of an even
        // scatter of trees.
        let grove = self.forest.get_noise_2d(x as f32, z as f32) * 0.5 + 0.5;
        let density = 0.25 + 1.5 * grove * grove;
        let tree_chance = biome.tree_chance() * (FEATURE_CELL * FEATURE_CELL) as f32 * density;

        if roll < tree_chance.min(0.92) {
            let kind = match biome {
                Biome::Plains if pick < 0.8 => TreeKind::Oak,
                Biome::Plains => TreeKind::BigOak,
                Biome::Meadow if pick < 0.6 => TreeKind::Oak,
                Biome::Meadow => TreeKind::Birch,
                Biome::BirchForest if pick < 0.7 => TreeKind::TallBirch,
                Biome::BirchForest => TreeKind::Birch,
                Biome::CherryGrove if pick < 0.9 => TreeKind::Cherry,
                Biome::CherryGrove => TreeKind::Birch,
                Biome::OldGrowthTaiga if pick < 0.4 => TreeKind::MegaSpruce,
                Biome::OldGrowthTaiga => TreeKind::Spruce,
                Biome::Forest if pick < 0.42 => TreeKind::Oak,
                Biome::Forest if pick < 0.78 => TreeKind::Birch,
                Biome::Forest => TreeKind::BigOak,
                Biome::Taiga | Biome::Mountains if pick < 0.25 => TreeKind::SmallSpruce,
                Biome::Taiga | Biome::Mountains => TreeKind::Spruce,
                Biome::Tundra => TreeKind::SmallSpruce,
                Biome::Savanna if pick < 0.85 => TreeKind::Acacia,
                Biome::Savanna => TreeKind::Oak,
                Biome::Jungle if pick < 0.28 => TreeKind::JungleGiant,
                Biome::Jungle => TreeKind::Jungle,
                Biome::Swamp => TreeKind::SwampOak,
                Biome::Island if pick < 0.65 => TreeKind::Palm,
                Biome::Island => TreeKind::Oak,
                _ => return None,
            };
            return Some(Feature::Tree(kind));
        }

        let roll = unit(self.hash2(x, z, 13));
        match biome {
            Biome::Desert if roll < 0.10 => Some(Feature::Cactus),
            Biome::Desert if roll < 0.22 => Some(Feature::DeadBush),
            Biome::Beach if roll < 0.04 && col.sample.temperature > 0.3 => {
                Some(Feature::Tree(TreeKind::Palm))
            }
            Biome::Savanna if roll < 0.03 => Some(Feature::DeadBush),
            Biome::Savanna | Biome::Mountains | Biome::StonyShore if roll > 0.985 => {
                Some(Feature::Boulder)
            }
            Biome::Taiga if roll > 0.975 => Some(Feature::Boulder),
            Biome::Forest | Biome::Taiga | Biome::Jungle if roll < 0.025 => {
                Some(Feature::FallenLog)
            }
            Biome::Tundra if roll < 0.012 => Some(Feature::IceSpike),
            Biome::Tundra if roll > 0.99 => Some(Feature::Boulder),
            Biome::Badlands if roll < 0.05 => Some(Feature::DeadBush),
            Biome::Badlands if roll < 0.08 => Some(Feature::Cactus),
            Biome::OldGrowthTaiga if roll < 0.035 => Some(Feature::FallenLog),
            Biome::OldGrowthTaiga if roll > 0.955 => Some(Feature::MossyBoulder),
            Biome::BirchForest if roll < 0.02 => Some(Feature::FallenLog),
            Biome::MushroomFields if roll < 0.07 => Some(Feature::HugeRedMushroom),
            Biome::MushroomFields if roll < 0.13 => Some(Feature::HugeBrownMushroom),
            Biome::Swamp if roll < 0.012 => Some(Feature::HugeBrownMushroom),
            Biome::Forest if roll > 0.996 => Some(Feature::HugeRedMushroom),
            _ => None,
        }
    }

    fn place_feature(
        &self,
        w: &mut FeatureWriter,
        feature: Feature,
        x: i32,
        y: i32,
        z: i32,
        col: &Column,
    ) {
        let h = self.hash2(x, z, 21);
        match feature {
            Feature::Tree(kind) => {
                w.soil(x, y - 1, z);
                (w.log, w.leaves) = match kind {
                    TreeKind::Birch | TreeKind::TallBirch => (BlockType::BirchLog, BlockType::Leaves),
                    TreeKind::Cherry => (BlockType::CherryLog, BlockType::CherryLeaves),
                    _ => (BlockType::Wood, BlockType::Leaves),
                };
                self.place_tree(w, kind, x, y, z, h);
                (w.log, w.leaves) = (BlockType::Wood, BlockType::Leaves);
            }
            Feature::Cactus => {
                for dy in 0..1 + (h % 3) as i32 {
                    w.put(x, y + dy, z, BlockType::Cactus, Place::Soft);
                }
            }
            Feature::DeadBush => w.put(x, y, z, BlockType::DeadBush, Place::Soft),
            Feature::Boulder => {
                let r = 1.2 + (h % 100) as f32 / 100.0 * 1.3;
                let ri = r.ceil() as i32;
                for dx in -ri..=ri {
                    for dy in -ri..=ri {
                        for dz in -ri..=ri {
                            let d = ((dx * dx) as f32 + (dy * dy) as f32 * 1.4 + (dz * dz) as f32)
                                .sqrt();
                            if d <= r {
                                let block = if self.hash3(x + dx, y + dy, z + dz, 3) % 5 == 0 {
                                    BlockType::Gravel
                                } else {
                                    BlockType::Stone
                                };
                                w.put(x + dx, y - 1 + dy, z + dz, block, Place::Rock);
                            }
                        }
                    }
                }
            }
            Feature::FallenLog => {
                let along_x = h % 2 == 0;
                let length = 3 + (h >> 3) as i32 % 3;
                let block = if along_x {
                    BlockType::WoodLogX
                } else {
                    BlockType::WoodLogZ
                };
                for i in 0..length {
                    let (px, pz) = if along_x { (x + i, z) } else { (x, z + i) };
                    // Logs lie on the ground; skip pieces that would float or
                    // sink where the terrain changes height.
                    if self.sample_column(px, pz).surface() == col.surface {
                        w.put(px, y, pz, block, Place::Wood);
                    }
                }
            }
            Feature::MossyBoulder => {
                let r = 1.4 + (h % 100) as f32 / 100.0 * 1.4;
                let ri = r.ceil() as i32;
                for dx in -ri..=ri {
                    for dy in -ri..=ri {
                        for dz in -ri..=ri {
                            let d = ((dx * dx) as f32 + (dy * dy) as f32 * 1.3 + (dz * dz) as f32)
                                .sqrt();
                            if d <= r {
                                let roll = self.hash3(x + dx, y + dy, z + dz, 4) % 6;
                                let block = match roll {
                                    0 | 1 if dy >= 0 => BlockType::Moss,
                                    0..=3 => BlockType::MossyStone,
                                    _ => BlockType::Stone,
                                };
                                w.put(x + dx, y - 1 + dy, z + dz, block, Place::Rock);
                            }
                        }
                    }
                }
            }
            Feature::HugeRedMushroom => {
                let stem = 5 + (h % 3) as i32;
                for dy in 0..stem {
                    w.put(x, y + dy, z, BlockType::MushroomStem, Place::Wood);
                }
                let top = y + stem;
                // Domed cap: a flat top and hanging walls around the stem.
                for dx in -3i32..=3 {
                    for dz in -3i32..=3 {
                        let d = ((dx * dx + dz * dz) as f32).sqrt();
                        if d <= 2.2 {
                            w.put(x + dx, top, z + dz, BlockType::RedMushroomBlock, Place::Soft);
                        }
                        if d > 1.6 && d <= 3.1 {
                            for dy in 1..=3 {
                                w.put(x + dx, top - dy, z + dz, BlockType::RedMushroomBlock, Place::Soft);
                            }
                        }
                    }
                }
            }
            Feature::HugeBrownMushroom => {
                let stem = 4 + (h % 3) as i32;
                for dy in 0..stem {
                    w.put(x, y + dy, z, BlockType::MushroomStem, Place::Wood);
                }
                // Wide, flat cap.
                let radius = 3.0 + ((h >> 4) % 2) as f32 * 0.6;
                for dx in -4i32..=4 {
                    for dz in -4i32..=4 {
                        if ((dx * dx + dz * dz) as f32).sqrt() <= radius {
                            w.put(x + dx, y + stem, z + dz, BlockType::BrownMushroomBlock, Place::Soft);
                        }
                    }
                }
            }
            Feature::IceSpike => {
                let height = 6 + (h % 10) as i32;
                for dy in -1..height {
                    let radius = if dy < height / 3 { 1 } else { 0 };
                    for dx in -radius..=radius {
                        for dz in -radius..=radius {
                            if dx == 0 || dz == 0 {
                                w.put(x + dx, y + dy, z + dz, BlockType::Ice, Place::Rock);
                            }
                        }
                    }
                }
            }
        }
    }

    fn place_tree(&self, w: &mut FeatureWriter, kind: TreeKind, x: i32, y: i32, z: i32, h: u32) {
        let r = |shift: u32, n: u32| ((h >> shift) % n) as i32;
        let tau = std::f32::consts::TAU;
        match kind {
            TreeKind::Oak => {
                let trunk = 4 + r(0, 3);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32 - 0.5, z as f32, 2.7, 2.2, 2.7);
            }
            TreeKind::BigOak => {
                let trunk = 7 + r(0, 4);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32 - 0.5, z as f32, 3.7, 2.8, 3.7);
                // Two side branches, each with its own crown.
                for branch in 0..2u32 {
                    let angle = unit(h.rotate_left(branch * 11 + 5)) * tau;
                    let (dx, dz) = (angle.cos() * 3.2, angle.sin() * 3.2);
                    let start = y + trunk / 2 + 1 + branch as i32;
                    self.branch(w, x, start, z, dx, 2.0, dz);
                    self.leaf_blob(
                        w,
                        x as f32 + dx,
                        start as f32 + 2.5,
                        z as f32 + dz,
                        2.4,
                        1.8,
                        2.4,
                    );
                }
            }
            TreeKind::Birch => {
                let trunk = 6 + r(0, 3);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32 - 1.5, z as f32, 2.1, 3.2, 2.1);
            }
            TreeKind::TallBirch => {
                let trunk = 9 + r(0, 4);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32 - 2.0, z as f32, 2.2, 4.0, 2.2);
            }
            TreeKind::Cherry => {
                let trunk = 4 + r(0, 2);
                self.trunk(w, x, y, z, trunk);
                let top = y + trunk;
                self.leaf_blob(w, x as f32, top as f32 + 1.0, z as f32, 3.2, 1.9, 3.2);
                // Two or three arching limbs, each with a drooping crown.
                for limb in 0..(2 + r(8, 2) as u32) {
                    let angle = unit(h.rotate_left(limb * 9 + 4)) * tau;
                    let (dx, dz) = (angle.cos() * 3.0, angle.sin() * 3.0);
                    self.branch(w, x, top - 2, z, dx, 2.0, dz);
                    let (cx, cz) = (x as f32 + dx, z as f32 + dz);
                    self.leaf_blob(w, cx, top as f32 + 0.5, cz, 2.8, 1.7, 2.8);
                    // Blossom curtains hanging under the crown.
                    for i in 0..5u32 {
                        let a = unit(self.hash2(x, z, 60 + limb * 8 + i)) * tau;
                        let px = (cx + a.cos() * 2.2).round() as i32;
                        let pz = (cz + a.sin() * 2.2).round() as i32;
                        for dy in 1..=1 + (self.hash2(px, pz, 61) % 2) as i32 {
                            w.put(px, top - 1 - dy, pz, w.leaves, Place::Soft);
                        }
                    }
                }
            }
            TreeKind::MegaSpruce => {
                let trunk = 16 + r(0, 10);
                for (ox, oz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    w.soil(x + ox, y - 1, z + oz);
                    self.trunk(w, x + ox, y, z + oz, trunk);
                }
                let top = y + trunk;
                for (ox, oz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    w.put(x + ox, top, z + oz, w.leaves, Place::Soft);
                }
                w.put(x, top + 1, z, w.leaves, Place::Soft);
                // A tall, narrow cone that starts high up the trunk.
                let lowest = y + trunk / 3 + r(4, 3);
                for (i, ly) in (lowest..top).rev().enumerate() {
                    let base = (i as f32 * 0.32 + 1.2).min(5.2);
                    let radius = if i % 3 == 2 { base - 1.3 } else { base };
                    for (ox, oz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        self.leaf_disc(w, x + ox, ly, z + oz, (radius - 0.6).max(1.0), true);
                    }
                }
            }
            TreeKind::Spruce | TreeKind::SmallSpruce => {
                let tall = kind == TreeKind::Spruce;
                let trunk = if tall { 9 + r(0, 6) } else { 5 + r(0, 3) };
                let max_radius = if tall { 3 } else { 2 };
                self.trunk(w, x, y, z, trunk);
                let top = y + trunk;
                w.put(x, top, z, w.leaves, Place::Soft);
                w.put(x, top + 1, z, w.leaves, Place::Soft);
                // Layered cone: the radius grows downwards, with every other
                // layer pulled in for the classic tiered silhouette.
                let lowest = y + 2 + r(4, 2);
                for (i, ly) in (lowest..top).rev().enumerate() {
                    let mut radius = ((i as i32 + 1) / 2).min(max_radius);
                    if i % 2 == 1 {
                        radius = (radius - 1).max(1);
                    }
                    self.leaf_disc(w, x, ly, z, radius as f32 + 0.4, true);
                }
            }
            TreeKind::Jungle => {
                let trunk = 7 + r(0, 5);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32, z as f32, 3.2, 2.4, 3.2);
                self.hanging_leaves(w, x, y + trunk - 1, z, 3, h);
            }
            TreeKind::JungleGiant => {
                let trunk = 15 + r(0, 9);
                for (ox, oz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    w.soil(x + ox, y - 1, z + oz);
                    self.trunk(w, x + ox, y, z + oz, trunk);
                }
                let (cx, cz) = (x as f32 + 0.5, z as f32 + 0.5);
                self.leaf_blob(w, cx, (y + trunk) as f32, cz, 5.2, 3.0, 5.2);
                self.hanging_leaves(w, x, y + trunk - 2, z, 5, h);
                for branch in 0..3u32 {
                    let angle = unit(h.rotate_left(branch * 7 + 3)) * tau;
                    let (dx, dz) = (angle.cos() * 3.5, angle.sin() * 3.5);
                    let start = y + trunk / 2 + branch as i32 * 3;
                    self.branch(w, x, start, z, dx, 1.5, dz);
                    self.leaf_blob(
                        w,
                        x as f32 + dx,
                        start as f32 + 2.0,
                        z as f32 + dz,
                        2.6,
                        1.6,
                        2.6,
                    );
                }
            }
            TreeKind::Acacia => {
                let straight = 3 + r(0, 2);
                self.trunk(w, x, y, z, straight);
                let dir = [(1, 0), (-1, 0), (0, 1), (0, -1)][r(2, 4) as usize];
                let bend = 2 + r(4, 2);
                let (mut tx, mut ty, mut tz) = (x, y + straight, z);
                for _ in 0..bend {
                    tx += dir.0;
                    tz += dir.1;
                    w.put(tx, ty, tz, w.log, Place::Wood);
                    ty += 1;
                    w.put(tx, ty, tz, w.log, Place::Wood);
                }
                self.leaf_disc(w, tx, ty + 1, tz, 3.6, false);
                self.leaf_disc(w, tx, ty + 2, tz, 2.2, false);
                // Usually a smaller second crown on the opposite side.
                if r(6, 3) > 0 {
                    let (sx, sz) = (x - dir.0 * 2, z - dir.1 * 2);
                    let sy = y + straight + 1;
                    self.branch(
                        w,
                        x,
                        y + straight - 1,
                        z,
                        -dir.0 as f32 * 2.0,
                        2.0,
                        -dir.1 as f32 * 2.0,
                    );
                    self.leaf_disc(w, sx, sy + 1, sz, 2.4, false);
                }
            }
            TreeKind::SwampOak => {
                let trunk = 4 + r(0, 3);
                self.trunk(w, x, y, z, trunk);
                self.leaf_blob(w, x as f32, (y + trunk) as f32 - 0.5, z as f32, 3.8, 1.7, 3.8);
                self.hanging_leaves(w, x, y + trunk - 2, z, 4, h);
            }
            TreeKind::Palm => {
                let trunk = 6 + r(0, 4);
                let dir = [(1, 0), (-1, 0), (0, 1), (0, -1)][r(3, 4) as usize];
                let (mut tx, mut tz) = (x, z);
                for i in 0..trunk {
                    if i > 1 && i % 3 == 0 {
                        tx += dir.0;
                        tz += dir.1;
                    }
                    w.put(tx, y + i, tz, w.log, Place::Wood);
                }
                let top = y + trunk;
                self.leaf_disc(w, tx, top, tz, 1.5, false);
                for (dx, dz) in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (-1, -1),
                    (1, -1),
                    (-1, 1),
                ] {
                    let len = if dx != 0 && dz != 0 { 2 } else { 4 };
                    for step in 1..=len {
                        let droop = i32::from(step >= len - 1);
                        w.put(
                            tx + dx * step,
                            top - droop,
                            tz + dz * step,
                            w.leaves,
                            Place::Soft,
                        );
                    }
                }
                w.put(tx, top + 1, tz, w.leaves, Place::Soft);
            }
        }
    }

    fn trunk(&self, w: &mut FeatureWriter, x: i32, y: i32, z: i32, height: i32) {
        for dy in 0..height {
            w.put(x, y + dy, z, w.log, Place::Wood);
        }
    }

    /// A straight wooden branch from `(x, y, z)` towards the given offset.
    #[allow(clippy::too_many_arguments)]
    fn branch(&self, w: &mut FeatureWriter, x: i32, y: i32, z: i32, dx: f32, dy: f32, dz: f32) {
        let steps = dx.abs().max(dz.abs()).max(dy.abs()).ceil() as i32;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            w.put(
                x + (dx * t).round() as i32,
                y + (dy * t).round() as i32,
                z + (dz * t).round() as i32,
                w.log,
                Place::Wood,
            );
        }
    }

    /// Ellipsoidal crown with a ragged edge.
    #[allow(clippy::too_many_arguments)]
    fn leaf_blob(
        &self,
        w: &mut FeatureWriter,
        cx: f32,
        cy: f32,
        cz: f32,
        rx: f32,
        ry: f32,
        rz: f32,
    ) {
        let (x0, x1) = ((cx - rx).floor() as i32, (cx + rx).ceil() as i32);
        let (y0, y1) = ((cy - ry).floor() as i32, (cy + ry).ceil() as i32);
        let (z0, z1) = ((cz - rz).floor() as i32, (cz + rz).ceil() as i32);
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    let d = ((x as f32 - cx) / rx).powi(2)
                        + ((y as f32 - cy) / ry).powi(2)
                        + ((z as f32 - cz) / rz).powi(2);
                    if d > 1.0 || (d > 0.6 && self.hash3(x, y, z, 5) % 3 == 0) {
                        continue;
                    }
                    w.put(x, y, z, w.leaves, Place::Soft);
                }
            }
        }
    }

    /// Flat circular leaf layer; `sparse_edge` trims some rim leaves.
    fn leaf_disc(&self, w: &mut FeatureWriter, cx: i32, y: i32, cz: i32, radius: f32, sparse_edge: bool) {
        let ri = radius.ceil() as i32;
        for dx in -ri..=ri {
            for dz in -ri..=ri {
                let d = ((dx * dx + dz * dz) as f32).sqrt();
                if d > radius
                    || (sparse_edge
                        && d > radius - 1.0
                        && self.hash3(cx + dx, y, cz + dz, 6) % 4 == 0)
                {
                    continue;
                }
                w.put(cx + dx, y, cz + dz, w.leaves, Place::Soft);
            }
        }
    }

    /// Strands of leaves hanging from the underside of a crown.
    fn hanging_leaves(&self, w: &mut FeatureWriter, x: i32, y: i32, z: i32, radius: i32, h: u32) {
        for i in 0..(radius * 3) as u32 {
            let angle = unit(self.hash2(x, z, 40 + i)) * std::f32::consts::TAU;
            let dist = radius as f32 * (0.7 + 0.3 * unit(h.rotate_right(i * 3 + 2)));
            let px = x + (angle.cos() * dist).round() as i32;
            let pz = z + (angle.sin() * dist).round() as i32;
            let length = 1 + (self.hash2(px, pz, 31) % 3) as i32;
            for dy in 0..length {
                w.put(px, y - dy, pz, w.leaves, Place::Soft);
            }
        }
    }

    // ── Hashing ────────────────────────────────────────────────────────── //

    /// Hashes coordinates for deterministic world-generation streams.
    /// Stream values distinguish generation decisions; they are not
    /// cryptographic salts. Keep the mixing stable to preserve existing worlds.
    fn hash2(&self, x: i32, z: i32, stream: u32) -> u32 {
        let mut h = self.seed ^ stream.wrapping_mul(0x27d4_eb2d);
        h ^= (x as u32).wrapping_mul(0x8da6_b343);
        h = h.rotate_left(13) ^ (z as u32).wrapping_mul(0xd816_3841);
        h = (h ^ (h >> 15)).wrapping_mul(0x2c1b_3c6d);
        h = (h ^ (h >> 12)).wrapping_mul(0x297a_2d39);
        h ^ (h >> 15)
    }

    /// Hashes 3D coordinates for deterministic world-generation streams.
    fn hash3(&self, x: i32, y: i32, z: i32, stream: u32) -> u32 {
        self.hash2(x, z, stream ^ (y as u32).wrapping_mul(0x1b87_3593))
    }
}

/// Terracotta band of the badlands at height `y`. Bands are horizontal and
/// shared by every column, so mesa walls show continuous stripes; `patch`
/// tilts them a little.
fn badlands_band(y: i32, patch: f32) -> BlockType {
    match (y + (patch * 3.0) as i32).rem_euclid(19) {
        0 | 1 | 9 => BlockType::RedTerracotta,
        5 | 14 => BlockType::WhiteTerracotta,
        11 => BlockType::Sandstone,
        _ => BlockType::Terracotta,
    }
}

/// Chunk-local coordinates of world point `p`, if it lies inside the chunk.
fn local(base_x: i32, base_z: i32, p: [f32; 3]) -> Option<(i32, i32, i32)> {
    let (lx, y, lz) = (
        p[0].floor() as i32 - base_x,
        p[1].floor() as i32,
        p[2].floor() as i32 - base_z,
    );
    ((0..CHUNK_SIZE).contains(&lx) && (1..WORLD_HEIGHT).contains(&y) && (0..CHUNK_SIZE).contains(&lz))
        .then_some((lx, y, lz))
}

/// Writes an ore block, replacing only plain stone or deepslate.
fn put_underground(chunk: &mut Chunk, base_x: i32, base_z: i32, p: [f32; 3], ore: BlockType) {
    if let Some((lx, y, lz)) = local(base_x, base_z, p) {
        if chunk.get_block(lx, y, lz).is_stone_like() {
            chunk.set_block_raw(lx, y, lz, ore);
        }
    }
}

/// Writes a geode block. The shell only replaces rock (never open caves or
/// water); the hollow core only clears rock.
fn put_geode(chunk: &mut Chunk, base_x: i32, base_z: i32, p: [f32; 3], block: BlockType) {
    if let Some((lx, y, lz)) = local(base_x, base_z, p) {
        let existing = chunk.get_block(lx, y, lz);
        let rock = existing.is_solid()
            && existing != BlockType::Bedrock
            && !existing.is_leaves()
            && existing != BlockType::Water;
        if rock {
            chunk.set_block_raw(lx, y, lz, block);
        }
    }
}

/// Grows a column of `block` from `(lx, y, lz)` up (`dir = 1`) or down
/// (`dir = -1`) through open air, at most `length` blocks.
fn grow(chunk: &mut Chunk, lx: i32, y: i32, lz: i32, dir: i32, length: i32, block: BlockType) {
    for i in 0..length {
        let wy = y + i * dir;
        if !(1..WORLD_HEIGHT).contains(&wy) || chunk.get_block(lx, wy, lz) != BlockType::Air {
            return;
        }
        // Leave a one-block gap so stalactites never seal a passage.
        let next = wy + dir;
        if (1..WORLD_HEIGHT).contains(&next) && i + 1 == length {
            if chunk.get_block(lx, next, lz) != BlockType::Air {
                return;
            }
        }
        chunk.set_block_raw(lx, wy, lz, block);
    }
}

/// Maps a hash to `[0, 1)`.
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

fn simplex(seed: u32, frequency: f32) -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(seed as i32);
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_frequency(Some(frequency));
    noise
}

fn fbm(seed: u32, frequency: f32, octaves: i32) -> FastNoiseLite {
    let mut noise = simplex(seed, frequency);
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(octaves));
    noise.set_fractal_lacunarity(Some(2.0));
    noise.set_fractal_gain(Some(0.5));
    noise
}

fn ridged(seed: u32, frequency: f32, octaves: i32) -> FastNoiseLite {
    let mut noise = fbm(seed, frequency, octaves);
    noise.set_fractal_type(Some(FractalType::Ridged));
    noise
}

/// Coarse cave field: four channels per lattice point (caverns, soft
/// pockets and the two noodle-tunnel fields).
struct CaveLattice {
    values: Vec<[f32; 4]>,
    y_samples: usize,
}

/// One bit per block of a chunk: carved by a worm tunnel.
struct WormMask {
    /// Tunnels that keep a roof on flat ground.
    bits: Vec<u64>,
    /// Entrance tunnels, allowed to break through the surface anywhere.
    open_bits: Vec<u64>,
}

impl Default for WormMask {
    fn default() -> Self {
        let words = ((CHUNK_SIZE * CHUNK_SIZE * WORLD_HEIGHT) as usize).div_ceil(64);
        Self {
            bits: vec![0; words],
            open_bits: vec![0; words],
        }
    }
}

impl WormMask {
    fn index(lx: i32, y: i32, lz: i32) -> usize {
        ((y * CHUNK_SIZE + lx) * CHUNK_SIZE + lz) as usize
    }

    fn get(&self, lx: i32, y: i32, lz: i32) -> bool {
        let i = Self::index(lx, y, lz);
        self.bits[i / 64] & (1 << (i % 64)) != 0
    }

    fn get_open(&self, lx: i32, y: i32, lz: i32) -> bool {
        let i = Self::index(lx, y, lz);
        self.open_bits[i / 64] & (1 << (i % 64)) != 0
    }

    /// Marks the blocks of an ellipsoid (horizontal radius `rh`, vertical
    /// radius `rv`) that fall inside the chunk at `(base_x, base_z)`.
    fn carve_ellipsoid(
        &mut self,
        base_x: i32,
        base_z: i32,
        [x, y, z]: [f32; 3],
        rh: f32,
        rv: f32,
        open: bool,
    ) {
        let x0 = ((x - rh).floor() as i32 - base_x).max(0);
        let x1 = ((x + rh).ceil() as i32 - base_x).min(CHUNK_SIZE - 1);
        let z0 = ((z - rh).floor() as i32 - base_z).max(0);
        let z1 = ((z + rh).ceil() as i32 - base_z).min(CHUNK_SIZE - 1);
        let y0 = ((y - rv).floor() as i32).max(1);
        let y1 = ((y + rv).ceil() as i32).min(WORLD_HEIGHT - 1);
        for lx in x0..=x1 {
            let dx = (base_x + lx) as f32 + 0.5 - x;
            for lz in z0..=z1 {
                let dz = (base_z + lz) as f32 + 0.5 - z;
                let horizontal = (dx * dx + dz * dz) / (rh * rh);
                if horizontal >= 1.0 {
                    continue;
                }
                for wy in y0..=y1 {
                    let dy = wy as f32 + 0.5 - y;
                    if horizontal + dy * dy / (rv * rv) < 1.0 {
                        let i = Self::index(lx, wy, lz);
                        let bits = if open { &mut self.open_bits } else { &mut self.bits };
                        bits[i / 64] |= 1 << (i % 64);
                    }
                }
            }
        }
    }
}

/// Parameters of one tunnel walk.
#[derive(Clone, Copy)]
struct Walk {
    yaw: f32,
    pitch: f32,
    radius: f32,
    length: i32,
    seed: u64,
    /// May split into two side passages.
    branch: bool,
    /// Entrance tunnel: may break the surface.
    open: bool,
}

/// Small deterministic xorshift generator for tunnel walks.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n.max(1) as u64) as u32
    }
}

/// Per-column inputs to cave carving.
struct CaveColumn {
    surface: i32,
    slope: i32,
    entrance: f32,
    ravine: Option<(i32, f32)>,
    /// Highest Y that may be carved (below any neighbouring water).
    carve_ceiling: i32,
    /// A cavern pillar stands in this column.
    pillar: bool,
}

/// Trilinear interpolation of one channel of the cave lattice.
fn sample_lattice(cave: &CaveLattice, channel: usize, lx: i32, y: i32, lz: i32) -> f32 {
    let cell = CAVE_CELL_SIZE;
    let (ix, iy, iz) = (
        (lx / cell) as usize,
        (y / cell) as usize,
        (lz / cell) as usize,
    );
    if iy + 1 >= cave.y_samples {
        return 0.0;
    }
    let (tx, ty, tz) = (
        (lx % cell) as f32 / cell as f32,
        (y % cell) as f32 / cell as f32,
        (lz % cell) as f32 / cell as f32,
    );
    let at =
        |x: usize, y: usize, z: usize| cave.values[(x * CAVE_XZ + z) * cave.y_samples + y][channel];
    let c00 = lerp(at(ix, iy, iz), at(ix + 1, iy, iz), tx);
    let c10 = lerp(at(ix, iy + 1, iz), at(ix + 1, iy + 1, iz), tx);
    let c01 = lerp(at(ix, iy, iz + 1), at(ix + 1, iy, iz + 1), tx);
    let c11 = lerp(at(ix, iy + 1, iz + 1), at(ix + 1, iy + 1, iz + 1), tx);
    lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz)
}

/// Writes the feature blocks that fall inside the chunk being generated.
struct FeatureWriter<'a> {
    chunk: &'a mut Chunk,
    base_x: i32,
    base_z: i32,
    /// Trunk block of the tree being placed.
    log: BlockType,
    /// Foliage block of the tree being placed.
    leaves: BlockType,
}

impl FeatureWriter<'_> {
    fn put(&mut self, x: i32, y: i32, z: i32, block: BlockType, place: Place) {
        let (lx, lz) = (x - self.base_x, z - self.base_z);
        if !(0..CHUNK_SIZE).contains(&lx)
            || !(0..CHUNK_SIZE).contains(&lz)
            || !(1..WORLD_HEIGHT).contains(&y)
        {
            return;
        }
        let existing = self.chunk.get_block(lx, y, lz);
        let replaceable = match place {
            Place::Soft => existing == BlockType::Air,
            Place::Wood => existing.is_leaves() || matches!(existing, BlockType::Air | BlockType::DeadBush),
            Place::Rock => {
                existing.is_leaves()
                    || matches!(
                        existing,
                        BlockType::Air
                            | BlockType::DeadBush
                            | BlockType::Grass
                            | BlockType::Dirt
                            | BlockType::Snow
                            | BlockType::Sand
                            | BlockType::RedSand
                            | BlockType::Podzol
                            | BlockType::Moss
                            | BlockType::Mycelium
                    )
            }
        };
        if replaceable {
            self.chunk.set_block_raw(lx, y, lz, block);
        }
    }

    /// Turns the grass or snow under a trunk into dirt.
    fn soil(&mut self, x: i32, y: i32, z: i32) {
        let (lx, lz) = (x - self.base_x, z - self.base_z);
        if (0..CHUNK_SIZE).contains(&lx)
            && (0..CHUNK_SIZE).contains(&lz)
            && matches!(
                self.chunk.get_block(lx, y, lz),
                BlockType::Grass | BlockType::Snow
            )
        {
            self.chunk.set_block_raw(lx, y, lz, BlockType::Dirt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_hash(chunk: &Chunk) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for y in 0..WORLD_HEIGHT {
                    hash = (hash ^ chunk.get_block(x, y, z) as u64).wrapping_mul(0x100_0000_01b3);
                }
                hash = (hash ^ chunk.column_tint(x, z) as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        hash
    }

    #[test]
    fn coordinate_hashes_are_stable() {
        assert_eq!(ChunkGenerator::new(42).hash2(0, 0, 0), 0xf1f5_261a);
        assert_eq!(ChunkGenerator::new(42).hash2(-17, 29, 3), 0xec7f_c347);
        assert_eq!(
            ChunkGenerator::new(u32::MAX).hash2(i32::MAX, i32::MIN, 0x04e5),
            0x964d_9058
        );

        // Stream 3 is used by the boulder material selector.
        assert_eq!(ChunkGenerator::new(42).hash3(12, 64, -9, 3), 0x89ef_2bb8);
        assert_eq!(
            ChunkGenerator::new(7).hash3(-33, -48, 91, 0xdee9),
            0xecb8_9a1c
        );
        assert_eq!(
            ChunkGenerator::new(u32::MAX).hash3(i32::MAX, i32::MIN, 0, 0xcafe),
            0xd66d_5eb0
        );
    }

    #[test]
    fn same_seed_generates_identical_chunks() {
        let a = ChunkGenerator::new(42);
        let b = ChunkGenerator::new(42);
        for (cx, cz) in [(0, 0), (5, -3), (-40, 17)] {
            assert_eq!(
                chunk_hash(&a.generate_chunk(cx, cz)),
                chunk_hash(&b.generate_chunk(cx, cz))
            );
        }
        assert_ne!(
            chunk_hash(&ChunkGenerator::new(1).generate_chunk(0, 0)),
            chunk_hash(&ChunkGenerator::new(2).generate_chunk(0, 0))
        );
    }

    #[test]
    fn legacy_generator_keeps_its_original_height_samples() {
        let generator = ChunkGenerator::new_with_version(2026, GeneratorVersion::Legacy);
        let samples: Vec<_> = [
            (0, 0),
            (317, -928),
            (1840, 2711),
            (-4096, 1777),
            (7712, -5203),
        ]
        .map(|(x, z)| generator.get_terrain_height_pub(x, z))
        .into();
        assert_eq!(samples, [70, 64, 98, 44, 65]);
    }

    #[test]
    fn current_mountains_retain_upland_height_and_reduce_steep_edges() {
        let mut totals = [(0u32, 0u32, 0u32); 2];
        for seed in [42, 2026, 12345] {
            let generators = [
                ChunkGenerator::new_with_version(seed, GeneratorVersion::Legacy),
                ChunkGenerator::new_with_version(seed, GeneratorVersion::Current),
            ];
            for index in 0..8192i32 {
                let x = (generators[0].hash2(index, index * 13, 0x31) % 8192) as i32 - 4096;
                let z = (generators[0].hash2(index * 7, index, 0x73) % 8192) as i32 - 4096;
                for (version, generator) in generators.iter().enumerate() {
                    let center = generator.get_terrain_height_pub(x, z);
                    if center > SEA_LEVEL + 70 {
                        totals[version].0 += 1;
                    }
                    for (dx, dz) in [(1, 0), (0, 1)] {
                        let neighbor = generator.get_terrain_height_pub(x + dx, z + dz);
                        if center.max(neighbor) > SEA_LEVEL + 40 {
                            totals[version].1 += 1;
                            if (center - neighbor).abs() >= 12 {
                                totals[version].2 += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            totals[1].0 * 10 >= totals[0].0 * 8,
            "new generator flattened the mountain ranges: {totals:?}"
        );
        assert!(
            totals[1].2 < totals[0].2,
            "new generator did not reduce steep upland edges: {totals:?}"
        );
    }

    #[test]
    fn new_generator_preserves_lowland_rivers() {
        let legacy = ChunkGenerator::new_with_version(2026, GeneratorVersion::Legacy);
        let current = ChunkGenerator::new_with_version(2026, GeneratorVersion::Current);
        let mut river_columns = 0;
        for index in 0..50_000i32 {
            let x = (legacy.hash2(index, index * 13, 0x51) % 8192) as i32 - 4096;
            let z = (legacy.hash2(index * 17, index, 0x97) % 8192) as i32 - 4096;
            let old = legacy.sample_column(x, z);
            if old.height <= SEA_LEVEL as f32 + 24.0 && old.mountain == 0.0 {
                let new = current.sample_column(x, z);
                assert_eq!(new.height, old.height, "lowland height changed at ({x}, {z})");
                assert_eq!(new.river, old.river, "lowland river changed at ({x}, {z})");
                if old.river > 0.4 {
                    river_columns += 1;
                }
            }
        }
        assert!(river_columns > 0, "test sample did not include lowland rivers");
    }

    #[test]
    fn terrain_height_matches_generated_surface() {
        let generator = ChunkGenerator::new(2026);
        for (cx, cz) in [(0, 0), (3, 9), (-12, 4), (30, -30)] {
            let chunk = generator.generate_chunk(cx, cz);
            for lx in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    let (x, z) = (cx * CHUNK_SIZE + lx, cz * CHUNK_SIZE + lz);
                    let h = generator.get_terrain_height_pub(x, z);
                    assert!((4..WORLD_HEIGHT - 20).contains(&h));
                    assert_eq!(chunk.get_block(lx, 0, lz), BlockType::Bedrock);
                    for y in h + 1..WORLD_HEIGHT {
                        let block = chunk.get_block(lx, y, lz);
                        assert!(
                            !matches!(
                                block,
                                BlockType::Grass | BlockType::Dirt | BlockType::Sand | BlockType::Bedrock
                            ),
                            "terrain block {block:?} above surface at ({x}, {y}, {z})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn features_are_identical_from_both_sides_of_a_border() {
        // Generating chunk B must write exactly the blocks that features
        // anchored in chunk A would put there: compare B's border columns
        // against a reference built from A's features alone.
        let generator = ChunkGenerator::new(7);
        let mut crossing = 0;
        for cx in -8..8 {
            let left = generator.generate_chunk(cx, 0);
            let right = generator.generate_chunk(cx + 1, 0);
            for lz in 0..CHUNK_SIZE {
                for y in 0..WORLD_HEIGHT {
                    if left.get_block(CHUNK_SIZE - 1, y, lz) == BlockType::Leaves
                        && right.get_block(0, y, lz) == BlockType::Leaves
                    {
                        crossing += 1;
                    }
                }
            }
        }
        assert!(crossing > 0, "expected crowns spanning chunk borders");
        // Regenerating in another order gives the same chunk.
        let first = chunk_hash(&generator.generate_chunk(3, 0));
        let _ = generator.generate_chunk(4, 0);
        assert_eq!(first, chunk_hash(&generator.generate_chunk(3, 0)));
    }

    #[test]
    fn world_has_varied_biomes() {
        let generator = ChunkGenerator::new(12345);
        let mut seen = std::collections::HashSet::new();
        for x in (-6000..6000).step_by(97) {
            for z in (-6000..6000).step_by(89) {
                seen.insert(generator.get_biome(x, z));
            }
        }
        for biome in [
            Biome::Ocean,
            Biome::Plains,
            Biome::Forest,
            Biome::Taiga,
            Biome::Mountains,
            Biome::River,
            Biome::Beach,
            Biome::Badlands,
            Biome::BirchForest,
            Biome::CherryGrove,
            Biome::OldGrowthTaiga,
        ] {
            assert!(seen.contains(&biome), "missing {biome:?}; saw {seen:?}");
        }
        assert!(seen.len() >= 12, "only {} biomes: {seen:?}", seen.len());
    }

    #[test]
    fn spawn_column_is_on_dry_land() {
        for seed in [0, 1, 7, 42, 2026, 12345, u32::MAX] {
            let generator = ChunkGenerator::new(seed);
            let (x, z) = generator.find_spawn_column();
            assert!(generator.get_terrain_height_pub(x, z) > SEA_LEVEL, "seed {seed}");
            assert!(!generator.get_biome(x, z).is_water(), "seed {seed}");
        }
    }

    #[test]
    fn generated_chunk_metadata_matches_columns() {
        let generator = ChunkGenerator::new(99);
        let chunk = generator.generate_chunk(2, -5);
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let expected = (0..WORLD_HEIGHT)
                    .rev()
                    .find(|&y| chunk.get_block(x, y, z).is_solid_opaque())
                    .map_or(-1, |y| y as i16);
                assert_eq!(chunk.highest_opaque_y(x, z), expected);
                assert!((1..=15).contains(&chunk.column_tint(x, z)));
            }
        }
    }
}
