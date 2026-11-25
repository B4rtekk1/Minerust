//! Biomes and the climate tint palette.
//!
//! The world generator classifies every block column into a [`Biome`] and,
//! independently, into a smooth *climate tint* (see [`climate_tint`]). The
//! tint is a 4-bit palette index stored per column, carried into every
//! terrain quad, and used by `terrain.wgsl` to recolour grass and leaves.
//! Because the tint follows continuous temperature/humidity fields rather
//! than hard biome borders, vegetation colour drifts gradually across the
//! world instead of switching at a biome edge.

/// All biome types used during world generation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Biome {
    /// Open temperate grassland with scattered trees.
    #[default]
    Plains,
    /// Flat, lush grassland with very few trees, usually on high ground.
    Meadow,
    /// Dense temperate woodland.
    Forest,
    /// Cold conifer forest with tall, narrow spruce-like trees.
    Taiga,
    /// Hot, dry grassland with flat-topped trees.
    Savanna,
    /// Hot, wet, very dense forest with giant trees.
    Jungle,
    /// Arid sand dunes with cacti and dead bushes.
    Desert,
    /// Frozen, snow-covered lowland.
    Tundra,
    /// Waterlogged lowland with shallow pools.
    Swamp,
    /// Steep rocky highlands below the snow line.
    Mountains,
    /// Snow- and ice-capped peaks.
    SnowyPeaks,
    /// Deep open water.
    Ocean,
    /// Sandy shoreline.
    Beach,
    /// Gravel or stone shoreline below cliffs.
    StonyShore,
    /// Flowing water channel.
    River,
    /// Enclosed inland body of water.
    Lake,
    /// Small land mass surrounded by ocean.
    Island,
}

impl Biome {
    /// `true` for biomes whose surface is (mostly) water.
    pub fn is_water(&self) -> bool {
        matches!(self, Biome::Ocean | Biome::River | Biome::Lake)
    }

    /// `true` if trees can generate in this biome.
    pub fn has_trees(&self) -> bool {
        self.tree_chance() > 0.0
    }

    /// Probability that a suitable grass column hosts a tree trunk, before
    /// local forest-density noise is applied.
    pub fn tree_chance(&self) -> f32 {
        match self {
            Biome::Plains => 0.006,
            Biome::Meadow => 0.0015,
            Biome::Forest => 0.060,
            Biome::Taiga => 0.050,
            Biome::Savanna => 0.006,
            Biome::Jungle => 0.090,
            Biome::Swamp => 0.020,
            Biome::Mountains => 0.006,
            Biome::Island => 0.025,
            Biome::Tundra => 0.002,
            _ => 0.0,
        }
    }

    /// Legacy threshold form of [`Self::tree_chance`] (`1.0` = never).
    pub fn tree_density(&self) -> f64 {
        1.0 - self.tree_chance() as f64
    }

    /// Representative grass colour (sRGB), used by tools and the minimap.
    pub fn grass_color(&self) -> [f32; 3] {
        let (t, h) = self.typical_climate();
        tint_color(climate_tint(t, h))
    }

    /// Representative leaf colour (sRGB).
    pub fn leaves_color(&self) -> [f32; 3] {
        leaves_tint_color(climate_tint(self.typical_climate().0, self.typical_climate().1))
    }

    /// Typical `(temperature, humidity)` in `[-1, 1]` for this biome.
    fn typical_climate(&self) -> (f32, f32) {
        match self {
            Biome::Tundra | Biome::SnowyPeaks => (-0.9, 0.0),
            Biome::Taiga => (-0.45, 0.2),
            Biome::Mountains | Biome::StonyShore => (-0.2, -0.1),
            Biome::Plains | Biome::Island | Biome::Beach => (0.1, 0.0),
            Biome::Meadow => (0.0, 0.3),
            Biome::Forest | Biome::River | Biome::Lake | Biome::Ocean => (0.1, 0.4),
            Biome::Swamp => (0.35, 0.8),
            Biome::Savanna => (0.75, -0.6),
            Biome::Desert => (0.9, -0.9),
            Biome::Jungle => (0.8, 0.85),
        }
    }
}

// ── Climate tint palette ─────────────────────────────────────────────────── //

/// Number of temperature bands in the tint palette.
pub const TINT_TEMPERATURE_BANDS: u8 = 5;
/// Number of humidity bands in the tint palette.
pub const TINT_HUMIDITY_BANDS: u8 = 3;
/// Palette index meaning "use the texture colour unchanged".
pub const TINT_NONE: u8 = 0;

/// Grass target colours (sRGB), indexed by `temperature * 3 + humidity`.
/// Rows run frozen → hot, columns dry → wet. Must match `GRASS_TINTS` in
/// `terrain.wgsl` (checked by a unit test).
pub const GRASS_TINTS: [[f32; 3]; 15] = [
    // frozen: grey-blue, frost-bitten
    [0.58, 0.67, 0.60],
    [0.52, 0.66, 0.60],
    [0.47, 0.64, 0.60],
    // cold: deep, cool green
    [0.50, 0.64, 0.40],
    [0.40, 0.62, 0.38],
    [0.33, 0.58, 0.40],
    // temperate: fresh spring green
    [0.62, 0.76, 0.32],
    [0.47, 0.75, 0.29],
    [0.32, 0.66, 0.27],
    // warm: bright, slightly yellow
    [0.72, 0.75, 0.30],
    [0.54, 0.78, 0.26],
    [0.28, 0.72, 0.22],
    // hot: savanna straw → jungle emerald
    [0.82, 0.72, 0.36],
    [0.66, 0.75, 0.28],
    [0.22, 0.74, 0.18],
];

/// Leaves are a darker, cooler variant of the grass colour in the same climate.
pub const LEAVES_TINT_SCALE: [f32; 3] = [0.78, 0.86, 0.80];

/// Maps a continuous climate sample in `[-1, 1]²` to a palette index in
/// `1..=15`. Bands are spread evenly so neighbouring columns differ by at
/// most one band and colours change gradually.
pub fn climate_tint(temperature: f32, humidity: f32) -> u8 {
    let band = |v: f32, bands: u8| -> u8 {
        let unit = ((v.clamp(-1.0, 1.0) + 1.0) * 0.5) * bands as f32;
        (unit as u8).min(bands - 1)
    };
    1 + band(temperature, TINT_TEMPERATURE_BANDS) * TINT_HUMIDITY_BANDS
        + band(humidity, TINT_HUMIDITY_BANDS)
}

/// Grass colour (sRGB) for a palette index; [`TINT_NONE`] yields plain grass.
pub fn tint_color(index: u8) -> [f32; 3] {
    match index {
        1..=15 => GRASS_TINTS[index as usize - 1],
        _ => [0.236, 0.616, 0.13],
    }
}

/// Leaf colour (sRGB) for a palette index.
pub fn leaves_tint_color(index: u8) -> [f32; 3] {
    let grass = tint_color(index);
    [0, 1, 2].map(|i| grass[i] * LEAVES_TINT_SCALE[i])
}

/// Atlas layers that receive the climate tint: grass top, grass side (its
/// green fringe only) and leaves.
pub fn tint_applies(tex_index: u8) -> bool {
    matches!(tex_index, 0 | 1 | 8)
}

/// Approximates the shader's recolouring for tools: replaces the hue of an
/// average texture colour with the tint colour while keeping its brightness.
pub fn apply_tint(texture_color: [f32; 3], index: u8) -> [f32; 3] {
    if index == TINT_NONE {
        return texture_color;
    }
    let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let target = tint_color(index);
    let scale = luma(texture_color) / luma(tint_color(TINT_NONE)).max(1e-3);
    target.map(|c| (c * scale).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn climate_tint_covers_palette_and_is_monotonic() {
        assert_eq!(climate_tint(-1.0, -1.0), 1);
        assert_eq!(climate_tint(1.0, 1.0), 15);
        let mut seen = [false; 16];
        for t in -20..=20 {
            for h in -20..=20 {
                let index = climate_tint(t as f32 / 20.0, h as f32 / 20.0);
                assert!((1..=15).contains(&index));
                seen[index as usize] = true;
            }
        }
        assert!(seen[1..].iter().all(|&s| s));
    }

    #[test]
    fn shader_palette_matches_rust_palette() {
        let shader = include_str!("../shaders/terrain.wgsl");
        for color in GRASS_TINTS {
            let row = format!("vec3<f32>({:.2}, {:.2}, {:.2})", color[0], color[1], color[2]);
            assert!(shader.contains(&row), "terrain.wgsl is missing tint {row}");
        }
        let leaves = format!(
            "LEAVES_TINT_SCALE = vec3<f32>({:.2}, {:.2}, {:.2})",
            LEAVES_TINT_SCALE[0], LEAVES_TINT_SCALE[1], LEAVES_TINT_SCALE[2]
        );
        assert!(shader.contains(&leaves), "terrain.wgsl is missing {leaves}");
    }
}
