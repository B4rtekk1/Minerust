/// All biome types used during world generation.
///
/// The biome at a given column determines surface block selection, grass and
/// leaf tint colors, tree density, and other generation parameters. The default
/// biome is [`Biome::Plains`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Biome {
    /// Open grassland. Moderate tree density, bright green grass.
    #[default]
    Plains,
    /// Dense woodland. Dark green foliage, high tree coverage.
    Forest,
    /// Arid sandy terrain. No trees, sand-colored grass tint.
    Desert,
    /// Cold, snowy landscape. Muted green tint, sparse trees.
    Tundra,
    /// High-altitude rocky terrain. Reduced vegetation, gray-green tint.
    Mountains,
    /// Waterlogged lowland. Dark, murky green tint, moderate trees.
    Swamp,
    /// Deep water body. No trees or land vegetation.
    Ocean,
    /// Sandy shoreline transition zone. No trees.
    Beach,
    /// Narrow flowing water channel. No trees.
    River,
    /// Enclosed body of water. No trees.
    Lake,
    /// Small land mass surrounded by water. Light tree coverage.
    Island,
}

impl Biome {
    /// Returns the RGB grass tint color for this biome.
    ///
    /// Used to tint the top face of [`BlockType::Grass`] blocks during meshing.
    /// Water biomes (`Ocean`, `River`, `Lake`) return a blue water color as a
    /// fallback since they have no grass surface.
    ///
    /// Components are in linear `[0.0, 1.0]` space.
    pub fn grass_color(&self) -> [f32; 3] {
        match self {
            Biome::Plains => [0.45, 0.75, 0.30],
            Biome::Forest => [0.25, 0.55, 0.20],
            Biome::Desert => [0.89, 0.83, 0.61],
            Biome::Tundra => [0.65, 0.75, 0.70],
            Biome::Mountains => [0.50, 0.60, 0.45],
            Biome::Swamp => [0.35, 0.50, 0.25],
            Biome::Ocean => [0.25, 0.46, 0.82],
            Biome::Beach => [0.89, 0.83, 0.61],
            Biome::River => [0.25, 0.46, 0.82],
            Biome::Lake => [0.25, 0.46, 0.82],
            Biome::Island => [0.40, 0.70, 0.30],
        }
    }

    /// Returns the RGB leaf tint color for [`BlockType::Leaves`] in this biome.
    ///
    /// Biomes without trees (`Desert`, `Ocean`, `Beach`, `River`, `Lake`)
    /// fall through to a neutral green default since leaves will not appear there
    /// under normal generation.
    ///
    /// Components are in linear `[0.0, 1.0]` space.
    pub fn leaves_color(&self) -> [f32; 3] {
        match self {
            Biome::Plains => [0.35, 0.65, 0.25],
            Biome::Forest => [0.20, 0.50, 0.15],
            Biome::Tundra => [0.30, 0.45, 0.35],
            Biome::Swamp => [0.30, 0.45, 0.20],
            Biome::Island => [0.35, 0.60, 0.25],
            _ => [0.30, 0.60, 0.20],
        }
    }

    /// Returns the minimum noise threshold above which a tree will be placed.
    ///
    /// The world generator compares this value against a `[0.0, 1.0]` noise
    /// sample; a tree is placed when `noise >= tree_density()`. Lower values
    /// therefore produce denser forests.
    ///
    /// Biomes that do not support trees (`Desert`, `Ocean`, `Beach`, `River`,
    /// `Lake`) return `1.0` so the threshold is never met. Prefer checking
    /// [`Self::has_trees`] before sampling noise to avoid unnecessary work.
    pub fn tree_density(&self) -> f64 {
        match self {
            Biome::Plains => 0.75,
            Biome::Forest => 0.45,
            Biome::Desert => 1.0,
            Biome::Tundra => 0.85,
            Biome::Mountains => 0.80,
            Biome::Swamp => 0.60,
            Biome::Ocean => 1.0,
            Biome::Beach => 1.0,
            Biome::River => 1.0,
            Biome::Lake => 1.0,
            Biome::Island => 0.65,
        }
    }

    /// Returns `true` if trees can generate in this biome.
    ///
    /// `false` for `Desert`, `Ocean`, `Beach`, `River`, and `Lake`. Use this
    /// as an early-out before evaluating [`Self::tree_density`] during
    /// world generation.
    pub fn has_trees(&self) -> bool {
        matches!(
            self,
            Biome::Plains
                | Biome::Forest
                | Biome::Tundra
                | Biome::Mountains
                | Biome::Swamp
                | Biome::Island
        )
    }
}
