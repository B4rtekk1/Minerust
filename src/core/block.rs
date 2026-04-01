use serde::{Deserialize, Serialize};

use crate::constants::*;

/// All block types that can exist in the world.
///
/// [`BlockType::Air`] is the default and represents empty space. Every other
/// variant is a placeable, solid, or fluid block. The enum is `Copy` and fits
/// in a single byte, making it cheap to store in the large 3-D arrays inside
/// [`SubChunk`](crate::core::chunk::SubChunk).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub enum BlockType {
    /// Empty space. The default block for uninitialised chunks.
    #[default]
    Air,
    /// Grass-covered dirt. Distinct top, side, and bottom textures.
    Grass,
    /// Plain dirt. Used beneath grass and in terrain generation.
    Dirt,
    /// Generic stone. Relatively slow to break.
    Stone,
    /// Sand. Found in deserts and beaches.
    Sand,
    /// Water. Transparent, non-solid fluid block rendered in a separate pass.
    Water,
    /// Wood log. Has distinct top/side textures.
    Wood,
    /// Tree leaves. Transparent and rendered with face-against-leaves culling.
    Leaves,
    /// Indestructible bedrock at the bottom of the world.
    Bedrock,
    /// Snow layer. Fast to break.
    Snow,
    /// Gravel. High-roughness stone variant.
    Gravel,
    /// Clay. Slightly reflective, found near water.
    Clay,
    /// Ice. Low roughness and a small metallic value, giving it a glossy look.
    Ice,
    /// Cactus. Solid, fast to break.
    Cactus,
    /// Dead bush. Non-solid decoration; instantly breakable.
    DeadBush,
    /// Wooden stair block. Transparent for culling purposes.
    WoodStairs,
}

impl BlockType {
    /// Returns the base RGB color used for vertex coloring and the minimap.
    ///
    /// Components are in linear `[0.0, 1.0]` space. [`BlockType::Air`] returns
    /// black (`[0, 0, 0]`) as a safe sentinel.
    pub fn color(&self) -> [f32; 3] {
        match self {
            BlockType::Air => [0.0, 0.0, 0.0],
            BlockType::Grass => [0.45, 0.32, 0.22],
            BlockType::Dirt => [0.52, 0.37, 0.26],
            BlockType::Stone => [0.55, 0.55, 0.55],
            BlockType::Sand => [0.89, 0.83, 0.61],
            BlockType::Water => [0.25, 0.46, 0.82],
            BlockType::Wood => [0.6, 0.4, 0.2],
            BlockType::Leaves => [0.3, 0.6, 0.2],
            BlockType::Bedrock => [0.2, 0.2, 0.2],
            BlockType::Snow => [0.95, 0.95, 0.98],
            BlockType::Gravel => [0.5, 0.5, 0.52],
            BlockType::Clay => [0.65, 0.65, 0.72],
            BlockType::Ice => [0.7, 0.85, 0.95],
            BlockType::Cactus => [0.2, 0.55, 0.2],
            BlockType::DeadBush => [0.55, 0.4, 0.25],
            BlockType::WoodStairs => [0.6, 0.4, 0.2],
        }
    }

    /// Returns the RGB color for the **top** face.
    ///
    /// Overridden for [`BlockType::Grass`], which uses a green tint on top.
    /// All other variants fall back to [`Self::color`].
    pub fn top_color(&self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.36, 0.7, 0.28],
            _ => self.color(),
        }
    }

    /// Returns the RGB color for the **bottom** face.
    ///
    /// Overridden for [`BlockType::Grass`], which shows a dirt color on the
    /// bottom. All other variants fall back to [`Self::color`].
    pub fn bottom_color(&self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.52, 0.37, 0.26],
            _ => self.color(),
        }
    }

    /// Returns `true` if this block physically obstructs movement.
    ///
    /// [`BlockType::Air`], [`BlockType::Water`], and [`BlockType::DeadBush`]
    /// are non-solid; everything else is solid.
    pub fn is_solid(&self) -> bool {
        !matches!(
            self,
            BlockType::Air | BlockType::Water | BlockType::DeadBush
        )
    }

    /// Returns `true` if this block allows light (and visibility) to pass through.
    ///
    /// Transparent blocks include: `Air`, `Water`, `Leaves`, `Ice`,
    /// `DeadBush`, and `WoodStairs`.
    pub fn is_transparent(&self) -> bool {
        matches!(
            self,
            BlockType::Air
                | BlockType::Water
                | BlockType::Leaves
                | BlockType::Ice
                | BlockType::DeadBush
                | BlockType::WoodStairs
        )
    }

    /// Returns `true` if this block is both non-transparent and non-air.
    ///
    /// Used by [`SubChunk::check_fully_opaque`](crate::core::chunk::SubChunk::check_fully_opaque)
    /// to determine whether an entire sub-chunk can occlude its neighbors.
    pub fn is_solid_opaque(&self) -> bool {
        !self.is_transparent() && *self != BlockType::Air
    }

    /// Returns `true` if a face of `self` should be rendered when `neighbor`
    /// is the adjacent block.
    ///
    /// # Rules (in priority order)
    /// 1. Always render against [`BlockType::Air`].
    /// 2. [`BlockType::Water`] never renders faces against any non-air block.
    /// 3. Any block renders against [`BlockType::Water`].
    /// 4. [`BlockType::Leaves`] renders against other leaves (avoids solid
    ///    interior artifacts).
    /// 5. Any block renders against [`BlockType::WoodStairs`] (partial geometry).
    /// 6. Any block renders against a transparent neighbor.
    pub fn should_render_face_against(&self, neighbor: BlockType) -> bool {
        if neighbor == BlockType::Air {
            return true;
        }
        if *self == BlockType::Water {
            return false;
        }
        if neighbor == BlockType::Water {
            return true;
        }
        if *self == BlockType::Leaves && neighbor == BlockType::Leaves {
            return true;
        }
        if neighbor == BlockType::WoodStairs {
            return true;
        }
        neighbor.is_transparent()
    }

    /// Returns the time in seconds for a player to break this block by hand.
    ///
    /// [`BlockType::Air`], [`BlockType::Water`], and [`BlockType::DeadBush`]
    /// return `0.0` (instant). [`BlockType::Bedrock`] returns
    /// [`f32::INFINITY`] (unbreakable).
    pub fn break_time(&self) -> f32 {
        match self {
            BlockType::Air => 0.0,
            BlockType::Grass => 0.6,
            BlockType::Dirt => 0.5,
            BlockType::Stone => 2.5,
            BlockType::Sand => 0.5,
            BlockType::Water => 0.0,
            BlockType::Wood => 2.0,
            BlockType::Leaves => 0.2,
            BlockType::Bedrock => f32::INFINITY,
            BlockType::Snow => 0.2,
            BlockType::Gravel => 0.6,
            BlockType::Clay => 0.6,
            BlockType::Ice => 0.5,
            BlockType::Cactus => 0.4,
            BlockType::DeadBush => 0.0,
            BlockType::WoodStairs => 2.0,
        }
    }

    /// Returns the texture atlas index for the **top** face.
    ///
    /// Indices correspond to constants defined in `crate::constants`
    /// (e.g. `TEX_GRASS_TOP`, `TEX_STONE`). [`BlockType::Air`] returns `0.0`.
    pub fn tex_top(&self) -> f32 {
        match self {
            BlockType::Air => 0.0,
            BlockType::Grass => TEX_GRASS_TOP,
            BlockType::Dirt => TEX_DIRT,
            BlockType::Stone => TEX_STONE,
            BlockType::Sand => TEX_SAND,
            BlockType::Water => TEX_WATER,
            BlockType::Wood => TEX_WOOD_TOP,
            BlockType::Leaves => TEX_LEAVES,
            BlockType::Bedrock => TEX_BEDROCK,
            BlockType::Snow => TEX_SNOW,
            BlockType::Gravel => TEX_GRAVEL,
            BlockType::Clay => TEX_CLAY,
            BlockType::Ice => TEX_ICE,
            BlockType::Cactus => TEX_CACTUS,
            BlockType::DeadBush => TEX_DEAD_BUSH,
            BlockType::WoodStairs => TEX_WOOD_TOP,
        }
    }

    /// Returns the texture atlas index for the **side** faces.
    ///
    /// Overridden for [`BlockType::Grass`] (grass-side texture) and
    /// [`BlockType::Wood`] (bark texture). All other variants fall back to
    /// [`Self::tex_top`].
    pub fn tex_side(&self) -> f32 {
        match self {
            BlockType::Grass => TEX_GRASS_SIDE,
            BlockType::Wood => TEX_WOOD_SIDE,
            _ => self.tex_top(),
        }
    }

    /// Returns the texture atlas index for the **bottom** face.
    ///
    /// Overridden for [`BlockType::Grass`] (dirt), [`BlockType::Wood`], and
    /// [`BlockType::WoodStairs`] (wood-top). All other variants fall back to
    /// [`Self::tex_top`].
    pub fn tex_bottom(&self) -> f32 {
        match self {
            BlockType::Grass => TEX_DIRT,
            BlockType::Wood => TEX_WOOD_TOP,
            BlockType::WoodStairs => TEX_WOOD_TOP,
            _ => self.tex_top(),
        }
    }

    /// Returns the PBR roughness value for this block (`0.0` = mirror, `1.0` = fully diffuse).
    ///
    /// Notable values:
    /// - Ice / Water: `0.1` (glossy)
    /// - Grass / Dirt: `1.0` (fully diffuse)
    /// - Stone / Bedrock / Gravel / Clay: `0.7`
    pub fn roughness(&self) -> f32 {
        match self {
            BlockType::Stone | BlockType::Bedrock | BlockType::Gravel | BlockType::Clay => 0.7,
            BlockType::Sand => 0.8,
            BlockType::Grass | BlockType::Dirt | BlockType::DeadBush => 1.0,
            BlockType::Leaves => 0.5,
            BlockType::Snow => 0.8,
            BlockType::Ice | BlockType::Water => 0.1,
            BlockType::Wood | BlockType::Cactus | BlockType::WoodStairs => 0.6,
            BlockType::Air => 1.0,
        }
    }

    /// Returns the PBR metallic value for this block (`0.0` = dielectric, `1.0` = metal).
    ///
    /// Only [`BlockType::Ice`] and [`BlockType::Water`] have a non-zero value
    /// (`0.05`) to produce a subtle specular sheen. All other blocks return `0.0`.
    pub fn metallic(&self) -> f32 {
        match self {
            BlockType::Ice | BlockType::Water => 0.05,
            _ => 0.0,
        }
    }

    /// Returns the human-readable name shown in the HUD and inventory.
    ///
    /// Returns a `'static` string slice; no allocation is performed.
    pub fn display_name(&self) -> &'static str {
        match self {
            BlockType::Air => "Air",
            BlockType::Grass => "Grass",
            BlockType::Dirt => "Dirt",
            BlockType::Stone => "Stone",
            BlockType::Sand => "Sand",
            BlockType::Water => "Water",
            BlockType::Wood => "Wood",
            BlockType::Leaves => "Leaves",
            BlockType::Bedrock => "Bedrock",
            BlockType::Snow => "Snow",
            BlockType::Gravel => "Gravel",
            BlockType::Clay => "Clay",
            BlockType::Ice => "Ice",
            BlockType::Cactus => "Cactus",
            BlockType::DeadBush => "Dead Bush",
            BlockType::WoodStairs => "Wood Stairs",
        }
    }
}
