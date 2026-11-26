use serde::{Deserialize, Serialize};

use crate::constants::*;

/// All block types that can exist in the world.
///
/// [`BlockType::Air`] is the default and represents empty space. Every other
/// variant is a placeable, solid, or fluid block. The enum is `Copy` and fits
/// in a single byte, making it cheap to store in dense
/// [`SubChunk`](crate::core::chunk::SubChunk) buffers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[repr(u8)]
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
    /// Horizontal wood log aligned along the X axis.
    WoodLogX,
    /// Horizontal wood log aligned along the Z axis.
    WoodLogZ,
    // ── Added with the richer world generator. New variants must be
    // appended so saved worlds and network ids keep their meaning. ──
    /// Stone with coal seams; common at every depth.
    CoalOre,
    /// Stone with iron deposits; mid depths.
    IronOre,
    /// Deepslate with gold veins; deep underground and in badlands.
    GoldOre,
    /// Deepslate with diamonds; only near the bottom of the world.
    DiamondOre,
    /// Dark, dense stone of the deep underground.
    Deepslate,
    /// Iron-rich sand of the badlands.
    RedSand,
    /// Orange fired clay, the main badlands layer.
    Terracotta,
    /// Red terracotta band.
    RedTerracotta,
    /// Pale terracotta band.
    WhiteTerracotta,
    /// Needle-covered forest floor of old-growth taiga.
    Podzol,
    /// Soft moss carpet block (lush caves, old forests).
    Moss,
    /// Moss-overgrown stone.
    MossyStone,
    /// Fungal soil of mushroom fields.
    Mycelium,
    /// Stem of a huge mushroom.
    MushroomStem,
    /// Cap of a huge red mushroom.
    RedMushroomBlock,
    /// Cap of a huge brown mushroom.
    BrownMushroomBlock,
    /// Compressed sand under deserts.
    Sandstone,
    /// White birch trunk.
    BirchLog,
    /// Pink cherry blossom foliage (alpha tested, like leaves).
    CherryLeaves,
    /// Glowing volcanic rock found in deep caves.
    Magma,
    /// Stalactites and stalagmites of dripstone caves.
    Dripstone,
    /// Purple crystal lining geodes.
    Amethyst,
    /// White mineral shell around geodes.
    Calcite,
    /// Dark cherry wood trunk.
    CherryLog,
    /// Simple test lamp that emits warm block light.
    Lamp,
    /// Golden mineral that glows in deep caves.
    Glowstone,
    /// Warm fungal growth found in lush caves.
    Shroomlight,
    /// Luminous crystal found in geodes and dripstone caves.
    GlowCrystal,
    /// Rough stone made from broken rock.
    Cobblestone,
    /// Cut wooden boards.
    Planks,
    /// Fired masonry blocks.
    Bricks,
    /// Soft volcanic stone found underground.
    Tuff,
    /// Dark volcanic rock with a distinct top face.
    Basalt,
    /// Dense, glassy volcanic rock.
    Obsidian,
}

impl BlockType {
    /// Every block type, in id order (`ALL[id as usize] as u8 == id`).
    pub const ALL: [BlockType; 52] = [
        BlockType::Air,
        BlockType::Grass,
        BlockType::Dirt,
        BlockType::Stone,
        BlockType::Sand,
        BlockType::Water,
        BlockType::Wood,
        BlockType::Leaves,
        BlockType::Bedrock,
        BlockType::Snow,
        BlockType::Gravel,
        BlockType::Clay,
        BlockType::Ice,
        BlockType::Cactus,
        BlockType::DeadBush,
        BlockType::WoodStairs,
        BlockType::WoodLogX,
        BlockType::WoodLogZ,
        BlockType::CoalOre,
        BlockType::IronOre,
        BlockType::GoldOre,
        BlockType::DiamondOre,
        BlockType::Deepslate,
        BlockType::RedSand,
        BlockType::Terracotta,
        BlockType::RedTerracotta,
        BlockType::WhiteTerracotta,
        BlockType::Podzol,
        BlockType::Moss,
        BlockType::MossyStone,
        BlockType::Mycelium,
        BlockType::MushroomStem,
        BlockType::RedMushroomBlock,
        BlockType::BrownMushroomBlock,
        BlockType::Sandstone,
        BlockType::BirchLog,
        BlockType::CherryLeaves,
        BlockType::Magma,
        BlockType::Dripstone,
        BlockType::Amethyst,
        BlockType::Calcite,
        BlockType::CherryLog,
        BlockType::Lamp,
        BlockType::Glowstone,
        BlockType::Shroomlight,
        BlockType::GlowCrystal,
        BlockType::Cobblestone,
        BlockType::Planks,
        BlockType::Bricks,
        BlockType::Tuff,
        BlockType::Basalt,
        BlockType::Obsidian,
    ];

    /// Converts a numeric id (as sent over the network) back to a block.
    /// Unknown ids become [`BlockType::Air`].
    pub fn from_u8(id: u8) -> BlockType {
        Self::ALL
            .get(id as usize)
            .copied()
            .unwrap_or(BlockType::Air)
    }

    /// `true` for any kind of tree foliage.
    pub fn is_leaves(&self) -> bool {
        matches!(self, BlockType::Leaves | BlockType::CherryLeaves)
    }

    /// `true` for any kind of vertical tree trunk.
    pub fn is_log(&self) -> bool {
        matches!(
            self,
            BlockType::Wood
                | BlockType::WoodLogX
                | BlockType::WoodLogZ
                | BlockType::BirchLog
                | BlockType::CherryLog
                | BlockType::MushroomStem
        )
    }

    /// `true` for the stone family that ores and cave decoration replace.
    pub fn is_stone_like(&self) -> bool {
        matches!(
            self,
            BlockType::Stone
                | BlockType::Deepslate
                | BlockType::Tuff
                | BlockType::Basalt
                | BlockType::Cobblestone
        )
    }

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
            BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ => [0.6, 0.4, 0.2],
            BlockType::Leaves => [0.3, 0.6, 0.2],
            BlockType::Bedrock => [0.2, 0.2, 0.2],
            BlockType::Snow => [0.95, 0.95, 0.98],
            BlockType::Gravel => [0.5, 0.5, 0.52],
            BlockType::Clay => [0.65, 0.65, 0.72],
            BlockType::Ice => [0.7, 0.85, 0.95],
            BlockType::Cactus => [0.2, 0.55, 0.2],
            BlockType::DeadBush => [0.55, 0.4, 0.25],
            BlockType::WoodStairs => [0.6, 0.4, 0.2],
            BlockType::CoalOre => [0.4, 0.4, 0.4],
            BlockType::IronOre => [0.6, 0.53, 0.48],
            BlockType::GoldOre => [0.45, 0.4, 0.25],
            BlockType::DiamondOre => [0.3, 0.45, 0.47],
            BlockType::Deepslate => [0.26, 0.26, 0.29],
            BlockType::RedSand => [0.74, 0.39, 0.16],
            BlockType::Terracotta => [0.62, 0.37, 0.26],
            BlockType::RedTerracotta => [0.57, 0.24, 0.18],
            BlockType::WhiteTerracotta => [0.83, 0.7, 0.64],
            BlockType::Podzol => [0.37, 0.25, 0.11],
            BlockType::Moss => [0.33, 0.42, 0.16],
            BlockType::MossyStone => [0.42, 0.48, 0.36],
            BlockType::Mycelium => [0.44, 0.38, 0.43],
            BlockType::MushroomStem => [0.8, 0.78, 0.74],
            BlockType::RedMushroomBlock => [0.77, 0.17, 0.16],
            BlockType::BrownMushroomBlock => [0.57, 0.43, 0.31],
            BlockType::Sandstone => [0.84, 0.77, 0.55],
            BlockType::BirchLog => [0.88, 0.87, 0.84],
            BlockType::CherryLeaves => [0.93, 0.66, 0.77],
            BlockType::Magma => [0.78, 0.31, 0.08],
            BlockType::Dripstone => [0.53, 0.42, 0.35],
            BlockType::Amethyst => [0.51, 0.38, 0.75],
            BlockType::Calcite => [0.84, 0.85, 0.83],
            BlockType::CherryLog => [0.24, 0.15, 0.19],
            BlockType::Lamp => [1.0, 0.72, 0.42],
            BlockType::Glowstone => [1.0, 0.72, 0.28],
            BlockType::Shroomlight => [1.0, 0.55, 0.29],
            BlockType::GlowCrystal => [0.55, 0.38, 0.9],
            BlockType::Cobblestone => [0.48, 0.48, 0.5],
            BlockType::Planks => [0.62, 0.43, 0.25],
            BlockType::Bricks => [0.62, 0.34, 0.29],
            BlockType::Tuff => [0.4, 0.41, 0.39],
            BlockType::Basalt => [0.24, 0.24, 0.27],
            BlockType::Obsidian => [0.16, 0.12, 0.25],
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
                | BlockType::CherryLeaves
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
        if self.is_leaves() && neighbor.is_leaves() {
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
            BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ => 2.0,
            BlockType::Leaves => 0.2,
            BlockType::Bedrock => f32::INFINITY,
            BlockType::Snow => 0.2,
            BlockType::Gravel => 0.6,
            BlockType::Clay => 0.6,
            BlockType::Ice => 0.5,
            BlockType::Cactus => 0.4,
            BlockType::DeadBush => 0.0,
            BlockType::WoodStairs => 2.0,
            BlockType::CoalOre | BlockType::IronOre => 3.0,
            BlockType::GoldOre | BlockType::DiamondOre => 4.0,
            BlockType::Deepslate => 3.5,
            BlockType::RedSand => 0.5,
            BlockType::Terracotta | BlockType::RedTerracotta | BlockType::WhiteTerracotta => 1.25,
            BlockType::Podzol | BlockType::Mycelium => 0.6,
            BlockType::Moss => 0.3,
            BlockType::MossyStone => 2.5,
            BlockType::MushroomStem
            | BlockType::RedMushroomBlock
            | BlockType::BrownMushroomBlock => 0.3,
            BlockType::Sandstone => 1.0,
            BlockType::BirchLog | BlockType::CherryLog => 2.0,
            BlockType::CherryLeaves => 0.2,
            BlockType::Magma => 0.8,
            BlockType::Dripstone => 1.5,
            BlockType::Amethyst | BlockType::Calcite => 1.0,
            BlockType::Lamp => 0.5,
            BlockType::Glowstone => 0.6,
            BlockType::Shroomlight => 0.4,
            BlockType::GlowCrystal => 1.2,
            BlockType::Cobblestone => 3.0,
            BlockType::Planks => 1.5,
            BlockType::Bricks => 2.0,
            BlockType::Tuff => 1.5,
            BlockType::Basalt => 2.5,
            BlockType::Obsidian => 8.0,
        }
    }

    /// Returns this block's light emission level (`0..=15`).
    pub fn light_emission(&self) -> u8 {
        match self {
            BlockType::Lamp => 15,
            BlockType::Glowstone => 15,
            BlockType::Shroomlight => 14,
            BlockType::GlowCrystal => 11,
            BlockType::Magma => 6,
            _ => 0,
        }
    }

    /// Returns `true` if this block emits block light.
    pub fn is_emissive(&self) -> bool {
        self.light_emission() > 0
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
            BlockType::WoodLogX | BlockType::WoodLogZ => TEX_WOOD_SIDE,
            BlockType::Leaves => TEX_LEAVES,
            BlockType::Bedrock => TEX_BEDROCK,
            BlockType::Snow => TEX_SNOW,
            BlockType::Gravel => TEX_GRAVEL,
            BlockType::Clay => TEX_CLAY,
            BlockType::Ice => TEX_ICE,
            BlockType::Cactus => TEX_CACTUS,
            BlockType::DeadBush => TEX_DEAD_BUSH,
            BlockType::WoodStairs => TEX_WOOD_TOP,
            BlockType::CoalOre => TEX_COAL_ORE,
            BlockType::IronOre => TEX_IRON_ORE,
            BlockType::GoldOre => TEX_GOLD_ORE,
            BlockType::DiamondOre => TEX_DIAMOND_ORE,
            BlockType::Deepslate => TEX_DEEPSLATE,
            BlockType::RedSand => TEX_RED_SAND,
            BlockType::Terracotta => TEX_TERRACOTTA,
            BlockType::RedTerracotta => TEX_RED_TERRACOTTA,
            BlockType::WhiteTerracotta => TEX_WHITE_TERRACOTTA,
            BlockType::Podzol => TEX_PODZOL_TOP,
            BlockType::Moss => TEX_MOSS,
            BlockType::MossyStone => TEX_MOSSY_STONE,
            BlockType::Mycelium => TEX_MYCELIUM_TOP,
            BlockType::MushroomStem => TEX_MUSHROOM_STEM,
            BlockType::RedMushroomBlock => TEX_RED_MUSHROOM,
            BlockType::BrownMushroomBlock => TEX_BROWN_MUSHROOM,
            BlockType::Sandstone => TEX_SANDSTONE,
            BlockType::BirchLog | BlockType::CherryLog => TEX_WOOD_TOP,
            BlockType::CherryLeaves => TEX_CHERRY_LEAVES,
            BlockType::Magma => TEX_MAGMA,
            BlockType::Dripstone => TEX_DRIPSTONE,
            BlockType::Amethyst => TEX_AMETHYST,
            BlockType::Calcite => TEX_CALCITE,
            BlockType::Lamp => TEX_LAMP,
            BlockType::Glowstone => TEX_GLOWSTONE,
            BlockType::Shroomlight => TEX_SHROOMLIGHT,
            BlockType::GlowCrystal => TEX_GLOW_CRYSTAL,
            BlockType::Cobblestone => TEX_COBBLESTONE,
            BlockType::Planks => TEX_PLANKS,
            BlockType::Bricks => TEX_BRICKS,
            BlockType::Tuff => TEX_TUFF,
            BlockType::Basalt => TEX_BASALT_TOP,
            BlockType::Obsidian => TEX_OBSIDIAN,
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
            BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ => TEX_WOOD_SIDE,
            BlockType::Podzol => TEX_PODZOL_SIDE,
            BlockType::Mycelium => TEX_MYCELIUM_SIDE,
            BlockType::BirchLog => TEX_BIRCH_LOG,
            BlockType::CherryLog => TEX_CHERRY_LOG,
            BlockType::Lamp => TEX_LAMP,
            BlockType::Basalt => TEX_BASALT_SIDE,
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
            BlockType::Grass | BlockType::Podzol | BlockType::Mycelium => TEX_DIRT,
            BlockType::Wood => TEX_WOOD_TOP,
            BlockType::WoodLogX | BlockType::WoodLogZ => TEX_WOOD_SIDE,
            BlockType::WoodStairs => TEX_WOOD_TOP,
            _ => self.tex_top(),
        }
    }

    /// Returns the texture atlas index for a world face direction.
    ///
    /// Face directions match the terrain mesher: `0/1 = X`, `2/3 = Y`,
    /// `4/5 = Z`. Horizontal logs use the cut texture only on their end axis.
    pub fn tex_for_face(&self, face_dir: i32) -> f32 {
        match self {
            BlockType::WoodLogX => {
                if matches!(face_dir, 0 | 1) {
                    TEX_WOOD_TOP
                } else {
                    TEX_WOOD_SIDE
                }
            }
            BlockType::WoodLogZ => {
                if matches!(face_dir, 4 | 5) {
                    TEX_WOOD_TOP
                } else {
                    TEX_WOOD_SIDE
                }
            }
            _ => match face_dir {
                2 => self.tex_bottom(),
                3 => self.tex_top(),
                _ => self.tex_side(),
            },
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
            BlockType::CoalOre
            | BlockType::IronOre
            | BlockType::Deepslate
            | BlockType::MossyStone
            | BlockType::Dripstone
            | BlockType::Calcite
            | BlockType::Sandstone
            | BlockType::Magma => 0.75,
            BlockType::GoldOre | BlockType::DiamondOre | BlockType::Amethyst => 0.35,
            BlockType::RedSand
            | BlockType::Terracotta
            | BlockType::RedTerracotta
            | BlockType::WhiteTerracotta => 0.85,
            BlockType::Podzol | BlockType::Moss | BlockType::Mycelium => 1.0,
            BlockType::CherryLeaves => 0.5,
            BlockType::MushroomStem
            | BlockType::RedMushroomBlock
            | BlockType::BrownMushroomBlock
            | BlockType::BirchLog
            | BlockType::CherryLog => 0.6,
            BlockType::Sand => 0.8,
            BlockType::Grass | BlockType::Dirt | BlockType::DeadBush => 1.0,
            BlockType::Leaves => 0.5,
            BlockType::Snow => 0.8,
            BlockType::Ice | BlockType::Water => 0.1,
            BlockType::Wood
            | BlockType::WoodLogX
            | BlockType::WoodLogZ
            | BlockType::Cactus
            | BlockType::WoodStairs => 0.6,
            BlockType::Air => 1.0,
            BlockType::Lamp => 0.7,
            BlockType::Glowstone | BlockType::Shroomlight => 0.65,
            BlockType::GlowCrystal => 0.3,
            BlockType::Cobblestone | BlockType::Bricks | BlockType::Tuff | BlockType::Basalt => 0.75,
            BlockType::Planks => 0.7,
            BlockType::Obsidian => 0.2,
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
            BlockType::WoodLogX | BlockType::WoodLogZ => "Fallen Wood",
            BlockType::Leaves => "Leaves",
            BlockType::Bedrock => "Bedrock",
            BlockType::Snow => "Snow",
            BlockType::Gravel => "Gravel",
            BlockType::Clay => "Clay",
            BlockType::Ice => "Ice",
            BlockType::Cactus => "Cactus",
            BlockType::DeadBush => "Dead Bush",
            BlockType::WoodStairs => "Wood Stairs",
            BlockType::CoalOre => "Coal Ore",
            BlockType::IronOre => "Iron Ore",
            BlockType::GoldOre => "Gold Ore",
            BlockType::DiamondOre => "Diamond Ore",
            BlockType::Deepslate => "Deepslate",
            BlockType::RedSand => "Red Sand",
            BlockType::Terracotta => "Terracotta",
            BlockType::RedTerracotta => "Red Terracotta",
            BlockType::WhiteTerracotta => "White Terracotta",
            BlockType::Podzol => "Podzol",
            BlockType::Moss => "Moss",
            BlockType::MossyStone => "Mossy Stone",
            BlockType::Mycelium => "Mycelium",
            BlockType::MushroomStem => "Mushroom Stem",
            BlockType::RedMushroomBlock => "Red Mushroom Block",
            BlockType::BrownMushroomBlock => "Brown Mushroom Block",
            BlockType::Sandstone => "Sandstone",
            BlockType::BirchLog => "Birch Log",
            BlockType::CherryLeaves => "Cherry Leaves",
            BlockType::Magma => "Magma",
            BlockType::Dripstone => "Dripstone",
            BlockType::Amethyst => "Amethyst",
            BlockType::Calcite => "Calcite",
            BlockType::CherryLog => "Cherry Log",
            BlockType::Lamp => "Lamp",
            BlockType::Glowstone => "Glowstone",
            BlockType::Shroomlight => "Shroomlight",
            BlockType::GlowCrystal => "Glow Crystal",
            BlockType::Cobblestone => "Cobblestone",
            BlockType::Planks => "Planks",
            BlockType::Bricks => "Bricks",
            BlockType::Tuff => "Tuff",
            BlockType::Basalt => "Basalt",
            BlockType::Obsidian => "Obsidian",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_through_all() {
        for (id, block) in BlockType::ALL.iter().enumerate() {
            assert_eq!(
                *block as u8 as usize, id,
                "{block:?} is out of order in ALL"
            );
            assert_eq!(BlockType::from_u8(id as u8), *block);
        }
        assert_eq!(BlockType::from_u8(250), BlockType::Air);
    }

    #[test]
    fn every_texture_index_fits_the_atlas() {
        for block in BlockType::ALL {
            for face in 0..6 {
                let tex = block.tex_for_face(face) as u32;
                assert!(
                    tex < TEXTURE_LAYERS,
                    "{block:?} face {face} uses layer {tex}"
                );
            }
        }
    }

    #[test]
    fn emissive_blocks_emit_block_light() {
        for (block, level) in [
            (BlockType::Lamp, 15),
            (BlockType::Glowstone, 15),
            (BlockType::Shroomlight, 14),
            (BlockType::GlowCrystal, 11),
            (BlockType::Magma, 6),
        ] {
            assert_eq!(block.light_emission(), level);
            assert!(block.is_emissive());
        }
        for block in BlockType::ALL {
            assert_eq!(block.is_emissive(), block.light_emission() > 0);
            if !matches!(
                block,
                BlockType::Lamp
                    | BlockType::Glowstone
                    | BlockType::Shroomlight
                    | BlockType::GlowCrystal
                    | BlockType::Magma
            ) {
                assert_eq!(block.light_emission(), 0, "{block:?} should not emit light");
            }
        }
    }
}
