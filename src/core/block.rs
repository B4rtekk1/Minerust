use serde::{Deserialize, Serialize};

use crate::constants::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
pub enum BlockType {
    #[default]
    Air,
    Grass,
    Dirt,
    Stone,
    Sand,
    Water,
    Wood,
    Leaves,
    Bedrock,
    Snow,
    Gravel,
    Clay,
    Ice,
    Cactus,
    DeadBush,
    WoodStairs,
}

impl BlockType {
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

    pub fn top_color(&self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.36, 0.7, 0.28],
            _ => self.color(),
        }
    }

    pub fn bottom_color(&self) -> [f32; 3] {
        match self {
            BlockType::Grass => [0.52, 0.37, 0.26],
            _ => self.color(),
        }
    }

    pub fn is_solid(&self) -> bool {
        !matches!(
            self,
            BlockType::Air | BlockType::Water | BlockType::DeadBush
        )
    }

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

    pub fn is_solid_opaque(&self) -> bool {
        !self.is_transparent() && *self != BlockType::Air
    }

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

    pub fn tex_side(&self) -> f32 {
        match self {
            BlockType::Grass => TEX_GRASS_SIDE,
            BlockType::Wood => TEX_WOOD_SIDE,
            _ => self.tex_top(),
        }
    }

    pub fn tex_bottom(&self) -> f32 {
        match self {
            BlockType::Grass => TEX_DIRT,
            BlockType::Wood => TEX_WOOD_TOP,
            BlockType::WoodStairs => TEX_WOOD_TOP,
            _ => self.tex_top(),
        }
    }

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

    pub fn metallic(&self) -> f32 {
        match self {
            BlockType::Ice | BlockType::Water => 0.05,
            _ => 0.0,
        }
    }
}
