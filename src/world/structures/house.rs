use super::Structure;
use crate::core::block::BlockType;

#[derive(Debug, Clone)]
pub struct House {
    pub structure: Structure,
}

impl House {
    pub fn new() -> Self {
        let mut structure = Structure::new("House", vec!["Plains", "Forest", "Desert"]);

        for x in 0..5 {
            for z in 0..5 {
                structure.blocks.push((x, 0, z, BlockType::Stone));
            }
        }

        for y in 1..=3 {
            for x in 0..5 {
                for z in 0..5 {
                    if x == 0 || x == 4 || z == 0 || z == 4 {
                        structure.blocks.push((x, y, z, BlockType::Wood));
                    } else {
                        structure.blocks.push((x, y, z, BlockType::Air));
                    }
                }
            }
        }

        for x in 0..5 {
            for z in 0..5 {
                structure.blocks.push((x, 4, z, BlockType::WoodStairs));
            }
        }

        for x in 1..4 {
            for z in 1..4 {
                structure.blocks.push((x, 5, z, BlockType::WoodStairs));
            }
        }

        structure.blocks.push((2, 6, 2, BlockType::WoodStairs));

        structure.blocks.push((2, 1, 0, BlockType::Air));
        structure.blocks.push((2, 2, 0, BlockType::Air));

        structure.blocks.push((0, 2, 2, BlockType::Air));
        structure.blocks.push((4, 2, 2, BlockType::Air));
        structure.blocks.push((2, 2, 4, BlockType::Air));

        Self { structure }
    }
}
