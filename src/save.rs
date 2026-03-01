use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::block::BlockType;
use crate::constants::*;

#[derive(Serialize, Deserialize)]
pub struct SavedChunk {
    pub cx: i32,
    pub cz: i32,
    pub subchunks: HashMap<u8, Vec<BlockType>>, // sy -> block data
}

#[derive(Serialize, Deserialize)]
pub struct SavedWorld {
    pub seed: u32,
    pub player_x: f32,
    pub player_y: f32,
    pub player_z: f32,
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub chunks: Vec<SavedChunk>,
}

impl SavedWorld {
    pub fn from_world<S: std::hash::BuildHasher>(
        chunks: &HashMap<(i32, i32), crate::chunk::Chunk, S>,
        seed: u32,
        player_pos: (f32, f32, f32),
        player_rot: (f32, f32),
    ) -> Self {
        let mut saved_chunks = Vec::new();

        for (&(cx, cz), chunk) in chunks.iter() {
            if !chunk.player_modified {
                continue;
            }

            let mut saved_subchunks = HashMap::new();
            for (sy, subchunk) in chunk.subchunks.iter().enumerate() {
                // Check if subchunk is actually modified or just empty
                if subchunk.is_empty {
                    continue;
                }

                let mut blocks = Vec::with_capacity(
                    CHUNK_SIZE as usize * SUBCHUNK_HEIGHT as usize * CHUNK_SIZE as usize,
                );
                for lx in 0..CHUNK_SIZE as usize {
                    for ly in 0..SUBCHUNK_HEIGHT as usize {
                        for lz in 0..CHUNK_SIZE as usize {
                            blocks.push(subchunk.blocks[lx][ly][lz]);
                        }
                    }
                }
                saved_subchunks.insert(sy as u8, blocks);
            }

            saved_chunks.push(SavedChunk {
                cx,
                cz,
                subchunks: saved_subchunks,
            });
        }

        SavedWorld {
            seed,
            player_x: player_pos.0,
            player_y: player_pos.1,
            player_z: player_pos.2,
            player_yaw: player_rot.0,
            player_pitch: player_rot.1,
            chunks: saved_chunks,
        }
    }
}

pub fn save_world<P: AsRef<Path>>(path: P, world: &SavedWorld) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Could not create file: {}", e))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, world).map_err(|e| format!("Serialization error: {}", e))
}

pub fn load_world<P: AsRef<Path>>(path: P) -> Result<SavedWorld, String> {
    let file = File::open(path).map_err(|e| format!("Could not open file: {}", e))?;
    let reader = BufReader::new(file);
    bincode::deserialize_from(reader).map_err(|e| format!("Deserialization error: {}", e))
}

pub const WORLD_FILE_EXTENSION: &str = "r3d";
pub const DEFAULT_WORLD_FILE: &str = "world.r3d";
