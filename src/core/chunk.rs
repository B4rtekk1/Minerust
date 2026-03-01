use cgmath::Vector3;

use crate::constants::*;
use crate::core::block::BlockType;
use crate::frustum::AABB;

pub struct SubChunk {
    pub blocks: [[[BlockType; CHUNK_SIZE as usize]; SUBCHUNK_HEIGHT as usize]; CHUNK_SIZE as usize],
    pub is_empty: bool,
    pub mesh_dirty: bool,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub num_indices: u32,
    pub water_vertex_buffer: Option<wgpu::Buffer>,
    pub water_index_buffer: Option<wgpu::Buffer>,
    pub num_water_indices: u32,
    pub aabb: AABB,
    pub is_fully_opaque: bool,
}

impl SubChunk {
    pub fn new(chunk_x: i32, subchunk_y: i32, chunk_z: i32) -> Self {
        let world_x = chunk_x * CHUNK_SIZE;
        let world_y = subchunk_y * SUBCHUNK_HEIGHT;
        let world_z = chunk_z * CHUNK_SIZE;

        SubChunk {
            blocks: [[[BlockType::Air; CHUNK_SIZE as usize]; SUBCHUNK_HEIGHT as usize];
                CHUNK_SIZE as usize],
            is_empty: true,
            is_fully_opaque: false,
            mesh_dirty: true,
            vertex_buffer: None,
            index_buffer: None,
            num_indices: 0,
            water_vertex_buffer: None,
            water_index_buffer: None,
            num_water_indices: 0,
            aabb: AABB::new(
                Vector3::new(world_x as f32, world_y as f32, world_z as f32),
                Vector3::new(
                    (world_x + CHUNK_SIZE) as f32,
                    (world_y + SUBCHUNK_HEIGHT) as f32,
                    (world_z + CHUNK_SIZE) as f32,
                ),
            ),
        }
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if x >= 0 && x < CHUNK_SIZE && y >= 0 && y < SUBCHUNK_HEIGHT && z >= 0 && z < CHUNK_SIZE {
            self.blocks[x as usize][y as usize][z as usize]
        } else {
            BlockType::Air
        }
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if x >= 0 && x < CHUNK_SIZE && y >= 0 && y < SUBCHUNK_HEIGHT && z >= 0 && z < CHUNK_SIZE {
            self.blocks[x as usize][y as usize][z as usize] = block;
            self.mesh_dirty = true;
            // Correctly track emptiness: placing a non-Air block means subchunk is not empty.
            // Note: we cannot cheaply reset is_empty=true here (would require full scan),
            // so check_empty() is used for that purpose after bulk generation.
            if block != BlockType::Air {
                self.is_empty = false;
            }
        }
    }

    pub fn check_empty(&mut self) {
        self.is_empty = true;
        for x in 0..CHUNK_SIZE as usize {
            for y in 0..SUBCHUNK_HEIGHT as usize {
                for z in 0..CHUNK_SIZE as usize {
                    if self.blocks[x][y][z] != BlockType::Air {
                        self.is_empty = false;
                        return;
                    }
                }
            }
        }
    }

    pub fn check_fully_opaque(&mut self) {
        for x in 0..CHUNK_SIZE as usize {
            for y in 0..SUBCHUNK_HEIGHT as usize {
                for z in 0..CHUNK_SIZE as usize {
                    if !self.blocks[x][y][z].is_solid_opaque() {
                        self.is_fully_opaque = false;
                        return;
                    }
                }
            }
        }
        self.is_fully_opaque = true;
    }
}

pub struct Chunk {
    pub subchunks: Vec<SubChunk>,
    pub player_modified: bool,
}

impl Chunk {
    pub fn new(x: i32, z: i32) -> Self {
        let mut subchunks = Vec::with_capacity(NUM_SUBCHUNKS as usize);
        for sy in 0..NUM_SUBCHUNKS {
            subchunks.push(SubChunk::new(x, sy, z));
        }
        Chunk {
            subchunks,
            player_modified: false,
        }
    }

    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if y < 0 || y >= WORLD_HEIGHT {
            return BlockType::Air;
        }
        let subchunk_idx = (y / SUBCHUNK_HEIGHT) as usize;
        let local_y = y % SUBCHUNK_HEIGHT;
        self.subchunks[subchunk_idx].get_block(x, local_y, z)
    }

    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: BlockType) {
        if y < 0 || y >= WORLD_HEIGHT {
            return;
        }
        let subchunk_idx = (y / SUBCHUNK_HEIGHT) as usize;
        let local_y = y % SUBCHUNK_HEIGHT;
        self.subchunks[subchunk_idx].set_block(x, local_y, z, block);
    }
}
