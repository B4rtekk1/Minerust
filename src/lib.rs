pub mod core;

pub mod player;

pub mod render;

pub mod render_core;

pub mod world;

pub mod biome {
    pub use crate::core::biome::*;
}
pub mod block {
    pub use crate::core::block::*;
}
pub mod chunk {
    pub use crate::core::chunk::*;
}
pub mod uniforms {
    pub use crate::core::uniforms::*;
}
pub mod vertex {
    pub use crate::core::vertex::*;
}

pub mod camera {
    pub use crate::player::camera::*;
}
pub mod input {
    pub use crate::player::input::*;
}

pub mod frustum {
    pub use crate::render::frustum::*;
}
pub mod mesh {
    pub use crate::render::mesh::*;
}
pub mod texture {
    pub use crate::render::texture::*;
}
pub mod mesh_loader {
    pub use crate::render::mesh_loader::*;
}

pub mod chunk_generator {
    pub use crate::world::generator::*;
}
pub mod chunk_loader {
    pub use crate::world::loader::*;
}

mod commands;
pub mod constants;
mod logger;
mod minerust_data;
pub mod save;
mod shader_utils;

pub use constants::*;
pub use constants::{get_active_cascade_count, get_chunk_worker_count, get_mesh_worker_count};
pub use core::{Biome, BlockType, Chunk, GameItem, ShadowConfig, SubChunk, Uniforms, Vertex};
pub use player::{Camera, DiggingState, InputState};
pub use render::{
    AABB, DrawIndexedIndirect, IndirectManager, MeshLoader, SubchunkKey, add_greedy_quad, add_quad,
    build_block_outline, build_crosshair, build_player_model, extract_frustum_planes,
    generate_texture_atlas, load_texture_atlas_from_file,
};
pub use save::{DEFAULT_WORLD_FILE, SavedWorld, load_world, save_world};
pub use vertex::OutlineVertex;
pub use world::{ChunkGenResult, ChunkGenerator, ChunkLoader, World};
