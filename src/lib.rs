// Core module with fundamental types
pub mod core;

// Player module with camera and input
pub mod player;

// Render module with graphics-related code
pub mod render;

// Render core module with CSM and advanced shadow techniques
pub mod render_core;

// World module with generation and terrain
pub mod world;

// Keep old module paths for backward compatibility (re-export from core)
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

// Re-export player modules for backward compatibility
pub mod camera {
    pub use crate::player::camera::*;
}
pub mod input {
    pub use crate::player::input::*;
}

// Re-export render modules for backward compatibility
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

// Re-export world modules for backward compatibility
pub mod chunk_generator {
    pub use crate::world::generator::*;
}
pub mod chunk_loader {
    pub use crate::world::loader::*;
}

// Other modules
pub mod constants;
pub mod save;

// Re-exports
pub use constants::*;
pub use constants::{get_chunk_worker_count, get_mesh_worker_count, get_active_cascade_count};
pub use core::{Biome, BlockType, Chunk, SubChunk, Uniforms, Vertex};
pub use player::{Camera, DiggingState, InputState};
pub use render::{
    AABB, DrawIndexedIndirect, IndirectManager, MeshLoader, SubchunkKey, add_greedy_quad, add_quad,
    build_crosshair, build_player_model, extract_frustum_planes, generate_texture_atlas,
    load_texture_atlas_from_file,
};
pub use save::{DEFAULT_WORLD_FILE, SavedWorld, load_world, save_world};
pub use world::{ChunkGenResult, ChunkGenerator, ChunkLoader, World};
