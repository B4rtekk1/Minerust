//! World generation and management modules
//! Contains chunk generation, loading, and world state.

pub mod generator;
pub mod loader;
pub mod structures;
pub mod terrain;
mod spline;

// Re-export commonly used types
pub use generator::ChunkGenerator;
pub use loader::{ChunkGenResult, ChunkLoader};
pub use terrain::World;
