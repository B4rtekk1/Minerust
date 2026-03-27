use crate::logger::{log, LogLevel};

pub const WORLD_HEIGHT: i32 = 256;
pub const CHUNK_SIZE: i32 = 16;
pub const SUBCHUNK_HEIGHT: i32 = 16;
pub const NUM_SUBCHUNKS: i32 = WORLD_HEIGHT / SUBCHUNK_HEIGHT;
pub const RENDER_DISTANCE: i32 = 12;
pub const SIMULATION_DISTANCE: i32 = RENDER_DISTANCE / 2;
pub const GENERATION_DISTANCE: i32 = RENDER_DISTANCE + 2;
pub const SEA_LEVEL: i32 = 64;
pub const CHUNK_UNLOAD_DISTANCE: i32 = RENDER_DISTANCE + 5;
pub const TEX_GRASS_TOP: f32 = 0.0;
pub const TEX_GRASS_SIDE: f32 = 1.0;
pub const TEX_DIRT: f32 = 2.0;
pub const TEX_STONE: f32 = 3.0;
pub const TEX_SAND: f32 = 4.0;
pub const TEX_WATER: f32 = 5.0;
pub const TEX_WOOD_SIDE: f32 = 6.0;
pub const TEX_WOOD_TOP: f32 = 7.0;
pub const TEX_LEAVES: f32 = 8.0;
pub const TEX_BEDROCK: f32 = 9.0;
pub const TEX_SNOW: f32 = 10.0;
pub const TEX_GRAVEL: f32 = 11.0;
pub const TEX_CLAY: f32 = 12.0;
pub const TEX_ICE: f32 = 13.0;
pub const TEX_CACTUS: f32 = 14.0;
pub const TEX_DEAD_BUSH: f32 = 15.0;
pub const TEXTURE_SIZE: u32 = 256;
pub const ATLAS_SIZE: u32 = 4;

pub const MAX_CHUNKS_PER_FRAME: usize = 8;
pub const MAX_MESH_BUILDS_PER_FRAME: usize = 8;
pub const ASYNC_WORKER_COUNT: usize = 4;

pub const PLAYER_HEIGHT: f32 = 1.8;
pub const PLAYER_CROUCH_HEIGHT: f32 = 1.7;
pub const PLAYER_WIDTH: f32 = 0.35;
pub const PLAYER_BASE_SPEED: f32 = 4.8;
pub const PLAYER_SPRINT_SPEED: f32 = 16.0;
pub const PLAYER_JUMP_HEIGHT: f32 = 1.0;

pub const CSM_CASCADE_COUNT: usize = 4;
pub const CSM_CASCADE_SPLITS: [f32; CSM_CASCADE_COUNT] = [16.0, 48.0, 128.0, 300.0];
pub const CSM_SHADOW_MAP_SIZE: u32 = 2048;

pub const DEFAULT_FOV: f32 = 70.0 * std::f32::consts::PI / 180.0;

pub const BLOCK_SIZE: f32 = 0.98;
pub const BLOCK_OFFSET: f32 = (1.0 - BLOCK_SIZE) / 2.0;

pub fn get_chunk_worker_count() -> usize {
    let cores = num_cpus::get();
    let workers = ((cores.saturating_sub(2)) / 2).max(2).min(8);
    log(LogLevel::Info, &format!("CPU cores: {}, chunk workers: {}", cores, workers));
    workers
}

pub fn get_mesh_worker_count() -> usize {
    let cores = num_cpus::get();
    ((cores.saturating_sub(2)) / 2).max(2).min(6)
}

pub fn get_active_cascade_count(render_distance: i32) -> usize {
    match render_distance {
        0..=6 => 2,
        7..=12 => 3,
        _ => 4,
    }
}
