use crate::logger::{LogLevel, log};

pub const WORLD_HEIGHT: i32 = 256;
pub const CHUNK_SIZE: i32 = 16;
pub const SUBCHUNK_HEIGHT: i32 = 16;
pub const NUM_SUBCHUNKS: i32 = WORLD_HEIGHT / SUBCHUNK_HEIGHT;
pub const RENDER_DISTANCE: i32 = 32;
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
pub const PLAYER_EYE_HEIGHT: f32 = 1.62;
pub const PLAYER_CROUCH_HEIGHT: f32 = 1.7;
pub const PLAYER_WIDTH: f32 = 0.35;
pub const PLAYER_BASE_SPEED: f32 = 4.8;
pub const PLAYER_SPRINT_SPEED: f32 = 16.0;
pub const PLAYER_CROUCH_SPEED_MULTIPLIER: f32 = 0.35;
pub const PLAYER_JUMP_HEIGHT: f32 = 1.0;

pub const CSM_CASCADE_COUNT: usize = 4;
pub const CSM_CASCADE_SPLITS: [f32; CSM_CASCADE_COUNT] = [16.0, 48.0, 128.0, 300.0];
pub const CSM_SHADOW_MAP_SIZE: u32 = 2048;
pub const CSM_CASCADE_SHADOW_MAP_SIZES: [u32; CSM_CASCADE_COUNT] = [2048, 1024, 1024, 512];
pub const CSM_SHADOW_ATLAS_WIDTH: u32 = 3584;
pub const CSM_SHADOW_ATLAS_HEIGHT: u32 = 2048;
pub const CSM_SHADOW_ATLAS_RECTS: [[u32; 4]; CSM_CASCADE_COUNT] = [
    [0, 0, 2048, 2048],
    [2048, 0, 1024, 1024],
    [2048, 1024, 1024, 1024],
    [3072, 0, 512, 512],
];
pub const CSM_PCF_SAMPLES: u32 = 16;
pub const SHADOW_MASK_DOWNSCALE: u32 = 2;
/// Maximum refresh rate for the CSM-only sun direction.
pub const SHADOW_SUN_UPDATE_HZ: f32 = 15.0;
/// Minimum angular change before the CSM sun direction is refreshed.
pub const SHADOW_SUN_MIN_ANGLE_STEP: f32 = 0.35_f32.to_radians();

pub const DEFAULT_FOV: f32 = 70.0 * std::f32::consts::PI / 180.0;

pub const BLOCK_SIZE: f32 = 0.98;
pub const BLOCK_OFFSET: f32 = (1.0 - BLOCK_SIZE) / 2.0;

pub fn get_chunk_worker_count() -> usize {
    let cores = num_cpus::get();
    let workers = ((cores.saturating_sub(2)) / 2).max(2).min(8);
    log(
        LogLevel::Info,
        &format!("CPU cores: {}, chunk workers: {}", cores, workers),
    );
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

pub fn get_shadow_mask_size(width: u32, height: u32) -> (u32, u32) {
    let scale = SHADOW_MASK_DOWNSCALE.max(1);
    (width.div_ceil(scale).max(1), height.div_ceil(scale).max(1))
}

pub fn get_csm_shadow_atlas_rects_normalized() -> [[f32; 4]; CSM_CASCADE_COUNT] {
    let atlas_width = CSM_SHADOW_ATLAS_WIDTH as f32;
    let atlas_height = CSM_SHADOW_ATLAS_HEIGHT as f32;
    let mut rects = [[0.0; 4]; CSM_CASCADE_COUNT];
    let mut i = 0;
    while i < CSM_CASCADE_COUNT {
        let rect = CSM_SHADOW_ATLAS_RECTS[i];
        rects[i] = [
            rect[0] as f32 / atlas_width,
            rect[1] as f32 / atlas_height,
            rect[2] as f32 / atlas_width,
            rect[3] as f32 / atlas_height,
        ];
        i += 1;
    }
    rects
}

pub fn get_csm_shadow_sizes() -> [f32; CSM_CASCADE_COUNT] {
    [
        CSM_CASCADE_SHADOW_MAP_SIZES[0] as f32,
        CSM_CASCADE_SHADOW_MAP_SIZES[1] as f32,
        CSM_CASCADE_SHADOW_MAP_SIZES[2] as f32,
        CSM_CASCADE_SHADOW_MAP_SIZES[3] as f32,
    ]
}
