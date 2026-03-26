use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, Viewport};
use wgpu;
use winit::window::Window;

use crate::multiplayer::player::RemotePlayer;
use crate::multiplayer::protocol::Packet;
use crate::ui::menu::{GameState, MenuState};
use minerust::chunk_loader::ChunkLoader;
use minerust::render_core::csm::CsmManager;
use minerust::{Camera, DiggingState, IndirectManager, InputState, World};

/// Central application state owned by the main thread.
///
/// `State` is the single source of truth for all GPU resources, world data,
/// camera, input, UI, and multiplayer state.  It is created once during
/// initialization and lives for the duration of the process.
///
/// # Field groupings
/// The fields are logically organized as follows (in declaration order):
///
/// - **wgpu surface & device** – `surface`, `device`, `queue`, `config`,
///   `surface_format`.
/// - **Render pipelines** – one pipeline per render pass
///   (`render_pipeline`, `water_pipeline`, `sun_pipeline`, etc.).
/// - **Static geometry buffers** – sun quad, crosshair.
/// - **Uniforms & bind groups** – shared uniform buffer and per-pass bind groups.
/// - **Render targets** – depth, MSAA, shadow cascade array, SSR, scene color,
///   Hi-Z pyramid.
/// - **World & camera** – the shared `World` behind an `RwLock`, camera, and
///   input state.
/// - **Frame timing & stats** – FPS counter, frame time, CPU update time.
/// - **UI buffers** – HUD vertex/index buffers (coordinates, progress bar,
///   hotbar) and all `glyphon` text buffers.
/// - **Multiplayer** – remote player map, player ID, network channels, and the
///   async Tokio runtime.
/// - **Streaming** – `ChunkLoader` for background generation and `MeshLoader`
///   for background meshing.
/// - **Indirect rendering** – `IndirectManager` for terrain and water, Hi-Z
///   pipeline and bind groups.
/// - **CSM shadows** – `CsmManager` and cascade-related buffers/views.
/// - **Post-processing** – composite pipeline and SSR resources.
pub struct State {
    // -------------------------------------------------------------------------
    // wgpu core
    // -------------------------------------------------------------------------
    /// The wgpu rendering surface backed by the OS window.
    pub surface: wgpu::Surface<'static>,
    /// Logical GPU device; used to create all GPU resources.
    pub device: wgpu::Device,
    /// Command queue for submitting work to the GPU.
    pub queue: wgpu::Queue,
    /// Surface configuration (size, format, present mode).
    pub config: wgpu::SurfaceConfiguration,
    /// Pixel format of the swap-chain surface (cached to avoid repeated lookups).
    pub surface_format: wgpu::TextureFormat,

    // -------------------------------------------------------------------------
    // Render pipelines
    // -------------------------------------------------------------------------
    /// Main opaque terrain render pipeline.
    pub render_pipeline: wgpu::RenderPipeline,
    /// Transparent water render pipeline (blended over opaque geometry).
    pub water_pipeline: wgpu::RenderPipeline,
    /// 3-D block outline overlay pipeline.
    pub outline_pipeline: wgpu::RenderPipeline,
    /// Sun disc render pipeline.
    pub sun_pipeline: wgpu::RenderPipeline,
    /// Sky background render pipeline.
    pub sky_pipeline: wgpu::RenderPipeline,
    /// Shadow-map generation pipeline (depth-only).
    pub shadow_pipeline: wgpu::RenderPipeline,
    /// Screen-space crosshair render pipeline.
    pub crosshair_pipeline: wgpu::RenderPipeline,
    /// Full-screen composite pipeline that resolves MSAA and applies post-FX.
    pub composite_pipeline: wgpu::RenderPipeline,
    /// Depth-resolve pipeline that copies the MSAA depth buffer to a 1-sample
    /// texture for use by the Hi-Z pass and SSR.
    pub depth_resolve_pipeline: wgpu::RenderPipeline,

    // -------------------------------------------------------------------------
    // Static geometry buffers
    // -------------------------------------------------------------------------
    /// Vertex buffer for the sun disc quad.
    pub sun_vertex_buffer: wgpu::Buffer,
    /// Index buffer for the sun disc quad.
    pub sun_index_buffer: wgpu::Buffer,
    /// Vertex buffer for the screen-space crosshair geometry.
    pub crosshair_vertex_buffer: wgpu::Buffer,
    /// Index buffer for the screen-space crosshair geometry.
    pub crosshair_index_buffer: wgpu::Buffer,
    /// Number of indices in the crosshair index buffer.
    pub num_crosshair_indices: u32,

    // -------------------------------------------------------------------------
    // Uniforms and bind groups
    // -------------------------------------------------------------------------
    /// Uniform buffer containing per-frame data (view-proj, sun direction, etc.).
    pub uniform_buffer: wgpu::Buffer,
    /// Small shadow settings buffer shared with the terrain shader.
    #[allow(dead_code)]
    pub shadow_config_buffer: wgpu::Buffer,
    /// Bind group that exposes `uniform_buffer` and the texture atlas to shaders.
    pub uniform_bind_group: wgpu::BindGroup,
    /// Empty placeholder bind group for terrain pipeline group(1).
    pub terrain_gbuffer_bind_group: wgpu::BindGroup,
    /// Empty placeholder bind group for terrain pipeline group(2).
    pub terrain_shadow_output_bind_group: wgpu::BindGroup,
    /// Bind group that exposes the shadow cascade array to the main render pass.
    pub shadow_bind_group: wgpu::BindGroup,
    /// Bind group for the water pass (SSR color/depth textures + sampler).
    pub water_bind_group: wgpu::BindGroup,
    /// Layout of `water_bind_group`; kept alive so the bind group can be rebuilt
    /// when the window resizes.
    pub water_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the composite pass (scene color + SSR resources).
    pub composite_bind_group: wgpu::BindGroup,
    /// Bind group for the depth-resolve pass.
    pub depth_resolve_bind_group: wgpu::BindGroup,

    // -------------------------------------------------------------------------
    // Render targets and textures
    // -------------------------------------------------------------------------
    /// Non-linear (sRGB) depth buffer view used by the main render pass.
    pub depth_texture: wgpu::TextureView,
    /// MSAA resolve target view (matches the surface format).
    pub msaa_texture_view: wgpu::TextureView,
    /// Full shadow cascade array texture view (all cascades as one 2-D array).
    pub shadow_texture_view: wgpu::TextureView,
    /// Screen-space shadow mask sampled by `terrain.wgsl`.
    #[allow(dead_code)]
    pub shadow_mask_texture: wgpu::Texture,
    /// View of the screen-space shadow mask texture.
    #[allow(dead_code)]
    pub shadow_mask_view: wgpu::TextureView,
    /// One `wgpu::TextureView` per shadow cascade for per-cascade rendering.
    pub shadow_cascade_views: Vec<wgpu::TextureView>,
    /// GPU buffer containing the packed `CascadeData` array for all cascades.
    pub shadow_cascade_buffer: wgpu::Buffer,
    /// Sampler used when reading the shadow cascade array in the main pass.
    /// Kept alive by the bind group; annotated `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub shadow_sampler: wgpu::Sampler,
    /// Bind group that exposes the screen-space shadow mask to `terrain.wgsl`.
    pub shadow_mask_bind_group: wgpu::BindGroup,
    /// Intermediate scene color texture rendered into before compositing.
    pub scene_color_texture: wgpu::Texture,
    /// View of `scene_color_texture`.
    pub scene_color_view: wgpu::TextureView,
    /// Color texture used as the SSR source (previous-frame or resolved).
    pub ssr_color_texture: wgpu::Texture,
    /// View of `ssr_color_texture`.
    pub ssr_color_view: wgpu::TextureView,
    /// Depth texture used by the SSR pass for ray-marching.
    pub ssr_depth_texture: wgpu::Texture,
    /// View of `ssr_depth_texture`.
    pub ssr_depth_view: wgpu::TextureView,
    /// Sampler used when reading SSR textures in the water and composite passes.
    pub ssr_sampler: wgpu::Sampler,
    /// The 16-layer `Texture2DArray` holding all block textures.
    /// Kept alive by the bind group; annotated `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub texture_atlas: wgpu::Texture,
    /// View of `texture_atlas` as a `D2Array`.
    /// Kept alive by the bind group; annotated `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub texture_view: wgpu::TextureView,
    /// Sampler used when reading the texture atlas in terrain/water shaders.
    /// Kept alive by the bind group; annotated `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub texture_sampler: wgpu::Sampler,
    /// Neutral flow-map texture used by the water shader.
    /// Owned by `State` so the texture stays alive as long as the view.
    #[allow(dead_code)]
    pub flow_map_texture: wgpu::Texture,
    /// View of the neutral flow-map texture.
    pub flow_map_view: wgpu::TextureView,
    /// Sampler used when reading the flow-map texture in the water shader.
    pub flow_sampler: wgpu::Sampler,

    // -------------------------------------------------------------------------
    // Hi-Z (hierarchical depth) occlusion culling
    // -------------------------------------------------------------------------
    /// Full Hi-Z mip-chain texture (R32Float, one mip per halving).
    pub hiz_texture: wgpu::Texture,
    /// View of the full Hi-Z mip chain (used by the culling shader).
    pub hiz_view: wgpu::TextureView,
    /// One view per mip level of `hiz_texture` (used as compute shader outputs).
    pub hiz_mips: Vec<wgpu::TextureView>,
    /// Compute pipeline that downsamples the depth buffer into the Hi-Z pyramid.
    pub hiz_pipeline: wgpu::ComputePipeline,
    /// One bind group per mip-to-mip downsampling step.
    pub hiz_bind_groups: Vec<wgpu::BindGroup>,
    /// Layout shared by all `hiz_bind_groups`.
    pub hiz_bind_group_layout: wgpu::BindGroupLayout,
    /// Pixel dimensions of the Hi-Z base level `[width, height]`.
    pub hiz_size: [u32; 2],

    // -------------------------------------------------------------------------
    // World, camera, and input
    // -------------------------------------------------------------------------
    /// Shared voxel world, protected by a reader-writer lock so background
    /// generation and meshing threads can read concurrently.
    pub world: Arc<parking_lot::RwLock<World>>,
    /// First-person camera (position, yaw, pitch, velocity).
    pub camera: Camera,
    /// Block currently under the crosshair and within reach, if any.
    pub highlighted_block: Option<(i32, i32, i32)>,
    /// Snapshot of keyboard and mouse button state updated each event.
    pub input: InputState,
    /// Block-breaking progress tracker for the currently targeted block.
    pub digging: DiggingState,
    /// The OS window; shared with the event loop and network thread.
    pub window: Arc<Window>,
    /// Whether the cursor is captured (hidden and locked to the window centre).
    pub mouse_captured: bool,
    /// Last known cursor position in logical pixels (used for menu interaction).
    pub cursor_position: Option<(f32, f32)>,

    // -------------------------------------------------------------------------
    // Frame timing and performance stats
    // -------------------------------------------------------------------------
    /// Total number of frames rendered since startup.
    pub frame_count: u32,
    /// `Instant` of the last FPS counter refresh.
    pub last_fps_update: Instant,
    /// Smoothed frames-per-second value displayed in the HUD.
    pub current_fps: f32,
    /// Last frame's total wall-clock time in milliseconds.
    pub frame_time_ms: f32,
    /// Last frame's CPU update (non-render) time in milliseconds.
    pub cpu_update_ms: f32,
    /// `Instant` of the last `request_redraw` call (used to throttle redraws).
    pub last_redraw: Instant,
    /// `Instant` at the start of the previous frame (used to compute `dt`).
    pub last_frame: Instant,
    /// `Instant` when the game session started (used for elapsed-time uniforms).
    pub game_start_time: Instant,
    /// Number of chunk columns that produced at least one draw call last frame.
    pub chunks_rendered: u32,
    /// Number of individual subchunks drawn last frame (post-culling).
    pub subchunks_rendered: u32,
    /// `Instant` of the last keyboard/mouse event (used for input timeout).
    pub last_input_time: Instant,
    /// Whether the GPU supports `multi_draw_indirect_count`; falls back to a
    /// fixed draw-count path when `false`.
    pub supports_indirect_count: bool,

    // -------------------------------------------------------------------------
    // Streaming: chunk generation and mesh building
    // -------------------------------------------------------------------------
    /// Submits chunk generation requests to background threads and collects results.
    pub chunk_loader: ChunkLoader,
    /// Chunk-column X coordinate of the player's position on the last generation scan.
    pub last_gen_player_cx: i32,
    /// Chunk-column Z coordinate of the player's position on the last generation scan.
    pub last_gen_player_cz: i32,
    /// Submits subchunk mesh-build requests to background threads and collects results.
    pub mesh_loader: minerust::MeshLoader,

    // -------------------------------------------------------------------------
    // Indirect rendering managers
    // -------------------------------------------------------------------------
    /// Manages the unified vertex/index buffers and GPU culling for terrain.
    pub indirect_manager: IndirectManager,
    /// Manages the unified vertex/index buffers and GPU culling for water.
    pub water_indirect_manager: IndirectManager,

    // -------------------------------------------------------------------------
    // Cascaded shadow maps (CSM)
    // -------------------------------------------------------------------------
    /// Computes and stores the per-cascade light-space view-projection matrices.
    pub csm: CsmManager,
    /// Active shadow-cascade mode selector (reserved for future multi-mode support).
    pub reflection_mode: u32,

    // -------------------------------------------------------------------------
    // HUD: coordinate display
    // -------------------------------------------------------------------------
    /// Vertex buffer for the coordinate HUD quad (rebuilt when position changes).
    pub coords_vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer for the coordinate HUD quad.
    pub coords_index_buffer: Option<wgpu::Buffer>,
    /// Number of indices in `coords_index_buffer`.
    pub coords_num_indices: u32,
    /// Block coordinates the coordinate HUD was last built for; used to skip
    /// rebuilds when the player has not moved to a new block.
    pub last_coords_position: (i32, i32, i32),

    // -------------------------------------------------------------------------
    // HUD: block-break progress bar
    // -------------------------------------------------------------------------
    /// Vertex buffer for the block-break progress bar quad.
    pub progress_bar_vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer for the block-break progress bar quad.
    pub progress_bar_index_buffer: Option<wgpu::Buffer>,

    // -------------------------------------------------------------------------
    // HUD: hotbar
    // -------------------------------------------------------------------------
    /// Currently selected hotbar slot index (0-based).
    pub hotbar_slot: usize,
    /// Vertex buffer for the hotbar background/selection quads.
    pub hotbar_vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer for the hotbar background/selection quads.
    pub hotbar_index_buffer: Option<wgpu::Buffer>,
    /// Number of indices in `hotbar_index_buffer`.
    pub hotbar_num_indices: u32,
    /// When `true` the hotbar geometry needs to be rebuilt before the next frame.
    pub hotbar_dirty: bool,
    /// Slot index the hotbar was last built for; used to detect slot changes.
    pub last_hotbar_slot: usize,

    // -------------------------------------------------------------------------
    // glyphon text rendering
    // -------------------------------------------------------------------------
    /// Manages font data and shaping for all text rendered via glyphon.
    pub font_system: FontSystem,
    /// Rasterises glyph outlines into the `text_atlas`.
    pub swash_cache: SwashCache,
    /// GPU glyph cache texture used by `text_renderer`.
    pub text_atlas: TextAtlas,
    /// Issues draw calls to render glyphon text into the current pass.
    pub text_renderer: TextRenderer,
    /// Tracks the logical viewport size for text layout.
    pub viewport: Viewport,

    /// FPS / performance stats overlay buffer.
    pub fps_buffer: glyphon::Buffer,

    // Main-menu text buffers.
    /// Large title text shown on the main menu.
    pub menu_title_buffer: glyphon::Buffer,
    /// Subtitle / version text shown below the title.
    pub menu_subtitle_buffer: glyphon::Buffer,
    /// Label "Server:" on the connection form.
    pub menu_server_label_buffer: glyphon::Buffer,
    /// Editable server address value on the connection form.
    pub menu_server_value_buffer: glyphon::Buffer,
    /// Label "Username:" on the connection form.
    pub menu_username_label_buffer: glyphon::Buffer,
    /// Editable username value on the connection form.
    pub menu_username_value_buffer: glyphon::Buffer,
    /// Rotating tip or instruction text shown at the bottom of the menu.
    pub menu_tips_buffer: glyphon::Buffer,
    /// "Connect" button label.
    pub menu_connect_button_buffer: glyphon::Buffer,
    /// "Singleplayer" button label.
    pub menu_singleplayer_button_buffer: glyphon::Buffer,
    /// Status / error message shown below the buttons (e.g. "Connecting…").
    pub menu_status_buffer: glyphon::Buffer,

    // In-game HUD text buffers.
    /// Item name label shown above the hotbar when the slot changes.
    pub hotbar_label_buffer: glyphon::Buffer,
    /// Pre-measured pixel width of `hotbar_label_buffer` for centering.
    pub hotbar_label_width: f32,
    /// One name-tag buffer per currently visible remote player.
    pub player_label_buffers: Vec<glyphon::Buffer>,

    // -------------------------------------------------------------------------
    // UI / game state
    // -------------------------------------------------------------------------
    /// Tracks whether the player is in the main menu, lobby, or in-game.
    pub game_state: GameState,
    /// Tracks focus / edit state of individual menu widgets.
    pub menu_state: MenuState,
    /// `1.0` when the camera eye is inside a water block; `0.0` otherwise.
    /// Passed to the composite shader to apply the underwater color tint.
    pub is_underwater: f32,

    // -------------------------------------------------------------------------
    // Multiplayer
    // -------------------------------------------------------------------------
    /// Map from player ID to the last-known state of each remote player.
    pub remote_players: HashMap<u32, RemotePlayer>,
    /// This client's own player ID assigned by the server (0 = not connected).
    pub my_player_id: u32,
    /// `Instant` of the last position packet sent to the server.
    pub last_position_send: Instant,
    /// Tokio async runtime used by the network thread (kept alive here).
    pub network_runtime: Option<tokio::runtime::Runtime>,
    /// Receives decoded packets forwarded from the network thread.
    pub network_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Packet>>,
    /// Sends packets from the game thread to the network thread for transmission.
    pub network_tx: Option<tokio::sync::mpsc::UnboundedSender<Packet>>,

    // -------------------------------------------------------------------------
    // Remote player model geometry
    // -------------------------------------------------------------------------
    /// Vertex buffer containing the combined geometry for all remote player models.
    pub player_model_vertex_buffer: Option<wgpu::Buffer>,
    /// Index buffer for the combined remote player model geometry.
    pub player_model_index_buffer: Option<wgpu::Buffer>,
    /// Number of indices in `player_model_index_buffer`.
    pub player_model_num_indices: u32,
    /// Allocated capacity of `player_model_vertex_buffer` in vertices.
    /// Used to detect when the buffer needs to be reallocated.
    pub player_model_vertex_capacity: u32,
    /// Allocated capacity of `player_model_index_buffer` in indices.
    pub player_model_index_capacity: u32,
}

/// A lightweight, read-only snapshot of world state collected under the read lock.
///
/// Gathering all read queries in one pass minimizes the time the lock is held
/// and avoids repeated acquisitions across the `update` method.
pub struct WorldSnapshot {
    /// Chunks within `GENERATION_DISTANCE` that are not yet loaded or pending.
    /// Each entry is `(chunk_x, chunk_z, squared_distance_priority)`.
    pub missing_chunks: Vec<(i32, i32, i32)>,
    /// Result of the block raycast: `(hit_x, hit_y, hit_z, face_nx, face_ny, face_nz)`,
    /// or `None` if the ray missed or no mouse button is held.
    pub raycast_result: Option<(i32, i32, i32, i32, i32, i32)>,
    /// Block type at the raycasted position, or `None` if the ray missed.
    pub target_block: Option<minerust::BlockType>,
    /// Block type at the camera eye position (used for the underwater effect).
    pub eye_block: minerust::BlockType,
}

/// Batches all world mutations that must occur under the write lock in one frame.
///
/// Collecting mutations during the read-locked snapshot phase and applying them
/// all at once in a single write-lock window minimizes contention with
/// background generation and mesh-building threads.
pub struct WorldWriteOps {
    /// Newly generated chunks ready to be inserted into the world map.
    /// Each entry is `(chunk_x, chunk_z, chunk_data)`.
    pub completed_chunks: Vec<(i32, i32, minerust::Chunk)>,
    /// Block coordinates to replace with `Air` this frame (player broke a block).
    pub block_break: Option<(i32, i32, i32)>,
    /// Block coordinates whose owning subchunk (and its neighbors) should be
    /// marked dirty for re-meshing.
    pub mark_dirty: Vec<(i32, i32, i32)>,
}
