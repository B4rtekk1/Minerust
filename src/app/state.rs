use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3};
use glyphon::{FontSystem, SwashCache, TextAtlas, TextRenderer, Viewport};
use wgpu;
use winit::{keyboard::ModifiersState, window::Window};

use crate::multiplayer::player::RemotePlayer;
use crate::multiplayer::protocol::Packet;
use crate::ui::menu::{GameState, MenuState};
use minerust::chunk_loader::ChunkLoader;
use minerust::{Camera, DiggingState, GpuFaceMesher, IndirectManager, InputState, World};

const MESH_UPLOAD_RING_REGIONS: usize = 3;
const INITIAL_MESH_UPLOAD_REGION_SIZE: u64 = 8 * 1024 * 1024;

/// Triple-buffered source storage for batched mesh uploads.
pub struct MeshUploadRing {
    buffers: Vec<wgpu::Buffer>,
    region_size: u64,
    next_region: usize,
}

impl MeshUploadRing {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buffers: Self::create_buffers(device, INITIAL_MESH_UPLOAD_REGION_SIZE),
            region_size: INITIAL_MESH_UPLOAD_REGION_SIZE,
            next_region: 0,
        }
    }

    fn create_buffers(device: &wgpu::Device, size: u64) -> Vec<wgpu::Buffer> {
        (0..MESH_UPLOAD_RING_REGIONS)
            .map(|index| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Mesh Upload Ring Region {index}")),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    pub fn next_buffer(&mut self, device: &wgpu::Device, required_size: u64) -> &wgpu::Buffer {
        if required_size > self.region_size {
            self.region_size = required_size.next_power_of_two();
            self.buffers = Self::create_buffers(device, self.region_size);
            self.next_region = 0;
        }
        let index = self.next_region;
        self.next_region = (self.next_region + 1) % MESH_UPLOAD_RING_REGIONS;
        &self.buffers[index]
    }
}

/// Tracks block placement while RMB is held so repeat placement stays in one line.
#[derive(Default)]
pub struct BlockPlacementState {
    pub anchor: Option<(i32, i32, i32)>,
    pub last: Option<(i32, i32, i32)>,
    pub axis: Option<usize>,
    pub cooldown: f32,
}

impl BlockPlacementState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

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
///   (`render_pipeline`, `sun_pipeline`, etc.).
/// - **Static geometry buffers** – sun quad, crosshair.
/// - **Uniforms & bind groups** – shared uniform buffer and per-pass bind groups.
/// - **Render targets** – depth, MSAA, scene color, Hi-Z pyramid.
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
/// - **Post-processing** – composite pipeline and its scene-color target.
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
    /// Transparent water render pipeline.
    pub water_pipeline: wgpu::RenderPipeline,
    /// 3-D block outline overlay pipeline.
    pub outline_pipeline: wgpu::RenderPipeline,
    /// Sun disc render pipeline.
    pub sun_pipeline: wgpu::RenderPipeline,
    /// Sky background render pipeline.
    pub sky_pipeline: wgpu::RenderPipeline,
    /// Full-screen bilinear upsample from the half-resolution sky target.
    pub sky_upsample_pipeline: wgpu::RenderPipeline,
    /// Screen-space crosshair render pipeline.
    pub crosshair_pipeline: wgpu::RenderPipeline,
    /// Full-screen composite pipeline that resolves MSAA and applies post-FX.
    pub composite_pipeline: wgpu::RenderPipeline,
    /// Compute pipeline that resolves the MSAA depth buffer into the Hi-Z seed
    /// level 0.
    pub depth_resolve_pipeline: wgpu::ComputePipeline,

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
    /// Whether the in-game crosshair and hotbar are rendered.
    pub show_crosshair: bool,

    // -------------------------------------------------------------------------
    // Uniforms and bind groups
    // -------------------------------------------------------------------------
    /// Uniform buffer containing per-frame data (view-proj, sun direction, etc.).
    pub uniform_buffer: wgpu::Buffer,
    /// Bind group that exposes `uniform_buffer` and the texture atlas to shaders.
    pub uniform_bind_group: wgpu::BindGroup,
    /// Bind group for the composite pass (scene color).
    pub composite_bind_group: wgpu::BindGroup,
    /// Bind group for the main-menu background image composite pass.
    pub menu_composite_bind_group: wgpu::BindGroup,
    /// Bind group for the depth-resolve compute pass.
    pub depth_resolve_bind_group: wgpu::BindGroup,
    /// Bind group used to sample the half-resolution procedural sky.
    pub sky_upsample_bind_group: wgpu::BindGroup,

    // -------------------------------------------------------------------------
    // Render targets and textures
    // -------------------------------------------------------------------------
    /// Non-linear (sRGB) depth buffer view used by the main render pass.
    pub depth_texture: wgpu::TextureView,
    /// MSAA resolve target view (matches the surface format).
    pub msaa_texture_view: wgpu::TextureView,
    /// Single-sampled half-resolution render target for the procedural sky.
    pub sky_texture: wgpu::Texture,
    /// View of `sky_texture` used for both rendering and upsampling.
    pub sky_view: wgpu::TextureView,
    /// Intermediate scene color texture rendered into before compositing.
    pub scene_color_texture: wgpu::Texture,
    /// View of `scene_color_texture`.
    pub scene_color_view: wgpu::TextureView,
    /// Static menu background image texture loaded from `assets/menu.png`.
    #[allow(dead_code)]
    pub menu_background_texture: wgpu::Texture,
    /// View of `menu_background_texture`.
    pub menu_background_view: wgpu::TextureView,
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
    /// Current camera transform. Frustum culling always uses this frame's matrix.
    pub current_view_proj: Mat4,
    /// Camera transform used to render the depth pyramid currently stored in Hi-Z.
    pub previous_view_proj: Mat4,
    /// Camera eye position associated with `previous_view_proj`.
    pub previous_hiz_camera_pos: Vec3,
    /// Camera forward direction associated with `previous_view_proj`.
    pub previous_hiz_forward: Vec3,
    /// True once Hi-Z contains a completed depth pyramid for `previous_view_proj`.
    pub hiz_history_valid: bool,

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
    /// Repeat-placement tracker used while the right mouse button is held.
    pub placement: BlockPlacementState,
    /// The OS window; shared with the event loop and network thread.
    pub window: Arc<Window>,
    /// Whether the cursor is captured (hidden and locked to the window center).
    pub mouse_captured: bool,
    /// Last known cursor position in logical pixels (used for menu interaction).
    pub cursor_position: Option<(f32, f32)>,
    /// Current keyboard modifier state, updated from winit modifier events.
    pub modifiers: ModifiersState,

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
    /// Triple-buffered staging source used to batch all mesh writes per frame.
    pub mesh_upload_ring: MeshUploadRing,
    /// FIFO of subchunks whose CPU block data changed and need async remeshing.
    pub dirty_mesh_queue: VecDeque<(i32, i32, i32)>,
    /// Deduplication set for `dirty_mesh_queue`.
    pub dirty_mesh_queued: HashSet<(i32, i32, i32)>,
    /// Cached list of currently loaded chunk columns inside the render radius.
    pub visible_chunk_columns: Vec<(i32, i32)>,
    /// Player chunk coordinate at which `visible_chunk_columns` was last rebuilt.
    pub visible_chunk_cache_center: (i32, i32),
    /// Forces `visible_chunk_columns` to be rebuilt before the next render.
    pub visible_chunk_columns_dirty: bool,

    // -------------------------------------------------------------------------
    // Indirect rendering managers
    // -------------------------------------------------------------------------
    /// Manages packed quad descriptors and GPU culling for terrain.
    pub indirect_manager: IndirectManager,
    /// Manages packed quad descriptors and GPU culling for water.
    pub water_indirect_manager: IndirectManager,
    /// Persistent compute mesher for ordinary voxel cubes.
    pub gpu_face_mesher: GpuFaceMesher,
    /// Storage bindings for procedural terrain quad expansion.
    pub terrain_quad_bind_group: wgpu::BindGroup,
    /// Storage bindings for procedural water quad expansion.
    pub water_quad_bind_group: wgpu::BindGroup,

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
    /// Whether the FPS/chunk debug overlay is rendered.
    pub show_debug_overlay: bool,

    // Main-menu text buffers.
    /// "multiplayer" menu label.
    pub menu_connect_button_buffer: glyphon::Buffer,
    /// "new world" menu label.
    pub menu_singleplayer_button_buffer: glyphon::Buffer,
    /// Render presentation mode toggle shown in the main menu.
    pub menu_render_mode_button_buffer: glyphon::Buffer,
    /// Server-address input text shown after clicking "multiplayer".
    pub menu_server_address_input_buffer: glyphon::Buffer,

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
    /// `true` after the player has entered a world at least once.
    ///
    /// The initial main menu uses the static menu background. Menus opened
    /// after gameplay has started use the current world render as their
    /// background instead.
    pub has_entered_world: bool,
    /// Tracks focus / edit state of individual menu widgets.
    pub menu_state: MenuState,
    /// `1.0` when the camera eye is inside a water block; `0.0` otherwise.
    /// Passed to the composite shader to apply the underwater color tint.
    pub is_underwater: f32,
    /// Smoothed open-sky visibility at the camera position.
    pub sky_visibility: f32,

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
    /// Approximate open-sky visibility above the camera eye.
    pub sky_visibility: f32,
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
    /// Block coordinates and types to place this frame.
    pub block_places: Vec<(i32, i32, i32, minerust::BlockType)>,
    /// Block coordinates whose owning subchunk (and its neighbors) should be
    /// marked dirty for re-meshing.
    pub mark_dirty: Vec<(i32, i32, i32)>,
}
