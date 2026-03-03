use std::sync::Arc;
use std::time::Instant;

use bytemuck;
use cgmath::{InnerSpace, Matrix4, Rad, SquareMatrix};
use glyphon::{
    Attrs, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea,
    TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowBuilder},
};

use crate::multiplayer;
use crate::ui;

use crate::app::texture_cache;

use clap::Parser;
use multiplayer::network::{connect_to_server, update_network};
use multiplayer::player::{RemotePlayer, queue_remote_players_labels};
use multiplayer::protocol::Packet;
use multiplayer::tcp::{TcpServer};
use std::collections::HashMap;
// use tokio::sync::mpsc;
use ui::menu::{GameState, MenuField, MenuState};
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Start as a server (host a game)
    #[arg(long, default_value_t = false)]
    server: bool,

    /// Port to bind the server to (default: 25565)
    #[arg(long, default_value_t = 25565)]
    port: u16,
}

use render3d::chunk_loader::ChunkLoader;
use render3d::render_core::csm::CsmManager;
use render3d::{
    BlockType, CHUNK_SIZE, Camera, DEFAULT_FOV, DEFAULT_WORLD_FILE, DiggingState,
    GENERATION_DISTANCE, IndirectManager, InputState, MAX_CHUNKS_PER_FRAME,
    MAX_MESH_BUILDS_PER_FRAME, NUM_SUBCHUNKS, RENDER_DISTANCE, SUBCHUNK_HEIGHT, SavedWorld,
    Uniforms, Vertex, World, build_crosshair, build_player_model, extract_frustum_planes,
    load_world, save_world,
};

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// Convert cgmath Vector4 frustum planes to [[f32; 4]; 6] for GPU culling shader.
/// SAFETY: cgmath::Vector4<f32> is #[repr(C)] with layout identical to [f32; 4].
#[inline(always)]
fn frustum_planes_to_array(planes: &[cgmath::Vector4<f32>; 6]) -> [[f32; 4]; 6] {
    unsafe { std::mem::transmute(*planes) }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    sun_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,
    sun_vertex_buffer: wgpu::Buffer,
    sun_index_buffer: wgpu::Buffer,
    crosshair_vertex_buffer: wgpu::Buffer,
    crosshair_index_buffer: wgpu::Buffer,
    num_crosshair_indices: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    msaa_texture_view: wgpu::TextureView,
    shadow_texture_view: wgpu::TextureView,
    shadow_cascade_views: Vec<wgpu::TextureView>,
    shadow_cascade_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    shadow_sampler: wgpu::Sampler,
    world: Arc<parking_lot::RwLock<World>>,
    camera: Camera,
    input: InputState,
    digging: DiggingState,
    window: Arc<Window>,
    frame_count: u32,
    last_fps_update: Instant,
    current_fps: f32,
    last_frame: Instant,
    mouse_captured: bool,
    chunks_rendered: u32,
    subchunks_rendered: u32,
    game_start_time: Instant,
    coords_vertex_buffer: Option<wgpu::Buffer>,
    coords_index_buffer: Option<wgpu::Buffer>,
    coords_num_indices: u32,
    last_coords_position: (i32, i32, i32),
    progress_bar_vertex_buffer: Option<wgpu::Buffer>,
    progress_bar_index_buffer: Option<wgpu::Buffer>,
    #[allow(dead_code)]
    texture_atlas: wgpu::Texture,
    #[allow(dead_code)]
    texture_view: wgpu::TextureView,
    #[allow(dead_code)]
    texture_sampler: wgpu::Sampler,
    game_state: GameState,
    menu_state: MenuState,
    /// Reflection mode: 0=off, 1=SSR (default)
    reflection_mode: u32,
    /// Cached underwater state (1.0 = underwater, 0.0 = above water), updated each tick
    is_underwater: f32,
    // Multiplayer
    remote_players: HashMap<u32, RemotePlayer>,
    my_player_id: u32,
    last_position_send: Instant,
    network_runtime: Option<tokio::runtime::Runtime>,
    network_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Packet>>,
    network_tx: Option<tokio::sync::mpsc::UnboundedSender<Packet>>,
    last_input_time: Instant,
    // Player model rendering
    player_model_vertex_buffer: Option<wgpu::Buffer>,
    player_model_index_buffer: Option<wgpu::Buffer>,
    player_model_num_indices: u32,
    // Async chunk loading
    chunk_loader: ChunkLoader,
    /// Cached player chunk coords — missing-chunk scan is skipped when unchanged
    last_gen_player_cx: i32,
    last_gen_player_cz: i32,
    // SSR (Screen Space Reflections) for water
    ssr_color_texture: wgpu::Texture,
    ssr_color_view: wgpu::TextureView,
    ssr_depth_texture: wgpu::Texture,
    ssr_depth_view: wgpu::TextureView,
    ssr_sampler: wgpu::Sampler,
    water_bind_group: wgpu::BindGroup,
    water_bind_group_layout: wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
    // Glyphon text rendering
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    fps_buffer: glyphon::Buffer,
    menu_buffer: glyphon::Buffer,
    player_label_buffers: Vec<glyphon::Buffer>,
    mesh_loader: render3d::MeshLoader,
    // SSAO (Screen Space Ambient Occlusion)
    ssao_enabled: bool,
    ssao_texture: wgpu::Texture,
    ssao_texture_view: wgpu::TextureView,
    ssao_blur_texture: wgpu::Texture,
    ssao_blur_view: wgpu::TextureView,
    ssao_noise_view: wgpu::TextureView,
    ssao_params_buffer: wgpu::Buffer,
    ssao_bind_group: wgpu::BindGroup,
    ssao_blur_bind_group: wgpu::BindGroup,
    ssao_pipeline: wgpu::RenderPipeline,
    ssao_blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group: wgpu::BindGroup,
    scene_color_texture: wgpu::Texture,
    scene_color_view: wgpu::TextureView,
    // GPU Indirect Drawing
    indirect_manager: IndirectManager,
    water_indirect_manager: IndirectManager,
    // Hi-Z Occlusion Culling
    hiz_texture: wgpu::Texture,
    hiz_view: wgpu::TextureView,
    hiz_mips: Vec<wgpu::TextureView>,
    hiz_pipeline: wgpu::ComputePipeline,
    hiz_bind_groups: Vec<wgpu::BindGroup>,
    hiz_bind_group_layout: wgpu::BindGroupLayout,
    /// Next-power-of-two size derived from the render resolution (≤ 4096).
    /// Dimensions of the Hi-Z texture (matches screen size).
    hiz_size: [u32; 2],

    // Depth resolve for SSR
    depth_resolve_pipeline: wgpu::RenderPipeline,
    depth_resolve_bind_group: wgpu::BindGroup,
}

struct WorldSnapshot {
    missing_chunks: Vec<(i32, i32, i32)>,
    raycast_result: Option<(i32, i32, i32, i32, i32, i32)>,
    target_block: Option<BlockType>,
    eye_block: BlockType,
}

struct WorldWriteOps {
    completed_chunks: Vec<(i32, i32, render3d::Chunk)>,
    block_break: Option<(i32, i32, i32)>, // (x, y, z)
    mark_dirty: Vec<(i32, i32, i32)>,
}

impl State {
    async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        tracing::info!("WGPU Instance created successfully");

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let info = adapter.get_info();
        tracing::info!(
            "Selected adapter: {} on {:?} backend",
            info.name,
            info.backend
        );
        if info.device_type == wgpu::DeviceType::Cpu {
            tracing::warn!(
                "Warning: Running on CPU (Software Renderer). Performance will be poor."
            );
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate, // No VSync - uncapped FPS
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // MSAA sample count (4x MSAA for quality anti-aliasing)
        // Note: Depth32Float only supports [1, 4] samples on most devices
        let msaa_sample_count: u32 = 4;

        let depth_texture = Self::create_depth_texture(&device, &config, msaa_sample_count);
        let msaa_texture_view =
            Self::create_msaa_texture(&device, &config, surface_format, msaa_sample_count);

        let hiz_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hi-Z Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz.wgsl").into()),
        });

        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terrain.wgsl").into()),
        });

        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water.wgsl").into()),
        });

        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
        });

        let sun_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sun Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sun.wgsl").into()),
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });

        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                view_proj: Matrix4::from_scale(1.0).into(),
                inv_view_proj: Matrix4::from_scale(1.0).into(),
                csm_view_proj: [[Matrix4::from_scale(1.0).into(); 1]; 4],
                csm_split_distances: [16.0, 48.0, 128.0, 300.0],
                camera_pos: [0.0, 0.0, 0.0],
                time: 0.0,
                sun_position: [0.4, -0.2, 0.3],
                is_underwater: 0.0,
                screen_size: [1920.0, 1080.0],
                water_level: 63.0,
                reflection_mode: 1.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });


        let (texture_atlas, texture_view, _atlas_width, _atlas_height) =
            texture_cache::load_or_generate_atlas(&device, &queue);

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        let shadow_map_size = 2048;
        let shadow_map_desc = wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: shadow_map_size,
                height: shadow_map_size,
                depth_or_array_layers: 4,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let shadow_texture = device.create_texture(&shadow_map_desc);
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Map Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let shadow_cascade_views = (0..4)
            .map(|i| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Shadow Map Cascade View {}", i)),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();

        let shadow_cascade_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Cascade Buffer"),
            size: 256 * 4, // 256 byte alignment * 4 cascades
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            });

        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // SSR textures for water reflections
        let ssr_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_color_view = ssr_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ssr_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_depth_view = ssr_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ssr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Water bind group layout with SSR textures
        let water_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    // SSR color texture (scene rendered before water)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // SSR depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // SSR sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&ssr_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&ssr_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&ssr_sampler),
                },
            ],
            label: Some("water_bind_group"),
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
            label: Some("uniform_bind_group"),
        });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shadow_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_cascade_buffer,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(80).unwrap()), // Exact size of data (mat4 + float)
                }),
            }],
            label: Some("shadow_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            immediate_size: 0,
        });

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Pipeline Layout"),
                bind_group_layouts: &[&shadow_bind_group_layout],
                immediate_size: 0,
            });

        let water_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Water Pipeline Layout"),
                bind_group_layouts: &[&water_bind_group_layout],
                immediate_size: 0,
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        // Reflection pipeline: same as render_pipeline but with reversed culling to handle mirrored winding

        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&water_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &water_shader,
                entry_point: Some("vs_water"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &water_shader,
                entry_point: Some("fs_water"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: Some("vs_ui"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_ui"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Cull back faces in the shadow pass — eliminates ~50% of shadow
                // rasterization work. Voxel geometry only exposes outward faces so
                // this is safe and prevents self-shadowing artefacts on rear faces.
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        let sun_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sun Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &sun_shader,
                entry_point: Some("vs_sun"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sun_shader,
                entry_point: Some("fs_sun"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        // Sky pipeline - renders a fullscreen quad with procedural sky
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        let sun_normal = Vertex::pack_normal([0.0, 0.0, 1.0]);
        let sun_color = Vertex::pack_color([1.0, 1.0, 1.0]);
        let sun_vertices = vec![
            Vertex {
                position: [-1.0, -1.0, 0.0],
                normal: sun_normal,
                color: sun_color,
                uv: [0.0, 0.0],
                tex_index: 0.0,
            },
            Vertex {
                position: [1.0, -1.0, 0.0],
                normal: sun_normal,
                color: sun_color,
                uv: [1.0, 0.0],
                tex_index: 0.0,
            },
            Vertex {
                position: [1.0, 1.0, 0.0],
                normal: sun_normal,
                color: sun_color,
                uv: [1.0, 1.0],
                tex_index: 0.0,
            },
            Vertex {
                position: [-1.0, 1.0, 0.0],
                normal: sun_normal,
                color: sun_color,
                uv: [0.0, 1.0],
                tex_index: 0.0,
            },
        ];
        let sun_indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        let sun_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sun Vertex Buffer"),
            contents: bytemuck::cast_slice(&sun_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sun_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sun Index Buffer"),
            contents: bytemuck::cast_slice(&sun_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        tracing::info!("Generating world...");
        let world = Arc::new(parking_lot::RwLock::new(World::new()));
        let spawn = world.read().find_spawn_point();
        let camera = Camera::new(spawn);
        tracing::info!("World generated! Spawn: {:?}", spawn);

        let seed = world.read().seed;
        let chunk_loader = ChunkLoader::new(seed);
        let mesh_loader =
            render3d::MeshLoader::new(Arc::clone(&world), render3d::get_mesh_worker_count());

        let (crosshair_vertices, crosshair_indices) = build_crosshair();
        let num_crosshair_indices = crosshair_indices.len() as u32;

        let crosshair_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Crosshair Vertex Buffer"),
                contents: bytemuck::cast_slice(&crosshair_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let crosshair_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Crosshair Index Buffer"),
            contents: bytemuck::cast_slice(&crosshair_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let mut font_system = FontSystem::new();
        // Load font
        font_system.db_mut().load_font_data(
            include_bytes!("../../assets/fonts/GoogleSans_17pt-Regular.ttf").to_vec(),
        );

        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut viewport = Viewport::new(&device, &cache);
        viewport.update(
            &queue,
            Resolution {
                width: config.width,
                height: config.height,
            },
        );

        let fps_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(40.0, 48.0));
        let menu_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(24.0, 32.0));

        // ============== DEPTH RESOLVE INITIALIZATION ==============
        let depth_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Depth Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_resolve.wgsl").into()),
        });

        let depth_resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Depth Resolve Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
            });

        let depth_resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Resolve Bind Group"),
            layout: &depth_resolve_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_texture),
            }],
        });

        let depth_resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Depth Resolve Pipeline Layout"),
                bind_group_layouts: &[&depth_resolve_bind_group_layout],
                immediate_size: 0,
            });

        let depth_resolve_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Depth Resolve Pipeline"),
                layout: Some(&depth_resolve_pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &depth_resolve_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &depth_resolve_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
            });

        // ============== SSAO INITIALIZATION ==============

        // Load SSAO shaders
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });

        let ssao_blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSAO Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao_blur.wgsl").into()),
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        let ssao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_texture_view = ssao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ssao_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Blur Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_blur_view = ssao_blur_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let scene_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_color_view =
            scene_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut ssao_kernel: [[f32; 4]; 64] = [[0.0; 4]; 64];
        let mut rng_state: u32 = 12345;
        for i in 0..64 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r1 = (rng_state as f32) / (u32::MAX as f32);
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r2 = (rng_state as f32) / (u32::MAX as f32);
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r3 = (rng_state as f32) / (u32::MAX as f32);

            let x = r1 * 2.0 - 1.0;
            let y = r2 * 2.0 - 1.0;
            let z = r3;
            let len = (x * x + y * y + z * z).sqrt().max(0.001);

            let scale = (i as f32) / 64.0;
            let scale = 0.1 + scale * scale * 0.9; // lerp(0.1, 1.0, scale^2)

            ssao_kernel[i] = [x / len * scale, y / len * scale, z / len * scale, 0.0];
        }

        let mut noise_data: [u8; 64] = [0; 64];
        for i in 0..16 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r1 = (rng_state as f32) / (u32::MAX as f32);
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r2 = (rng_state as f32) / (u32::MAX as f32);

            noise_data[i * 4] = ((r1 * 2.0 - 1.0) * 127.5 + 127.5) as u8;
            noise_data[i * 4 + 1] = ((r2 * 2.0 - 1.0) * 127.5 + 127.5) as u8;
            noise_data[i * 4 + 2] = 128; // Z = 0 (after denormalization)
            noise_data[i * 4 + 3] = 255; // Alpha
        }

        let ssao_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise Texture"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ssao_noise_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );

        let ssao_noise_view =
            ssao_noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // SSAO params uniform
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SSAOParams {
            proj: [[f32; 4]; 4],
            inv_proj: [[f32; 4]; 4],
            samples: [[f32; 4]; 64],
            noise_scale: [f32; 2],
            radius: f32,
            bias: f32,
            intensity: f32,
            aspect_ratio: f32,
            _padding: [f32; 2],
        }

        let proj_mat: [[f32; 4]; 4] = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            config.width as f32 / config.height as f32,
            0.1,
            500.0,
        )
        .into();

        let inv_proj_mat = cgmath::perspective(
            Rad(std::f32::consts::FRAC_PI_2),
            config.width as f32 / config.height as f32,
            0.1,
            500.0,
        )
        .invert()
        .unwrap_or(Matrix4::identity());

        let aspect_ratio = config.width as f32 / config.height as f32;

        let ssao_params = SSAOParams {
            proj: proj_mat,
            inv_proj: inv_proj_mat.into(),
            samples: ssao_kernel,
            noise_scale: [config.width as f32 / 4.0, config.height as f32 / 4.0],
            radius: 0.5,
            bias: 0.025,
            intensity: 1.5,
            aspect_ratio,
            _padding: [0.0; 2],
        };

        let ssao_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Params Buffer"),
            contents: bytemuck::bytes_of(&ssao_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // SSAO bind group layout
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let ssao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ssao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Bind Group"),
            layout: &ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ssao_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ssr_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ssao_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                },
            ],
        });

        // SSAO blur bind group layout
        let ssao_blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Blur Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let ssao_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Blur Bind Group"),
            layout: &ssao_blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ssao_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ssr_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                },
            ],
        });

        // Composite bind group layout
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ssao_blur_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&composite_sampler),
                },
            ],
        });

        // SSAO Pipeline Layout
        let ssao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[&ssao_bind_group_layout],
            immediate_size: 0,
        });

        let ssao_blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO Blur Pipeline Layout"),
                bind_group_layouts: &[&ssao_blur_bind_group_layout],
                immediate_size: 0,
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                immediate_size: 0,
            });

        // SSAO Pipeline
        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Pipeline"),
            layout: Some(&ssao_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        // SSAO Blur Pipeline
        let ssao_blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Blur Pipeline"),
            layout: Some(&ssao_blur_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &ssao_blur_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_blur_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        // Composite Pipeline
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        // ============== END SSAO INITIALIZATION ==============

        // Create IndirectManagers before Self (device is moved into Self)
        let mut indirect_manager = IndirectManager::new(&device);
        let mut water_indirect_manager = IndirectManager::new(&device);

        // Initialize shadow culling resources
        indirect_manager.init_shadow_resources(&device);
        water_indirect_manager.init_shadow_resources(&device);

        // Hi-Z Initialization: match screen size exactly to allow resolving in the same pass as SSR depth.
        let hiz_size = [config.width, config.height];
        let hiz_max_dim = config.width.max(config.height);
        let hiz_mips_count = (hiz_max_dim as f32).log2().floor() as u32 + 1;
        let hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z Texture"),
            size: wgpu::Extent3d {
                width: hiz_size[0],
                height: hiz_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: hiz_mips_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let hiz_mips = (0..hiz_mips_count)
            .map(|i| {
                hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Hi-Z Mip View {}", i)),
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();

        let hiz_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Hi-Z Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let hiz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hi-Z Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hi-Z Pipeline Layout"),
                    bind_group_layouts: &[&hiz_bind_group_layout],
                    immediate_size: 0,
                }),
            ),
            module: &hiz_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hiz_bind_groups = (0..hiz_mips_count - 1)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Hi-Z Bind Group {}", i)),
                    layout: &hiz_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&hiz_mips[i as usize]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &hiz_mips[(i + 1) as usize],
                            ),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>();

        // Initial bind group update
        indirect_manager.update_bind_group(&device, &hiz_view);
        water_indirect_manager.update_bind_group(&device, &hiz_view);

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            water_pipeline,
            sun_pipeline,
            sky_pipeline,
            shadow_pipeline,
            crosshair_pipeline,
            sun_vertex_buffer,
            sun_index_buffer,
            crosshair_vertex_buffer,
            crosshair_index_buffer,
            num_crosshair_indices,
            uniform_buffer,
            uniform_bind_group,
            shadow_bind_group,
            depth_texture,
            msaa_texture_view,
            shadow_texture_view,
            shadow_cascade_views,
            shadow_cascade_buffer,
            #[allow(dead_code)]
            shadow_sampler,
            world,
            mesh_loader,
            camera,
            input: InputState::default(),
            digging: DiggingState::default(),
            window,
            frame_count: 0,
            last_fps_update: Instant::now(),
            current_fps: 0.0,
            last_frame: Instant::now(),
            mouse_captured: false,
            chunks_rendered: 0,
            subchunks_rendered: 0,
            game_start_time: Instant::now(),
            coords_vertex_buffer: None,
            coords_index_buffer: None,
            coords_num_indices: 0,
            last_coords_position: (i32::MIN, i32::MIN, i32::MIN),
            progress_bar_vertex_buffer: None,
            progress_bar_index_buffer: None,
            texture_atlas,
            texture_view,
            texture_sampler,
            game_state: GameState::Menu,
            menu_state: MenuState::default(),
            reflection_mode: 1, // Default: SSR only
            is_underwater: 0.0,
            // Multiplayer
            remote_players: HashMap::new(),
            my_player_id: 0,
            last_position_send: Instant::now(),
            network_runtime: Some(
                tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"),
            ),
            network_rx: None,
            network_tx: None,
            last_input_time: Instant::now(),
            // Player model rendering
            player_model_vertex_buffer: None,
            player_model_index_buffer: None,
            player_model_num_indices: 0,
            // Async chunk loading
            chunk_loader,
            last_gen_player_cx: i32::MIN,
            last_gen_player_cz: i32::MIN,
            // SSR (Screen Space Reflections) for water
            ssr_color_texture,
            ssr_color_view,
            ssr_depth_texture,
            ssr_depth_view,
            ssr_sampler,
            water_bind_group,
            water_bind_group_layout,
            surface_format,
            // Glyphon
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            fps_buffer,
            menu_buffer,
            player_label_buffers: Vec::new(),
            // SSAO (Screen Space Ambient Occlusion)
            ssao_enabled: true,
            ssao_texture,
            ssao_texture_view,
            ssao_blur_texture,
            ssao_blur_view,
            ssao_noise_view,
            ssao_params_buffer,
            ssao_bind_group,
            ssao_blur_bind_group,
            ssao_pipeline,
            ssao_blur_pipeline,
            composite_pipeline,
            composite_bind_group,
            scene_color_texture,
            scene_color_view,
            // GPU Indirect Drawing
            indirect_manager,
            water_indirect_manager,
            hiz_texture,
            hiz_view,
            hiz_mips,
            hiz_pipeline,
            hiz_bind_groups,
            hiz_bind_group_layout,
            hiz_size,
            // Depth Resolve
            depth_resolve_pipeline,
            depth_resolve_bind_group,
        }
    }

    fn update_coords_ui(&mut self) {
        if let Some((vb, ib, num_indices)) = ui::ui::update_coords_ui(
            &self.device,
            self.camera.position,
            &mut self.last_coords_position,
        ) {
            self.coords_vertex_buffer = Some(vb);
            self.coords_index_buffer = Some(ib);
            self.coords_num_indices = num_indices;
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_msaa_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            // MSAA sample count (must match initialization)
            let msaa_sample_count: u32 = 4;
            self.depth_texture =
                Self::create_depth_texture(&self.device, &self.config, msaa_sample_count);
            self.msaa_texture_view = Self::create_msaa_texture(
                &self.device,
                &self.config,
                self.surface_format,
                msaa_sample_count,
            );

            // Recreate SSR textures at new size
            self.ssr_color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SSR Color Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ssr_color_view = self
                .ssr_color_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.ssr_depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SSR Depth Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ssr_depth_view = self
                .ssr_depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.ssr_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SSR Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            });

            // Recreate water bind group with new texture views
            self.water_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.water_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.shadow_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.ssr_sampler),
                    },
                ],
                label: Some("water_bind_group"),
            });

            // Recreate depth resolve bind group
            self.depth_resolve_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Depth Resolve Bind Group"),
                    layout: &self.depth_resolve_pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.depth_texture),
                    }],
                });

            self.viewport.update(
                &self.queue,
                Resolution {
                    width: new_size.width,
                    height: new_size.height,
                },
            );

            // Recreate SSAO textures at new size
            self.ssao_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SSAO Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ssao_texture_view = self
                .ssao_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.ssao_blur_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SSAO Blur Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ssao_blur_view = self
                .ssao_blur_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            self.scene_color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Scene Color Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.scene_color_view = self
                .scene_color_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // Recreate SSAO bind groups with new texture views
            let ssao_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SSAO Sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            // Need to get bind group layout from existing pipeline
            let ssao_bind_group_layout = self.ssao_pipeline.get_bind_group_layout(0);
            self.ssao_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Bind Group"),
                layout: &ssao_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.ssao_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.ssao_noise_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                    },
                ],
            });

            let ssao_blur_bind_group_layout = self.ssao_blur_pipeline.get_bind_group_layout(0);
            self.ssao_blur_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Blur Bind Group"),
                layout: &ssao_blur_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.ssao_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                    },
                ],
            });

            let composite_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Composite Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let composite_bind_group_layout = self.composite_pipeline.get_bind_group_layout(0);
            self.composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group"),
                layout: &composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.ssao_blur_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&composite_sampler),
                    },
                ],
            });

            // Recreate Hi-Z pyramid to match new render resolution exactly
            let new_hiz_size = [new_size.width, new_size.height];
            if new_hiz_size != self.hiz_size {
                self.hiz_size = new_hiz_size;
                let hiz_max_dim = new_size.width.max(new_size.height);
                let hiz_mips_count = (hiz_max_dim as f32).log2().floor() as u32 + 1;
                let hiz_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Hi-Z Texture"),
                    size: wgpu::Extent3d {
                        width: new_hiz_size[0],
                        height: new_hiz_size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: hiz_mips_count,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let new_hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());
                let new_hiz_mips: Vec<_> = (0..hiz_mips_count)
                    .map(|i| {
                        hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                            label: Some(&format!("Hi-Z Mip View {}", i)),
                            base_mip_level: i,
                            mip_level_count: Some(1),
                            ..Default::default()
                        })
                    })
                    .collect();
                let new_hiz_bind_groups: Vec<_> = (0..hiz_mips_count - 1)
                    .map(|i| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("Hi-Z Bind Group {}", i)),
                            layout: &self.hiz_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &new_hiz_mips[i as usize],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &new_hiz_mips[(i + 1) as usize],
                                    ),
                                },
                            ],
                        })
                    })
                    .collect();
                self.indirect_manager
                    .update_bind_group(&self.device, &new_hiz_view);
                self.water_indirect_manager
                    .update_bind_group(&self.device, &new_hiz_view);
                self.hiz_texture = hiz_texture;
                self.hiz_view = new_hiz_view;
                self.hiz_mips = new_hiz_mips;
                self.hiz_bind_groups = new_hiz_bind_groups;
            }
        }
    }

    fn update_subchunk_mesh(&mut self, result: render3d::mesh_loader::MeshResult) {
        let cx = result.cx;
        let cz = result.cz;
        let sy = result.sy;

        let aabb_copy = {
            let mut world = self.world.write();

            // Use entry API for single lookup:
            let chunk = match world.chunks.get_mut(&(cx, cz)) {
                Some(chunk) => chunk,
                None => return, // Early return if chunk doesn't exist
            };

            let subchunk = &mut chunk.subchunks[sy as usize];
            let aabb = subchunk.aabb; // Copy AABB before releasing lock

            // Update mesh counts (GPU buffers managed exclusively by IndirectManager)
            subchunk.num_indices = result.terrain.1.len() as u32;
            subchunk.num_water_indices = result.water.1.len() as u32;
            subchunk.mesh_dirty = false;
            aabb // Return AABB
        }; // Lock released

        // Now upload to IndirectManager (no world lock needed)
        if !result.terrain.0.is_empty() && !result.terrain.1.is_empty() {
            let key = render3d::render::indirect::SubchunkKey {
                chunk_x: cx,
                chunk_z: cz,
                subchunk_y: sy,
            };
            self.indirect_manager.upload_subchunk(
                &self.queue,
                key,
                &result.terrain.0,
                &result.terrain.1,
                &aabb_copy,
            );
        }

        if !result.water.0.is_empty() && !result.water.1.is_empty() {
            let key = render3d::render::indirect::SubchunkKey {
                chunk_x: cx,
                chunk_z: cz,
                subchunk_y: sy,
            };
            self.water_indirect_manager.upload_subchunk(
                &self.queue,
                key,
                &result.water.0,
                &result.water.1,
                &aabb_copy,
            );
        }
    }

    fn update(&mut self) {
        self.update_network();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let completed_chunks = self.chunk_loader.poll_results(MAX_CHUNKS_PER_FRAME);

        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let player_chunk_moved =
            player_cx != self.last_gen_player_cx || player_cz != self.last_gen_player_cz;

        let snapshot = {
            let world = self.world.read();

            self.camera.update(&*world, dt, &self.input);

            // Only scan for missing chunks when transitioning to a new chunk or when the loader queue has space.
            // This ensures we eventually fill the entire generation radius even if stationary.
            let mut missing_chunks = Vec::new();
            if player_chunk_moved || self.chunk_loader.pending_count() < 32 {
                for cx in (player_cx - GENERATION_DISTANCE)..=(player_cx + GENERATION_DISTANCE) {
                    for cz in (player_cz - GENERATION_DISTANCE)..=(player_cz + GENERATION_DISTANCE)
                    {
                        if !world.chunks.contains_key(&(cx, cz))
                            && !self.chunk_loader.is_pending(cx, cz)
                        {
                            let dx = cx - player_cx;
                            let dz = cz - player_cz;
                            let priority = dx * dx + dz * dz;
                            missing_chunks.push((cx, cz, priority));
                        }
                    }
                }
            }

            let (raycast_result, target_block) = if self.mouse_captured && self.input.left_mouse {
                let raycast = self.camera.raycast(&*world, 5.0);
                if let Some((bx, by, bz, _, _, _)) = raycast {
                    let block = world.get_block(bx, by, bz);
                    (Some((bx, by, bz, 0, 0, 0)), Some(block))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            let eye_pos = self.camera.eye_position();
            let eye_block = world.get_block(
                eye_pos.x.floor() as i32,
                eye_pos.y.floor() as i32,
                eye_pos.z.floor() as i32,
            );

            WorldSnapshot {
                missing_chunks,
                raycast_result,
                target_block,
                eye_block,
            }
        };

        if player_chunk_moved {
            self.last_gen_player_cx = player_cx;
            self.last_gen_player_cz = player_cz;
        }

        let mut requests = snapshot.missing_chunks;
        requests.sort_by_key(|&(_, _, priority)| priority);
        // Request up to 8 chunks per frame (if missing) to fill the worker queue faster
        for (cx, cz, priority) in requests.into_iter().take(MAX_CHUNKS_PER_FRAME * 2) {
            self.chunk_loader.request_chunk(cx, cz, priority);
        }

        let mut write_ops = WorldWriteOps {
            completed_chunks: completed_chunks
                .into_iter()
                .map(|r| (r.cx, r.cz, r.chunk))
                .collect(),
            block_break: None,
            mark_dirty: Vec::new(),
        };

        if let Some(target_block) = snapshot.target_block {
            if let Some((bx, by, bz, _, _, _)) = snapshot.raycast_result {
                let target = (bx, by, bz);
                let break_time = target_block.break_time();

                if break_time.is_finite() && break_time > 0.0 {
                    if self.digging.target == Some(target) {
                        self.digging.progress += dt;
                        if self.digging.progress >= break_time {
                            write_ops.block_break = Some((bx, by, bz));
                            write_ops.mark_dirty.push((bx, by, bz));
                            self.digging.target = None;
                            self.digging.progress = 0.0;
                        }
                    } else {
                        self.digging.target = Some(target);
                        self.digging.progress = 0.0;
                        // Zapisz break_time do DiggingState
                        self.digging.break_time = break_time;
                    }
                }
            }
        } else {
            self.digging.target = None;
            self.digging.progress = 0.0;
        }

        if !write_ops.completed_chunks.is_empty()
            || write_ops.block_break.is_some()
            || !write_ops.mark_dirty.is_empty()
        {
            let mut world = self.world.write();

            for (cx, cz, chunk) in write_ops.completed_chunks {
                world.chunks.insert((cx, cz), chunk);
            }

            if let Some((bx, by, bz)) = write_ops.block_break {
                world.set_block_player(bx, by, bz, BlockType::Air);
            }

            world.update_chunks_around_player(self.camera.position.x, self.camera.position.z);
        }
        for (bx, by, bz) in write_ops.mark_dirty {
            self.mark_chunk_dirty(bx, by, bz);
        }

        self.is_underwater = if snapshot.eye_block == BlockType::Water {
            1.0
        } else {
            0.0
        };

        self.update_coords_ui();

        while let Some(result) = self.mesh_loader.poll_result() {
            self.update_subchunk_mesh(result);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // 1. Build remote player meshes before starting render pass
        if !self.remote_players.is_empty() && self.game_state != GameState::Menu {
            let mut all_vertices = Vec::with_capacity(self.remote_players.len() * 16);
            let mut all_indices = Vec::with_capacity(self.remote_players.len() * 24);

            for (_id, player) in &self.remote_players {
                let (vertices, indices) =
                    build_player_model(player.x, player.y, player.z, player.yaw);
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            self.player_model_num_indices = all_indices.len() as u32;

            if !all_vertices.is_empty() {
                self.player_model_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Player Model Vertex Buffer"),
                        contents: bytemuck::cast_slice(&all_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
                self.player_model_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Player Model Index Buffer"),
                        contents: bytemuck::cast_slice(&all_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
            }
        } else {
            self.player_model_num_indices = 0;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let aspect = self.config.width as f32 / self.config.height as f32;
        // DEFAULT_FOV is vertical FOV (like Minecraft and most games)
        let proj = cgmath::perspective(Rad(DEFAULT_FOV), aspect, 0.1, 500.0);
        let view_mat = self.camera.view_matrix();
        let view_proj = OPENGL_TO_WGPU_MATRIX * proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.into();

        let time = self.game_start_time.elapsed().as_secs_f32();

        let day_cycle_speed = 0.005;
        // Start at noon (sun at top) by adding PI/2 offset
        let sun_angle = time * day_cycle_speed + std::f32::consts::FRAC_PI_2;

        let sun_x = 0.0;
        let sun_y = sun_angle.sin();
        let sun_z = sun_angle.cos();
        let sun_dir = cgmath::Vector3::new(sun_x, sun_y, sun_z).normalize();

        // Use CsmManager to compute cascade matrices with texel snapping for stability
        let mut csm = CsmManager::new();
        let fov_y = DEFAULT_FOV;
        csm.update(
            &view_mat, sun_dir, 0.1,   // near
            300.0, // far (render distance)
            aspect, fov_y,
        );

        // Convert cascade matrices to array format for uniforms
        let csm_view_proj: [[[[f32; 4]; 4]; 1]; 4] = [
            [csm.cascades[0].view_proj.into()],
            [csm.cascades[1].view_proj.into()],
            [csm.cascades[2].view_proj.into()],
            [csm.cascades[3].view_proj.into()],
        ];
        let csm_split_distances: [f32; 4] = [
            csm.cascades[0].split_distance,
            csm.cascades[1].split_distance,
            csm.cascades[2].split_distance,
            csm.cascades[3].split_distance,
        ];

        let inv_view_proj = view_proj.invert().unwrap_or(Matrix4::identity());
        let inv_view_proj_array: [[f32; 4]; 4] = inv_view_proj.into();

        // Check if camera is underwater (cached from update())
        let eye_pos = self.camera.eye_position();
        let is_underwater = self.is_underwater;

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                view_proj: view_proj_array,
                inv_view_proj: inv_view_proj_array,
                csm_view_proj,
                csm_split_distances,
                camera_pos: eye_pos.into(),
                time,
                sun_position: [sun_x, sun_y, sun_z],
                is_underwater,
                screen_size: [self.config.width as f32, self.config.height as f32],
                water_level: 63.0,
                reflection_mode: self.reflection_mode as f32,
            }]),
        );

        let frustum_planes = extract_frustum_planes(&view_proj);

        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        // Render 4 cascades for shadows
        for i in 0..4 {
            let cascade_matrix: [[f32; 4]; 4] = csm.cascades[i].view_proj.into();
            let mut shadow_uniform_data = [0f32; 64];
            shadow_uniform_data[0..16].copy_from_slice(cascade_matrix.as_flattened());
            shadow_uniform_data[16] = time;

            self.queue.write_buffer(
                &self.shadow_cascade_buffer,
                (i * 256) as u64,
                bytemuck::cast_slice(&shadow_uniform_data),
            );

            // Select cascade view-proj matrix using dynamic uniform offset
            let offset = (i * 256) as u32;

            // Extract light frustum planes for this cascade to perform GPU culling
            let cascade_view_proj = csm.cascades[i].view_proj;
            let shadow_frustum = extract_frustum_planes(&cascade_view_proj);
            let shadow_frustum_array = frustum_planes_to_array(&shadow_frustum);

            // Dispatch GPU culling for this shadow cascade
            self.indirect_manager.dispatch_shadow_culling(
                &mut encoder,
                &self.queue,
                i,
                &shadow_frustum_array,
            );
            self.water_indirect_manager.dispatch_shadow_culling(
                &mut encoder,
                &self.queue,
                i,
                &shadow_frustum_array,
            );

            const SHADOW_PASS_LABELS: [&str; 4] = [
                "Shadow Pass Cascade 0",
                "Shadow Pass Cascade 1",
                "Shadow Pass Cascade 2",
                "Shadow Pass Cascade 3",
            ];
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(SHADOW_PASS_LABELS[i]),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_cascade_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.shadow_bind_group, &[offset]);

            // 1. Draw Terrain (using GPU Indirect Drawing)
            shadow_pass.set_vertex_buffer(0, self.indirect_manager.vertex_buffer().slice(..));
            shadow_pass.set_index_buffer(
                self.indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            shadow_pass.multi_draw_indexed_indirect(
                self.indirect_manager.shadow_draw_commands(i),
                0,
                self.indirect_manager.active_count(),
            );
        }

        // Optimization: Pre-allocate since we limit to max_meshes_per_frame
        let mut meshes_to_request: Vec<(i32, i32, i32)> = Vec::with_capacity(8);
        let mut chunks_rendered = 0u32;
        let mut subchunks_rendered = 0u32;

        {
            let world = self.world.read();
            for cx in (player_cx - RENDER_DISTANCE)..=(player_cx + RENDER_DISTANCE) {
                for cz in (player_cz - RENDER_DISTANCE)..=(player_cz + RENDER_DISTANCE) {
                    if let Some(chunk) = world.chunks.get(&(cx, cz)) {
                        let mut chunk_has_visible = false;
                        for (sy, subchunk) in chunk.subchunks.iter().enumerate() {
                            if subchunk.is_empty {
                                continue;
                            }

                            // Collect dirty meshes
                            if subchunk.mesh_dirty {
                                meshes_to_request.push((cx, cz, sy as i32));
                            }

                            // Count active subchunks (submitted to GPU culling)
                            if subchunk.num_indices > 0 || subchunk.num_water_indices > 0 {
                                subchunks_rendered += 1;
                                chunk_has_visible = true;
                            }
                        }
                        if chunk_has_visible {
                            chunks_rendered += 1;
                        }
                    }
                }
            }
        }

        let max_meshes_per_frame = MAX_MESH_BUILDS_PER_FRAME;
        // Prioritize meshes closer to the player
        meshes_to_request.sort_by_key(|&(cx, cz, _sy)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });
        meshes_to_request.truncate(max_meshes_per_frame);

        for (cx, cz, sy) in &meshes_to_request {
            self.mesh_loader.request_mesh(*cx, *cz, *sy);
        }

        // Mark chunks as not dirty to avoid re-requesting
        if !meshes_to_request.is_empty() {
            let mut world = self.world.write();
            for (cx, cz, sy) in &meshes_to_request {
                if let Some(chunk) = world.chunks.get_mut(&(*cx, *cz)) {
                    chunk.subchunks[*sy as usize].mesh_dirty = false;
                }
            }
        }

        let day_factor = sun_dir.y.max(0.0).min(1.0);
        let night_factor = (-sun_dir.y).max(0.0).min(1.0);
        let sunset_factor = 1.0 - sun_dir.y.abs();

        let day_sky = (0.53, 0.81, 0.98);
        let sunset_sky = (1.0, 0.5, 0.2);
        let night_sky = (0.001, 0.001, 0.005);

        let sky_r: f32 = (day_sky.0 * day_factor
            + sunset_sky.0 * sunset_factor * 0.5
            + night_sky.0 * night_factor)
            .min(1.0);
        let sky_g: f32 = (day_sky.1 * day_factor
            + sunset_sky.1 * sunset_factor * 0.5
            + night_sky.1 * night_factor)
            .min(1.0);
        let sky_b: f32 = (day_sky.2 * day_factor
            + sunset_sky.2 * sunset_factor * 0.5
            + night_sky.2 * night_factor)
            .min(1.0);

        self.chunks_rendered = chunks_rendered;
        self.subchunks_rendered = subchunks_rendered;

        // Dispatch GPU frustum and occlusion culling for indirect drawing
        let frustum_planes_array = frustum_planes_to_array(&frustum_planes);

        let hiz_size_f = [self.hiz_size[0] as f32, self.hiz_size[1] as f32];
        self.indirect_manager.dispatch_culling(
            &mut encoder,
            &self.queue,
            &view_proj,
            &frustum_planes_array,
            self.camera.position.into(),
            hiz_size_f,
            [self.config.width as f32, self.config.height as f32],
        );
        self.water_indirect_manager.dispatch_culling(
            &mut encoder,
            &self.queue,
            &view_proj,
            &frustum_planes_array,
            self.camera.position.into(),
            hiz_size_f,
            [self.config.width as f32, self.config.height as f32],
        );

        // Opaque Pass: Sky, Terrain, Players, Sun
        {
            let mut opaque_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Opaque Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(&self.ssr_color_view), // Resolve to SSR Color for reflections
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: sky_r as f64,
                            g: sky_g as f64,
                            b: sky_b as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // 1. Draw Sky
            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

            // 2. Draw Terrain
            opaque_pass.set_pipeline(&self.render_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.indirect_manager.vertex_buffer().slice(..));
            opaque_pass.set_index_buffer(
                self.indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            opaque_pass.multi_draw_indexed_indirect(
                self.indirect_manager.draw_commands(),
                0,
                self.indirect_manager.active_count(),
            );

            // 3. Draw Players
            if self.player_model_num_indices > 0 {
                if let (Some(vb), Some(ib)) = (
                    &self.player_model_vertex_buffer,
                    &self.player_model_index_buffer,
                ) {
                    opaque_pass.set_pipeline(&self.render_pipeline);
                    opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    opaque_pass.set_vertex_buffer(0, vb.slice(..));
                    opaque_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    opaque_pass.draw_indexed(0..self.player_model_num_indices, 0, 0..1);
                }
            }

            // 4. Draw Sun
            opaque_pass.set_pipeline(&self.sun_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

        // Depth Resolve Pass: Resolve MSAA depth to SSR depth texture AND Hi-Z Mip 0
        {
            let mut depth_resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Resolve Pass (SSR + Hi-Z)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hiz_mips[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), // Clear to far depth (1.0)
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.ssr_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Set viewport to screen size to resolve into top-left of the potentially larger Hi-Z texture
            depth_resolve_pass.set_viewport(
                0.0,
                0.0,
                self.config.width as f32,
                self.config.height as f32,
                0.0,
                1.0,
            );

            depth_resolve_pass.set_pipeline(&self.depth_resolve_pipeline);
            depth_resolve_pass.set_bind_group(0, &self.depth_resolve_bind_group, &[]);
            depth_resolve_pass.draw(0..3, 0..1);
        }

        // 6. Generate Hi-Z Pyramid (Depth Pyramid) from just-resolved Mip 0
        // We use separate compute passes for each level to ensure proper synchronization
        // and satisfy WGPU resource usage rules (barrier between write level i and read level i).
        for i in 0..self.hiz_bind_groups.len() {
            let mut hiz_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hi-Z Generation Pass Level"),
                timestamp_writes: None,
            });
            hiz_pass.set_pipeline(&self.hiz_pipeline);
            hiz_pass.set_bind_group(0, &self.hiz_bind_groups[i], &[]);

            // Each mip is half the size of previous
            let div = 1 << (i + 1);
            let mip_width = (self.hiz_size[0] / div).max(1);
            let mip_height = (self.hiz_size[1] / div).max(1);

            hiz_pass.dispatch_workgroups((mip_width + 15) / 16, (mip_height + 15) / 16, 1);
        }

        // Determine final resolve target (screen or SSAO input)
        let resolve_target = if self.ssao_enabled {
            &self.scene_color_view
        } else {
            &view
        };

        // Transparent Pass: Water
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(resolve_target), // Final resolve to screen/SSAO
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw Water
            transparent_pass.set_pipeline(&self.water_pipeline);
            transparent_pass.set_bind_group(0, &self.water_bind_group, &[]);
            transparent_pass
                .set_vertex_buffer(0, self.water_indirect_manager.vertex_buffer().slice(..));
            transparent_pass.set_index_buffer(
                self.water_indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            transparent_pass.multi_draw_indexed_indirect(
                self.water_indirect_manager.draw_commands(),
                0,
                self.water_indirect_manager.active_count(),
            );
        }

        // ============== SSAO POST-PROCESS PASSES ==============
        if self.ssao_enabled {
            // SSAO Pass: Generate ambient occlusion from depth buffer
            {
                let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssao_texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                ssao_pass.set_pipeline(&self.ssao_pipeline);
                ssao_pass.set_bind_group(0, &self.ssao_bind_group, &[]);
                ssao_pass.draw(0..3, 0..1); // Full-screen triangle
            }

            // SSAO Blur Pass: Bilateral blur for smoother results
            {
                let mut blur_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Blur Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssao_blur_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                blur_pass.set_pipeline(&self.ssao_blur_pipeline);
                blur_pass.set_bind_group(0, &self.ssao_blur_bind_group, &[]);
                blur_pass.draw(0..3, 0..1); // Full-screen triangle
            }

            // Composite Pass: Combine scene with SSAO and output to swapchain
            {
                let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Composite Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                composite_pass.set_pipeline(&self.composite_pipeline);
                composite_pass.set_bind_group(0, &self.composite_bind_group, &[]);
                composite_pass.draw(0..3, 0..1); // Full-screen triangle
            }
        }
        // ============== END SSAO POST-PROCESS PASSES ==============

        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            ui_pass.set_pipeline(&self.crosshair_pipeline);
            ui_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
            ui_pass.set_index_buffer(
                self.crosshair_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            ui_pass.draw_indexed(0..self.num_crosshair_indices, 0, 0..1);

            if let (Some(vb), Some(ib)) = (&self.coords_vertex_buffer, &self.coords_index_buffer) {
                if self.coords_num_indices > 0 {
                    ui_pass.set_vertex_buffer(0, vb.slice(..));
                    ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    ui_pass.draw_indexed(0..self.coords_num_indices, 0, 0..1);
                }
            }
        }

        if self.digging.target.is_some() && self.digging.break_time > 0.0 {
            let progress = (self.digging.progress / self.digging.break_time).min(1.0);

            let bar_width = 0.15;
            let bar_height = 0.015;
            let bar_y = -0.05;

            let bg_color = Vertex::pack_color([0.2, 0.2, 0.2]);
            let prog_color = Vertex::pack_color([1.0 - progress, progress, 0.0]);

            let mut vertices = Vec::with_capacity(8);
            let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

            vertices.push(Vertex {
                position: [-bar_width, bar_y - bar_height, 0.0],
                normal,
                color: bg_color,
                uv: [0.0, 0.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [bar_width, bar_y - bar_height, 0.0],
                normal,
                color: bg_color,
                uv: [1.0, 0.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [bar_width, bar_y + bar_height, 0.0],
                normal,
                color: bg_color,
                uv: [1.0, 1.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [-bar_width, bar_y + bar_height, 0.0],
                normal,
                color: bg_color,
                uv: [0.0, 1.0],
                tex_index: 0.0,
            });

            let prog_width = bar_width * 2.0 * progress - bar_width;
            vertices.push(Vertex {
                position: [-bar_width + 0.005, bar_y - bar_height + 0.003, 0.0],
                normal,
                color: prog_color,
                uv: [0.0, 0.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [prog_width - 0.005, bar_y - bar_height + 0.003, 0.0],
                normal,
                color: prog_color,
                uv: [1.0, 0.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [prog_width - 0.005, bar_y + bar_height - 0.003, 0.0],
                normal,
                color: prog_color,
                uv: [1.0, 1.0],
                tex_index: 0.0,
            });
            vertices.push(Vertex {
                position: [-bar_width + 0.005, bar_y + bar_height - 0.003, 0.0],
                normal,
                color: prog_color,
                uv: [0.0, 1.0],
                tex_index: 0.0,
            });

            // Optimization: Reuse buffers if they exist, otherwise create them
            let indices: [u32; 12] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

            if self.progress_bar_vertex_buffer.is_none() {
                self.progress_bar_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar VB"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                self.progress_bar_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar IB"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
            } else {
                // Update existing buffer with new vertex data
                self.queue.write_buffer(
                    self.progress_bar_vertex_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&vertices),
                );
            }

            let progress_vb = self.progress_bar_vertex_buffer.as_ref().unwrap();
            let progress_ib = self.progress_bar_index_buffer.as_ref().unwrap();

            let mut progress_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Progress Bar Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            progress_pass.set_pipeline(&self.crosshair_pipeline);
            progress_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            progress_pass.set_vertex_buffer(0, progress_vb.slice(..));
            progress_pass.set_index_buffer(progress_ib.slice(..), wgpu::IndexFormat::Uint32);
            progress_pass.draw_indexed(0..12, 0, 0..1);
        }

        // Render menu overlay if in menu state
        if self.game_state == GameState::Menu {
            self.render_menu(&mut encoder, &view);
        } else {
            // Render other players
            self.render_remote_players(
                &view_proj,
                self.config.width as f32,
                self.config.height as f32,
            );
        }

        // Final UI and Text rendering
        {
            // 1. Update all buffers first (requires &mut self)
            let fps_text = format!("FPS: {:.0}", self.current_fps);
            self.fps_buffer.set_text(
                &mut self.font_system,
                &fps_text,
                &Attrs::new().family(Family::SansSerif),
                Shaping::Advanced,
                None,
            );
            self.fps_buffer.set_size(
                &mut self.font_system,
                Some(self.config.width as f32),
                Some(self.config.height as f32),
            );

            let labels = if self.game_state == GameState::Menu {
                self.prepare_menu_text();
                Vec::new()
            } else {
                let labels = queue_remote_players_labels(
                    &self.remote_players,
                    &view_proj,
                    self.config.width as f32,
                    self.config.height as f32,
                );

                // Ensure we have enough buffers
                while self.player_label_buffers.len() < labels.len() {
                    self.player_label_buffers.push(glyphon::Buffer::new(
                        &mut self.font_system,
                        Metrics::new(24.0, 32.0),
                    ));
                }

                for (i, label) in labels.iter().enumerate() {
                    let buffer = &mut self.player_label_buffers[i];
                    buffer.set_text(
                        &mut self.font_system,
                        &label.username,
                        &Attrs::new()
                            .family(Family::SansSerif)
                            .color(Color::rgb(76, 255, 76)),
                        Shaping::Advanced,
                        None,
                    );
                    buffer.set_size(
                        &mut self.font_system,
                        Some(self.config.width as f32),
                        Some(self.config.height as f32),
                    );
                }
                labels
            };

            // 2. Now create TextAreas (immutable borrow of self)
            let mut text_areas = Vec::with_capacity(4);

            text_areas.push(TextArea {
                buffer: &self.fps_buffer,
                left: 10.0,
                top: 10.0,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                },
                default_color: Color::rgb(255, 255, 255),
                custom_glyphs: &[],
            });

            if self.game_state == GameState::Menu {
                text_areas.push(TextArea {
                    buffer: &self.menu_buffer,
                    left: 0.0,
                    top: 0.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            } else {
                for (i, label) in labels.iter().enumerate() {
                    text_areas.push(TextArea {
                        buffer: &self.player_label_buffers[i],
                        left: label.screen_x,
                        top: label.screen_y,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: 0,
                            top: 0,
                            right: self.config.width as i32,
                            bottom: self.config.height as i32,
                        },
                        default_color: Color::rgb(255, 255, 255),
                        custom_glyphs: &[],
                    });
                }
            }

            self.text_renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.font_system,
                    &mut self.text_atlas,
                    &self.viewport,
                    text_areas,
                    &mut self.swash_cache,
                )
                .unwrap();

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .unwrap();
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Prepare the menu text buffer
    fn prepare_menu_text(&mut self) {
        let mut text = String::new();

        // Title
        text.push_str("MULTIPLAYER\n\n");

        // Instructions
        text.push_str(
            "Click a field to edit, Tab to switch, Enter to connect, Esc to play solo\n\n\n",
        );

        // Server address
        let addr_selected = self.menu_state.selected_field == MenuField::ServerAddress;
        if addr_selected {
            text.push_str("> ");
        }
        text.push_str(&format!(
            "Server Address: {}\n",
            self.menu_state.server_address
        ));

        // Username
        let user_selected = self.menu_state.selected_field == MenuField::Username;
        if user_selected {
            text.push_str("> ");
        }
        text.push_str(&format!("Username: {}\n\n", self.menu_state.username));

        // Hints
        text.push_str("[ENTER] Connect to Server\n");
        text.push_str("[ESC] Play Singleplayer\n\n");

        // Status/Error
        if let Some(ref err) = self.menu_state.error_message {
            text.push_str(&format!("Error: {}\n", err));
        } else if let Some(ref status) = self.menu_state.status_message {
            text.push_str(&format!("Status: {}\n", status));
        }

        self.menu_buffer.set_text(
            &mut self.font_system,
            &text,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );
    }

    fn render_menu(&mut self, _encoder: &mut wgpu::CommandEncoder, _view: &wgpu::TextureView) {}

    fn connect_to_server(&mut self) {
        connect_to_server(
            &mut self.menu_state,
            &mut self.game_state,
            &self.network_runtime,
            &mut self.network_rx,
            &mut self.network_tx,
        );
    }

    fn update_network(&mut self) {
        update_network(
            &mut self.my_player_id,
            &self.camera.position,
            self.camera.yaw,
            self.camera.pitch,
            &mut self.last_position_send,
            &self.network_tx,
            &mut self.network_rx,
            &mut self.remote_players,
            &mut self.game_state,
            &mut self.mouse_captured,
            &self.window,
        );
    }
    fn render_remote_players(&mut self, _view_proj: &Matrix4<f32>, _width: f32, _height: f32) {
        // Build combined mesh for all remote players
        if !self.remote_players.is_empty() {
            let mut all_vertices = Vec::new();
            let mut all_indices = Vec::new();

            for (_id, player) in &self.remote_players {
                let (vertices, indices) =
                    build_player_model(player.x, player.y, player.z, player.yaw);
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            self.player_model_num_indices = all_indices.len() as u32;

            if !all_vertices.is_empty() {
                self.player_model_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Player Model Vertex Buffer"),
                        contents: bytemuck::cast_slice(&all_vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
                self.player_model_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Player Model Index Buffer"),
                        contents: bytemuck::cast_slice(&all_indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
            }
        } else {
            self.player_model_num_indices = 0;
        }
    }

    fn handle_mouse_input(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.input.left_mouse = pressed,
            MouseButton::Right => self.input.right_mouse = pressed,
            _ => {}
        }

        if !self.mouse_captured {
            return;
        }

        if button == MouseButton::Right && pressed {
            let target = self.camera.raycast(&*self.world.read(), 5.0);
            if let Some((_, _, _, px, py, pz)) = target {
                // Check if the new block would intersect with the local player
                if self.camera.intersects_block(px, py, pz) {
                    return;
                }

                // Check if it would intersect with any remote players
                for player in self.remote_players.values() {
                    let player_pos = cgmath::Point3::new(player.x, player.y, player.z);
                    if render3d::camera::check_intersection(player_pos, px, py, pz) {
                        return;
                    }
                }

                self.world
                    .write()
                    .set_block_player(px, py, pz, BlockType::Stone);
                self.mark_chunk_dirty(px, py, pz);
            }
        }
    }

    fn mark_chunk_dirty(&mut self, x: i32, y: i32, z: i32) {
        let cx = (x as f32 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (z as f32 / CHUNK_SIZE as f32).floor() as i32;
        let sy = y / SUBCHUNK_HEIGHT;

        let mut world = self.world.write();

        if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
            if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                chunk.subchunks[sy as usize].mesh_dirty = true;
            }
        }

        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);
        let ly = y.rem_euclid(SUBCHUNK_HEIGHT);

        if lx == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx - 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lx == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx + 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lz == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz - 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lz == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz + 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if ly == 0 && sy > 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy - 1) as usize].mesh_dirty = true;
            }
        }
        if ly == SUBCHUNK_HEIGHT - 1 && sy < NUM_SUBCHUNKS - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy + 1) as usize].mesh_dirty = true;
            }
        }
    }
}

/// Main game entry point - call this from the actual main() function
pub fn run_game() {
    // Parse CLI arguments
    let args = Args::parse();

    // If server mode, run server and return (headless)
    if args.server {
        let addr = format!("0.0.0.0:{}", args.port);
        tracing::info!("====================================================");
        tracing::info!("Starting Headless Dedicated Server on {}...", addr);
        tracing::info!("Note: This is a console-only server. No game window will appear.");
        tracing::info!("To play the game, run the application without --server.");
        tracing::info!("Press Ctrl+C to stop the server.");
        tracing::info!("====================================================");

        use std::io::Write;
        std::io::stdout().flush().unwrap();

        let rt = tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime. Check if you have enough system resources.");
        rt.block_on(async {
            match TcpServer::bind(&addr).await {
                Ok(server_inst) => {
                    let server = Arc::new(server_inst);
                    println!("Server is now listening on {}", addr);
                    println!("Waiting for connections...");
                    std::io::stdout().flush().unwrap();

                    // Accept connections in a loop
                    loop {
                        match server.accept().await {
                            Ok((id, conn)) => {
                                println!("Client {} connected from {}", id, conn.addr());
                                let server_clone = server.clone();

                                // Handle client in separate task
                                tokio::spawn(async move {
                                    loop {
                                        match conn.recv().await {
                                            Ok(mut packet) => {
                                                // Ensure player_id matches the one assigned by server for all relevant packets
                                                match packet {
                                                    Packet::Connect {
                                                        ref mut player_id, ..
                                                    } => {
                                                        *player_id = id;
                                                        // Send acknowledgement back to client
                                                        let ack = Packet::ConnectAck {
                                                            success: true,
                                                            player_id: id,
                                                        };
                                                        let _ = conn.send(&ack).await;
                                                    }
                                                    Packet::Position {
                                                        ref mut player_id, ..
                                                    } => {
                                                        *player_id = id;
                                                    }
                                                    Packet::Rotation {
                                                        ref mut player_id, ..
                                                    } => {
                                                        *player_id = id;
                                                    }
                                                    Packet::Chat {
                                                        ref mut player_id, ..
                                                    } => {
                                                        *player_id = id;
                                                    }
                                                    Packet::Disconnect {
                                                        ref mut player_id,
                                                        ..
                                                    } => {
                                                        *player_id = id;
                                                    }
                                                    _ => {}
                                                }

                                                // Broadcast to everyone else
                                                let _ = server_clone
                                                    .broadcast_except(&packet, id)
                                                    .await;
                                            }
                                            Err(_) => {
                                                println!("Client {} disconnected", id);

                                                // Inform others about disconnection
                                                let disconnect_packet =
                                                    Packet::Disconnect { player_id: id };
                                                let _ = server_clone
                                                    .broadcast_except(&disconnect_packet, id)
                                                    .await;

                                                // Clean up connection from server
                                                server_clone.remove_client(id).await;
                                                break;
                                            }
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                eprintln!("Accept error: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to start server: {}", e);
                }
            }
        });
        return;
    }

    // Client mode
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Mini Minecraft 256x256 | Loading...")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut state = pollster::block_on(State::new(window));

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    state.resize(size);
                    state.window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    state.frame_count += 1;
                    let now = Instant::now();
                    let elapsed = now.duration_since(state.last_fps_update).as_secs_f32();

                    if elapsed >= 0.5 {
                        state.current_fps = state.frame_count as f32 / elapsed;
                        state.frame_count = 0;
                        state.last_fps_update = now;
                    }

                    state.update();

                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size()),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("Render error: {:?}", e),
                    }

                    state.window.request_redraw();
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    physical_key: PhysicalKey::Code(key),
                                    state: key_state,
                                    text,
                                    ..
                                },
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = key_state == ElementState::Pressed;

                    // Handle text input for menu
                    if state.game_state == GameState::Menu && pressed {
                        if let Some(ref txt) = text {
                            for ch in txt.chars() {
                                state.menu_state.handle_char(ch);
                            }
                        }
                    }

                    // Handle based on game state
                    if state.game_state == GameState::Menu {
                        // Menu mode - handle text input
                        if pressed {
                            match key {
                                KeyCode::Tab => {
                                    state.menu_state.next_field();
                                }
                                KeyCode::Enter => {
                                    // Try to connect using menu data
                                    state.connect_to_server();
                                }
                                KeyCode::Escape => {
                                    // Play singleplayer - exit menu
                                    state.game_state = GameState::Playing;
                                }
                                KeyCode::Backspace => {
                                    state.menu_state.handle_backspace();
                                }
                                KeyCode::F11 => {
                                    if state.window.fullscreen().is_some() {
                                        state.window.set_fullscreen(None);
                                    } else {
                                        state.window.set_fullscreen(Some(
                                            winit::window::Fullscreen::Borderless(None),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                        // Playing mode - normal game controls
                        match key {
                            KeyCode::KeyW => state.input.forward = pressed,
                            KeyCode::KeyS => state.input.backward = pressed,
                            KeyCode::KeyA => state.input.left = pressed,
                            KeyCode::KeyD => state.input.right = pressed,
                            KeyCode::Space => state.input.jump = pressed,
                            KeyCode::ShiftLeft => state.input.sprint = pressed,
                            KeyCode::Escape if pressed => {
                                // Return to menu OR release mouse
                                if state.mouse_captured {
                                    state.mouse_captured = false;
                                    let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                                    state.window.set_cursor_visible(true);
                                } else {
                                    state.game_state = GameState::Menu;
                                }
                            }
                            KeyCode::F11 if pressed => {
                                if state.window.fullscreen().is_some() {
                                    state.window.set_fullscreen(None);
                                } else {
                                    state.window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                }
                            }
                            KeyCode::KeyR if pressed => {
                                // Toggle reflection mode: 0=Off, 1=SSR
                                state.reflection_mode = (state.reflection_mode + 1) % 2;
                                let mode_name = match state.reflection_mode {
                                    0 => "Off",
                                    1 => "SSR",
                                    _ => "Unknown",
                                };
                                println!("Reflection mode: {}", mode_name);
                            }
                            KeyCode::KeyO if pressed => {
                                // Toggle SSAO
                                state.ssao_enabled = !state.ssao_enabled;
                                println!("SSAO: {}", if state.ssao_enabled { "ON" } else { "OFF" });
                            }
                            KeyCode::F5 if pressed => {
                                let world = state.world.read();
                                let saved = SavedWorld::from_world(
                                    &world.chunks,
                                    world.seed,
                                    (
                                        state.camera.position.x,
                                        state.camera.position.y,
                                        state.camera.position.z,
                                    ),
                                    (state.camera.yaw, state.camera.pitch),
                                );
                                if let Err(e) = save_world(DEFAULT_WORLD_FILE, &saved) {
                                    eprintln!("Failed to save world: {}", e);
                                } else {
                                    println!("World saved to {}", DEFAULT_WORLD_FILE);
                                }
                            }
                            KeyCode::F9 if pressed => match load_world(DEFAULT_WORLD_FILE) {
                                Ok(saved) => {
                                    println!("Regenerating world with seed {}...", saved.seed);
                                    {
                                        let mut world = state.world.write();
                                        *world = World::new_with_seed(saved.seed);
                                    }
                                    state.camera.position.x = saved.player_x;
                                    state.camera.position.y = saved.player_y;
                                    state.camera.position.z = saved.player_z;
                                    state.camera.yaw = saved.player_yaw;
                                    state.camera.pitch = saved.player_pitch;

                                    {
                                        let mut world = state.world.write();
                                        for chunk_data in &saved.chunks {
                                            let cx = chunk_data.cx;
                                            let cz = chunk_data.cz;

                                            for (&sy, block_data) in &chunk_data.subchunks {
                                                if let Some(chunk) = world.chunks.get_mut(&(cx, cz))
                                                {
                                                    if (sy as usize) < chunk.subchunks.len() {
                                                        let subchunk =
                                                            &mut chunk.subchunks[sy as usize];
                                                        let mut n = 0;
                                                        for lx in 0..CHUNK_SIZE as usize {
                                                            for ly in 0..SUBCHUNK_HEIGHT as usize {
                                                                for lz in 0..CHUNK_SIZE as usize {
                                                                    if n < block_data.len() {
                                                                        subchunk.blocks[lx][ly]
                                                                            [lz] = block_data[n];
                                                                        n += 1;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        subchunk.is_empty = false;
                                                        subchunk.mesh_dirty = true;
                                                    }
                                                    chunk.player_modified = true;
                                                }
                                            }
                                        }
                                    }
                                    // Mark all neighbor chunks as dirty to ensure geometry joins correctly
                                    {
                                        let mut world = state.world.write();
                                        for chunk in world.chunks.values_mut() {
                                            for subchunk in &mut chunk.subchunks {
                                                subchunk.mesh_dirty = true;
                                            }
                                        }
                                    }

                                    println!(
                                        "World loaded from {} (seed: {})",
                                        DEFAULT_WORLD_FILE, saved.seed
                                    );
                                }
                                Err(e) => println!("Error loading: {}", e),
                            },
                            _ => {}
                        }
                    }
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            state: btn_state,
                            button,
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = btn_state == ElementState::Pressed;

                    if pressed && !state.mouse_captured {
                        state.mouse_captured = true;
                        let _ = state
                            .window
                            .set_cursor_grab(CursorGrabMode::Confined)
                            .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                        state.window.set_cursor_visible(false);
                        let _ = state.window.set_cursor_position(PhysicalPosition::new(
                            state.config.width / 2,
                            state.config.height / 2,
                        ));
                    } else {
                        state.handle_mouse_input(button, pressed);
                    }
                }
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    if state.mouse_captured {
                        let sensitivity = 0.002;
                        state.camera.yaw += delta.0 as f32 * sensitivity;
                        state.camera.pitch -= delta.1 as f32 * sensitivity;
                        state.camera.pitch = state.camera.pitch.clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.1,
                            std::f32::consts::FRAC_PI_2 - 0.1,
                        );
                    }
                }
                Event::AboutToWait => {
                    let is_idle = state.last_input_time.elapsed().as_secs() >= 30;
                    if is_idle {
                        // Limit to ~30 FPS (33.3ms per frame)
                        let next_frame = Instant::now() + std::time::Duration::from_millis(33);
                        elwt.set_control_flow(ControlFlow::WaitUntil(next_frame));
                    } else {
                        elwt.set_control_flow(ControlFlow::Poll);
                    }
                    state.window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => elwt.exit(),
                _ => {}
            }
        })
        .unwrap();
}
