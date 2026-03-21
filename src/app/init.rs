use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use bytemuck;
use cgmath::Matrix4;
use glyphon::{
    Cache, FontSystem, Metrics, Resolution, SwashCache, TextAtlas, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::app::texture_cache;
use crate::ui::menu::{GameState, MenuState};
use minerust::chunk_loader::ChunkLoader;
use minerust::{
    Camera, DiggingState, IndirectManager, InputState, SEA_LEVEL, Uniforms, Vertex, World,
    build_crosshair,
};

use super::state::State;

/// Converts an OpenGL-style clip-space matrix to wgpu's NDC convention.
///
/// wgpu (like Metal and DirectX) uses a depth range of [0, 1] in NDC,
/// whereas OpenGL uses [-1, 1]. This matrix remaps the Z axis accordingly.
/// It should be applied **after** the projection matrix when computing the
/// final `view_proj` uniform that is uploaded to the GPU.
///
/// ```text
/// depth_wgpu = depth_gl * 0.5 + 0.5
/// ```
#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// Converts an array of six frustum planes from `cgmath::Vector4<f32>` into
/// a plain `[[f32; 4]; 6]` that can be sent directly to a GPU buffer.
///
/// # Safety
/// `cgmath::Vector4<f32>` has the same memory layout as `[f32; 4]`
/// (four tightly-packed 32-bit floats), so the `transmute` is sound.
///
/// # Parameters
/// - `planes` – Six frustum planes (left, right, top, bottom, near, far) in
///   world space, each encoded as `(nx, ny, nz, d)` where `nx·x + ny·y + nz·z + d = 0`.
///
/// # Returns
/// The same data as a raw `[[f32; 4]; 6]` array ready for `bytemuck::cast_slice`.
#[inline(always)]
pub fn frustum_planes_to_array(planes: &[cgmath::Vector4<f32>; 6]) -> [[f32; 4]; 6] {
    unsafe { std::mem::transmute(*planes) }
}

impl State {
    /// Initializes the complete rendering state for the application.
    ///
    /// This is a large, one-shot async constructor that performs every wgpu
    /// setup step in sequence:
    ///
    /// 1. **Surface & adapter selection** – creates the OS window surface,
    ///    picks the highest-performance GPU adapter, and logs its name and backend.
    /// 2. **Device & queue** – requests a logical device, enabling
    ///    `MULTI_DRAW_INDIRECT_COUNT` when the adapter supports it so the
    ///    indirect draw manager can cull invisible chunks on the GPU.
    /// 3. **Swap-chain configuration** – prefers an sRGB surface format and
    ///    `PresentMode::Immediate` (uncapped frame rate) with 4× MSAA.
    /// 4. **Shader compilation** – compiles all WGSL shaders (terrain, water,
    ///    shadow, sky, sun, UI, Hi-Z, depth-resolve, composite).
    /// 5. **Buffers & textures** – allocates the uniform buffer, shadow map
    ///    cascade array, SSR color/depth targets, MSAA resolve targets, and
    ///    the hierarchical-Z (Hi-Z) mip chain.
    /// 6. **Bind group layouts & bind groups** – wires textures, samplers, and
    ///    buffers to the correct shader bindings for each pipeline.
    /// 7. **Render pipelines** – builds one `RenderPipeline` per pass:
    ///    terrain, water (alpha-blended), crosshair UI, shadow depth, sun
    ///    billboard, sky dome, depth-resolve, and the final composite blit.
    /// 8. **Compute pipelines** – builds the Hi-Z downsampling compute pipeline
    ///    with one bind group per adjacent mip level pair.
    /// 9. **World & camera** – constructs the voxel `World`, finds a safe spawn
    ///    point, and positions the `Camera` there.
    /// 10. **Text rendering** – initialises `glyphon` with a bundled Google Sans
    ///     font and pre-allocates `Buffer` objects for every piece of on-screen
    ///     text (FPS counter, menu labels, hotbar slot name, etc.).
    /// 11. **Indirect draw managers** – creates `IndirectManager` instances for
    ///     opaque terrain and water, and wires them to the Hi-Z texture so GPU
    ///     occlusion culling works correctly.
    ///
    /// # Panics
    /// Panics if:
    /// - No compatible GPU adapter is found.
    /// - The logical device cannot be created.
    /// - The window surface cannot be created.
    /// - The Tokio runtime for networking cannot be created.
    pub async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        // ------------------------------------------------------------------ //
        // Instance & surface
        // ------------------------------------------------------------------ //

        // Use all available backends; on Windows, wgpu will prefer DX12 over
        // Vulkan because DX12 typically yields better frame times there.
        let backend = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backend,
            ..Default::default()
        });

        // The surface must be created before adapter selection so that wgpu
        // can guarantee the chosen adapter can present to this window.
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        // ------------------------------------------------------------------ //
        // Adapter selection
        // ------------------------------------------------------------------ //

        // Request the highest-performance (discrete) GPU.  If two adapters are
        // equally capable, wgpu falls back to its own scoring heuristic.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let info = adapter.get_info();
        tracing::info!(
            "Selected adapter: {} on {:?} backend",
            info.name,
            info.backend
        );

        // ------------------------------------------------------------------ //
        // Feature negotiation
        // ------------------------------------------------------------------ //

        // `MULTI_DRAW_INDIRECT_COUNT` allows the GPU to determine at draw time
        // how many indirect draw calls to execute (i.e. the count itself lives
        // in a GPU buffer rather than being supplied by the CPU).  This enables
        // fully GPU-side occlusion culling: chunks that fail the Hi-Z test are
        // simply never emitted into the draw list.
        let adapter_features = adapter.features();
        let supports_indirect_count =
            adapter_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let requested_features = if supports_indirect_count {
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        } else {
            wgpu::Features::empty()
        };

        // ------------------------------------------------------------------ //
        // Logical device & queue
        // ------------------------------------------------------------------ //

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: requested_features,
                // Inherit adapter limits so we can use maximum buffer sizes,
                // bind group counts, etc. that the hardware exposes.
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create GPU device");

        // ------------------------------------------------------------------ //
        // Swap-chain (surface) configuration
        // ------------------------------------------------------------------ //

        let surface_caps = surface.get_capabilities(&adapter);
        // Prefer sRGB so that texture colors appear physically correct;
        // fall back to whatever the surface offers if sRGB is unavailable.
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
            // `Immediate` disables vsync so the frame rate is uncapped.
            // Switch to `Fifo` (vsync) to reduce GPU power consumption.
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ------------------------------------------------------------------ //
        // MSAA & depth textures
        // ------------------------------------------------------------------ //

        // 4× MSAA reduces aliasing on geometry edges with a reasonable
        // memory/bandwidth cost.  All color render passes write to the MSAA
        // texture; it is resolved to the swap-chain image at the end of each
        // frame.
        let msaa_sample_count: u32 = 4;

        // A multisampled Depth32Float texture is used for all geometry passes
        // (terrain, water, sun, sky).  A separate single-sampled depth texture
        // is used for SSR so that the water shader can sample the opaque scene
        // depth at full precision.
        let depth_texture = Self::create_depth_texture(&device, &config, msaa_sample_count);
        let msaa_texture_view =
            Self::create_msaa_texture(&device, &config, surface_format, msaa_sample_count);

        // ------------------------------------------------------------------ //
        // Shader compilation
        // ------------------------------------------------------------------ //

        // All shaders are embedded at compile time via `include_str!` so no
        // file-system access is required at runtime.

        /// Compiles a WGSL shader from a string literal embedded in the binary.
        // (macro-like helper used only inside this function for brevity)

        let hiz_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hi-Z Shader"),
            // Downsamples the depth buffer into a mip chain for GPU occlusion culling.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz.wgsl").into()),
        });
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            // Main opaque geometry pass: texture atlas lookup, CSM shadow
            // comparison, and per-vertex AO.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terrain.wgsl").into()),
        });
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            // Translucent water pass: SSR reflection, refraction, foam edge
            // detection, Fresnel blend.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water.wgsl").into()),
        });
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            // Renders flat UI geometry (crosshair, hotbar) in screen space with
            // alpha blending; no depth test.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
        });
        let sun_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sun Shader"),
            // Renders the sun / moon disc billboard oriented toward the camera.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sun.wgsl").into()),
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            // Depth-only pass that writes each CSM cascade's shadow map.
            // Fragment stage is omitted entirely for maximum throughput.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            // Procedural sky dome rendered at the far plane; uses
            // `LessEqual` depth compare so it appears behind all geometry.
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });

        // ------------------------------------------------------------------ //
        // Uniform buffer
        // ------------------------------------------------------------------ //

        // A single `Uniforms` struct is uploaded once per frame and shared by
        // all shader stages.  The initial values are placeholder identity/zero
        // matrices; they are overwritten before the first draw call.
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                view_proj: Matrix4::from_scale(1.0).into(),
                inv_view_proj: Matrix4::from_scale(1.0).into(),
                // `csm_view_proj` holds four 4×4 matrices – one per cascade.
                csm_view_proj: [Matrix4::from_scale(1.0).into(); 4],
                // Split distances (world-space) for the four CSM cascades.
                // Tune these to balance shadow resolution vs. coverage range.
                csm_split_distances: [16.0, 48.0, 128.0, 300.0],
                camera_pos: [0.0, 0.0, 0.0],
                time: 0.0,
                sun_position: [0.4, -0.2, 0.3],
                is_underwater: 0.0,
                screen_size: [1920.0, 1080.0],
                // Y coordinate (in world blocks) of the water surface.
                water_level: SEA_LEVEL as f32 - 1.0,
                // 1.0 = SSR enabled, 0.0 = flat reflection fallback.
                reflection_mode: 1.0,
                moon_position: [-0.4, 0.2, -0.3],
                _pad1_moon: 0.0,
                moon_intensity: 0.0,
                wind_dir: [0.8, 0.6],
                wind_speed: 1.0,
                _pad: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ------------------------------------------------------------------ //
        // Texture atlas
        // ------------------------------------------------------------------ //

        // The texture atlas packs all block textures into a single 2D array
        // texture.  It is either loaded from a disk cache or generated from the
        // raw asset images on first run.
        let (texture_atlas, texture_view, _atlas_width, _atlas_height) =
            texture_cache::load_or_generate_atlas(&device, &queue);

        // Anisotropic filtering (16×) significantly reduces blurring on
        // steeply-angled surfaces like cliff faces.
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        // ------------------------------------------------------------------ //
        // Shadow map (Cascaded Shadow Maps – CSM)
        // ------------------------------------------------------------------ //

        // A 2 K × 2 K Depth32Float texture array with 4 layers, one per
        // cascade.  Increasing `shadow_map_size` improves shadow sharpness at
        // the cost of VRAM and shadow-pass render time.
        let shadow_map_size = 2048;
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: shadow_map_size,
                height: shadow_map_size,
                depth_or_array_layers: 4, // one layer per CSM cascade
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // `D2Array` view used by the terrain fragment shader to sample all
        // four cascades in a single `textureSampleCompareLevel` call.
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow Map Array View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Individual `D2` views, one per cascade, used as render targets in
        // the shadow pass (wgpu render attachments cannot target array layers
        // through an array view).
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

        // Dynamic-offset uniform buffer that stores the per-cascade light-space
        // view-projection matrix.  Using a dynamic offset means we can switch
        // cascades by simply changing the bind-group offset rather than
        // rebinding a different buffer.
        let shadow_cascade_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow Cascade Buffer"),
            // 256 bytes × 4 cascades; 256-byte alignment satisfies the
            // `min_uniform_buffer_offset_alignment` requirement (typically 256 B).
            size: 256 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Comparison sampler used in the terrain shader for hardware PCF
        // (percentage closer filtering).  `LessEqual` matches the convention
        // that shadow depth is stored as the distance from the light.
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

        // ------------------------------------------------------------------ //
        // Bind group layouts
        // ------------------------------------------------------------------ //

        // Layout shared by the terrain, sky, and sun pipelines.
        // Bindings:
        //   0 – Uniforms (vertex + fragment)
        //   1 – Texture atlas array (fragment)
        //   2 – Atlas sampler (fragment)
        //   3 – Shadow map array (fragment, depth texture for comparison)
        //   4 – Shadow comparison sampler (fragment)
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

        // Layout for the shadow depth pass.
        // Binding 0 uses a **dynamic offset** so the same bind group can be
        // reused for all four cascades; only the offset changes between draws.
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

        // ------------------------------------------------------------------ //
        // SSR (Screen-Space Reflections) targets
        // ------------------------------------------------------------------ //

        // The terrain pass renders into these textures first.  The water
        // shader then samples them to produce planar reflections of the scene
        // above the water surface.
        let ssr_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, // SSR targets are single-sampled (no MSAA)
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

        // Nearest-neighbor sampler for SSR lookups; bilinear filtering would
        // blur the reflected image and cause incorrect depth comparisons.
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

        // Neutral flow map so the shader can enable distortion without an
        // extra asset dependency.
        let flow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Flow Map Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
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
                texture: &flow_map_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[128, 128, 128, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let flow_map_view = flow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let flow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Flow Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // ------------------------------------------------------------------ //
        // Water bind group layout & bind group
        // ------------------------------------------------------------------ //

        // Extends the terrain layout with SSR and flow-map bindings:
        //   5 – SSR color texture (fragment)
        //   6 – SSR depth texture  (fragment)
        //   7 – SSR sampler        (fragment)
        //   8 – flow map texture   (fragment)
        //   9 – flow sampler       (fragment)
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
                    // SSR color – the opaque scene rendered before the water pass.
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
                    // SSR depth – used to detect where the reflection ray intersects
                    // the opaque scene for refraction and ray termination.
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
                    // SSR sampler – nearest neighbor for correct texel fetches.
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
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
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&flow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&flow_sampler),
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

        // Bind the shadow cascade buffer at offset 0 (range = 80 bytes, which
        // covers one 4×4 f32 matrix = 64 bytes + padding).  At draw time the
        // dynamic offset selects which cascade's matrix to use.
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shadow_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shadow_cascade_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(80),
                }),
            }],
            label: Some("shadow_bind_group"),
        });

        // ------------------------------------------------------------------ //
        // Pipeline layouts
        // ------------------------------------------------------------------ //

        // Terrain / sky / sun / crosshair all share the same uniform layout.
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

        // ------------------------------------------------------------------ //
        // Render pipelines
        // ------------------------------------------------------------------ //

        // --- Terrain (opaque geometry) ---
        // Back-face culled, depth write enabled, 4× MSAA.
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

        // --- Water (translucent, alpha-blended) ---
        // No back-face culling so water surfaces are visible from below.
        // Depth writes are disabled: water contributes to color but must not
        // occlude geometry drawn in later transparent passes.
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
                cull_mode: None, // visible from both sides
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // read-only depth test
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

        // --- Crosshair / UI ---
        // No depth test at all so the crosshair always draws on top.
        // Sample count is 1 because the crosshair is drawn after MSAA resolve
        // (directly onto the swap-chain image).
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
            depth_stencil: None, // no depth test for UI
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        // --- Shadow depth pass ---
        // Fragment shader is intentionally omitted; we only need the depth
        // values written by the vertex stage.  A depth bias is applied to
        // combat shadow acne (self-shadowing artifacts on angled surfaces).
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
            fragment: None, // depth-only – no color output
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
                bias: wgpu::DepthBiasState {
                    // `constant` and `slope_scale` push shadow-map depth values
                    // slightly away from the camera to avoid self-shadowing.
                    // Tune these if you see shadow acne or peter-panning.
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        // --- Sun / Moon billboard ---
        // Rendered as a quad in world space; alpha-blended and depth-tested
        // (but no depth write) so the disc clips correctly behind terrain.
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

        // --- Sky dome ---
        // Uses `LessEqual` so it renders at depth = 1.0 (the far plane) and
        // appears behind every piece of geometry.  No depth writes so it does
        // not occlude anything.
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
                // `LessEqual` rather than `Less` because the sky sits at
                // exactly depth 1.0 and we want it to pass rather than fail.
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

        // ------------------------------------------------------------------ //
        // Sun billboard geometry
        // ------------------------------------------------------------------ //

        // A single unit quad in local space; the vertex shader billboards it
        // toward the camera and translates it to the sun/moon direction.
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

        // ------------------------------------------------------------------ //
        // World, camera, chunk loader
        // ------------------------------------------------------------------ //

        tracing::info!("Generating world...");
        let world = Arc::new(parking_lot::RwLock::new(World::new()));

        // `find_spawn_point` searches downward from a candidate column until
        // it finds a non-air block, ensuring the player spawns on solid ground.
        let spawn = world.read().find_spawn_point();
        let camera = Camera::new(spawn);
        tracing::info!("World generated! Spawn: {:?}", spawn);

        let seed = world.read().seed;
        // `ChunkLoader` generates chunk data (terrain noise, biomes, structures)
        // on background threads.  It is seeded from the world so that chunk
        // generation is deterministic and seamlessly continuous across sessions.
        let chunk_loader = ChunkLoader::new(seed);

        // `MeshLoader` converts raw chunk block data into GPU vertex/index
        // buffers.  It runs on a pool of worker threads whose count is chosen
        // by `get_mesh_worker_count` (typically `num_cpus - 1`).
        let mesh_loader =
            minerust::MeshLoader::new(Arc::clone(&world), minerust::get_mesh_worker_count());

        // ------------------------------------------------------------------ //
        // Crosshair geometry
        // ------------------------------------------------------------------ //

        // `build_crosshair` returns a pre-built vertex/index list for a small
        // plus-sign rendered at the screen center in NDC coordinates.
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

        // ------------------------------------------------------------------ //
        // Text rendering (glyphon)
        // ------------------------------------------------------------------ //

        // `glyphon` is a GPU text renderer built on top of `cosmic-text` for
        // shaping and `swash` for rasterization.  Each `glyphon::Buffer` holds
        // shaped text for one UI element and is re-set whenever the displayed
        // string changes.

        let mut font_system = FontSystem::new();
        // Load the bundled Google Sans font so that text looks consistent
        // across all platforms regardless of system fonts installed.
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

        // Pre-allocate one `glyphon::Buffer` per on-screen text element.
        // The `Metrics` (font size, line height) are set here and do not change
        // at runtime; only the text content is updated each frame.

        /// FPS counter displayed in the top-left corner.
        let fps_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(40.0, 48.0));

        // --- Main-menu text buffers ---
        let menu_title_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(44.0, 52.0));
        let menu_subtitle_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(22.0, 30.0));
        let menu_server_label_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(18.0, 24.0));
        let menu_server_value_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(24.0, 32.0));
        let menu_username_label_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(18.0, 24.0));
        let menu_username_value_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(24.0, 32.0));
        /// Random tip shown in the menu footer.
        let menu_tips_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(18.0, 24.0));
        let menu_connect_button_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(20.0, 28.0));
        let menu_singleplayer_button_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(20.0, 28.0));
        /// Connection status / error message shown below the buttons.
        let menu_status_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(18.0, 24.0));

        // Hotbar slot name (e.g., "Stone Sword") displayed above the hotbar.
        let hotbar_label_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(22.0, 28.0));

        // ------------------------------------------------------------------ //
        // Depth-resolve pipeline
        // ------------------------------------------------------------------ //

        // After all MSAA geometry passes we need a single-sampled depth texture
        // for the SSR depth lookup.  The depth-resolve pipeline reads from the
        // multisampled depth buffer (sample 0) and writes to the SSR depth
        // target in a full-screen triangle pass.

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
                        multisampled: true, // must match the MSAA depth texture
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
                    // No vertex buffer – the vertex shader generates a full-screen
                    // triangle from `gl_VertexIndex` alone (clip-space trick).
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &depth_resolve_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    // Output is R32Float so the water shader can fetch the raw
                    // floating-point depth value without a comparison sampler.
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
                    // `Always` so every pixel is written regardless of its
                    // depth value (we want an exact copy, not a cull).
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(), // single-sampled output
                multiview_mask: None,
            });

        // ------------------------------------------------------------------ //
        // Composite pipeline (post-processing blit)
        // ------------------------------------------------------------------ //

        // After all scene passes have written to `scene_color_texture`, the
        // composite pass applies any post-processing (e.g., underwater fog,
        // vignette) and blits the result to the swap-chain surface.
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        // Intermediate single-sampled color target.  All MSAA-resolved
        // geometry ends up here before the final composite blit.
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

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    // Uniforms for the composite shader (camera, time, underwater flag, etc.)
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
                    // The resolved scene color to be composited onto the swap-chain.
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
                    resource: wgpu::BindingResource::Sampler(&composite_sampler),
                },
            ],
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                immediate_size: 0,
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[], // full-screen triangle from vertex index
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, // opaque blit – no blending needed
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // no depth test for the final blit
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        // ------------------------------------------------------------------ //
        // Indirect draw managers
        // ------------------------------------------------------------------ //

        // `IndirectManager` maintains GPU-side indirect draw argument buffers
        // and a compute shader that populates them after the Hi-Z occlusion
        // cull step.  One manager for opaque terrain, one for water.
        let mut indirect_manager = IndirectManager::new(&device);
        let mut water_indirect_manager = IndirectManager::new(&device);
        // Initialize the per-cascade shadow draw argument buffers.
        indirect_manager.init_shadow_resources(&device);
        water_indirect_manager.init_shadow_resources(&device);

        // ------------------------------------------------------------------ //
        // Hierarchical-Z (Hi-Z) occlusion buffer
        // ------------------------------------------------------------------ //

        // The Hi-Z buffer is a full mip-chain of R32Float textures that
        // approximates the scene depth at progressively coarser resolutions.
        // The occlusion-cull compute shader compares each chunk's AABB against
        // the nearest (finest) mip level that covers the AABB's projected
        // screen extent, rejecting chunks whose farthest depth sample is
        // shallower than the depth at that mip level.

        let hiz_size = [config.width, config.height];
        let hiz_max_dim = config.width.max(config.height);
        // Number of mip levels needed to downsample to 1×1.
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
            usage: wgpu::TextureUsages::STORAGE_BINDING   // written by compute
                | wgpu::TextureUsages::TEXTURE_BINDING    // read by compute & cull
                | wgpu::TextureUsages::RENDER_ATTACHMENT, // mip 0 written by depth-resolve
            view_formats: &[],
        });

        // Full-mip view used when the cull shader needs to sample any level.
        let hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Individual single-mip views used as the source/destination pair in
        // each Hi-Z downsampling dispatch.
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

        // Hi-Z compute bind group layout.
        // Binding 0 – read from mip N (Texture2D<f32>, non-filterable)
        // Binding 1 – write to mip N+1 (StorageTexture WriteOnly R32Float)
        let hiz_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Hi-Z Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            // Non-filterable because we take the maximum of a 2×2
                            // region manually in the shader (conservative depth).
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

        // One bind group per adjacent mip pair (N → N+1).  At runtime we
        // dispatch the Hi-Z pipeline once per bind group in descending-resolution
        // order to build the full mip chain.
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

        // Give both indirect managers access to the Hi-Z texture so the GPU
        // cull shader can sample it during the indirect dispatch.
        indirect_manager.update_bind_group(&device, &hiz_view);
        water_indirect_manager.update_bind_group(&device, &hiz_view);

        // ------------------------------------------------------------------ //
        // Assemble and return State
        // ------------------------------------------------------------------ //

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
            frame_time_ms: 0.0,
            cpu_update_ms: 0.0,
            last_redraw: Instant::now(),
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
            reflection_mode: 1,
            is_underwater: 0.0,
            remote_players: HashMap::new(),
            my_player_id: 0,
            last_position_send: Instant::now(),
            network_runtime: Some(
                tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"),
            ),
            network_rx: None,
            network_tx: None,
            last_input_time: Instant::now(),
            player_model_vertex_buffer: None,
            player_model_index_buffer: None,
            player_model_num_indices: 0,
            player_model_vertex_capacity: 0,
            player_model_index_capacity: 0,
            chunk_loader,
            last_gen_player_cx: i32::MIN,
            last_gen_player_cz: i32::MIN,
            ssr_color_texture,
            ssr_color_view,
            ssr_depth_texture,
            ssr_depth_view,
            ssr_sampler,
            flow_map_texture,
            flow_map_view,
            flow_sampler,
            water_bind_group,
            water_bind_group_layout,
            surface_format,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            fps_buffer,
            menu_title_buffer,
            menu_subtitle_buffer,
            menu_server_label_buffer,
            menu_server_value_buffer,
            menu_username_label_buffer,
            menu_username_value_buffer,
            menu_tips_buffer,
            menu_connect_button_buffer,
            menu_singleplayer_button_buffer,
            menu_status_buffer,
            hotbar_label_buffer,
            hotbar_label_width: 0.0,
            last_hotbar_slot: usize::MAX,
            player_label_buffers: Vec::new(),
            composite_pipeline,
            composite_bind_group,
            scene_color_texture,
            scene_color_view,
            indirect_manager,
            water_indirect_manager,
            hiz_texture,
            hiz_view,
            hiz_mips,
            hiz_pipeline,
            hiz_bind_groups,
            hiz_bind_group_layout,
            hiz_size,
            depth_resolve_pipeline,
            depth_resolve_bind_group,
            supports_indirect_count,
            csm: minerust::render_core::csm::CsmManager::new(),
            hotbar_slot: 0,
            hotbar_vertex_buffer: None,
            hotbar_index_buffer: None,
            hotbar_num_indices: 0,
            hotbar_dirty: true,
            cursor_position: None,
        }
    }

    /// Creates a (possibly multisampled) depth texture and returns a view into it.
    ///
    /// The texture uses `Depth32Float` for maximum precision, which is
    /// required for the Hi-Z chain (which stores raw floating-point depth
    /// values rather than normalized integers).
    ///
    /// # Parameters
    /// - `device`       – Active wgpu logical device.
    /// - `config`       – Current surface configuration; width/height are read
    ///                    from here so the depth texture always matches the
    ///                    swap-chain resolution.
    /// - `sample_count` – Number of MSAA samples.  Pass `1` for a
    ///                    single-sampled texture (e.g., SSR targets) or `4`
    ///                    for the main multisampled depth buffer.
    ///
    /// # Returns
    /// A `TextureView` wrapping the newly created depth texture.
    pub fn create_depth_texture(
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
            // `TEXTURE_BINDING` is needed so the depth-resolve shader can read
            // the multisampled depth as a texture.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Creates a multisampled color texture used as the MSAA render target.
    ///
    /// All geometry passes render into this texture.  At the end of each frame
    /// it is resolved to the single-sampled `scene_color_texture` (and
    /// ultimately to the swap-chain surface) by the wgpu resolve attachment
    /// mechanism.
    ///
    /// # Parameters
    /// - `device`       – Active wgpu logical device.
    /// - `config`       – Current surface configuration.
    /// - `format`       – Surface pixel format (sRGB if available).
    /// - `sample_count` – Number of MSAA samples (typically 4).
    ///
    /// # Returns
    /// A `TextureView` wrapping the newly created MSAA colour texture.
    pub fn create_msaa_texture(
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
            // Only used as a render attachment; the resolved result is written
            // elsewhere so `TEXTURE_BINDING` is not needed here.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}
