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
use render3d::chunk_loader::ChunkLoader;
use render3d::{
    build_crosshair, Camera, DiggingState, IndirectManager, InputState, Uniforms, Vertex, World,
};

use super::state::State;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// Convert cgmath Vector4 frustum planes to [[f32; 4]; 6] for GPU culling shader.
/// SAFETY: cgmath::Vector4<f32> is #[repr(C)] with layout identical to [f32; 4].
#[inline(always)]
pub fn frustum_planes_to_array(planes: &[cgmath::Vector4<f32>; 6]) -> [[f32; 4]; 6] {
    unsafe { std::mem::transmute(*planes) }
}

impl State {
    pub async fn new(window: Window) -> Self {
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

        let adapter_features = adapter.features();
        let supports_indirect_count =
            adapter_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        let requested_features = if supports_indirect_count {
            wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
        } else {
            wgpu::Features::empty()
        };
        tracing::info!(
            "MULTI_DRAW_INDIRECT_COUNT supported: {}",
            supports_indirect_count
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: requested_features,
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
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // MSAA sample count (4x MSAA for quality anti-aliasing)
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
                moon_position: [-0.4, 0.2, -0.3],
                _pad1_moon: 0.0,
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
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear, // Nearest -> Linear
            anisotropy_clamp: 16,                          // 1 -> 16
            ..Default::default()
        });

        let shadow_map_size = 2048;
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
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
        });
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
            size: 256 * 4,
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

        // SSR textures
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
        let ssr_color_view =
            ssr_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
        let ssr_depth_view =
            ssr_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                    size: Some(std::num::NonZeroU64::new(80).unwrap()),
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
            Vertex { position: [-1.0, -1.0, 0.0], normal: sun_normal, color: sun_color, uv: [0.0, 0.0], tex_index: 0.0 },
            Vertex { position: [1.0, -1.0, 0.0], normal: sun_normal, color: sun_color, uv: [1.0, 0.0], tex_index: 0.0 },
            Vertex { position: [1.0, 1.0, 0.0], normal: sun_normal, color: sun_color, uv: [1.0, 1.0], tex_index: 0.0 },
            Vertex { position: [-1.0, 1.0, 0.0], normal: sun_normal, color: sun_color, uv: [0.0, 1.0], tex_index: 0.0 },
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/depth_resolve.wgsl").into(),
            ),
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

        // ============== POST-PROCESS INITIALIZATION ==============
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });
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

        let mut indirect_manager = IndirectManager::new(&device);
        let mut water_indirect_manager = IndirectManager::new(&device);
        indirect_manager.init_shadow_resources(&device);
        water_indirect_manager.init_shadow_resources(&device);

        // Hi-Z
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
            water_bind_group,
            water_bind_group_layout,
            surface_format,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            fps_buffer,
            menu_buffer,
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
            csm: render3d::render_core::csm::CsmManager::new(),
        }
    }

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

