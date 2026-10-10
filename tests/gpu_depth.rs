//! Explicit headless GPU verification and timestamp benchmark. These are
//! ignored in ordinary CI because a Vulkan adapter is required.
use minerust::render::depth::hiz_base_size;
use std::time::Duration;

const DEPTH_FIXTURE: &str = r#"
@vertex
fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
    return vec4<f32>(positions[id], 0., 1.);
}
@fragment
fn fs(@builtin(position) position: vec4<f32>, @builtin(sample_index) sample: u32)
    -> @builtin(frag_depth) f32 {
    let x = u32(position.x);
    let y = u32(position.y);
    if (x + y + sample) % 11u == 0u { return 1.; }
    return f32((x * 17u + y * 31u + sample * 113u) % 1024u) / 1024.;
}
"#;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fixture: wgpu::RenderPipeline,
    resolve: wgpu::ComputePipeline,
    reference: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
}

impl Gpu {
    fn new(timestamps: bool) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("explicit GPU tests require a Vulkan adapter");
        println!("GPU adapter: {:?}", adapter.get_info());
        let features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            ..Default::default()
        }))
        .unwrap();
        let fixture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Per-sample depth fixture"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_FIXTURE.into()),
        });
        let fixture = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Per-sample depth fixture"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &fixture_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fixture_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let compute = |source: &'static str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let resolve = compute(include_str!("../src/shaders/depth_resolve.wgsl"));
        let reference = compute(include_str!("fixtures/depth_resolve_reference.wgsl"));
        let reduce = compute(include_str!("../src/shaders/hiz.wgsl"));
        Self {
            device,
            queue,
            fixture,
            resolve,
            reference,
            reduce,
        }
    }

    fn wait(&self) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.wait();
        rx.recv().unwrap().unwrap();
        let result = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        buffer.unmap();
        result
    }
}

struct Pyramid {
    texture: wgpu::Texture,
    size: [u32; 2],
    resolve: wgpu::BindGroup,
    reductions: Vec<wgpu::BindGroup>,
    reference: bool,
}

impl Pyramid {
    fn new(gpu: &Gpu, depth: &wgpu::TextureView, surface: [u32; 2], reference: bool) -> Self {
        let size = if reference {
            surface
        } else {
            hiz_base_size(surface)
        };
        let levels = size[0].max(size[1]).ilog2() + 1;
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z test pyramid"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let views: Vec<_> = (0..levels)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let ssr = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Legacy unused SSR depth"),
            size: wgpu::Extent3d {
                width: surface[0],
                height: surface[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let ssr_view = ssr.create_view(&Default::default());
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
        ];
        if reference {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&ssr_view),
            });
        }
        let pipeline = if reference {
            &gpu.reference
        } else {
            &gpu.resolve
        };
        let resolve = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let reductions = views
            .windows(2)
            .map(|pair| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &gpu.reduce.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&pair[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&pair[1]),
                        },
                    ],
                })
            })
            .collect();
        Self {
            texture,
            size,
            resolve,
            reductions,
            reference,
        }
    }

    fn encode(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        query: Option<(&wgpu::QuerySet, u32)>,
    ) {
        for level in 0..=self.reductions.len() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: query.and_then(|(query_set, index)| {
                    if level == 0 || level == self.reductions.len() {
                        Some(wgpu::ComputePassTimestampWrites {
                            query_set,
                            beginning_of_pass_write_index: (level == 0).then_some(index),
                            end_of_pass_write_index: (level == self.reductions.len())
                                .then_some(index + 1),
                        })
                    } else {
                        None
                    }
                }),
            });
            if level == 0 {
                pass.set_pipeline(if self.reference {
                    &gpu.reference
                } else {
                    &gpu.resolve
                });
                pass.set_bind_group(0, &self.resolve, &[]);
            } else {
                pass.set_pipeline(&gpu.reduce);
                pass.set_bind_group(0, &self.reductions[level - 1], &[]);
            }
            pass.dispatch_workgroups(
                (self.size[0] >> level).max(1).div_ceil(16),
                (self.size[1] >> level).max(1).div_ceil(16),
                1,
            );
        }
    }

    fn read_level(&self, gpu: &Gpu, level: u32) -> Vec<f32> {
        let width = (self.size[0] >> level).max(1);
        let height = (self.size[1] >> level).max(1);
        let pitch = (width * 4).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (pitch * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);
        gpu.read(&buffer)
            .chunks_exact(pitch as usize)
            .flat_map(|row| {
                row[..width as usize * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v))
            })
            .collect()
    }
}

fn render_fixture(gpu: &Gpu, size: [u32; 2]) -> wgpu::TextureView {
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 4,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = depth.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&gpu.fixture);
        pass.draw(0..3, 0..1);
    }
    gpu.queue.submit([encoder.finish()]);
    view
}

fn cpu_depth(size: [u32; 2]) -> Vec<f32> {
    (0..size[1])
        .flat_map(|y| {
            (0..size[0]).map(move |x| {
                (0..4)
                    .map(|sample| {
                        if (x + y + sample) % 11 == 0 {
                            1.0
                        } else {
                            ((x * 17 + y * 31 + sample * 113) % 1024) as f32 / 1024.0
                        }
                    })
                    .fold(0.0, f32::max)
            })
        })
        .collect()
}

fn cpu_reduce(depth: &[f32], size: [u32; 2]) -> Vec<f32> {
    let dst = hiz_base_size(size);
    (0..dst[1])
        .flat_map(|y| {
            (0..dst[0]).map(move |x| {
                let mut max_depth: f32 = 0.0;
                for dy in 0..2 + size[1] % 2 {
                    for dx in 0..2 + size[0] % 2 {
                        let sx = (x * 2 + dx).min(size[0] - 1);
                        let sy = (y * 2 + dy).min(size[1] - 1);
                        max_depth = max_depth.max(depth[(sy * size[0] + sx) as usize]);
                    }
                }
                max_depth
            })
        })
        .collect()
}

#[test]
#[ignore = "requires a Vulkan adapter; verifies executed MSAA resolve and every Hi-Z mip"]
fn fused_depth_matches_original_pyramid_and_cpu_reference() {
    let gpu = Gpu::new(false);
    for size in [
        [1, 1],
        [1, 17],
        [17, 1],
        [2, 2],
        [3, 3],
        [16, 16],
        [17, 19],
        [31, 32],
        [65, 67],
        [257, 129],
    ] {
        let depth = render_fixture(&gpu, size);
        let reference = Pyramid::new(&gpu, &depth, size, true);
        let optimized = Pyramid::new(&gpu, &depth, size, false);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        reference.encode(&gpu, &mut encoder, None);
        optimized.encode(&gpu, &mut encoder, None);
        gpu.queue.submit([encoder.finish()]);
        let mut expected = cpu_reduce(&cpu_depth(size), size);
        let mut mip_size = hiz_base_size(size);
        for level in 0..=optimized.reductions.len() as u32 {
            let actual = optimized.read_level(&gpu, level);
            assert_eq!(
                actual, expected,
                "CPU comparison: surface {size:?}, mip {level}"
            );
            let original_level = if size == [1, 1] { 0 } else { level + 1 };
            assert_eq!(
                actual,
                reference.read_level(&gpu, original_level),
                "original comparison: surface {size:?}, mip {level}"
            );
            expected = cpu_reduce(&expected, mip_size);
            mip_size = hiz_base_size(mip_size);
        }
        println!("Verified all Hi-Z mips for {size:?}");
    }
}

#[test]
#[ignore = "GPU timestamp microbenchmark; run explicitly on the target GPU"]
fn benchmark_depth_pyramid_gpu_timestamps() {
    let gpu = Gpu::new(true);
    for size in [[1280, 720], [1920, 1080], [1919, 1079]] {
        let depth = render_fixture(&gpu, size);
        let reference = Pyramid::new(&gpu, &depth, size, true);
        let optimized = Pyramid::new(&gpu, &depth, size, false);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for _ in 0..4 {
            reference.encode(&gpu, &mut encoder, None);
            optimized.encode(&gpu, &mut encoder, None);
        }
        gpu.queue.submit([encoder.finish()]);
        gpu.wait();
        let pairs = 40;
        let query = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: pairs * 4,
        });
        let bytes = (pairs * 4 * 8) as u64;
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for i in 0..pairs {
            // Alternate order to reduce systematic cache/order bias.
            if i % 2 == 0 {
                reference.encode(&gpu, &mut encoder, Some((&query, i * 4)));
                optimized.encode(&gpu, &mut encoder, Some((&query, i * 4 + 2)));
            } else {
                optimized.encode(&gpu, &mut encoder, Some((&query, i * 4 + 2)));
                reference.encode(&gpu, &mut encoder, Some((&query, i * 4)));
            }
        }
        encoder.resolve_query_set(&query, 0..pairs * 4, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, bytes);
        gpu.queue.submit([encoder.finish()]);
        let timestamps: Vec<_> = gpu
            .read(&readback)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| u64::from_le_bytes(*v))
            .collect();
        let mut old = Vec::new();
        let mut new = Vec::new();
        for pair in timestamps.as_chunks::<4>().0 {
            old.push(
                (pair[1] - pair[0]) as f64 * gpu.queue.get_timestamp_period() as f64 / 1_000_000.0,
            );
            new.push(
                (pair[3] - pair[2]) as f64 * gpu.queue.get_timestamp_period() as f64 / 1_000_000.0,
            );
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!(
            "{size:?}: original median={:.3} ms, fused median={:.3} ms (40 paired samples)",
            old[20], new[20]
        );
    }
}
