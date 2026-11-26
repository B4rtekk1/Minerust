use glyphon::Resolution;

use super::state::State;

pub(super) fn create_ssr_targets(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };

    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSR Color Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    (color_texture, color_view)
}

pub(super) fn create_ssr_hiz_targets(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView, Vec<wgpu::TextureView>, [u32; 2]) {
    let size = [config.width, config.height];
    // Water reflections only read the resolved full-resolution depth.
    let mip_level_count = 1;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("SSR Hi-Z Texture"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mips = (0..mip_level_count)
        .map(|level| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("SSR Hi-Z Mip View {level}")),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    (texture, view, mips, size)
}

/// Quarter-resolution (per axis) target of the ray-marched god-rays pass.
pub(super) fn create_god_rays_target(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> (wgpu::Texture, wgpu::TextureView, [u32; 2]) {
    let size = [config.width.div_ceil(4).max(1), config.height.div_ceil(4).max(1)];
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("God Rays Texture"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view, size)
}

pub(super) fn create_god_rays_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    depth_view: &wgpu::TextureView,
    shadow_grid: &minerust::render::shadow_grid::ShadowGrid,
    output_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("God Rays Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: shadow_grid.bits_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shadow_grid.header_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    })
}

impl State {
    /// Responds to a window resize event by rebuilding every GPU resource
    /// whose dimensions are tied to the surface size.
    ///
    /// wgpu does not automatically resize textures when the OS window changes
    /// size, so every resolution-dependent resource must be explicitly
    /// recreated here.  The method is a no-op when either dimension is zero
    /// (e.g., a minimized window) to avoid creating zero-sized textures, which
    /// are invalid on most backends.
    ///
    /// # Resources rebuilt on every resize
    ///
    /// | Resource | Reason |
    /// |---|---|
    /// | Surface configuration | Swap-chain must match the new pixel dimensions. |
    /// | Depth texture (MSAA) | Multisampled depth must match the color target size. |
    /// | MSAA color texture | Render target size changed. |
    /// | Opaque scene color texture + water bind group | Reflection targets must match the surface. |
    /// | `depth_resolve_bind_group` | References the new multisampled depth view. |
    /// | `glyphon` viewport | Text renderer needs the physical resolution for HiDPI. |
    /// | Scene color texture + view | MSAA resolve target for the composite pass. |
    /// | `composite_bind_group` | References the new scene color view. |
    /// | Hi-Z texture + mips + bind groups | When the half-resolution base dimensions change. |
    ///
    /// # Hi-Z conditional rebuild
    /// The Hi-Z base dimensions are `max(1, floor(surface / 2))` per axis.
    /// Because both the texture dimensions and the mip count depend on the
    /// surface size, the Hi-Z resources are rebuilt when their base dimensions
    /// differ from the previously stored `hiz_size`.
    /// Rebuilding also rewires the two `IndirectManager` bind groups so the
    /// GPU cull compute shader continues to read from the correct texture.
    ///
    /// # Parameters
    /// - `new_size` – Physical pixel dimensions reported by winit.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.recreate_size_dependent_targets();
        self.recreate_ssr_hiz_targets();
        self.viewport.update(
            &self.queue,
            Resolution {
                width: new_size.width,
                height: new_size.height,
            },
        );
        self.rebuild_water_bind_group();
        self.rebuild_composite_bind_groups();
        self.hiz_valid = false;
        self.recreate_hiz_targets();
        self.rebuild_depth_resolve_bind_group();
    }

    fn recreate_size_dependent_targets(&mut self) {
        self.depth_texture =
            Self::create_depth_texture(&self.device, &self.config, self.msaa_samples);
        self.msaa_texture_view = Self::create_msaa_texture(
            &self.device,
            &self.config,
            self.surface_format,
            self.msaa_samples,
        );

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
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.scene_color_view = self
            .scene_color_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let (color_texture, color_view) =
            create_ssr_targets(&self.device, &self.config, self.surface_format);
        self.ssr_color_texture = color_texture;
        self.ssr_color_view = color_view;
    }

    fn rebuild_water_bind_group(&mut self) {
        self.water_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water_bind_group"),
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
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.ssr_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.ssr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&self.wave_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&self.wave_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&self.ssr_hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.shadow_grid.bits_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.shadow_grid.header_buffer().as_entire_binding(),
                },
            ],
        });
    }

    fn recreate_ssr_hiz_targets(&mut self) {
        let (texture, view, mips, size) = create_ssr_hiz_targets(&self.device, &self.config);
        let resolve_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Hi-Z Resolve Bind Group"),
            layout: &self.ssr_hiz_resolve_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mips[0]),
                },
            ],
        });
        self.ssr_hiz_texture = texture;
        self.ssr_hiz_view = view;
        self.ssr_hiz_mips = mips;
        self.ssr_hiz_size = size;
        self.ssr_hiz_resolve_bind_group = resolve_bind_group;

        let (texture, view, size) = create_god_rays_target(&self.device, &self.config);
        self.god_rays_bind_group = create_god_rays_bind_group(
            &self.device,
            &self.god_rays_pipeline.get_bind_group_layout(0),
            &self.uniform_buffer,
            &self.ssr_hiz_view,
            &self.shadow_grid,
            &view,
        );
        self.god_rays_texture = texture;
        self.god_rays_view = view;
        self.god_rays_size = size;
    }

    fn rebuild_composite_bind_groups(&mut self) {
        let layout = self.composite_pipeline.get_bind_group_layout(0);
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &layout,
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
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.god_rays_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.ssr_hiz_view),
                },
            ],
        });
        self.menu_composite_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Menu Composite Bind Group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.menu_background_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.god_rays_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.ssr_hiz_view),
                    },
                ],
            });
    }

    fn recreate_hiz_targets(&mut self) {
        let new_hiz_size = minerust::render::depth::hiz_base_size([self.config.width, self.config.height]);
        if new_hiz_size == self.hiz_size {
            return;
        }

        let hiz_max_dim = new_hiz_size[0].max(new_hiz_size[1]);
        let hiz_mips_count = hiz_max_dim.ilog2() + 1;

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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let hiz_mips: Vec<_> = (0..hiz_mips_count)
            .map(|i| {
                hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Hi-Z Mip View {i}")),
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let hiz_bind_groups: Vec<_> = (0..hiz_mips_count - 1)
            .map(|i| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Hi-Z Bind Group {i}")),
                    layout: &self.hiz_bind_group_layout,
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
            .collect();

        self.indirect_manager.update_bind_group(
            &self.device,
            &hiz_view,
            &self.water_indirect_manager,
        );
        self.hiz_valid = false;
        self.hiz_size = new_hiz_size;
        self.hiz_texture = hiz_texture;
        self.hiz_view = hiz_view;
        self.hiz_mips = hiz_mips;
        self.hiz_bind_groups = hiz_bind_groups;
    }

    fn rebuild_depth_resolve_bind_group(&mut self) {
        self.depth_resolve_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Resolve Bind Group"),
            layout: &self.depth_resolve_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.hiz_mips[0]),
                },
            ],
        });
    }
}
