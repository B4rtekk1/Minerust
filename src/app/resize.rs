use glyphon::Resolution;

use super::state::State;

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
    /// | SSR color texture + view | SSR reads scene pixels 1:1; must stay in sync. |
    /// | SSR depth texture + view | Same reason – used for refraction depth lookups. |
    /// | SSR sampler | Recreated alongside its textures for clarity. |
    /// | `water_bind_group` | References the new SSR views. |
    /// | `depth_resolve_bind_group` | References the new multisampled depth view. |
    /// | `glyphon` viewport | Text renderer needs the physical resolution for HiDPI. |
    /// | Scene color texture + view | MSAA resolve target for the composite pass. |
    /// | `composite_bind_group` | References the new scene color view. |
    /// | Hi-Z texture + mips + bind groups | Only when the mip count changes (see below). |
    ///
    /// # Hi-Z conditional rebuild
    /// The hierarchical-Z texture mip count is `⌊log₂(max(w, h))⌋ + 1`.
    /// Because the mip count can change when the window crosses a
    /// power-of-two boundary, the Hi-Z resources are only rebuilt when
    /// `[width, height]` differs from the previously stored `hiz_size`.
    /// Rebuilding also rewires the two `IndirectManager` bind groups so the
    /// GPU cull compute shader continues to read from the correct texture.
    ///
    /// # Parameters
    /// - `new_size` – Physical pixel dimensions reported by winit.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Guard: zero-sized surfaces are invalid on all backends and occur
        // transiently when the window is minimized on some platforms.
        if new_size.width > 0 && new_size.height > 0 {
            // ── Swap-chain reconfiguration ────────────────────────────────── //
            self.config.width  = new_size.width;
            self.config.height = new_size.height;
            // Reconfiguring the surface implicitly invalidates the old
            // swap-chain textures; any `SurfaceTexture` acquired before this
            // call must already have been presented or dropped.
            self.surface.configure(&self.device, &self.config);

            // ── MSAA color and depth targets ─────────────────────────────── //
            // Both must exactly match the new surface dimensions; mismatched
            // sizes cause validation errors when beginning render passes.
            let msaa_sample_count: u32 = 4;
            self.depth_texture = Self::create_depth_texture(
                &self.device, &self.config, msaa_sample_count,
            );
            self.msaa_texture_view = Self::create_msaa_texture(
                &self.device,
                &self.config,
                self.surface_format,
                msaa_sample_count,
            );

            // ── SSR (Screen-Space Reflections) targets ────────────────────── //
            // Both the color and depth SSR textures must be single-sampled
            // (the water shader reads them with a non-comparison sampler) and
            // must match the new surface dimensions for correct texel mapping.

            // SSR color: receives the resolved opaque scene for reflection sampling.
            self.ssr_color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SSR Color Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1, // single-sampled – water shader cannot use MSAA textures
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.ssr_color_view = self
                .ssr_color_texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // SSR depth: single-sampled resolved depth for refraction ray marching.
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

            // Nearest-neighbor sampler for SSR lookups; bilinear filtering
            // would blur the reflected image and produce incorrect depth reads.
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

            // ── Water bind group ──────────────────────────────────────────── //
            // The water shader's bind group contains direct references to the
            // SSR texture views, so it must be recreated whenever those views
            // change.  All other bindings (uniforms, atlas, shadow map) are
            // resolution-independent and are simply re-bound from their
            // existing handles.
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
                        // ← new SSR color view from the recreated texture
                        resource: wgpu::BindingResource::TextureView(&self.ssr_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        // ← new SSR depth view from the recreated texture
                        resource: wgpu::BindingResource::TextureView(&self.ssr_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.ssr_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&self.flow_map_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(&self.flow_sampler),
                    },
                ],
                label: Some("water_bind_group"),
            });

            // ── Depth-resolve bind group ──────────────────────────────────── //
            // The depth-resolve shader reads the multisampled depth texture to
            // produce the single-sampled Hi-Z seed and SSR depth.  The bind
            // group must reference the freshly-created `depth_texture` view.
            // Layout is retrieved from the pipeline to avoid storing a
            // redundant handle on `State`.
            self.depth_resolve_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Depth Resolve Bind Group"),
                    layout: &self.depth_resolve_pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        // ← new multisampled depth view
                        resource: wgpu::BindingResource::TextureView(&self.depth_texture),
                    }],
                });

            // ── glyphon viewport ──────────────────────────────────────────── //
            // The text renderer uses the physical resolution to convert between
            // pixel and subpixel coordinates; it must be kept in sync so text
            // appears at the correct size and position after a resize.
            self.viewport.update(
                &self.queue,
                Resolution {
                    width:  new_size.width,
                    height: new_size.height,
                },
            );

            // ── Scene color texture (composite pass input) ───────────────── //
            // After all MSAA passes resolve into this texture, the composite
            // shader reads it and writes post-processed output to the swap-chain.
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

            // ── Composite bind group ──────────────────────────────────────── //
            // Must reference the new `scene_color_view`.  The sampler is
            // bilinear (unlike the nearest-neighbor SSR sampler) because the
            // composite shader may apply a slight blur or scale during post-
            // processing.
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
                        // ← new scene color view
                        resource: wgpu::BindingResource::TextureView(&self.scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&composite_sampler),
                    },
                ],
            });

            // ── Hierarchical-Z (Hi-Z) texture rebuild ─────────────────────── //
            // The Hi-Z mip count is `⌊log₂(max(width, height))⌋ + 1`.  This
            // value can change when the window crosses a power-of-two boundary,
            // which means the number of bind groups would also need to change.
            // Rather than trying to patch the existing resources, we simply
            // recreate all Hi-Z objects whenever the surface dimensions change.
            let new_hiz_size = [new_size.width, new_size.height];
            if new_hiz_size != self.hiz_size {
                self.hiz_size = new_hiz_size;
                let hiz_max_dim    = new_size.width.max(new_size.height);
                let hiz_mips_count = (hiz_max_dim as f32).log2().floor() as u32 + 1;

                let hiz_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Hi-Z Texture"),
                    size: wgpu::Extent3d {
                        width:  new_hiz_size[0],
                        height: new_hiz_size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: hiz_mips_count,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    // STORAGE_BINDING  – written by the Hi-Z compute shader.
                    // TEXTURE_BINDING  – read by the cull compute shader.
                    // RENDER_ATTACHMENT – mip 0 is written as a color attachment
                    //                    in the depth-resolve pass.
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });

                // Full-mip view used when any level needs to be sampled.
                let new_hiz_view =
                    hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Individual single-mip views for the compute downsampling pairs.
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

                // One bind group per adjacent mip pair (N reads → N+1 writes).
                // The layout is retrieved from `State` (stored during `new`)
                // so it doesn't need to be recreated here.
                let new_hiz_bind_groups: Vec<_> = (0..hiz_mips_count - 1)
                    .map(|i| {
                        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("Hi-Z Bind Group {}", i)),
                            layout: &self.hiz_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    // Source mip (read-only texture)
                                    resource: wgpu::BindingResource::TextureView(
                                        &new_hiz_mips[i as usize],
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    // Destination mip (write-only storage texture)
                                    resource: wgpu::BindingResource::TextureView(
                                        &new_hiz_mips[(i + 1) as usize],
                                    ),
                                },
                            ],
                        })
                    })
                    .collect();

                // Rewire the indirect-draw managers so their cull compute
                // shaders sample the new Hi-Z texture on the next frame.
                self.indirect_manager
                    .update_bind_group(&self.device, &new_hiz_view);
                self.water_indirect_manager
                    .update_bind_group(&self.device, &new_hiz_view);

                // Commit all new Hi-Z resources to State, dropping the old ones.
                self.hiz_texture    = hiz_texture;
                self.hiz_view       = new_hiz_view;
                self.hiz_mips       = new_hiz_mips;
                self.hiz_bind_groups = new_hiz_bind_groups;
            }
        }
    }
}
