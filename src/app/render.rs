use cgmath::{InnerSpace, Matrix4, Rad, SquareMatrix};
use glyphon::{
    Attrs, Color, Family, Metrics, Shaping, TextArea, TextBounds,
};
use wgpu::util::DeviceExt;

use render3d::{
    build_player_model, extract_frustum_planes, BlockType, CHUNK_SIZE, DEFAULT_FOV,
    RENDER_DISTANCE, Uniforms, Vertex,
};

use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::GameState;

use super::init::OPENGL_TO_WGPU_MATRIX;
use super::init::frustum_planes_to_array;
use super::state::State;

impl State {
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
                let needed_verts = all_vertices.len() as u32;
                let needed_idxs = all_indices.len() as u32;

                // Reallocate only when buffer is too small (grow-only)
                if needed_verts > self.player_model_vertex_capacity
                    || self.player_model_vertex_buffer.is_none()
                {
                    let new_cap = (needed_verts * 2).max(256);
                    self.player_model_vertex_buffer = Some(self.device.create_buffer(
                        &wgpu::BufferDescriptor {
                            label: Some("Player Model Vertex Buffer"),
                            size: (new_cap as usize * size_of::<Vertex>()) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        },
                    ));
                    self.player_model_vertex_capacity = new_cap;
                }
                if needed_idxs > self.player_model_index_capacity
                    || self.player_model_index_buffer.is_none()
                {
                    let new_cap = (needed_idxs * 2).max(512);
                    self.player_model_index_buffer = Some(self.device.create_buffer(
                        &wgpu::BufferDescriptor {
                            label: Some("Player Model Index Buffer"),
                            size: (new_cap as usize * size_of::<u32>()) as u64,
                            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        },
                    ));
                    self.player_model_index_capacity = new_cap;
                }

                self.queue.write_buffer(
                    self.player_model_vertex_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&all_vertices),
                );
                self.queue.write_buffer(
                    self.player_model_index_buffer.as_ref().unwrap(),
                    0,
                    bytemuck::cast_slice(&all_indices),
                );
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
        let proj = cgmath::perspective(Rad(DEFAULT_FOV), aspect, 0.1, 500.0);
        let view_mat = self.camera.view_matrix();
        let view_proj = OPENGL_TO_WGPU_MATRIX * proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.into();

        let time = self.game_start_time.elapsed().as_secs_f32();

        let day_cycle_speed = 0.005;
        let sun_angle = time * day_cycle_speed + std::f32::consts::FRAC_PI_2;
        let sun_x = 0.0;
        let sun_y = sun_angle.sin();
        let sun_z = sun_angle.cos();
        let sun_dir = cgmath::Vector3::new(sun_x, sun_y, sun_z).normalize();

        let moon_x = 0.0f32;
        let moon_y = -sun_y;
        let moon_z = -sun_z;

        let mut csm = &mut self.csm;
        let fov_y = DEFAULT_FOV;
        csm.update(&view_mat, sun_dir, 0.1, 300.0, aspect, fov_y);

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
                moon_position: [moon_x, moon_y, moon_z],
                _pad1_moon: 0.0,
            }]),
        );

        let frustum_planes = extract_frustum_planes(&view_proj);

        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        // Dynamic cascade count based on render distance for performance
        let active_cascades = render3d::get_active_cascade_count(RENDER_DISTANCE);

        // --- Shadow pass: compute upload + dispatch first for active cascades ---
        // Precompute frustum arrays for active cascades only
        let mut shadow_frustum_arrays = [[[0f32; 4]; 6]; 4];
        for i in 0..active_cascades {
            let cascade_matrix: [[f32; 4]; 4] = csm.cascades[i].view_proj.into();
            let mut shadow_uniform_data = [0f32; 64];
            shadow_uniform_data[0..16].copy_from_slice(cascade_matrix.as_flattened());
            shadow_uniform_data[16] = time;

            self.queue.write_buffer(
                &self.shadow_cascade_buffer,
                (i * 256) as u64,
                bytemuck::cast_slice(&shadow_uniform_data),
            );

            let cascade_view_proj = csm.cascades[i].view_proj;
            let shadow_frustum = extract_frustum_planes(&cascade_view_proj);
            shadow_frustum_arrays[i] = frustum_planes_to_array(&shadow_frustum);
        }

        // Dispatch shadow culling compute passes for active cascades (allows GPU overlap)
        for i in 0..active_cascades {
            self.indirect_manager.dispatch_shadow_culling(
                &mut encoder,
                &self.queue,
                i,
                &shadow_frustum_arrays[i],
            );
            self.water_indirect_manager.dispatch_shadow_culling(
                &mut encoder,
                &self.queue,
                i,
                &shadow_frustum_arrays[i],
            );
        }

        // Then execute shadow render passes for active cascades (after all culling is dispatched)
        const SHADOW_PASS_LABELS: [&str; 4] = [
            "Shadow Pass Cascade 0",
            "Shadow Pass Cascade 1",
            "Shadow Pass Cascade 2",
            "Shadow Pass Cascade 3",
        ];
        for i in 0..active_cascades {
            let offset = (i * 256) as u32;
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
            shadow_pass.set_vertex_buffer(0, self.indirect_manager.vertex_buffer().slice(..));
            shadow_pass.set_index_buffer(
                self.indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            if self.supports_indirect_count {
                shadow_pass.multi_draw_indexed_indirect_count(
                    self.indirect_manager.shadow_draw_commands(i),
                    0,
                    self.indirect_manager.shadow_visible_count_buffer(i),
                    0,
                    self.indirect_manager.active_count(),
                );
            } else {
                shadow_pass.multi_draw_indexed_indirect(
                    self.indirect_manager.shadow_draw_commands(i),
                    0,
                    self.indirect_manager.active_count(),
                );
            }
        }

        let mut meshes_to_request: Vec<(i32, i32, i32)> = Vec::new();
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
                            if subchunk.mesh_dirty
                                && !self.mesh_loader.is_pending(cx, cz, sy as i32)
                            {
                                meshes_to_request.push((cx, cz, sy as i32));
                            }
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

        meshes_to_request.sort_by_key(|&(cx, cz, _sy)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });
        for (cx, cz, sy) in &meshes_to_request {
            self.mesh_loader.request_mesh(*cx, *cz, *sy);
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
                    resolve_target: Some(&self.ssr_color_view),
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

            // 1. Sky
            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

            // 2. Terrain
            opaque_pass.set_pipeline(&self.render_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.indirect_manager.vertex_buffer().slice(..));
            opaque_pass.set_index_buffer(
                self.indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            if self.supports_indirect_count {
                opaque_pass.multi_draw_indexed_indirect_count(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.visible_count_buffer(),
                    0,
                    self.indirect_manager.active_count(),
                );
            } else {
                opaque_pass.multi_draw_indexed_indirect(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.active_count(),
                );
            }

            // 3. Players
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

            // 4. Sun
            opaque_pass.set_pipeline(&self.sun_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

        // Depth Resolve Pass
        {
            let mut depth_resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Depth Resolve Pass (SSR + Hi-Z)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.hiz_mips[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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

        // Hi-Z Generation
        for i in 0..self.hiz_bind_groups.len() {
            let mut hiz_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hi-Z Generation Pass Level"),
                timestamp_writes: None,
            });
            hiz_pass.set_pipeline(&self.hiz_pipeline);
            hiz_pass.set_bind_group(0, &self.hiz_bind_groups[i], &[]);
            let div = 1 << (i + 1);
            let mip_width = (self.hiz_size[0] / div).max(1);
            let mip_height = (self.hiz_size[1] / div).max(1);
            hiz_pass.dispatch_workgroups((mip_width + 15) / 16, (mip_height + 15) / 16, 1);
        }

        let resolve_target = &self.scene_color_view;

        // Transparent Pass: Water
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(resolve_target),
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

            transparent_pass.set_pipeline(&self.water_pipeline);
            transparent_pass.set_bind_group(0, &self.water_bind_group, &[]);
            transparent_pass
                .set_vertex_buffer(0, self.water_indirect_manager.vertex_buffer().slice(..));
            transparent_pass.set_index_buffer(
                self.water_indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            if self.supports_indirect_count {
                transparent_pass.multi_draw_indexed_indirect_count(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.visible_count_buffer(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            } else {
                transparent_pass.multi_draw_indexed_indirect(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            }
        }

        // Composite pass
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
            composite_pass.draw(0..3, 0..1);
        }

        // UI Pass
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

        // Progress bar
        if self.digging.target.is_some() && self.digging.break_time > 0.0 {
            let progress = (self.digging.progress / self.digging.break_time).min(1.0);

            let bar_width = 0.15;
            let bar_height = 0.015;
            let bar_y = -0.05;

            let bg_color = Vertex::pack_color([0.2, 0.2, 0.2]);
            let prog_color = Vertex::pack_color([1.0 - progress, progress, 0.0]);
            let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

            let mut vertices = Vec::with_capacity(8);
            vertices.push(Vertex { position: [-bar_width, bar_y - bar_height, 0.0], normal, color: bg_color, uv: [0.0, 0.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [bar_width, bar_y - bar_height, 0.0], normal, color: bg_color, uv: [1.0, 0.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [bar_width, bar_y + bar_height, 0.0], normal, color: bg_color, uv: [1.0, 1.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [-bar_width, bar_y + bar_height, 0.0], normal, color: bg_color, uv: [0.0, 1.0], tex_index: 0.0 });

            let prog_width = bar_width * 2.0 * progress - bar_width;
            vertices.push(Vertex { position: [-bar_width + 0.005, bar_y - bar_height + 0.003, 0.0], normal, color: prog_color, uv: [0.0, 0.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [prog_width - 0.005, bar_y - bar_height + 0.003, 0.0], normal, color: prog_color, uv: [1.0, 0.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [prog_width - 0.005, bar_y + bar_height - 0.003, 0.0], normal, color: prog_color, uv: [1.0, 1.0], tex_index: 0.0 });
            vertices.push(Vertex { position: [-bar_width + 0.005, bar_y + bar_height - 0.003, 0.0], normal, color: prog_color, uv: [0.0, 1.0], tex_index: 0.0 });

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

        // Menu overlay
        if self.game_state == GameState::Menu {
            self.render_menu(&mut encoder, &view);
        } else {
            self.render_remote_players(
                &view_proj,
                self.config.width as f32,
                self.config.height as f32,
            );
        }

        // Text rendering
        {
            let fps_text = format!(
                "FPS: {:.0}\nFrame: {:.2} ms\nCPU update: {:.2} ms\nChunks: {}\nSubchunks: {}",
                self.current_fps,
                self.frame_time_ms,
                self.cpu_update_ms,
                self.chunks_rendered,
                self.subchunks_rendered
            );
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

    pub fn prepare_menu_text(&mut self) {
        use crate::ui::menu::MenuField;
        let mut text = String::new();
        text.push_str("MULTIPLAYER\n\n");
        text.push_str(
            "Click a field to edit, Tab to switch, Enter to connect, Esc to play solo\n\n\n",
        );

        let addr_selected = self.menu_state.selected_field == MenuField::ServerAddress;
        if addr_selected { text.push_str("> "); }
        text.push_str(&format!("Server Address: {}\n", self.menu_state.server_address));

        let user_selected = self.menu_state.selected_field == MenuField::Username;
        if user_selected { text.push_str("> "); }
        text.push_str(&format!("Username: {}\n\n", self.menu_state.username));

        text.push_str("[ENTER] Connect to Server\n");
        text.push_str("[ESC] Play Singleplayer\n\n");

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

    pub fn render_menu(&mut self, _encoder: &mut wgpu::CommandEncoder, _view: &wgpu::TextureView) {}

    pub fn render_remote_players(
        &mut self,
        _view_proj: &Matrix4<f32>,
        _width: f32,
        _height: f32,
    ) {
        // Player model buffers are already built at the start of render().
    }
}

