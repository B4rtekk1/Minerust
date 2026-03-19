use cgmath::{InnerSpace, Matrix4, Rad, SquareMatrix};
use glyphon::{Attrs, Color, Family, Metrics, Shaping, TextArea, TextBounds};
use wgpu::util::DeviceExt;

use minerust::{
    CHUNK_SIZE, DEFAULT_FOV, RENDER_DISTANCE, Uniforms, Vertex, build_player_model,
    extract_frustum_planes,
};

use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::{GameState, MenuField, MenuLayout, Rect};

use super::init::OPENGL_TO_WGPU_MATRIX;
use super::init::frustum_planes_to_array;
use super::state::State;

fn px_to_ndc_x(x: f32, width: f32) -> f32 {
    (x / width) * 2.0 - 1.0
}

fn px_to_ndc_y(y: f32, height: f32) -> f32 {
    1.0 - (y / height) * 2.0
}

fn push_rect(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    rect: Rect,
    color: [u8; 4],
    width: f32,
    height: f32,
) {
    let base = vertices.len() as u32;
    let x0 = px_to_ndc_x(rect.x, width);
    let y0 = px_to_ndc_y(rect.y, height);
    let x1 = px_to_ndc_x(rect.x + rect.w, width);
    let y1 = px_to_ndc_y(rect.y + rect.h, height);
    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
        vertices.push(Vertex {
            position: [x, y, 0.0],
            normal,
            color,
            uv: [0.0, 0.0],
            tex_index: 0.0,
        });
    }

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn rgba(color: [f32; 4]) -> [u8; 4] {
    Vertex::pack_color_rgba(color)
}

impl State {
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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

                if needed_verts > self.player_model_vertex_capacity
                    || self.player_model_vertex_buffer.is_none()
                {
                    let new_cap = (needed_verts * 2).max(256);
                    self.player_model_vertex_buffer =
                        Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Player Model Vertex Buffer"),
                            size: (new_cap as usize * size_of::<Vertex>()) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    self.player_model_vertex_capacity = new_cap;
                }
                if needed_idxs > self.player_model_index_capacity
                    || self.player_model_index_buffer.is_none()
                {
                    let new_cap = (needed_idxs * 2).max(512);
                    self.player_model_index_buffer =
                        Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Player Model Index Buffer"),
                            size: (new_cap as usize * size_of::<u32>()) as u64,
                            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    self.player_model_index_capacity = new_cap;
                }

                self.queue.write_buffer(
                    self.player_model_vertex_buffer
                        .as_ref()
                        .expect("Player model vertex buffer should be initialized"),
                    0,
                    bytemuck::cast_slice(&all_vertices),
                );
                self.queue.write_buffer(
                    self.player_model_index_buffer
                        .as_ref()
                        .expect("Player model index buffer should be initialized"),
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
        let far_plane = (RENDER_DISTANCE as f32 * CHUNK_SIZE as f32 * 1.5).max(400.0);
        let proj = cgmath::perspective(Rad(DEFAULT_FOV), aspect, 0.1, far_plane);
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

        let csm = &mut self.csm;
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
        let active_cascades = minerust::get_active_cascade_count(RENDER_DISTANCE);

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

            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

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

            opaque_pass.set_pipeline(&self.sun_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

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

            if self.game_state != GameState::Menu {
                if self.hotbar_dirty || self.hotbar_vertex_buffer.is_none() {
                    let aspect = self.config.width as f32 / self.config.height as f32;
                    let (vb, ib, count) = crate::ui::ui::build_hotbar(
                        &self.device,
                        self.hotbar_slot,
                        aspect,
                    );
                    self.hotbar_vertex_buffer = Some(vb);
                    self.hotbar_index_buffer = Some(ib);
                    self.hotbar_num_indices = count;
                    self.hotbar_dirty = false;
                }
                if let (Some(vb), Some(ib)) =
                    (&self.hotbar_vertex_buffer, &self.hotbar_index_buffer)
                {
                    if self.hotbar_num_indices > 0 {
                        ui_pass.set_vertex_buffer(0, vb.slice(..));
                        ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        ui_pass.draw_indexed(0..self.hotbar_num_indices, 0, 0..1);
                    }
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
            let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

            let mut vertices = Vec::with_capacity(8);
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
                    self.progress_bar_vertex_buffer
                        .as_ref()
                        .expect("Progress bar vertex buffer should be initialized"),
                    0,
                    bytemuck::cast_slice(&vertices),
                );
            }

            let progress_vb = self
                .progress_bar_vertex_buffer
                .as_ref()
                .expect("Progress bar vertex buffer should be initialized");
            let progress_ib = self
                .progress_bar_index_buffer
                .as_ref()
                .expect("Progress bar index buffer should be initialized");

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

        if self.game_state == GameState::Menu {
            self.render_menu(&mut encoder, &view);
        } else {
            self.render_remote_players(
                &view_proj,
                self.config.width as f32,
                self.config.height as f32,
            );
        }

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

            if self.game_state != GameState::Menu && self.last_hotbar_slot != self.hotbar_slot {
                let block = crate::ui::ui::HOTBAR_SLOTS[self.hotbar_slot];
                let label = block.display_name();
                self.hotbar_label_buffer.set_text(
                    &mut self.font_system,
                    label,
                    &Attrs::new()
                        .family(Family::SansSerif)
                        .color(Color::rgb(255, 238, 200)),
                    Shaping::Advanced,
                    None,
                );
                self.hotbar_label_buffer.set_size(
                    &mut self.font_system,
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                let font_size = 22.0;
                let char_width = font_size * 0.6;
                self.hotbar_label_width = label.chars().count() as f32 * char_width;
                self.last_hotbar_slot = self.hotbar_slot;
            }

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
                let layout = MenuLayout::new(self.config.width, self.config.height);
                let title_x = layout.header.x + 10.0;
                let title_y = layout.header.y + 6.0;
                let subtitle_x = layout.header.x + 10.0;
                let subtitle_y = layout.header.y + 56.0;
                let server_label_y = layout.server_label.y - 6.0;
                let username_label_y = layout.username_label.y - 6.0;
                let server_value_y = layout.server_field.y + 12.0;
                let username_value_y = layout.username_field.y + 12.0;
                let tips_y = layout.quick_card.y + 86.0;
                let button_text_y = layout.connect_button.y + 15.0;
                let single_text_y = layout.singleplayer_button.y + 15.0;
                let status_y = layout.status_pill.y + 8.0;

                text_areas.push(TextArea {
                    buffer: &self.menu_title_buffer,
                    left: title_x,
                    top: title_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(242, 227, 187),
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_subtitle_buffer,
                    left: subtitle_x,
                    top: subtitle_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(186, 201, 214),
                    custom_glyphs: &[],
                });

                text_areas.push(TextArea {
                    buffer: &self.menu_server_label_buffer,
                    left: layout.server_label.x + 2.0,
                    top: server_label_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(140, 153, 167),
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_server_value_buffer,
                    left: layout.server_field.x + 16.0,
                    top: server_value_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(248, 250, 252),
                    custom_glyphs: &[],
                });

                text_areas.push(TextArea {
                    buffer: &self.menu_username_label_buffer,
                    left: layout.username_label.x + 2.0,
                    top: username_label_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(140, 153, 167),
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_username_value_buffer,
                    left: layout.username_field.x + 16.0,
                    top: username_value_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(248, 250, 252),
                    custom_glyphs: &[],
                });

                text_areas.push(TextArea {
                    buffer: &self.menu_tips_buffer,
                    left: layout.quick_card.x + 20.0,
                    top: tips_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(171, 189, 202),
                    custom_glyphs: &[],
                });

                let connect_estimate = 7.0 * 10.5;
                let single_estimate = 12.0 * 10.5;
                text_areas.push(TextArea {
                    buffer: &self.menu_connect_button_buffer,
                    left: layout.connect_button.x + (layout.connect_button.w - connect_estimate) * 0.5,
                    top: button_text_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(245, 249, 255),
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_singleplayer_button_buffer,
                    left: layout.singleplayer_button.x
                        + (layout.singleplayer_button.w - single_estimate) * 0.5,
                    top: single_text_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(220, 228, 236),
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_status_buffer,
                    left: layout.status_pill.x + 16.0,
                    top: status_y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: if self.menu_state.error_message.is_some() {
                        Color::rgb(255, 124, 124)
                    } else if self.menu_state.status_message.is_some() {
                        Color::rgb(124, 224, 208)
                    } else {
                        Color::rgb(219, 229, 239)
                    },
                    custom_glyphs: &[],
                });
            } else {
                let label_width = self.hotbar_label_width.min(self.config.width as f32);
                let label_left = (self.config.width as f32 - label_width) * 0.5;
                let label_top = (self.config.height as f32 - 170.0).max(0.0);
                text_areas.push(TextArea {
                    buffer: &self.hotbar_label_buffer,
                    left: label_left,
                    top: label_top,
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
                .map_err(|e| {
                    tracing::error!("Failed to prepare text: {:?}", e);
                    wgpu::SurfaceError::Lost
                })?;

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
                .map_err(|e| {
                    tracing::error!("Failed to render text: {:?}", e);
                    wgpu::SurfaceError::Lost
                })?;
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    pub fn prepare_menu_text(&mut self) {
        let selected = self.menu_state.selected_field;

        let title = "Render 3D";
        let subtitle = "Voxel sandbox with multiplayer and custom UI";
        let server_label = if selected == MenuField::ServerAddress {
            "SERVER ADDRESS  •  active"
        } else {
            "SERVER ADDRESS"
        };
        let username_label = if selected == MenuField::Username {
            "USERNAME  •  active"
        } else {
            "USERNAME"
        };
        let server_value = self.menu_state.server_address.as_str();
        let username_value = self.menu_state.username.as_str();
        let tips = "TAB switch field\nENTER connect\nESC singleplayer\nF11 fullscreen";
        let connect_button = "CONNECT";
        let singleplayer_button = "SINGLEPLAYER";

        let status_text = if let Some(ref err) = self.menu_state.error_message {
            format!("ERROR: {}", err)
        } else if let Some(ref status) = self.menu_state.status_message {
            format!("STATUS: {}", status)
        } else {
            "READY: ENTER joins multiplayer, ESC starts solo".to_string()
        };

        self.menu_title_buffer.set_text(
            &mut self.font_system,
            title,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_title_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_subtitle_buffer.set_text(
            &mut self.font_system,
            subtitle,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_subtitle_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_server_label_buffer.set_text(
            &mut self.font_system,
            server_label,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_server_label_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_server_value_buffer.set_text(
            &mut self.font_system,
            server_value,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_server_value_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_username_label_buffer.set_text(
            &mut self.font_system,
            username_label,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_username_label_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_username_value_buffer.set_text(
            &mut self.font_system,
            username_value,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_username_value_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_tips_buffer.set_text(
            &mut self.font_system,
            tips,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_tips_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_connect_button_buffer.set_text(
            &mut self.font_system,
            connect_button,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_connect_button_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_singleplayer_button_buffer.set_text(
            &mut self.font_system,
            singleplayer_button,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_singleplayer_button_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_status_buffer.set_text(
            &mut self.font_system,
            &status_text,
            &Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
            None,
        );
        self.menu_status_buffer.set_size(
            &mut self.font_system,
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );
    }

    pub fn render_menu(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let layout = MenuLayout::new(self.config.width, self.config.height);
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        let panel = layout.panel;
        let hovered = self
            .cursor_position
            .and_then(|(x, y)| layout.hit_test(x, y));

        let mut vertices = Vec::with_capacity(96);
        let mut indices = Vec::with_capacity(144);

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: 0.0,
                y: 0.0,
                w: width,
                h: height,
            },
            rgba([0.03, 0.05, 0.08, 0.94]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x - 10.0,
                y: panel.y - 10.0,
                w: panel.w + 20.0,
                h: panel.h + 20.0,
            },
            rgba([0.05, 0.1, 0.16, 0.55]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            panel,
            rgba([0.07, 0.09, 0.12, 0.96]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x,
                y: panel.y,
                w: panel.w,
                h: 6.0,
            },
            rgba([0.95, 0.72, 0.24, 1.0]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x + 20.0,
                y: panel.y + 18.0,
                w: 180.0,
                h: 34.0,
            },
            rgba([0.12, 0.16, 0.21, 0.95]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x + 16.0,
                y: panel.y + 16.0,
                w: 8.0,
                h: 40.0,
            },
            rgba([0.97, 0.74, 0.24, 1.0]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.quick_card.x,
                y: layout.quick_card.y,
                w: layout.quick_card.w,
                h: layout.quick_card.h,
            },
            rgba([0.11, 0.14, 0.18, 0.98]),
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.quick_card.x,
                y: layout.quick_card.y,
                w: 4.0,
                h: layout.quick_card.h,
            },
            rgba([0.35, 0.8, 0.78, 1.0]),
            width,
            height,
        );

        let field_color = if self.menu_state.selected_field == MenuField::ServerAddress {
            rgba([0.13, 0.2, 0.27, 1.0])
        } else {
            rgba([0.1, 0.13, 0.17, 1.0])
        };
        let username_color = if self.menu_state.selected_field == MenuField::Username {
            rgba([0.13, 0.2, 0.27, 1.0])
        } else {
            rgba([0.1, 0.13, 0.17, 1.0])
        };

        push_rect(
            &mut vertices,
            &mut indices,
            layout.server_field,
            rgba([0.02, 0.03, 0.04, 1.0]),
            width,
            height,
        );
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.server_field.x + 2.0,
                y: layout.server_field.y + 2.0,
                w: layout.server_field.w - 4.0,
                h: layout.server_field.h - 4.0,
            },
            field_color,
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            layout.username_field,
            rgba([0.02, 0.03, 0.04, 1.0]),
            width,
            height,
        );
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.username_field.x + 2.0,
                y: layout.username_field.y + 2.0,
                w: layout.username_field.w - 4.0,
                h: layout.username_field.h - 4.0,
            },
            username_color,
            width,
            height,
        );

        let connect_fill = if matches!(hovered, Some(crate::ui::menu::MenuHit::Connect)) {
            rgba([0.24, 0.52, 0.84, 1.0])
        } else {
            rgba([0.2, 0.45, 0.74, 1.0])
        };
        let single_fill = if matches!(hovered, Some(crate::ui::menu::MenuHit::Singleplayer)) {
            rgba([0.19, 0.22, 0.28, 1.0])
        } else {
            rgba([0.16, 0.19, 0.24, 1.0])
        };

        push_rect(
            &mut vertices,
            &mut indices,
            layout.connect_button,
            rgba([0.16, 0.33, 0.55, 1.0]),
            width,
            height,
        );
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.connect_button.x + 2.0,
                y: layout.connect_button.y + 2.0,
                w: layout.connect_button.w - 4.0,
                h: layout.connect_button.h - 4.0,
            },
            connect_fill,
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            layout.singleplayer_button,
            rgba([0.1, 0.11, 0.14, 1.0]),
            width,
            height,
        );
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.singleplayer_button.x + 2.0,
                y: layout.singleplayer_button.y + 2.0,
                w: layout.singleplayer_button.w - 4.0,
                h: layout.singleplayer_button.h - 4.0,
            },
            single_fill,
            width,
            height,
        );

        push_rect(
            &mut vertices,
            &mut indices,
            layout.status_pill,
            rgba([0.08, 0.1, 0.13, 0.96]),
            width,
            height,
        );

        let selected_field_x = match self.menu_state.selected_field {
            MenuField::ServerAddress => Some(layout.server_field),
            MenuField::Username => Some(layout.username_field),
            MenuField::None => None,
        };
        if let Some(field) = selected_field_x {
            push_rect(
                &mut vertices,
                &mut indices,
                Rect {
                    x: field.x - 2.0,
                    y: field.y - 2.0,
                    w: field.w + 4.0,
                    h: 3.0,
                },
                rgba([0.97, 0.74, 0.24, 1.0]),
                width,
                height,
            );
        }

        let active_field = match self.menu_state.selected_field {
            MenuField::ServerAddress => Some((layout.server_field, self.menu_state.server_address.as_str())),
            MenuField::Username => Some((layout.username_field, self.menu_state.username.as_str())),
            MenuField::None => None,
        };

        if let Some((field, value)) = active_field {
            let char_count = value.chars().count() as f32;
            let cursor_x = (field.x + 16.0 + char_count * 11.0).min(field.x + field.w - 12.0);
            push_rect(
                &mut vertices,
                &mut indices,
                Rect {
                    x: cursor_x,
                    y: field.y + 8.0,
                    w: 2.0,
                    h: field.h - 16.0,
                },
                rgba([0.97, 0.74, 0.24, 0.95]),
                width,
                height,
            );
        }

        if !vertices.is_empty() {
            let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Menu UI VB"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Menu UI IB"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Menu UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            pass.set_pipeline(&self.crosshair_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
    }

    pub fn render_remote_players(&mut self, _view_proj: &Matrix4<f32>, _width: f32, _height: f32) {}
}
