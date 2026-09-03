use glam::{Mat4, Vec3};
use glyphon::{Attrs, Color, Family, Metrics, Shaping, TextArea, TextBounds};
use wgpu::util::DeviceExt;

use minerust::{
    BlockType, CHUNK_SIZE, DEFAULT_FOV, MAX_MESH_BUILDS_PER_FRAME, RENDER_DISTANCE, SEA_LEVEL,
    SUN_MOVEMENT_SPEED, Uniforms, Vertex, World, build_block_outline, build_player_model,
    extract_frustum_planes,
};

use crate::logger::{LogLevel, log};
use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::{GameState, MenuField, MenuHit, MenuLayout};

use super::init::frustum_planes_to_array;
use super::state::State;

#[derive(Debug)]
pub enum RenderError {
    Surface(wgpu::CurrentSurfaceTexture),
    Text,
}

/// Computes which faces of the highlighted block should be outlined.
///
/// The outline follows the same face-visibility rules as block meshing so the
/// overlay only draws exposed faces.
fn visible_outline_faces(world: &World, bx: i32, by: i32, bz: i32) -> [bool; 6] {
    let block = world.get_block(bx, by, bz);
    if block == BlockType::Air {
        return [false; 6];
    }

    [
        block.should_render_face_against(world.get_block(bx + 1, by, bz)),
        block.should_render_face_against(world.get_block(bx - 1, by, bz)),
        block.should_render_face_against(world.get_block(bx, by + 1, bz)),
        block.should_render_face_against(world.get_block(bx, by - 1, bz)),
        block.should_render_face_against(world.get_block(bx, by, bz + 1)),
        block.should_render_face_against(world.get_block(bx, by, bz - 1)),
    ]
}

fn append_ui_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    for (i, (x, y)) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        .into_iter()
        .enumerate()
    {
        vertices.push(Vertex {
            position: [x, y, 0.0],
            packed: Vertex::pack_ui(normal, color, 0, i as u8),
        });
    }

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn build_menu_input_box(
    rect: crate::ui::menu::Rect,
    screen_width: f32,
    screen_height: f32,
    focused: bool,
    selected: bool,
) -> (Vec<Vertex>, Vec<u32>) {
    let x0 = rect.x / screen_width * 2.0 - 1.0;
    let x1 = (rect.x + rect.w) / screen_width * 2.0 - 1.0;
    let y0 = 1.0 - (rect.y + rect.h) / screen_height * 2.0;
    let y1 = 1.0 - rect.y / screen_height * 2.0;

    let border_x = 4.0 / screen_width;
    let border_y = 4.0 / screen_height;
    let border_color = if focused {
        [0.78, 0.92, 1.0, 0.95]
    } else {
        [0.38, 0.50, 0.58, 0.85]
    };
    let inner_color = if selected {
        [0.20, 0.38, 0.60, 0.86]
    } else {
        [0.10, 0.14, 0.18, 0.78]
    };

    let mut vertices = Vec::with_capacity(8);
    let mut indices = Vec::with_capacity(12);
    append_ui_quad(&mut vertices, &mut indices, x0, y0, x1, y1, border_color);
    append_ui_quad(
        &mut vertices,
        &mut indices,
        x0 + border_x,
        y0 + border_y,
        x1 - border_x,
        y1 - border_y,
        inner_color,
    );

    (vertices, indices)
}

impl State {
    fn rebuild_visible_chunk_cache(&mut self, player_cx: i32, player_cz: i32) {
        if !self.visible_chunk_columns_dirty
            && self.visible_chunk_cache_center == (player_cx, player_cz)
        {
            return;
        }

        self.visible_chunk_columns.clear();
        {
            let world = self.world.read();
            for cx in (player_cx - RENDER_DISTANCE)..=(player_cx + RENDER_DISTANCE) {
                for cz in (player_cz - RENDER_DISTANCE)..=(player_cz + RENDER_DISTANCE) {
                    if world.chunks.contains_key(&(cx, cz)) {
                        self.visible_chunk_columns.push((cx, cz));
                    }
                }
            }
        }

        self.visible_chunk_columns.sort_by_key(|&(cx, cz)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });
        self.visible_chunk_cache_center = (player_cx, player_cz);
        self.visible_chunk_columns_dirty = false;
    }

    /// Produces one complete frame and presents it to the OS window.
    ///
    /// # Render pipeline overview
    ///
    /// The frame is built from the following render / compute passes in order:
    ///
    /// 1. **Player model update** – re-builds the combined vertex/index buffers
    ///    for all visible remote players if any exist.
    /// 2. **Uniform upload** – computes the camera matrices, advances the day
    ///    cycle, and uploads the `Uniforms` struct.
    /// 3. **Mesh request** – walks the visible chunk grid, queues dirty sub-chunk
    ///    meshes for background rebuild, and tallies rendered counts.
    /// 4. **Main cull dispatch** – GPU frustum + Hi-Z occlusion cull for both
    ///    the opaque terrain and water indirect managers.
    /// 5. **Opaque pass** – sky dome -> terrain -> remote player models -> sun/moon.
    ///    Resolves MSAA into `ssr_color_view` for later water reflections.
    /// 6. **Depth resolve compute** – resolves the multisampled depth buffer
    ///    into `ssr_depth_view` (for water refraction) and the first Hi-Z mip
    ///    level (for next-frame occlusion culling).
    /// 7. **Hi-Z generation** (compute) – downsamples the depth mip chain.
    /// 8. **Transparent pass** – water surfaces, alpha-blended on top of the
    ///    opaque result.  Resolves MSAA into `scene_color_view`.
    /// 9. **Composite pass** – post-processing blit from `scene_color_view`
    ///     to the swap-chain surface (underwater fog, vignette, etc.).
    /// 10. **UI pass** – crosshair, coordinate debug overlay, hotbar.
    /// 11. **Progress bar pass** – block-breaking progress indicator (only
    ///     when the player is actively mining).
    /// 12. **Menu / HUD** – either the main-menu overlay or remote-player
    ///     name labels, depending on `game_state`.
    /// 13. **Text pass** – all `glyphon` text areas (FPS counter, menu
    ///     labels, hotbar slot name, player name tags).
    /// 14. **Submit** – the completed command buffer is submitted and the
    ///     swap-chain texture is presented.
    ///
    /// # Errors
    /// Returns an error when the swap-chain texture cannot
    /// be acquired (e.g., the window is minimized or the surface is lost).
    /// The caller should handle `Lost` / `Outdated` by calling `resize`.
    pub fn render(&mut self) -> Result<(), RenderError> {
        // ── Acquire swap-chain texture ────────────────────────────────────── //
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            status => return Err(RenderError::Surface(status)),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let menu_visible = self.game_state != GameState::Playing;
        let world_has_loaded_chunks = menu_visible && !self.world.read().chunks.is_empty();
        let menu_uses_world_background =
            menu_visible && (self.has_entered_world || world_has_loaded_chunks);
        let render_world_scene = !menu_visible || menu_uses_world_background;

        // ── Remote player model buffers ───────────────────────────────────── //
        // All remote player meshes are concatenated into a single vertex/index
        // buffer pair that grows on demand (doubling strategy).  This avoids
        // per-player draw calls and keeps buffer management simple.
        if !self.remote_players.is_empty() && render_world_scene {
            let mut all_vertices = Vec::with_capacity(self.remote_players.len() * 16);
            let mut all_indices = Vec::with_capacity(self.remote_players.len() * 24);

            for (_id, player) in &self.remote_players {
                let (vertices, indices) =
                    build_player_model(player.x, player.y, player.z, player.yaw);
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                // Remap local indices to the combined buffer's address space.
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            self.player_model_num_indices = all_indices.len() as u32;

            if !all_vertices.is_empty() {
                let needed_verts = all_vertices.len() as u32;
                let needed_idxs = all_indices.len() as u32;

                // Grow the vertex buffer if it no longer fits all players.
                // New capacity = 2× required, minimum 256 vertices.
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
                // Same doubling strategy for the index buffer.
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
            // No remote players or we're in the menu – skip the draw later.
            self.player_model_num_indices = 0;
        }

        // ── Command encoder ───────────────────────────────────────────────── //
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // ── Camera & projection matrices ──────────────────────────────────── //
        let aspect = self.config.width as f32 / self.config.height as f32;
        // Extend the far plane beyond RENDER_DISTANCE so chunks at the horizon
        // are not clipped by the projection; 400 blocks is a sensible floor.
        let far_plane = (RENDER_DISTANCE as f32 * CHUNK_SIZE as f32 * 1.5).max(400.0);
        let proj = Mat4::perspective_rh(DEFAULT_FOV, aspect, 0.1, far_plane);
        let view_mat = self.camera.view_matrix();
        // `glam`'s RH projection helpers already use WebGPU's [0, 1] depth range.
        let view_proj = proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.to_cols_array_2d();

        // ── Day/night cycle ───────────────────────────────────────────────── //
        let time = self.game_start_time.elapsed().as_secs_f32();

        // Keep the speed in constants so the day cycle can be re-enabled
        // without changing render logic. It is currently fixed at noon.
        let day_cycle_speed = SUN_MOVEMENT_SPEED;
        // Offset by π/2 so the sun starts at noon (Y = +1) rather than
        // the horizon.
        let sun_angle = time * day_cycle_speed + std::f32::consts::FRAC_PI_2;
        let sun_x = 0.0;
        let sun_y = sun_angle.sin(); // +1 = overhead noon, −1 = midnight
        let sun_z = sun_angle.cos();
        let sun_dir = Vec3::new(sun_x, sun_y, sun_z).normalize();
        let moon_intensity = (-sun_dir.y).clamp(0.0, 1.0);

        // The moon is always opposite the sun direction.
        let moon_position = [-sun_dir.x, -sun_dir.y, -sun_dir.z];

        // Chunk coordinates of the camera, used to center the render window.
        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        self.rebuild_visible_chunk_cache(player_cx, player_cz);

        // Inverse view-projection is used by the composite / water shaders to
        // reconstruct world-space positions from screen-space depth samples.
        let inv_view_proj = view_proj.inverse();
        let inv_view_proj_array: [[f32; 4]; 4] = inv_view_proj.to_cols_array_2d();

        let eye_pos = self.camera.eye_position();
        let is_underwater = self.is_underwater;

        // ── Upload uniforms ───────────────────────────────────────────────── //
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                view_proj: view_proj_array,
                inv_view_proj: inv_view_proj_array,
                camera_pos: eye_pos.to_array(),
                time,
                sun_position: [sun_x, sun_y, sun_z],
                is_underwater,
                screen_size: [self.config.width as f32, self.config.height as f32],
                water_level: SEA_LEVEL as f32 - 1.0,
                reflection_mode: self.reflection_mode as f32,
                moon_position,
                _pad1_moon: 0.0,
                moon_intensity,
                wind_dir: [0.8, 0.6],
                wind_speed: 1.0,
                rain_factor: 0.0,
                sky_visibility: self.sky_visibility,
                menu_blur: if menu_visible { 1.0 } else { 0.0 },
                _pad_uniforms: 0.0,
            }]),
        );

        // ── Frustum planes (main camera) ──────────────────────────────────── //
        // Six half-space planes derived from the combined view-projection
        // matrix, used both for CPU-side mesh gating and the GPU cull shader.
        let frustum_planes = extract_frustum_planes(&view_proj);

        // ── Mesh rebuild requests ─────────────────────────────────────────── //
        // Walk all chunks within RENDER_DISTANCE.  For each sub-chunk whose
        // mesh is stale and not already being rebuilt on a worker thread,
        // queue a rebuild request.  Requests are sorted nearest-first so the
        // closest geometry always appears first.
        let mut meshes_to_request: Vec<(i32, i32, i32)> = Vec::new();
        let mut chunks_rendered = 0u32;
        let mut subchunks_rendered = 0u32;

        {
            let world = self.world.read();
            for &(cx, cz) in &self.visible_chunk_columns {
                if let Some(chunk) = world.chunks.get(&(cx, cz)) {
                    let mut chunk_has_visible = false;
                    for (sy, subchunk) in chunk.subchunks.iter().enumerate() {
                        if subchunk.is_empty {
                            continue; // skip fully empty sub-chunks early
                        }
                        if subchunk.mesh_dirty && !self.mesh_loader.is_pending(cx, cz, sy as i32) {
                            meshes_to_request.push((cx, cz, sy as i32));
                        }
                        if subchunk.num_quads > 0 || subchunk.num_water_quads > 0 {
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
        for (cx, cz, sy) in meshes_to_request.iter().take(MAX_MESH_BUILDS_PER_FRAME) {
            self.mesh_loader.request_mesh(*cx, *cz, *sy);
        }

        // ── Sky color interpolation ──────────────────────────────────────── //
        // Three anchor colors (day, sunset, night) are blended based on the
        // sun's Y component so the sky transitions smoothly through the day.
        let day_factor = sun_dir.y.max(0.0).min(1.0); // 1 at noon
        let night_factor = (-sun_dir.y).max(0.0).min(1.0); // 1 at midnight
        let sunset_factor = 1.0 - sun_dir.y.abs(); // 1 at horizon

        let day_sky = (0.53, 0.81, 0.98); // light blue
        let sunset_sky = (1.0, 0.5, 0.2); // orange
        let night_sky = (0.001, 0.001, 0.005); // near-black

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

        // ── Main camera GPU cull dispatch ─────────────────────────────────── //
        // The indirect manager's compute shader reads the Hi-Z texture and
        // frustum planes to populate per-chunk indirect draw arguments.
        let frustum_planes_array = frustum_planes_to_array(&frustum_planes);
        let hiz_size_f = [self.hiz_size[0] as f32, self.hiz_size[1] as f32];

        self.indirect_manager.dispatch_culling(
            &mut encoder,
            &self.queue,
            &self.water_indirect_manager,
            &view_proj,
            &frustum_planes_array,
            self.camera.position.into(),
            hiz_size_f,
            [self.config.width as f32, self.config.height as f32],
        );

        // ── Opaque pass ───────────────────────────────────────────────────── //
        // Renders: sky dome → terrain chunks → remote player models → sun/moon.
        // Writes to the 4× MSAA color target which is resolved simultaneously
        // into `ssr_color_view` (used by the water pass for reflections).
        {
            let mut opaque_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Opaque Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    // Resolve MSAA into the SSR color target so the water
                    // shader can sample the opaque scene for reflections.
                    resolve_target: Some(&self.ssr_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Clear to the sky color computed above.
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
                        // This is the only terrain rasterization. Its depth is
                        // resolved into Hi-Z after the pass for next-frame culling.
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // --- Sky dome ---
            // Uses LessEqual depth compare so it renders at depth 1.0 without
            // being clipped, and the same quad geometry as the sun billboard.
            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

            // --- Terrain chunks (indirect) ---
            // `multi_draw_indirect[_count]` emits one draw call per
            // visible chunk; the GPU cull pass already filtered the list.
            opaque_pass.set_pipeline(&self.render_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_bind_group(1, &self.terrain_quad_bind_group, &[]);
            if self.supports_indirect_count {
                opaque_pass.multi_draw_indirect_count(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.visible_count_buffer(),
                    0,
                    self.indirect_manager.active_count(),
                );
            } else {
                opaque_pass.multi_draw_indirect(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.active_count(),
                );
            }

            // --- Remote player models ---
            // Drawn with the terrain pipeline so lighting and fog stay
            // consistent with the surrounding world geometry.
            if self.player_model_num_indices > 0 {
                if let (Some(vb), Some(ib)) = (
                    &self.player_model_vertex_buffer,
                    &self.player_model_index_buffer,
                ) {
                    opaque_pass.set_pipeline(&self.player_model_pipeline);
                    opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    opaque_pass.set_vertex_buffer(0, vb.slice(..));
                    opaque_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    opaque_pass.draw_indexed(0..self.player_model_num_indices, 0, 0..1);
                }
            }

            // --- Sun / moon billboard ---
            // No depth write; depth test enabled so the disc is occluded by
            // terrain on the horizon.
            opaque_pass.set_pipeline(&self.sun_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

        // ── Depth resolve + Hi-Z generation ───────────────────────────────── //
        // The opaque pass above produced the depth buffer, eliminating the
        // former terrain depth prepass. Culling at the start of this frame reads
        // the previous pyramid; this freshly built pyramid is consumed next frame.
        if render_world_scene {
            let mut depth_resolve_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Depth Resolve Compute Pass"),
                timestamp_writes: None,
            });
            depth_resolve_pass.set_pipeline(&self.depth_resolve_pipeline);
            depth_resolve_pass.set_bind_group(0, &self.depth_resolve_bind_group, &[]);
            depth_resolve_pass.dispatch_workgroups(
                (self.config.width + 15) / 16,
                (self.config.height + 15) / 16,
                1,
            );
            drop(depth_resolve_pass);

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
        }

        // ── Transparent (water) pass ──────────────────────────────────────── //
        // Loads (does not clear) the existing MSAA color and depth buffers so
        // water is composited on top of the opaque scene.  Resolves into
        // `scene_color_view` for the composite pass.
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(&self.scene_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // keep opaque scene color
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, // keep opaque depth for z-test
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            transparent_pass.set_pipeline(&self.water_pipeline);
            transparent_pass.set_bind_group(0, &self.water_bind_group, &[]);
            transparent_pass.set_bind_group(1, &self.water_quad_bind_group, &[]);
            if self.supports_indirect_count {
                transparent_pass.multi_draw_indirect_count(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.visible_count_buffer(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            } else {
                transparent_pass.multi_draw_indirect(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            }
        }

        // ── Block outline pass ───────────────────────────────────────────── //
        // Draw the targeted block outline before the composite pass so the
        // resolved scene color includes the visible edges. The pass uses the
        // MSAA color target and the main depth buffer so hidden edges are
        // rejected by depth testing instead of being painted over the scene.
        if self.game_state == GameState::Playing {
            let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Outline Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(&self.scene_color_view),
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
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let Some((bx, by, bz)) = self.highlighted_block {
                let visible_faces = {
                    let world = self.world.read();
                    visible_outline_faces(&*world, bx, by, bz)
                };
                let (outline_vertices, outline_indices) =
                    build_block_outline(bx, by, bz, visible_faces);
                if !outline_vertices.is_empty() && !outline_indices.is_empty() {
                    let outline_vb =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Block Outline VB"),
                                contents: bytemuck::cast_slice(&outline_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let outline_ib =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Block Outline IB"),
                                contents: bytemuck::cast_slice(&outline_indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                    outline_pass.set_pipeline(&self.outline_pipeline);
                    outline_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    outline_pass.set_vertex_buffer(0, outline_vb.slice(..));
                    outline_pass.set_index_buffer(outline_ib.slice(..), wgpu::IndexFormat::Uint32);
                    outline_pass.draw_indexed(0..outline_indices.len() as u32, 0, 0..1);
                }
            }
        }

        // ── Composite pass (post-processing blit) ─────────────────────────── //
        // Reads from `scene_color_view` (the fully composited opaque + water
        // scene) and writes the post-processed result directly to the
        // swap-chain surface.  The composite shader handles underwater fog
        // color grading, vignette, and similar full-screen effects.
        {
            let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, // write directly to the swap-chain image
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, // no depth test for a full-screen blit
                ..Default::default()
            });

            composite_pass.set_pipeline(&self.composite_pipeline);
            let composite_bind_group = if menu_visible && !menu_uses_world_background {
                &self.menu_composite_bind_group
            } else {
                &self.composite_bind_group
            };
            composite_pass.set_bind_group(0, composite_bind_group, &[]);
            composite_pass.draw(0..3, 0..1); // full-screen triangle
        }

        // ── UI pass ───────────────────────────────────────────────────────── //
        // Draws the crosshair, coordinate debug overlay, and hotbar using the
        // same `crosshair_pipeline` (alpha-blended, no depth test).  All
        // elements are drawn directly onto the swap-chain surface on top of
        // the composited scene.
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // keep the composited scene
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            ui_pass.set_pipeline(&self.crosshair_pipeline);
            ui_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            if menu_visible {
                if self.menu_state.server_address_input_visible {
                    let layout = MenuLayout::new(self.config.width, self.config.height);
                    let focused = self.menu_state.selected_field == MenuField::ServerAddress;
                    let selected = focused && self.menu_state.selected_all;
                    let (vertices, indices) = build_menu_input_box(
                        layout.server_address_input,
                        self.config.width as f32,
                        self.config.height as f32,
                        focused,
                        selected,
                    );
                    let input_vb =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Menu Server Address Input VB"),
                                contents: bytemuck::cast_slice(&vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let input_ib =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Menu Server Address Input IB"),
                                contents: bytemuck::cast_slice(&indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    ui_pass.set_vertex_buffer(0, input_vb.slice(..));
                    ui_pass.set_index_buffer(input_ib.slice(..), wgpu::IndexFormat::Uint32);
                    ui_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                }
            } else {
                // --- Crosshair ---
                if self.show_crosshair {
                    ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
                    ui_pass.set_index_buffer(
                        self.crosshair_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    ui_pass.draw_indexed(0..self.num_crosshair_indices, 0, 0..1);
                }

                // --- Coordinate debug overlay ---
                // Only drawn when `coords_vertex_buffer` has been populated.
                if let (Some(vb), Some(ib)) =
                    (&self.coords_vertex_buffer, &self.coords_index_buffer)
                {
                    if self.coords_num_indices > 0 {
                        ui_pass.set_vertex_buffer(0, vb.slice(..));
                        ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        ui_pass.draw_indexed(0..self.coords_num_indices, 0, 0..1);
                    }
                }

                // --- Hotbar ---
                if self.show_crosshair {
                    if self.hotbar_dirty || self.hotbar_vertex_buffer.is_none() {
                        let aspect = self.config.width as f32 / self.config.height as f32;
                        let (vb, ib, count) =
                            crate::ui::ui::build_hotbar(&self.device, self.hotbar_slot, aspect);
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
        }

        // ── Block-breaking progress bar ───────────────────────────────────── //
        // A two-quad horizontal bar (background + foreground) displayed just
        // below the crosshair while the player is mining a block.
        // The color interpolates from red (0%) through yellow to green (100%).
        if self.digging.target.is_some() && self.digging.break_time > 0.0 {
            let progress = (self.digging.progress / self.digging.break_time).min(1.0);

            // Bar dimensions in NDC space (centred horizontally, slightly
            // below the crosshair at y = −0.05).
            let bar_width = 0.15;
            let bar_height = 0.015;
            let bar_y = -0.05;

            let bg_color = [0.2, 0.2, 0.2];
            // Color shifts from red (0%) → yellow (50%) → green (100%).
            let prog_color = [1.0 - progress, progress, 0.0];
            let normal_idx = Vertex::pack_normal([0.0, 0.0, 1.0]);

            // Background quad (full-width gray bar).
            let mut vertices = Vec::with_capacity(8);
            for (i, (x, y)) in [
                (-bar_width, bar_y - bar_height),
                (bar_width, bar_y - bar_height),
                (bar_width, bar_y + bar_height),
                (-bar_width, bar_y + bar_height),
            ]
            .into_iter()
            .enumerate()
            {
                vertices.push(Vertex {
                    position: [x, y, 0.0],
                    packed: Vertex::pack_ui(
                        normal_idx,
                        [bg_color[0], bg_color[1], bg_color[2], 1.0],
                        0,
                        i as u8,
                    ),
                });
            }

            // Foreground quad (colored fill, inset by 0.005/0.003 on each
            // side so the gray border remains visible all around).
            let prog_width = bar_width * 2.0 * progress - bar_width;
            let fg_corners = [
                (-bar_width + 0.005, bar_y - bar_height + 0.003),
                (prog_width - 0.005, bar_y - bar_height + 0.003),
                (prog_width - 0.005, bar_y + bar_height - 0.003),
                (-bar_width + 0.005, bar_y + bar_height - 0.003),
            ];
            for (i, (x, y)) in fg_corners.into_iter().enumerate() {
                vertices.push(Vertex {
                    position: [x, y, 0.0],
                    packed: Vertex::pack_ui(
                        normal_idx,
                        [prog_color[0], prog_color[1], prog_color[2], 1.0],
                        0,
                        i as u8,
                    ),
                });
            }

            // Indices for two quads (bg = 0-3, fg = 4-7).
            let indices: [u32; 12] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

            // Lazy-create the vertex buffer on first use; update it every
            // frame thereafter because the progress value changes continuously.
            if self.progress_bar_vertex_buffer.is_none() {
                self.progress_bar_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar VB"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                // Index buffer is constant (same two-quad layout every frame).
                self.progress_bar_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar IB"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
            } else {
                // Buffer already exists – overwrite only the vertex data.
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

        // ── Menu overlay or remote player labels ──────────────────────────── //
        if menu_visible {
            self.render_menu(&mut encoder, &view);
        } else {
            // Projects each remote player's world position into screen space
            // so the text pass can draw their name above their head.
            self.render_remote_players(
                &view_proj,
                self.config.width as f32,
                self.config.height as f32,
            );
        }

        // ── Text pass (glyphon) ───────────────────────────────────────────── //
        // All on-screen text is batched into a single `TextRenderer::prepare`
        // call and rendered in one pass.  The individual `glyphon::Buffer`
        // objects are updated lazily (only when the underlying text changes)
        // to avoid redundant re-shaping work.
        {
            // ---- FPS counter (in-game only) ----
            if !menu_visible && self.show_debug_overlay {
                let fps_text = format!(
                    "FPS: {:.0}\nFrame: {:.2} ms\nCPU time avg: {:.2} ms\nChunks: {}\nSubchunks: {}",
                    self.current_fps,
                    self.frame_time_ms,
                    self.cpu_update_ms,
                    self.chunks_rendered,
                    self.subchunks_rendered
                );
                self.fps_buffer.set_text(
                    &fps_text,
                    &Attrs::new().family(Family::SansSerif),
                    Shaping::Advanced,
                    None,
                );
                self.fps_buffer
                    .shape_until_scroll(&mut self.font_system, false);
                self.fps_buffer.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
            }

            // ---- Hotbar slot label (in-game only, updated on slot change) ----
            if !menu_visible && self.last_hotbar_slot != self.hotbar_slot {
                let block = crate::ui::ui::HOTBAR_SLOTS[self.hotbar_slot];
                let label = block.display_name();
                self.hotbar_label_buffer.set_text(
                    label,
                    &Attrs::new()
                        .family(Family::SansSerif)
                        .color(Color::rgb(255, 238, 200)),
                    Shaping::Advanced,
                    None,
                );
                self.hotbar_label_buffer
                    .shape_until_scroll(&mut self.font_system, false);
                self.hotbar_label_buffer.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                // Approximate pixel width for centring the label above the
                // hotbar.  0.6 × font_size is a reasonable estimate for the
                // average glyph advance of sans-serif digits and Latin text.
                let font_size = 22.0;
                let char_width = font_size * 0.6;
                self.hotbar_label_width = label.chars().count() as f32 * char_width;
                self.last_hotbar_slot = self.hotbar_slot;
            }

            // ---- Remote player name labels / menu text ----
            // In menu mode: update all menu label buffers via `prepare_menu_text`.
            // In game mode: project remote player positions and grow the label
            // buffer pool as needed (one `glyphon::Buffer` per player).
            let labels = if menu_visible {
                self.prepare_menu_text();
                Vec::new() // menu text is rendered through dedicated buffers
            } else {
                let labels = queue_remote_players_labels(
                    &self.remote_players,
                    &view_proj,
                    self.config.width as f32,
                    self.config.height as f32,
                );
                // Grow the buffer pool lazily so we always have at least as
                // many buffers as there are visible remote players.
                while self.player_label_buffers.len() < labels.len() {
                    self.player_label_buffers.push(glyphon::Buffer::new(
                        &mut self.font_system,
                        Metrics::new(24.0, 32.0),
                    ));
                }
                for (i, label) in labels.iter().enumerate() {
                    let buffer = &mut self.player_label_buffers[i];
                    buffer.set_text(
                        &label.username,
                        &Attrs::new()
                            .family(Family::SansSerif)
                            .color(Color::rgb(76, 255, 76)), // bright green name tags
                        Shaping::Advanced,
                        None,
                    );
                    buffer.shape_until_scroll(&mut self.font_system, false);
                    buffer.set_size(
                        Some(self.config.width as f32),
                        Some(self.config.height as f32),
                    );
                }
                labels
            };

            // ---- Assemble TextArea list ----
            // Each `TextArea` pairs a `glyphon::Buffer` with its screen
            // position, clipping bounds, and default color.
            let mut text_areas = Vec::with_capacity(4);

            if !menu_visible && self.show_debug_overlay {
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
            }

            if menu_visible {
                let layout = MenuLayout::new(self.config.width, self.config.height);
                let hovered = self
                    .cursor_position
                    .and_then(|(x, y)| layout.hit_test(x, y));
                let new_world_hovered = matches!(hovered, Some(MenuHit::NewWorld));
                let multiplayer_hovered = matches!(hovered, Some(MenuHit::Multiplayer));
                let render_mode_hovered = matches!(hovered, Some(MenuHit::RenderMode));
                let hover_text_color = Color::rgb(255, 255, 255);
                let menu_text_top_offset = 3.0;
                let menu_text_bounds = TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                };
                let new_world_color = if new_world_hovered {
                    hover_text_color
                } else {
                    Color::rgb(238, 241, 236)
                };
                let multiplayer_color = if multiplayer_hovered {
                    hover_text_color
                } else {
                    Color::rgb(211, 226, 238)
                };
                let render_mode_color = if render_mode_hovered {
                    hover_text_color
                } else {
                    Color::rgb(184, 205, 220)
                };

                text_areas.push(TextArea {
                    buffer: &self.menu_render_mode_button_buffer,
                    left: layout.render_mode_text.x,
                    top: layout.render_mode_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: render_mode_color,
                    custom_glyphs: &[],
                });

                text_areas.push(TextArea {
                    buffer: &self.menu_singleplayer_button_buffer,
                    left: layout.new_world_text.x,
                    top: layout.new_world_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: new_world_color,
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_connect_button_buffer,
                    left: layout.multiplayer_text.x,
                    top: layout.multiplayer_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: multiplayer_color,
                    custom_glyphs: &[],
                });
                if self.menu_state.server_address_input_visible {
                    let input = layout.server_address_input;
                    let input_focused = self.menu_state.selected_field == MenuField::ServerAddress;
                    let input_text_color = if input_focused {
                        Color::rgb(255, 255, 255)
                    } else {
                        Color::rgb(215, 230, 238)
                    };

                    text_areas.push(TextArea {
                        buffer: &self.menu_server_address_input_buffer,
                        left: input.x + 12.0,
                        top: input.y + 8.0,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: input.x as i32,
                            top: input.y as i32,
                            right: (input.x + input.w) as i32,
                            bottom: (input.y + input.h) as i32,
                        },
                        default_color: input_text_color,
                        custom_glyphs: &[],
                    });
                }
            } else {
                // ---- In-game HUD text ----

                if self.show_crosshair {
                    // Hotbar slot name: centred above the hotbar, clamped to the
                    // screen width.
                    let label_width = self.hotbar_label_width.min(self.config.width as f32);
                    let label_left = (self.config.width as f32 - label_width) * 0.5;
                    // 170 px above the bottom edge keeps the label above the hotbar.
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
                }

                // Remote player name tags (one per visible player).
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

            // Upload shaped glyph data and rasterize new glyphs into the atlas.
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
                    log(LogLevel::Error, &format!("Failed to prepare text: {:?}", e));
                    RenderError::Text
                })?;

            // Render all glyphs in a single pass on top of everything else.
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
                    log(LogLevel::Error, &format!("Failed to render text: {:?}", e));
                    RenderError::Text
                })?;
        }

        // ── Submit & present ──────────────────────────────────────────────── //
        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);
        Ok(())
    }

    /// Updates the main-menu labels and the optional server-address input.
    pub fn prepare_menu_text(&mut self) {
        let layout = MenuLayout::new(self.config.width, self.config.height);

        self.menu_connect_button_buffer.set_text(
            "MULTIPLAYER",
            &Attrs::new().family(Family::Name("Google Sans")),
            Shaping::Advanced,
            None,
        );
        self.menu_connect_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_connect_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_singleplayer_button_buffer.set_text(
            "NEW WORLD",
            &Attrs::new().family(Family::Name("Google Sans")),
            Shaping::Advanced,
            None,
        );
        self.menu_singleplayer_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_singleplayer_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        let render_mode_text = match self.config.present_mode {
            wgpu::PresentMode::Fifo => "RENDER MODE: VSYNC",
            _ => "RENDER MODE: INSTANT",
        };
        self.menu_render_mode_button_buffer.set_text(
            render_mode_text,
            &Attrs::new().family(Family::Name("Google Sans")),
            Shaping::Advanced,
            None,
        );
        self.menu_render_mode_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_render_mode_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        let show_cursor = self.menu_state.selected_field == MenuField::ServerAddress
            && !self.menu_state.selected_all;
        let server_address_text = if show_cursor {
            format!("{}_", self.menu_state.server_address)
        } else {
            self.menu_state.server_address.clone()
        };
        self.menu_server_address_input_buffer.set_text(
            &server_address_text,
            &Attrs::new().family(Family::Name("Google Sans")),
            Shaping::Advanced,
            None,
        );
        self.menu_server_address_input_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_server_address_input_buffer.set_size(
            Some((layout.server_address_input.w - 24.0).max(32.0)),
            Some(layout.server_address_input.h),
        );
    }

    /// Menu blur is applied in the composite shader; menu labels are drawn in the text pass.
    pub fn render_menu(&mut self, _encoder: &mut wgpu::CommandEncoder, _view: &wgpu::TextureView) {}

    /// Projects remote player world positions into screen space for name-tag
    /// rendering.
    ///
    /// Currently a stub; the actual projection logic is handled by
    /// [`queue_remote_players_labels`] in the multiplayer player module and
    /// the results are consumed directly in `render`.  This method exists as
    /// a hook for future per-player rendering work (e.g., health bars, custom
    /// skins) that would require a dedicated render pass rather than a text
    /// overlay.
    ///
    /// # Parameters
    /// - `_view_proj` – Combined view-projection matrix (unused by the stub).
    /// - `_width`     – Surface width in pixels (unused by the stub).
    /// - `_height`    – Surface height in pixels (unused by the stub).
    pub fn render_remote_players(&mut self, _view_proj: &glam::Mat4, _width: f32, _height: f32) {}
}
