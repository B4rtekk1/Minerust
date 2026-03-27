use glam::{Mat4, Vec3};
use glyphon::{Attrs, Color, Family, Metrics, Shaping, TextArea, TextBounds};
use wgpu::util::DeviceExt;

use minerust::{
    BlockType, CHUNK_SIZE, DEFAULT_FOV, RENDER_DISTANCE, SEA_LEVEL, Uniforms, Vertex, World,
    build_block_outline, build_player_model, extract_frustum_planes,
};

use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::{GameState, MenuField, MenuLayout, Rect};
use crate::logger::{log, LogLevel};

use super::init::OPENGL_TO_WGPU_MATRIX;
use super::init::frustum_planes_to_array;
use super::state::State;

// ─────────────────────────────────────────────────────────────────────────────
// NDC conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Converts a horizontal pixel coordinate to Normalised Device Coordinates.
///
/// wgpu's NDC X axis runs from −1.0 (left edge) to +1.0 (right edge).
///
/// # Parameters
/// - `x`     – Pixel coordinate, origin at the left edge of the window.
/// - `width` – Current surface width in physical pixels.
fn px_to_ndc_x(x: f32, width: f32) -> f32 {
    (x / width) * 2.0 - 1.0
}

/// Converts a vertical pixel coordinate to Normalised Device Coordinates.
///
/// wgpu's NDC Y axis runs from +1.0 (top) to −1.0 (bottom), which is the
/// opposite of the typical screen-space convention where Y increases downward.
///
/// # Parameters
/// - `y`      – Pixel coordinate, origin at the top edge of the window.
/// - `height` – Current surface height in physical pixels.
fn px_to_ndc_y(y: f32, height: f32) -> f32 {
    1.0 - (y / height) * 2.0
}

/// Appends a screen-space rectangle to shared vertex and index lists.
///
/// The rectangle is specified in pixel space (origin = top-left corner of the
/// window) and is converted to NDC internally.  Four vertices and two
/// triangles (six indices) are appended; the index base is derived from the
/// current length of `vertices` so that multiple rectangles can share the
/// same buffers.
///
/// # Parameters
/// - `vertices` – Target vertex list (extended in-place).
/// - `indices`  – Target index list (extended in-place).
/// - `rect`     – Position and size in pixel space.
/// - `color`    – Pre-packed RGBA color produced by [`rgba`].
/// - `width`    – Surface width used for the NDC conversion.
/// - `height`   – Surface height used for the NDC conversion.
fn push_rect(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    rect: Rect,
    color: [f32; 4],
    width: f32,
    height: f32,
) {
    let base = vertices.len() as u32;
    let x0 = px_to_ndc_x(rect.x, width);
    let y0 = px_to_ndc_y(rect.y, height);
    let x1 = px_to_ndc_x(rect.x + rect.w, width);
    let y1 = px_to_ndc_y(rect.y + rect.h, height);
    let normal_idx = Vertex::pack_normal([0.0, 0.0, 1.0]);

    // Top-left → top-right → bottom-right → bottom-left (corner_idx 0..3)
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    for (i, &(x, y)) in corners.iter().enumerate() {
        vertices.push(Vertex {
            position: [x, y, 0.0],
            packed: Vertex::pack_ui(normal_idx, color, 0, i as u8),
        });
    }

    // Two counter-clockwise triangles covering the quad.
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
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

impl State {
    /// Produces one complete frame and presents it to the OS window.
    ///
    /// # Render pipeline overview
    ///
    /// The frame is built from the following render / compute passes in order:
    ///
    /// 1. **Player model update** – re-builds the combined vertex/index buffers
    ///    for all visible remote players if any exist.
    /// 2. **Uniform upload** – computes the camera matrices, advances the day
    ///    cycle, updates CSM cascades, and uploads the `Uniforms` struct.
    /// 3. **Shadow cull + shadow passes** (×`active_cascades`) – each cascade
    ///    runs a GPU culling dispatch followed by a depth-only draw into its
    ///    shadow map layer.
    /// 4. **Mesh request** – walks the visible chunk grid, queues dirty sub-chunk
    ///    meshes for background rebuild, and tallies rendered counts.
    /// 5. **Main cull dispatch** – GPU frustum + Hi-Z occlusion cull for both
    ///    the opaque terrain and water indirect managers.
    /// 6. **Opaque pass** – sky dome → terrain → remote player models → sun/moon.
    ///    Resolves MSAA into `ssr_color_view` for later water reflections.
    /// 7. **Depth resolve compute** – resolves the multisampled depth buffer
    ///    into `ssr_depth_view` (for water refraction) and the first Hi-Z mip
    ///    level (for next-frame occlusion culling).
    /// 8. **Hi-Z generation** (compute) – downsamples the depth mip chain.
    /// 9. **Transparent pass** – water surfaces, alpha-blended on top of the
    ///    opaque result.  Resolves MSAA into `scene_color_view`.
    /// 10. **Composite pass** – post-processing blit from `scene_color_view`
    ///     to the swap-chain surface (underwater fog, vignette, etc.).
    /// 11. **UI pass** – crosshair, coordinate debug overlay, hotbar.
    /// 12. **Progress bar pass** – block-breaking progress indicator (only
    ///     when the player is actively mining).
    /// 13. **Menu / HUD** – either the main-menu overlay or remote-player
    ///     name labels, depending on `game_state`.
    /// 14. **Text pass** – all `glyphon` text areas (FPS counter, menu
    ///     labels, hotbar slot name, player name tags).
    /// 15. **Submit** – the completed command buffer is submitted and the
    ///     swap-chain texture is presented.
    ///
    /// # Errors
    /// Returns `Err(wgpu::SurfaceError)` when the swap-chain texture cannot
    /// be acquired (e.g., the window is minimized or the surface is lost).
    /// The caller should handle `Lost` / `Outdated` by calling `resize`.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // ── Acquire swap-chain texture ────────────────────────────────────── //
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // ── Remote player model buffers ───────────────────────────────────── //
        // All remote player meshes are concatenated into a single vertex/index
        // buffer pair that grows on demand (doubling strategy).  This avoids
        // per-player draw calls and keeps buffer management simple.
        if !self.remote_players.is_empty() && self.game_state != GameState::Menu {
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
        // Combine projection, view, and the OpenGL→wgpu NDC correction into
        // one matrix uploaded to the GPU once per frame.
        let view_proj = OPENGL_TO_WGPU_MATRIX * proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.to_cols_array_2d();

        // ── Day/night cycle ───────────────────────────────────────────────── //
        let time = self.game_start_time.elapsed().as_secs_f32();

        // `day_cycle_speed` controls how fast the sun orbits.  At 0.005 rad/s
        // a full day takes ~1257 seconds (≈21 minutes).
        let day_cycle_speed = 0.005;
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

        // ── CSM (Cascaded Shadow Maps) update ─────────────────────────────── //
        // `CsmManager::update` computes the four tight orthographic light-space
        // matrices that cover successive depth ranges of the camera frustum.
        let csm = &mut self.csm;
        let fov_y = DEFAULT_FOV;
        csm.update(&view_mat, sun_dir, 0.1, 300.0, aspect, fov_y);

        // Pack cascade view-projection matrices into the uniform struct format.
        let csm_view_proj: [[[f32; 4]; 4]; 4] = [
            csm.cascades[0].view_proj.to_cols_array_2d(),
            csm.cascades[1].view_proj.to_cols_array_2d(),
            csm.cascades[2].view_proj.to_cols_array_2d(),
            csm.cascades[3].view_proj.to_cols_array_2d(),
        ];
        // Split distances tell the terrain shader which cascade to sample for
        // a given fragment based on its camera-space depth.
        let csm_split_distances: [f32; 4] = [
            csm.cascades[0].split_distance,
            csm.cascades[1].split_distance,
            csm.cascades[2].split_distance,
            csm.cascades[3].split_distance,
        ];

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
                csm_view_proj,
                csm_split_distances,
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
                _pad: 0.0,
                rain_factor: 0.0,
            }]),
        );

        // ── Frustum planes (main camera) ──────────────────────────────────── //
        // Six half-space planes derived from the combined view-projection
        // matrix, used both for CPU-side mesh gating and the GPU cull shader.
        let frustum_planes = extract_frustum_planes(&view_proj);

        // Chunk coordinates of the camera, used to center the render window.
        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        // Fewer cascades are needed at short render distances because the far
        // splits collapse below useful thresholds.
        let active_cascades = minerust::get_active_cascade_count(RENDER_DISTANCE);

        // ── Shadow cascade buffer upload + shadow cull ────────────────────── //
        let mut shadow_frustum_arrays = [[[0f32; 4]; 6]; 4];
        for i in 0..active_cascades {
            // Pack the cascade's light-space matrix into a 256-byte aligned
            // uniform slot so the dynamic-offset shadow bind group can select
            // the correct cascade without rebinding.
            let cascade_matrix: [[f32; 4]; 4] = csm.cascades[i].view_proj.to_cols_array_2d();
            let mut shadow_uniform_data = [0f32; 64]; // 64 × 4 bytes = 256 bytes
            shadow_uniform_data[0..16].copy_from_slice(cascade_matrix.as_flattened());

            self.queue.write_buffer(
                &self.shadow_cascade_buffer,
                (i * 256) as u64,
                bytemuck::cast_slice(&shadow_uniform_data),
            );

            // Extract the light-space frustum planes for this cascade so the
            // GPU can cull chunks that are outside the cascade's projection.
            let cascade_view_proj = csm.cascades[i].view_proj;
            let shadow_frustum = extract_frustum_planes(&cascade_view_proj);
            shadow_frustum_arrays[i] = frustum_planes_to_array(&shadow_frustum);
        }

        // Dispatch GPU occlusion + frustum culling for each active cascade,
        // for both opaque terrain and water chunks.
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

        // ── Shadow depth passes (one per active cascade) ──────────────────── //
        // Each pass renders opaque terrain into one layer of the shadow map
        // array using the corresponding light-space matrix.  The fragment
        // shader is absent; only depth values are written.
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
                color_attachments: &[], // depth-only pass, no color output
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_cascade_views[i],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // clear to max depth
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            shadow_pass.set_pipeline(&self.shadow_pipeline);
            // Dynamic offset selects cascade i's light-space matrix in the
            // 256-byte-aligned shadow cascade buffer.
            shadow_pass.set_bind_group(0, &self.shadow_bind_group, &[offset]);
            shadow_pass.set_vertex_buffer(0, self.indirect_manager.vertex_buffer().slice(..));
            shadow_pass.set_index_buffer(
                self.indirect_manager.index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            // Use count-based indirect if supported so only GPU-visible chunks
            // are drawn; fall back to a fixed count otherwise.
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
            for cx in (player_cx - RENDER_DISTANCE)..=(player_cx + RENDER_DISTANCE) {
                for cz in (player_cz - RENDER_DISTANCE)..=(player_cz + RENDER_DISTANCE) {
                    if let Some(chunk) = world.chunks.get(&(cx, cz)) {
                        let mut chunk_has_visible = false;
                        for (sy, subchunk) in chunk.subchunks.iter().enumerate() {
                            if subchunk.is_empty {
                                continue; // skip fully empty sub-chunks early
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

        // Nearest-first ordering ensures that the player sees close geometry
        // rebuild quickly after entering a new area.
        meshes_to_request.sort_by_key(|&(cx, cz, _sy)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });
        for (cx, cz, sy) in &meshes_to_request {
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

        // ── Opaque pass ───────────────────────────────────────────────────── //
        // Renders: sky dome → terrain chunks → remote player models → sun/moon.
        // Writes to the 4× MSAA color target which is resolved simultaneously
        // into `ssr_color_view` (used by the water pass for reflections).
        {
            let opaque_resolve_target = if self.game_state == GameState::Menu {
                &view
            } else {
                &self.ssr_color_view
            };

            let mut opaque_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Opaque Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    // Resolve MSAA into the SSR color target so the water
                    // shader can sample the opaque scene for reflections.
                    resolve_target: Some(opaque_resolve_target),
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
                        load: wgpu::LoadOp::Clear(1.0), // 1.0 = max depth
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
            opaque_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            opaque_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            opaque_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

            // --- Terrain chunks (indirect) ---
            // `multi_draw_indexed_indirect[_count]` emits one draw call per
            // visible chunk; the GPU cull pass already filtered the list.
            opaque_pass.set_pipeline(&self.render_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            opaque_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            opaque_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
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

            // --- Remote player models ---
            // Drawn with the terrain pipeline so they receive shadow and fog
            // effects consistent with the surrounding world geometry.
            if self.player_model_num_indices > 0 {
                if let (Some(vb), Some(ib)) = (
                    &self.player_model_vertex_buffer,
                    &self.player_model_index_buffer,
                ) {
                    opaque_pass.set_pipeline(&self.render_pipeline);
                    opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    opaque_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
                    opaque_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
                    opaque_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
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
            opaque_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            opaque_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            opaque_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

        // ── Depth resolve compute pass ───────────────────────────────────── //
        // Resolve the MSAA depth buffer into two single-sampled outputs:
        //   • `hiz_mips[0]`    – conservative max-depth seed for Hi-Z
        //   • `ssr_depth_view` – closest-depth copy for water refraction
        {
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
        }

        // ── Hi-Z mip chain generation (compute) ───────────────────────────── //
        // Downsample the depth texture from mip 0 down to 1×1 using the
        // conservative max-depth rule: each output texel stores the maximum
        // depth of the 2×2 source block, ensuring that a chunk whose AABB
        // projects entirely within one mip texel is only passed if there is
        // guaranteed to be no closer geometry in that region.
        for i in 0..self.hiz_bind_groups.len() {
            let mut hiz_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hi-Z Generation Pass Level"),
                timestamp_writes: None,
            });
            hiz_pass.set_pipeline(&self.hiz_pipeline);
            hiz_pass.set_bind_group(0, &self.hiz_bind_groups[i], &[]);
            // Each workgroup covers a 16×16 tile; dispatch enough groups to
            // cover the entire mip level even if its dimensions are not
            // multiples of 16.
            let div = 1 << (i + 1);
            let mip_width = (self.hiz_size[0] / div).max(1);
            let mip_height = (self.hiz_size[1] / div).max(1);
            hiz_pass.dispatch_workgroups((mip_width + 15) / 16, (mip_height + 15) / 16, 1);
        }

        // ── Transparent (water) pass ──────────────────────────────────────── //
        // Loads (does not clear) the existing MSAA color and depth buffers so
        // water is composited on top of the opaque scene.  Resolves into
        // `scene_color_view` for the composite pass.
        let resolve_target = &self.scene_color_view;

        if self.game_state != GameState::Menu {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(resolve_target), // → scene_color_view
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

        // ── Block outline pass ───────────────────────────────────────────── //
        // Draw the targeted block outline before the composite pass so the
        // resolved scene color includes the visible edges. The pass uses the
        // MSAA color target and the main depth buffer so hidden edges are
        // rejected by depth testing instead of being painted over the scene.
        if self.game_state != GameState::Menu {
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
                    outline_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
                    outline_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
                    outline_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
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
        if self.game_state != GameState::Menu {
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
            composite_pass.set_bind_group(0, &self.composite_bind_group, &[]);
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

            // --- Crosshair ---
            ui_pass.set_pipeline(&self.crosshair_pipeline);
            ui_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            ui_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            ui_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            ui_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
            ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
            ui_pass.set_index_buffer(
                self.crosshair_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            ui_pass.draw_indexed(0..self.num_crosshair_indices, 0, 0..1);

            // --- Coordinate debug overlay ---
            // Only drawn when `coords_vertex_buffer` has been populated (i.e.,
            // when the player has moved to a new chunk and the overlay was
            // rebuilt by `update`).
            if let (Some(vb), Some(ib)) = (&self.coords_vertex_buffer, &self.coords_index_buffer) {
                if self.coords_num_indices > 0 {
                    ui_pass.set_vertex_buffer(0, vb.slice(..));
                    ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    ui_pass.draw_indexed(0..self.coords_num_indices, 0, 0..1);
                }
            }

            // --- Hotbar ---
            // Only drawn in-game (not on the menu).  Rebuilt lazily when
            // `hotbar_dirty` is true (e.g., after a slot change).
            if self.game_state != GameState::Menu {
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
            for (i, (x, y)) in [(-bar_width, bar_y - bar_height), (bar_width, bar_y - bar_height), (bar_width, bar_y + bar_height), (-bar_width, bar_y + bar_height)].into_iter().enumerate() {
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
            progress_pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            progress_pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            progress_pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
            progress_pass.set_vertex_buffer(0, progress_vb.slice(..));
            progress_pass.set_index_buffer(progress_ib.slice(..), wgpu::IndexFormat::Uint32);
            progress_pass.draw_indexed(0..12, 0, 0..1);
        }

        // ── Menu overlay or remote player labels ──────────────────────────── //
        if self.game_state == GameState::Menu {
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
            // ---- FPS counter (always visible) ----
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

            // ---- Hotbar slot label (in-game only, updated on slot change) ----
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
            let labels = if self.game_state == GameState::Menu {
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
                        &mut self.font_system,
                        &label.username,
                        &Attrs::new()
                            .family(Family::SansSerif)
                            .color(Color::rgb(76, 255, 76)), // bright green name tags
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

            // ---- Assemble TextArea list ----
            // Each `TextArea` pairs a `glyphon::Buffer` with its screen
            // position, clipping bounds, and default color.
            let mut text_areas = Vec::with_capacity(4);

            // FPS counter – top-left, always on top of all other UI.
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
                // ---- Menu text layout ----
                // `MenuLayout` computes all element rectangles from the current
                // surface size so menu text scales correctly at any resolution.
                let layout = MenuLayout::new(self.config.width, self.config.height);

                // Small offsets (+6, +56, etc.) fine-tune vertical alignment
                // within each panel so text sits inside its background rect
                // with comfortable padding.
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
                    default_color: Color::rgb(242, 227, 187), // warm gold
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
                    default_color: Color::rgb(186, 201, 214), // muted blue-grey
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

                // Buttons are centred by estimating the text width
                // (chars × ~10.5 px) and offsetting accordingly.
                let connect_estimate = 7.0 * 10.5; // "CONNECT" ≈ 7 chars
                let single_estimate = 12.0 * 10.5; // "SINGLEPLAYER" ≈ 12 chars
                text_areas.push(TextArea {
                    buffer: &self.menu_connect_button_buffer,
                    left: layout.connect_button.x
                        + (layout.connect_button.w - connect_estimate) * 0.5,
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

                // Status pill color reflects the current state:
                //   red   → connection error
                //   teal  → in-progress status (connecting…)
                //   gray  → idle / ready
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
                        Color::rgb(255, 124, 124) // error red
                    } else if self.menu_state.status_message.is_some() {
                        Color::rgb(124, 224, 208) // progress teal
                    } else {
                        Color::rgb(219, 229, 239) // idle grey
                    },
                    custom_glyphs: &[],
                });
            } else {
                // ---- In-game HUD text ----

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
                    wgpu::SurfaceError::Lost
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
                    wgpu::SurfaceError::Lost
                })?;
        }

        // ── Submit & present ──────────────────────────────────────────────── //
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    /// Updates all `glyphon::Buffer` objects that display menu text.
    ///
    /// This method is called once per frame while in `GameState::Menu`.  It
    /// reads from `menu_state` and `config` to produce the correct strings,
    /// then calls `set_text` + `set_size` on every relevant buffer.
    ///
    /// Separating text content update from `TextArea` assembly (which happens
    /// in `render`) keeps the render function focused on GPU commands and makes
    /// it easy to add or remove menu fields without touching the render loop.
    ///
    /// # Active-field label decoration
    /// When a text field is selected, its label gains a `"  •  active"` suffix
    /// so the player has a clear visual indication of where keyboard input goes.
    pub fn prepare_menu_text(&mut self) {
        let selected = self.menu_state.selected_field;

        let title = "Minerust";
        let subtitle = "Voxel sandbox with multiplayer and custom UI";

        // Append an activity indicator to the label of the focused field.
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

        // Status pill: prefer error > status > idle ready message.
        let status_text = if let Some(ref err) = self.menu_state.error_message {
            format!("ERROR: {}", err)
        } else if let Some(ref status) = self.menu_state.status_message {
            format!("STATUS: {}", status)
        } else {
            "READY: ENTER joins multiplayer, ESC starts solo".to_string()
        };

        // ---- Update each buffer ----
        // All buffers use the same pattern: `set_text` to reshape the string,
        // then `set_size` to update the wrap/clip width.

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

    /// Renders the main-menu overlay as a series of flat colored rectangles.
    ///
    /// All geometry is built in CPU memory each frame using [`push_rect`] and
    /// uploaded via `create_buffer_init` (the buffers are too small and
    /// change-heavy to justify a persistent mapped buffer).  The pass reuses
    /// the `crosshair_pipeline` because the menu quads share the same vertex
    /// format and require the same alpha-blended, no-depth-test rendering.
    ///
    /// # Visual structure (back to front)
    /// 1. Full-screen semi-transparent dark overlay.
    /// 2. Panel drop-shadow (slightly larger than the panel itself).
    /// 3. Panel background.
    /// 4. Panel top accent stripe (gold).
    /// 5. Title badge background + left accent stripe.
    /// 6. Quick-tips card + left accent stripe.
    /// 7. Server address field (border + fill, highlight when active).
    /// 8. Username field (border + fill, highlight when active).
    /// 9. Connect button (border + fill, highlight on hover).
    /// 10. Singleplayer button (border + fill, highlight on hover).
    /// 11. Status pill background.
    /// 12. Active-field top underline (gold, only when a field is selected).
    /// 13. Text cursor (blinking gold bar inside the active field).
    ///
    /// # Parameters
    /// - `encoder` – Command encoder to append the render pass to.
    /// - `view`    – Swap-chain texture view to draw into.
    pub fn render_menu(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let layout = MenuLayout::new(self.config.width, self.config.height);
        let width = self.config.width as f32;
        let height = self.config.height as f32;
        let panel = layout.panel;

        // Determine which interactive element the cursor is currently over so
        // hover highlight colors can be applied to the correct button.
        let hovered = self
            .cursor_position
            .and_then(|(x, y)| layout.hit_test(x, y));

        let mut vertices = Vec::with_capacity(96);
        let mut indices = Vec::with_capacity(144);

        // 1. Full-screen backdrop. Keep it fully transparent so the world
        //    stays visible behind the menu.
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: 0.0,
                y: 0.0,
                w: width,
                h: height,
            },
            [0.0, 0.0, 0.0, 0.0],
            width,
            height,
        );

        // 2. Panel drop-shadow (10 px bleed on each side).
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x - 10.0,
                y: panel.y - 10.0,
                w: panel.w + 20.0,
                h: panel.h + 20.0,
            },
            [0.08, 0.12, 0.18, 0.28],
            width,
            height,
        );

        // 3. Panel background.
        push_rect(
            &mut vertices,
            &mut indices,
            panel,
            [0.12, 0.15, 0.20, 0.72],
            width,
            height,
        );

        // 4. Gold top accent stripe (6 px high).
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x,
                y: panel.y,
                w: panel.w,
                h: 6.0,
            },
            [0.95, 0.72, 0.24, 1.0],
            width,
            height,
        );

        // 5a. Title badge background.
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x + 20.0,
                y: panel.y + 18.0,
                w: 180.0,
                h: 34.0,
            },
            [0.16, 0.20, 0.26, 0.82],
            width,
            height,
        );

        // 5b. Title badge left accent stripe (gold, 8 px wide).
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: panel.x + 16.0,
                y: panel.y + 16.0,
                w: 8.0,
                h: 40.0,
            },
            [0.97, 0.74, 0.24, 1.0],
            width,
            height,
        );

        // 6a. Quick-tips card background.
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.quick_card.x,
                y: layout.quick_card.y,
                w: layout.quick_card.w,
                h: layout.quick_card.h,
            },
            [0.15, 0.19, 0.24, 0.76],
            width,
            height,
        );

        // 6b. Quick-tips card left accent stripe (teal, 4 px wide).
        push_rect(
            &mut vertices,
            &mut indices,
            Rect {
                x: layout.quick_card.x,
                y: layout.quick_card.y,
                w: 4.0,
                h: layout.quick_card.h,
            },
            [0.35, 0.8, 0.78, 1.0],
            width,
            height,
        );

        // 7. Server address field (active = slightly brighter fill).
        let field_color = if self.menu_state.selected_field == MenuField::ServerAddress {
            [0.13, 0.2, 0.27, 0.88]
        } else {
            [0.13, 0.17, 0.22, 0.78]
        };
        // Outer dark border (1 px implied by the 2 px inset of the inner rect).
        push_rect(
            &mut vertices,
            &mut indices,
            layout.server_field,
            [0.04, 0.05, 0.07, 0.78],
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

        // 8. Username field (same pattern as server field).
        let username_color = if self.menu_state.selected_field == MenuField::Username {
            [0.13, 0.2, 0.27, 0.88]
        } else {
            [0.13, 0.17, 0.22, 0.78]
        };
        push_rect(
            &mut vertices,
            &mut indices,
            layout.username_field,
            [0.04, 0.05, 0.07, 0.78],
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

        // 9. Connect button (brighter fill on hover).
        let connect_fill = if matches!(hovered, Some(crate::ui::menu::MenuHit::Connect)) {
            [0.24, 0.52, 0.84, 1.0]
        } else {
            [0.2, 0.45, 0.74, 1.0]
        };
        push_rect(
            &mut vertices,
            &mut indices,
            layout.connect_button,
            [0.16, 0.33, 0.55, 1.0],
            width,
            height,
        ); // border
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

        // 10. Singleplayer button (same pattern, darker palette).
        let single_fill = if matches!(hovered, Some(crate::ui::menu::MenuHit::Singleplayer)) {
            [0.19, 0.22, 0.28, 1.0]
        } else {
            [0.16, 0.19, 0.24, 1.0]
        };
        push_rect(
            &mut vertices,
            &mut indices,
            layout.singleplayer_button,
            [0.1, 0.11, 0.14, 1.0],
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

        // 11. Status pill background.
        push_rect(
            &mut vertices,
            &mut indices,
            layout.status_pill,
            [0.12, 0.15, 0.19, 0.82],
            width,
            height,
        );

        // 12. Active-field top underline (gold, 3 px high).
        // Only drawn when a field is actually selected.
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
                [0.97, 0.74, 0.24, 1.0],
                width,
                height,
            );
        }

        // 13. Text cursor (2 px wide gold bar inside the active field).
        // Positioned after the last character; clamped so it never leaves
        // the field bounds.  A proper blinking cursor would require time-based
        // alpha, which can be added by sampling `self.game_start_time`.
        let active_field = match self.menu_state.selected_field {
            MenuField::ServerAddress => {
                Some((layout.server_field, self.menu_state.server_address.as_str()))
            }
            MenuField::Username => Some((layout.username_field, self.menu_state.username.as_str())),
            MenuField::None => None,
        };
        if let Some((field, value)) = active_field {
            let char_count = value.chars().count() as f32;
            // 11 px per character is an approximation for the menu font size.
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
                [0.97, 0.74, 0.24, 0.95],
                width,
                height,
            );
        }

        // ── Upload and draw ───────────────────────────────────────────────── //
        if !vertices.is_empty() {
            // Allocate fresh buffers every frame; the menu geometry is small
            // enough that the allocation overhead is negligible.
            let vb = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Menu UI VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                        // Preserve the already composited world and draw the
                        // menu overlay on top of it.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.crosshair_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.terrain_gbuffer_bind_group, &[]);
            pass.set_bind_group(2, &self.terrain_shadow_output_bind_group, &[]);
            pass.set_bind_group(3, &self.shadow_mask_bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }
    }

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
