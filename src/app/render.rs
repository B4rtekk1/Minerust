use std::sync::LazyLock;
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use glyphon::{Color, Metrics, Shaping, TextArea, TextBounds};
use wgpu::util::DeviceExt;

use minerust::{
    BlockType, CHUNK_SIZE, DEFAULT_FOV, MAX_MESH_BUILDS_PER_FRAME, RENDER_DISTANCE, SEA_LEVEL,
    SUN_PATH_TILT, Uniforms, Vertex, World, build_block_outline, build_item_model,
    build_player_model, extract_frustum_planes,
};

use crate::logger::{LogLevel, log};
use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::{
    GameState, MenuAction, MenuContext, MenuField, MenuLayout, MenuScreen, MenuItemKind,
    build_menu_geometry,
};
use super::graphics::WaterQuality;

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

// Keep all HUD query slots current even when a GPU stage is disabled.
fn write_empty_gpu_interval(
    encoder: &mut wgpu::CommandEncoder,
    query_set: Option<&wgpu::QuerySet>,
    start: u32,
    end: u32,
) {
    if let Some(query_set) = query_set {
        let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Inactive GPU Stage Timestamps"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(start),
                end_of_pass_write_index: Some(end),
            }),
        });
    }
}

// Sorting relative coordinates once preserves the previous distance/tie order
// without sorting thousands of visible columns after each streamed insertion.
static RENDER_OFFSETS: LazyLock<Vec<(i32, i32)>> = LazyLock::new(|| {
    let mut offsets: Vec<_> = (-RENDER_DISTANCE..=RENDER_DISTANCE)
        .flat_map(|dx| (-RENDER_DISTANCE..=RENDER_DISTANCE).map(move |dz| (dx, dz)))
        .filter(|&(dx, dz)| dx * dx + dz * dz <= RENDER_DISTANCE * RENDER_DISTANCE)
        .collect();
    offsets.sort_by_key(|&(dx, dz)| dx * dx + dz * dz);
    offsets
});

fn collect_visible_columns(
    out: &mut Vec<(i32, i32)>,
    center: (i32, i32),
    mut is_loaded: impl FnMut(i32, i32) -> bool,
) {
    out.clear();
    for &(dx, dz) in RENDER_OFFSETS.iter() {
        let (cx, cz) = (center.0 + dx, center.1 + dz);
        if is_loaded(cx, cz) {
            out.push((cx, cz));
        }
    }
}

impl State {
    fn rebuild_visible_chunk_cache(&mut self, player_cx: i32, player_cz: i32) {
        if !self.visible_chunk_columns_dirty
            && self.visible_chunk_cache_center == (player_cx, player_cz)
        {
            return;
        }

        {
            let world = self.world.read();
            collect_visible_columns(
                &mut self.visible_chunk_columns,
                (player_cx, player_cz),
                |cx, cz| world.chunks.contains_key(&(cx, cz)),
            );
        }

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
    /// 3. **Mesh request** – drains a bounded dirty-subchunk queue into the
    ///    background mesher.
    /// 4. **Main cull dispatch** – GPU frustum + Hi-Z occlusion cull for both
    ///    the opaque terrain and water indirect managers.
    /// 5. **Opaque pass** – sky dome -> terrain -> remote player models -> sun/moon.
    ///    Resolves MSAA into `ssr_color_view` for later water reflections.
    /// 6. **Depth resolve compute** – resolves the multisampled depth buffer
    ///    directly into a half-resolution Hi-Z base for next-frame culling.
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
        // Drive completed map callbacks without waiting for the GPU. Timestamp
        // values are therefore intentionally from an earlier frame.
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(profiler) = &mut self.gpu_timestamp_profiler {
            if let Some(profile) = profiler.poll() {
                self.gpu_frame_profile = Some(profile);
                self.gpu_profile_log.record(&profile);
            }
        }

        // ── Acquire swap-chain texture ────────────────────────────────────── //
        let render_start = Instant::now();
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
        let preparation_start = Instant::now();
        self.frame_profile.acquire_ms = (preparation_start - render_start).as_secs_f32() * 1000.0;

        // ── Dynamic world-model buffers ──────────────────────────────────── //
        // Remote players and dropped items share one vertex/index pair. This
        // avoids a draw call per item while preserving terrain lighting, fog,
        // depth testing and terrain-atlas textures.
        if (!self.remote_players.is_empty() || !self.item_entities.is_empty()) && render_world_scene {
            let mut all_vertices = Vec::with_capacity(
                self.remote_players.len() * 16 + self.item_entities.len() * 24,
            );
            let mut all_indices = Vec::with_capacity(
                self.remote_players.len() * 24 + self.item_entities.len() * 36,
            );

            for (_id, player) in &self.remote_players {
                let (vertices, indices) =
                    build_player_model(player.x, player.y, player.z, player.yaw);
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                // Remap local indices to the combined buffer's address space.
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            let elapsed = self.game_start_time.elapsed().as_secs_f32();
            for item in &self.item_entities {
                let Some(block) = minerust::block_for_item(item.stack.item) else {
                    continue;
                };
                let bob = (elapsed * 3.0 + item.id as f32 * 0.71).sin() * 0.06;
                let (vertices, indices) = build_item_model(
                    [item.position.x, item.position.y + bob, item.position.z],
                    block,
                    elapsed * 1.8 + item.id as f32 * 0.37,
                );
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
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
        self.frame_profile.remote_players_ms = preparation_start.elapsed().as_secs_f32() * 1000.0;

        // ── Command encoder ───────────────────────────────────────────────── //
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // A handle clone avoids holding an immutable borrow of `self` across
        // the CPU-side preparation work below.
        let timestamp_query_set = self
            .gpu_timestamp_profiler
            .as_ref()
            .map(|p| p.query_set.clone());
        let timestamp_readback_slot = if render_world_scene {
            self.gpu_timestamp_profiler
                .as_ref()
                .and_then(|profiler| profiler.acquire_readback())
        } else {
            None
        };

        // ── Camera & projection matrices ──────────────────────────────────── //
        let section_start = Instant::now();
        let aspect = self.config.width as f32 / self.config.height as f32;
        // Extend the far plane beyond RENDER_DISTANCE so chunks at the horizon
        // are not clipped by the projection; 400 blocks is a sensible floor.
        let far_plane = (RENDER_DISTANCE as f32 * CHUNK_SIZE as f32 * 1.5).max(400.0);
        let proj = Mat4::perspective_rh(DEFAULT_FOV, aspect, 0.1, far_plane);
        let view_mat = self.camera.view_matrix();
        // `glam`'s RH projection helpers already use WebGPU's [0, 1] depth range.
        let view_proj = proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.to_cols_array_2d();
        self.frame_profile.camera_matrices_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        // ── Day/night cycle ───────────────────────────────────────────────── //
        let time = self.game_start_time.elapsed().as_secs_f32();

        // The sun angle is advanced in `update` (day cycle and the `[` / `]`
        // keys); π/2 is noon. The arc is tilted sideways so shadows never
        // fall straight down.
        let sun_angle = self.sun_angle;
        let sun_dir = Vec3::new(SUN_PATH_TILT, sun_angle.sin(), sun_angle.cos()).normalize();
        let (sun_x, sun_y, sun_z) = (sun_dir.x, sun_dir.y, sun_dir.z);
        let moon_intensity = (-sun_dir.y).clamp(0.0, 1.0);

        // The moon is always opposite the sun direction.
        let moon_position = [-sun_dir.x, -sun_dir.y, -sun_dir.z];

        // Chunk coordinates of the camera, used to center the render window.
        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let section_start = Instant::now();
        self.rebuild_visible_chunk_cache(player_cx, player_cz);
        self.frame_profile.visible_cache_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        // Inverse view-projection is used by the composite / water shaders to
        // reconstruct world-space positions from screen-space depth samples.
        let inv_view_proj = view_proj.inverse();
        let inv_view_proj_array: [[f32; 4]; 4] = inv_view_proj.to_cols_array_2d();

        let eye_pos = self.camera.eye_position();
        let is_underwater = self.is_underwater;

        // ── Sun-shadow voxel grid ─────────────────────────────────────────── //
        // Streams changed chunk columns into the occupancy window (a few
        // per frame, nearest first).
        let section_start = Instant::now();
        if render_world_scene {
            let world = self.world.read();
            self.shadow_grid.update(
                &self.queue,
                &world,
                player_cx,
                player_cz,
                self.graphics.shadows,
                self.graphics.water == WaterQuality::Realistic,
            );
            drop(world);
            self.shadow_grid.encode_distance_field(&self.queue, &mut encoder);
        }
        self.frame_profile.shadow_grid_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        // ── Upload uniforms ───────────────────────────────────────────────── //
        let section_start = Instant::now();
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
                water_quality: self.graphics.water.shader_value(),
                moon_position,
                _pad1_moon: 0.0,
                moon_intensity,
                wind_dir: [0.8, 0.6],
                wind_speed: 1.0,
                rain_factor: 0.0,
                sky_visibility: self.sky_visibility,
                menu_blur: if menu_visible { 1.0 } else { 0.0 },
                fog_far: (minerust::RENDER_DISTANCE * minerust::CHUNK_SIZE) as f32,
            }]),
        );
        self.frame_profile.uniform_upload_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        // ── Frustum planes (main camera) ──────────────────────────────────── //
        // Six half-space planes derived from the combined view-projection
        // matrix, used both for CPU-side mesh gating and the GPU cull shader.
        let frustum_planes = extract_frustum_planes(&view_proj);

        // ── Mesh rebuild requests ─────────────────────────────────────────── //
        // Dirtying code enqueues a key immediately, so the hot render path
        // never traverses every visible chunk/subchunk merely to discover that
        // nothing changed.  The queue is deduplicated at insertion time.
        let section_start = Instant::now();
        for _ in 0..MAX_MESH_BUILDS_PER_FRAME {
            let Some(key) = self.pop_dirty_mesh() else {
                break;
            };
            // A synchronous player-edit rebuild may have cleaned this entry
            // after it was queued.  Inspect only this dequeued key, never the
            // whole visible world.
            let still_dirty = self
                .world
                .read()
                .chunks
                .get(&(key.chunk_x, key.chunk_z))
                .and_then(|chunk| chunk.subchunks.get(key.subchunk_y as usize))
                .is_some_and(|subchunk| subchunk.mesh_dirty && !subchunk.is_empty);
            if !still_dirty {
                continue;
            }
            // Meshing against a missing neighbour bakes in fake boundary
            // faces. Skip it: inserting the neighbour dirties this subchunk
            // again, so it is queued once its surroundings are complete.
            let neighbours_loaded = {
                let world = self.world.read();
                [(-1, 0), (1, 0), (0, -1), (0, 1)].iter().all(|&(dx, dz)| {
                    world
                        .chunks
                        .contains_key(&(key.chunk_x + dx, key.chunk_z + dz))
                })
            };
            if !neighbours_loaded {
                continue;
            }
            if !self
                .mesh_loader
                .request_mesh(key.chunk_x, key.chunk_z, key.subchunk_y)
            {
                // The bounded worker channel is full.  Retain the work for a
                // later frame instead of falling back to a global re-scan.
                self.requeue_dirty_mesh_front(key);
                break;
            }
        }
        self.frame_profile.render_chunk_scan_ms = section_start.elapsed().as_secs_f32() * 1000.0;
        self.frame_profile.mesh_request_submit_ms = 0.0;
        self.frame_profile.frame_preparation_ms =
            preparation_start.elapsed().as_secs_f32() * 1000.0;

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

        // ── Main camera GPU cull dispatch ─────────────────────────────────── //
        // The indirect manager's compute shader reads the Hi-Z texture and
        // frustum planes to populate per-chunk indirect draw arguments.
        let frustum_planes_array = frustum_planes_to_array(&frustum_planes);
        // A zero Hi-Z size makes the compute shader accept every
        // frustum-visible subchunk. F4 uses this to distinguish temporal
        // occlusion artifacts from a meshing-streaming backlog.
        let hiz_size_f = if self.occlusion_culling_enabled && self.hiz_valid {
            [self.hiz_size[0] as f32, self.hiz_size[1] as f32]
        } else {
            [0.0, 0.0]
        };

        self.indirect_manager.flush_slot_clears(&self.queue);
        self.indirect_manager.dispatch_culling(
            &mut encoder,
            &self.queue,
            timestamp_query_set.as_ref(),
            0,
            1,
            &self.water_indirect_manager,
            &view_proj,
            &frustum_planes_array,
            // The eye, not the feet: back-facing direction culling must use
            // the point the image is actually rendered from.
            eye_pos.into(),
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
                    // shader can sample the opaque scene for reflections
                    // (copied below instead when anti-aliasing is off).
                    resolve_target: (self.msaa_samples > 1).then_some(&self.ssr_color_view),
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
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
                ..Default::default()
            });

            // --- Terrain chunks (indirect) ---
            // The GPU cull pass emits one draw per visible subchunk face
            // direction (back-facing directions are skipped).
            // Opaque blocks first (no `discard`, so early-Z works), then the
            // alpha-tested leaves and dead bushes with their own pipeline.
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_bind_group(1, &self.terrain_quad_bind_group, &[]);
            // Indexed: 4 pulled vertices per quad via a shared index pattern.
            opaque_pass.set_index_buffer(
                self.indirect_manager.quad_index_buffer().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for (pipeline, commands, count, max_count) in [
                (
                    &self.render_pipeline,
                    self.indirect_manager.draw_commands(),
                    self.indirect_manager.visible_count_buffer(),
                    self.indirect_manager.max_terrain_draws(),
                ),
                (
                    &self.cutout_pipeline,
                    self.indirect_manager.cutout_draw_commands(),
                    self.indirect_manager.cutout_count_buffer(),
                    self.indirect_manager.max_cutout_draws(),
                ),
            ] {
                if max_count == 0 {
                    continue;
                }
                opaque_pass.set_pipeline(pipeline);
                if self.supports_indirect_count {
                    opaque_pass.multi_draw_indexed_indirect_count(commands, 0, count, 0, max_count);
                } else {
                    // Unused entries were cleared to zero-index draws.
                    opaque_pass.multi_draw_indexed_indirect(commands, 0, max_count);
                }
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

            // --- Sky dome ---
            // Drawn after all opaque geometry at depth 1.0 with LessEqual and
            // no depth write: early-Z skips every pixel already covered by
            // terrain, so the expensive sky shader only runs where sky shows.
            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

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

        self.copy_single_sample_color(&mut encoder, &self.ssr_color_view);

        if render_world_scene {
            let mut ssr_resolve_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scene Depth Resolve Pass"),
                timestamp_writes: None,
            });
            ssr_resolve_pass.set_pipeline(&self.ssr_hiz_resolve_pipeline);
            ssr_resolve_pass.set_bind_group(0, &self.ssr_hiz_resolve_bind_group, &[]);
            ssr_resolve_pass.dispatch_workgroups(
                self.ssr_hiz_size[0].div_ceil(16),
                self.ssr_hiz_size[1].div_ceil(16),
                1,
            );
            // Water reflections reuse this depth for visible hits and the
            // god-rays pass reconstructs world positions from it.
            drop(ssr_resolve_pass);

            // Quarter-resolution ray-marched light shafts, upsampled and
            // added by the composite pass.
            let mut god_rays_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("God Rays Pass"),
                timestamp_writes: None,
            });
            god_rays_pass.set_pipeline(&self.god_rays_pipeline);
            god_rays_pass.set_bind_group(0, &self.god_rays_bind_group, &[]);
            god_rays_pass.dispatch_workgroups(
                self.god_rays_size[0].div_ceil(8),
                self.god_rays_size[1].div_ceil(8),
                1,
            );
            drop(god_rays_pass);
        }

        // ── Depth resolve + Hi-Z generation ───────────────────────────────── //
        // The opaque pass above produced the depth buffer, eliminating the
        // former terrain depth prepass. Culling at the start of this frame reads
        // the previous pyramid; this freshly built pyramid is consumed next frame.
        if render_world_scene && self.occlusion_culling_enabled {
            let mut depth_resolve_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Depth Resolve Compute Pass"),
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }
                }),
            });
            depth_resolve_pass.set_pipeline(&self.depth_resolve_pipeline);
            depth_resolve_pass.set_bind_group(0, &self.depth_resolve_bind_group, &[]);
            depth_resolve_pass.dispatch_workgroups(
                self.hiz_size[0].div_ceil(16),
                self.hiz_size[1].div_ceil(16),
                1,
            );
            drop(depth_resolve_pass);

            for i in 0..self.hiz_bind_groups.len() {
                let mut hiz_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hi-Z Generation Pass Level"),
                    timestamp_writes: timestamp_query_set.as_ref().and_then(|query_set| {
                        if i == 0 || i + 1 == self.hiz_bind_groups.len() {
                            Some(wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: (i == 0).then_some(6),
                                end_of_pass_write_index: (i + 1 == self.hiz_bind_groups.len())
                                    .then_some(7),
                            })
                        } else {
                            None
                        }
                    }),
                });
                hiz_pass.set_pipeline(&self.hiz_pipeline);
                hiz_pass.set_bind_group(0, &self.hiz_bind_groups[i], &[]);
                let div = 1 << (i + 1);
                let mip_width = (self.hiz_size[0] / div).max(1);
                let mip_height = (self.hiz_size[1] / div).max(1);
                hiz_pass.dispatch_workgroups((mip_width + 15) / 16, (mip_height + 15) / 16, 1);
            }
            // A 1x1 base has no downsampling passes but still needs fresh
            // timestamps; otherwise the HUD would reuse queries from old frames.
            if self.hiz_bind_groups.is_empty() {
                write_empty_gpu_interval(&mut encoder, timestamp_query_set.as_ref(), 6, 7);
            }
            self.hiz_valid = true;
        } else {
            self.hiz_valid = false;
            write_empty_gpu_interval(&mut encoder, timestamp_query_set.as_ref(), 4, 5);
            write_empty_gpu_interval(&mut encoder, timestamp_query_set.as_ref(), 6, 7);
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
                    resolve_target: (self.msaa_samples > 1).then_some(&self.scene_color_view),
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
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(8),
                        end_of_pass_write_index: Some(9),
                    }
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

        self.copy_single_sample_color(&mut encoder, &self.scene_color_view);

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
                    resolve_target: (self.msaa_samples > 1).then_some(&self.scene_color_view),
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

        if self.game_state == GameState::Playing {
            // The outline pass drew into the main target again.
            self.copy_single_sample_color(&mut encoder, &self.scene_color_view);
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
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(10),
                        end_of_pass_write_index: Some(11),
                    }
                }),
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
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(12),
                        end_of_pass_write_index: Some(13),
                    }
                }),
                ..Default::default()
            });

            ui_pass.set_pipeline(&self.crosshair_pipeline);
            ui_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            if menu_visible {
                let context = MenuContext {
                    has_world: self.has_entered_world,
                    save_available: std::path::Path::new(minerust::DEFAULT_WORLD_FILE).exists(),
                };
                let layout = MenuLayout::new(self.config.width, self.config.height, self.menu_state.screen, context);
                let hovered = self.cursor_position.and_then(|(x, y)| layout.hit_test(x, y));
                let selected = self.menu_state.selected_action(&layout);
                let (vertices, indices) = build_menu_geometry(
                    self.config.width,
                    self.config.height,
                    &layout,
                    hovered.or(selected),
                    self.menu_state.selected_field,
                );
                let menu_vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Menu Geometry VB"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let menu_ib = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Menu Geometry IB"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                ui_pass.set_vertex_buffer(0, menu_vb.slice(..));
                ui_pass.set_index_buffer(menu_ib.slice(..), wgpu::IndexFormat::Uint32);
                ui_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            } else {
                // --- Inventory overlay ---
                if self.inventory_open {
                    let size = [self.config.width, self.config.height];
                    if self.inventory_vertex_buffer.is_none()
                        || self.last_inventory_revision != self.inventory.revision()
                        || self.inventory_buffer_size != size
                    {
                        let (vertices, indices) = crate::ui::inventory::build_geometry(&self.inventory, size[0], size[1]);
                        self.inventory_vertex_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Inventory UI VB"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX,
                        }));
                        self.inventory_index_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Inventory UI IB"), contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX,
                        }));
                        self.inventory_num_indices = indices.len() as u32;
                        self.last_inventory_revision = self.inventory.revision();
                        self.inventory_buffer_size = size;
                    }
                    if let (Some(vb), Some(ib)) = (&self.inventory_vertex_buffer, &self.inventory_index_buffer) {
                        ui_pass.set_vertex_buffer(0, vb.slice(..));
                        ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        ui_pass.draw_indexed(0..self.inventory_num_indices, 0, 0..1);
                    }
                    if let (Some(stack), Some((x, y))) = (&self.inventory_ui.cursor_stack, self.cursor_position) {
                        let vertices = crate::ui::inventory::cursor_geometry(stack, x, y, size[0], size[1]);
                        if self.inventory_cursor_vertex_buffer.is_none() {
                            self.inventory_cursor_vertex_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Inventory cursor VB"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            }));
                            self.inventory_cursor_index_buffer = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Inventory cursor IB"), contents: bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]), usage: wgpu::BufferUsages::INDEX,
                            }));
                        } else if let Some(vb) = &self.inventory_cursor_vertex_buffer {
                            self.queue.write_buffer(vb, 0, bytemuck::cast_slice(&vertices));
                        }
                        if let (Some(vb), Some(ib)) = (&self.inventory_cursor_vertex_buffer, &self.inventory_cursor_index_buffer) {
                            ui_pass.set_vertex_buffer(0, vb.slice(..));
                            ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            ui_pass.draw_indexed(0..6, 0, 0..1);
                        }
                    }
                }

                // --- Crosshair ---
                if self.show_crosshair && !self.inventory_open {
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
                    if self.hotbar_dirty || self.hotbar_vertex_buffer.is_none() || self.last_hotbar_inventory_revision != self.inventory.revision() {
                        let aspect = self.config.width as f32 / self.config.height as f32;
                        let (vb, ib, count) =
                            crate::ui::ui::build_hotbar(&self.device, &self.inventory, aspect);
                        self.hotbar_vertex_buffer = Some(vb);
                        self.hotbar_index_buffer = Some(ib);
                        self.hotbar_num_indices = count;
                        self.hotbar_dirty = false;
                        self.last_hotbar_inventory_revision = self.inventory.revision();
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
        let text_start = Instant::now();
        self.frame_profile.encode_ms = (text_start - preparation_start).as_secs_f32() * 1000.0
            - self.frame_profile.frame_preparation_ms;
        // All on-screen text is batched into a single `TextRenderer::prepare`
        // call and rendered in one pass.  The individual `glyphon::Buffer`
        // objects are updated lazily (only when the underlying text changes)
        // to avoid redundant re-shaping work.
        {
            // ---- FPS counter (in-game only) ----
            // Formatting and shaping this multi-line buffer on every redraw
            // makes the profiler affect the result it reports.  Keep drawing
            // the already prepared buffer every frame, while refreshing its
            // dynamic contents at 10 Hz.
            if !menu_visible
                && self.show_debug_overlay
                && self.last_debug_text_update.elapsed() >= Duration::from_millis(100)
            {
                let profile = &self.frame_profile;
                let update_children_ms = profile.update_children_ms();
                let gpu_text = if let Some(gpu) = self.gpu_frame_profile {
                    format!(
                        "GPU frame: {:.2} ms\n  Cull: {:.2} ms\n  Opaque: {:.2} ms\n  Depth resolve: {:.2} ms\n  Hi-Z: {:.2} ms\n  Water: {:.2} ms\n  Composite: {:.2} ms\n  UI: {:.2} ms\n  Text: {:.2} ms\n\n",
                        gpu.frame_ms,
                        gpu.cull_ms,
                        gpu.opaque_ms,
                        gpu.depth_resolve_ms,
                        gpu.hiz_ms,
                        gpu.water_ms,
                        gpu.composite_ms,
                        gpu.ui_ms,
                        gpu.text_ms,
                    )
                } else if self.gpu_timestamp_profiler.is_some() {
                    "GPU frame: collecting timestamp queries...\n\n".to_owned()
                } else {
                    "GPU frame: unavailable (TIMESTAMP_QUERY unsupported)\n\n".to_owned()
                };
                let fps_text = format!(
                    "FPS: {:.0}\nFrame avg: {:.2} ms\nCPU time avg: {:.2} ms\n\n{}Update: {:.2} ms\n  Network: {:.2}\n  Chunk poll: {:.2}\n  Physics/snapshot: {:.2}\n  Requests: {:.2}\n  Chunk commit: {:.2}\n  Mesh commit: {:.2}\n  Children sum: {:.2}\n  Unaccounted: {:.2}\n\nFrame preparation: {:.2} ms\n  Camera/matrices: {:.2}\n  Visible cache: {:.2}\n  Uniform upload: {:.2}\n  Dirty mesh queue: {:.2}\n  Mesh requests: {:.2}\n  Remote players: {:.2}\n\nMeshing: queued {} | pending {}\nHi-Z occlusion: {}\n\nChunks: {}\nSubchunks: {}",
                    self.current_fps,
                    self.frame_time_ms,
                    self.cpu_update_ms,
                    gpu_text,
                    profile.update_ms,
                    profile.network_ms,
                    profile.chunk_poll_ms,
                    profile.physics_snapshot_ms,
                    profile.chunk_requests_ms,
                    profile.chunk_commit_ms,
                    profile.mesh_commit_ms,
                    update_children_ms,
                    profile.update_ms - update_children_ms,
                    profile.frame_preparation_ms,
                    profile.camera_matrices_ms,
                    profile.visible_cache_ms,
                    profile.uniform_upload_ms,
                    profile.render_chunk_scan_ms,
                    profile.mesh_request_submit_ms,
                    profile.remote_players_ms,
                    self.dirty_mesh_queue.len(),
                    self.mesh_loader.pending_count(),
                    if self.occlusion_culling_enabled { "on" } else { "OFF" },
                    self.chunks_rendered,
                    self.subchunks_rendered
                );
                self.fps_buffer.set_text(
                    &fps_text,
                    &self.ui_font.attrs(),
                    Shaping::Advanced,
                    None,
                );
                self.fps_buffer
                    .shape_until_scroll(&mut self.font_system, false);
                self.fps_buffer.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                self.last_debug_text_update = std::time::Instant::now();
            }

            // ---- Hotbar slot label (in-game only, updated on slot change) ----
            if !menu_visible && self.last_hotbar_slot != self.inventory.selected_hotbar as usize {
                let label = self.inventory.selected_stack().map(|stack| minerust::item_registry().get(stack.item).display_name).unwrap_or("");
                self.hotbar_label_buffer.set_text(
                    label,
                    &self.ui_font.colored(Color::rgb(255, 238, 200)),
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
                self.last_hotbar_slot = self.inventory.selected_hotbar as usize;
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
                        &self.ui_font.colored(Color::rgb(76, 255, 76)), // bright green name tags
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
                let context = MenuContext {
                    has_world: self.has_entered_world,
                    save_available: std::path::Path::new(minerust::DEFAULT_WORLD_FILE).exists(),
                };
                let no_save_message = !context.save_available
                    && matches!(self.menu_state.screen, MenuScreen::Main | MenuScreen::Pause);
                let layout = MenuLayout::new(self.config.width, self.config.height, self.menu_state.screen, context);
                let hovered = self.cursor_position.and_then(|(x, y)| layout.hit_test(x, y));
                let selected = self.menu_state.selected_action(&layout);
                let active = hovered.or(selected);
                let bounds = TextBounds {
                    left: layout.panel.x as i32,
                    top: layout.panel.y as i32,
                    right: (layout.panel.x + layout.panel.w) as i32,
                    bottom: (layout.panel.y + layout.panel.h) as i32,
                };
                let scale = layout.scale;
                let title_width = self.menu_title_buffer.layout_runs().next().map(|run| run.line_w).unwrap_or(0.0) * scale;
                text_areas.push(TextArea {
                    buffer: &self.menu_title_buffer,
                    left: layout.title.x + (layout.title.w - title_width) * 0.5,
                    top: layout.title.y + (layout.title.h - 50.0 * scale) * 0.5,
                    scale,
                    bounds,
                    default_color: Color::rgb(238, 246, 248),
                    custom_glyphs: &[],
                });

                for (index, item) in layout.items.iter().enumerate() {
                    let Some(buffer) = self.menu_text_buffers.get(index) else { continue; };
                    let line_width = buffer.layout_runs().next().map(|run| run.line_w).unwrap_or(0.0) * scale;
                    let color = if !item.enabled {
                        Color::rgb(115, 124, 132)
                    } else if active == item.action || (item.kind == MenuItemKind::TextField && self.menu_state.is_editing()) {
                        Color::rgb(255, 255, 255)
                    } else if item.kind == MenuItemKind::Text {
                        Color::rgb(176, 195, 205)
                    } else {
                        Color::rgb(211, 224, 230)
                    };
                    text_areas.push(TextArea {
                        buffer,
                        left: item.rect.x + (item.rect.w - line_width) * 0.5,
                        top: item.rect.y + (item.rect.h - 32.0 * scale) * 0.5,
                        scale,
                        bounds,
                        default_color: color,
                        custom_glyphs: &[],
                    });
                }
                if (self.menu_state.error_message.is_some() || self.menu_state.status_message.is_some() || no_save_message)
                    && let Some(buffer) = self.menu_text_buffers.get(layout.items.len())
                {
                    text_areas.push(TextArea {
                        buffer,
                        left: layout.message.x,
                        top: layout.message.y,
                        scale: scale * 0.82,
                        bounds,
                        default_color: if self.menu_state.error_message.is_some() || no_save_message { Color::rgb(255, 150, 128) } else { Color::rgb(145, 210, 180) },
                        custom_glyphs: &[],
                    });
                }
            } else {
                // ---- In-game HUD text ----

                // Stack counts are ordinary Glyphon text, so inventory uses
                // the same Windows-backed UI font as every other label.
                if self.inventory_open {
                    let counts: Vec<(usize, u16)> = (0..36)
                        .filter_map(|slot| self.inventory.get_flat(slot).map(|item| (slot, item.count)))
                        .collect();
                    while self.inventory_count_buffers.len() < counts.len() {
                        self.inventory_count_buffers.push(glyphon::Buffer::new(
                            &mut self.font_system,
                            Metrics::new(18.0, 20.0),
                        ));
                    }
                    let layout = crate::ui::inventory::InventoryLayout::new(
                        self.config.width,
                        self.config.height,
                    );
                    for (index, (_, quantity)) in counts.iter().enumerate() {
                        let buffer = &mut self.inventory_count_buffers[index];
                        buffer.set_text(
                            &quantity.to_string(),
                            &self.ui_font.colored(Color::rgb(255, 255, 255)),
                            Shaping::Advanced,
                            None,
                        );
                        buffer.shape_until_scroll(&mut self.font_system, false);
                    }
                    for (index, (slot, _)) in counts.iter().enumerate() {
                        let buffer = &self.inventory_count_buffers[index];
                        let (x, y, width, height) = layout.slot_rect(*slot);
                        text_areas.push(TextArea {
                            buffer,
                            left: x + width - 24.0,
                            top: y + height - 25.0,
                            scale: 1.0,
                            bounds: TextBounds {
                                left: x as i32,
                                top: y as i32,
                                right: (x + width) as i32,
                                bottom: (y + height) as i32,
                            },
                            default_color: Color::rgb(255, 255, 255),
                            custom_glyphs: &[],
                        });
                    }
                }

                if self.show_crosshair {
                    while self.hotbar_count_buffers.len() < 9 {
                        self.hotbar_count_buffers.push(glyphon::Buffer::new(&mut self.font_system, Metrics::new(19.0, 21.0)));
                    }
                    let slot_width = self.config.width as f32 * 0.04;
                    let gap = self.config.width as f32 * 0.002;
                    let total_width = slot_width * 9.0 + gap * 8.0;
                    let start_x = (self.config.width as f32 - total_width) * 0.5;
                    for slot in 0..9 {
                        let buffer = &mut self.hotbar_count_buffers[slot];
                        let count = self.inventory.get_flat(27 + slot).map(|stack| stack.count.to_string()).unwrap_or_default();
                        let count_attrs = self
                            .ui_font
                            .colored(Color::rgb(255, 255, 255))
                            .weight(glyphon::Weight::BOLD);
                        buffer.set_text(&count, &count_attrs, Shaping::Advanced, None);
                        buffer.shape_until_scroll(&mut self.font_system, false);
                    }
                    for slot in 0..9 {
                        let buffer = &self.hotbar_count_buffers[slot];
                        let left = start_x + slot as f32 * (slot_width + gap) + slot_width - 20.0;
                        // The hotbar quad ends a few pixels above the bottom
                        // edge.  Keep the quantity baseline inside that quad;
                        // the old `height - 31` anchor put the glyph descenders
                        // below the slot on shorter screens.
                        let count_top = (self.config.height as f32 * 0.975 - 25.0).max(0.0);
                        text_areas.push(TextArea { buffer, left, top: count_top, scale: 1.0,
                            bounds: TextBounds { left: left as i32 - 4, top: (count_top as i32 - 4).max(0), right: (left + 28.0) as i32, bottom: (count_top as i32 + 24).min(self.config.height as i32) },
                            default_color: Color::rgb(255, 255, 255), custom_glyphs: &[] });
                    }
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
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(14),
                        end_of_pass_write_index: Some(15),
                    }
                }),
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
        let submit_start = Instant::now();
        self.frame_profile.text_ms = (submit_start - text_start).as_secs_f32() * 1000.0;
        if let (Some(profiler), Some(slot)) =
            (&self.gpu_timestamp_profiler, timestamp_readback_slot)
        {
            profiler.encode_resolve(&mut encoder, slot);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        if let (Some(profiler), Some(slot)) =
            (&mut self.gpu_timestamp_profiler, timestamp_readback_slot)
        {
            profiler.request_readback(slot);
        }
        self.queue.present(output);
        self.frame_profile.submit_ms = submit_start.elapsed().as_secs_f32() * 1000.0;
        self.frame_profile.render_ms = render_start.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    /// Updates the main-menu labels and the optional server-address input.
    /// With anti-aliasing off the main color target has a single sample and
    /// cannot be resolved; copy it to where a resolve would have written.
    fn copy_single_sample_color(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        if self.msaa_samples > 1 {
            return;
        }
        encoder.copy_texture_to_texture(
            self.msaa_texture_view.texture().as_image_copy(),
            target.texture().as_image_copy(),
            wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn prepare_menu_text(&mut self) {
        let context = MenuContext {
            has_world: self.has_entered_world,
            save_available: std::path::Path::new(minerust::DEFAULT_WORLD_FILE).exists(),
        };
        let layout = MenuLayout::new(self.config.width, self.config.height, self.menu_state.screen, context);
        let scale = layout.scale;
        let title = match self.menu_state.screen {
            MenuScreen::Main => "MINERUST",
            MenuScreen::Pause => "PAUSED",
            MenuScreen::Multiplayer => "MULTIPLAYER",
            MenuScreen::Settings => "SETTINGS",
            MenuScreen::Controls => "CONTROLS",
        };
        self.menu_title_buffer.set_text(title, &self.ui_font.attrs(), Shaping::Advanced, None);
        self.menu_title_buffer.set_size(Some(layout.title.w / scale), Some(layout.title.h / scale));
        self.menu_title_buffer.shape_until_scroll(&mut self.font_system, false);

        for (buffer, item) in self.menu_text_buffers.iter_mut().zip(&layout.items) {
            let text = match item.action {
                Some(MenuAction::FocusServerAddress) => {
                    let cursor = self.menu_state.selected_field == MenuField::ServerAddress && !self.menu_state.selected_all;
                    format!("SERVER ADDRESS   {}{}", self.menu_state.server_address, if cursor { "_" } else { "" })
                }
                Some(MenuAction::FocusUsername) => {
                    let cursor = self.menu_state.selected_field == MenuField::Username && !self.menu_state.selected_all;
                    format!("USERNAME   {}{}", self.menu_state.username, if cursor { "_" } else { "" })
                }
                Some(MenuAction::ToggleRenderMode) => match self.config.present_mode {
                    wgpu::PresentMode::Fifo => "RENDER MODE: VSYNC".to_string(),
                    _ => "RENDER MODE: AUTO".to_string(),
                },
                Some(MenuAction::ToggleWater) => self.graphics.water.menu_label().to_string(),
                Some(MenuAction::CycleAntiAliasing) => {
                    let aa = self.graphics.anti_aliasing;
                    if aa.sample_count() == self.msaa_samples { aa.menu_label().to_string() }
                    else { format!("{} (RESTART)", aa.menu_label()) }
                }
                Some(MenuAction::ToggleShadows) => format!("SUN SHADOWS: {}", if self.graphics.shadows { "ON" } else { "OFF" }),
                _ => item.label.clone(),
            };
            buffer.set_text(&text, &self.ui_font.attrs(), Shaping::Advanced, None);
            buffer.set_size(Some(item.rect.w / scale), Some(item.rect.h / scale));
            buffer.shape_until_scroll(&mut self.font_system, false);
        }

        let message = self.menu_state.error_message.as_deref()
            .or(self.menu_state.status_message.as_deref())
            .or_else(|| {
                (!context.save_available && matches!(self.menu_state.screen, MenuScreen::Main | MenuScreen::Pause))
                    .then_some("No saved world found")
            });
        if let Some(message) = message {
            if let Some(buffer) = self.menu_text_buffers.get_mut(layout.items.len()) {
                buffer.set_text(message, &self.ui_font.attrs(), Shaping::Advanced, None);
                buffer.set_size(Some(layout.message.w / scale), Some(layout.message.h / scale));
                buffer.shape_until_scroll(&mut self.font_system, false);
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_visibility_order_matches_full_sort_for_sparse_and_loaded_worlds() {
        for center in [(0, 0), (-21, 34)] {
            for sparse in [false, true] {
                let is_loaded = |cx: i32, cz: i32| !sparse || (cx + cz).rem_euclid(3) == 0;
                let mut expected = Vec::new();
                for cx in center.0 - RENDER_DISTANCE..=center.0 + RENDER_DISTANCE {
                    for cz in center.1 - RENDER_DISTANCE..=center.1 + RENDER_DISTANCE {
                        let (dx, dz) = (cx - center.0, cz - center.1);
                        if dx * dx + dz * dz <= RENDER_DISTANCE * RENDER_DISTANCE
                            && is_loaded(cx, cz)
                        {
                            expected.push((cx, cz));
                        }
                    }
                }
                expected.sort_by_key(|&(cx, cz)| (cx - center.0).pow(2) + (cz - center.1).pow(2));
                let mut actual = vec![(i32::MIN, i32::MIN)];
                collect_visible_columns(&mut actual, center, is_loaded);
                assert_eq!(actual, expected);
                collect_visible_columns(&mut actual, center, |_, _| false);
                assert!(actual.is_empty());
            }
        }
    }
}
