use std::time::Instant;

use minerust::{
    BlockType, CHUNK_SIZE, GENERATION_DISTANCE, MAX_CHUNKS_PER_FRAME, MAX_MESH_BUILDS_PER_FRAME,
    NUM_SUBCHUNKS, SUBCHUNK_HEIGHT,
};

use crate::multiplayer::network::update_network;
use crate::ui;

use super::state::{State, WorldSnapshot, WorldWriteOps};

impl State {
    /// Rebuilds the on-screen coordinate HUD if the camera has moved since the
    /// last update.
    ///
    /// When the player position changes, new vertex and index buffers are
    /// generated and stored on `self` so the next render pass picks them up.
    /// Does nothing if the position is unchanged.
    pub fn update_coords_ui(&mut self) {
        if let Some((vb, ib, num_indices)) = ui::ui::update_coords_ui(
            &self.device,
            self.camera.position,
            &mut self.last_coords_position,
        ) {
            self.coords_vertex_buffer = Some(vb);
            self.coords_index_buffer = Some(ib);
            self.coords_num_indices = num_indices;
        }
    }

    /// Applies a completed mesh result to the GPU indirect draw buffers.
    ///
    /// Called after a background mesh worker finishes building a subchunk.
    /// The method:
    /// 1. Updates the subchunk's index counts and clears its `mesh_dirty` flag
    ///    under a brief write lock.
    /// 2. Uploads the terrain and water meshes to the respective
    ///    `IndirectManager` instances.
    /// 3. If either upload fails (buffer full), marks the subchunk dirty again
    ///    so it will be retried on the next frame.
    ///
    /// Does nothing if the parent chunk has been unloaded since the mesh was
    /// requested.
    pub fn update_subchunk_mesh(&mut self, result: minerust::mesh_loader::MeshResult) {
        let cx = result.cx;
        let cz = result.cz;
        let sy = result.sy;

        // Update CPU-side bookkeeping under a short write lock, then release
        // before touching the GPU buffers to minimize lock contention.
        let aabb_copy = {
            let mut world = self.world.write();
            let chunk = match world.chunks.get_mut(&(cx, cz)) {
                Some(chunk) => chunk,
                None => return, // Chunk was unloaded while the mesh was in flight.
            };
            let subchunk = &mut chunk.subchunks[sy as usize];
            let aabb = subchunk.aabb;
            subchunk.num_indices = result.terrain.1.len() as u32;
            subchunk.num_water_indices = result.water.1.len() as u32;
            subchunk.mesh_dirty = false;
            aabb
        };

        let key = minerust::render::indirect::SubchunkKey {
            chunk_x: cx,
            chunk_z: cz,
            subchunk_y: sy,
        };

        let terrain_uploaded = self.indirect_manager.upload_subchunk(
            &self.queue,
            key,
            &result.terrain.0,
            &result.terrain.1,
            &aabb_copy,
        );

        let water_uploaded = self.water_indirect_manager.upload_subchunk(
            &self.queue,
            key,
            &result.water.0,
            &result.water.1,
            &aabb_copy,
        );

        // If either buffer was full the upload was skipped; re-dirty the
        // subchunk so the mesh is requested again once space becomes available.
        if !terrain_uploaded || !water_uploaded {
            let mut world = self.world.write();
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[sy as usize].mesh_dirty = true;
            }
        }
    }

    /// Main per-frame update: advances physics, processes input, loads chunks,
    /// builds meshes, and applies world mutations.
    ///
    /// This method coordinates all frame-rate-dependent logic in a deliberate
    /// order to minimize the time the world write-lock is held:
    ///
    /// 1. **Network** – flush incoming packets and send position updates.
    /// 2. **Delta time** – compute `dt`, clamped to 100 ms to survive hitches.
    /// 3. **Chunk streaming** – poll completed chunk generation results and
    ///    determine which chunks are still missing within `GENERATION_DISTANCE`.
    /// 4. **Read-locked snapshot** – run camera physics and collect all
    ///    read-only world queries (raycast, eye-block check) in one pass to
    ///    avoid repeated lock acquisitions.
    /// 5. **Chunk requests** – sort missing chunks by squared distance and
    ///    submit up to `MAX_CHUNKS_PER_FRAME * 2` requests to the loader.
    /// 6. **Digging** – accumulate break progress for the targeted block.
    /// 7. **World write** – insert newly generated chunks, break blocks, and
    ///    evict out-of-range chunks (all in a single write-lock window).
    /// 8. **Mesh uploads** – drain up to `MAX_MESH_BUILDS_PER_FRAME` completed
    ///    mesh results from the background workers.
    pub fn update(&mut self) {
        // --- 1. Network ---
        self.update_network_state();

        // --- 2. Delta time ---
        let now = Instant::now();
        // Cap dt to 100 ms so a single long frame doesn't cause the player to
        // tunnel through terrain or fly out of bounds.
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // --- 3. Chunk streaming ---
        let completed_chunks = self.chunk_loader.poll_results(MAX_CHUNKS_PER_FRAME);

        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let player_chunk_moved =
            player_cx != self.last_gen_player_cx || player_cz != self.last_gen_player_cz;

        // --- 4. Read-locked snapshot ---
        // Acquire the read lock once and do all read-only queries inside a
        // single block so the lock is held for the shortest possible time.
        let snapshot = {
            let world = self.world.read();

            self.camera.update(&*world, dt, &self.input);

            // Collect chunks that need to be generated.
            let mut missing_chunks = Vec::new();
            if player_chunk_moved || self.chunk_loader.pending_count() < 32 {
                for cx in (player_cx - GENERATION_DISTANCE)..=(player_cx + GENERATION_DISTANCE) {
                    for cz in (player_cz - GENERATION_DISTANCE)..=(player_cz + GENERATION_DISTANCE)
                    {
                        if !world.chunks.contains_key(&(cx, cz))
                            && !self.chunk_loader.is_pending(cx, cz)
                        {
                            // Use squared distance as the priority so nearer
                            // chunks are generated first (no sqrt needed).
                            let dx = cx - player_cx;
                            let dz = cz - player_cz;
                            let priority = dx * dx + dz * dz;
                            missing_chunks.push((cx, cz, priority));
                        }
                    }
                }
            }

            // Raycast whenever the player is actively controlling the camera
            // so the targeted block outline stays visible without requiring a
            // mouse button press.
            let (raycast_result, target_block) = if self.mouse_captured {
                let raycast = self.camera.raycast(&*world, 5.0);
                if let Some((bx, by, bz, _, _, _)) = raycast {
                    let block = world.get_block(bx, by, bz);
                    (Some((bx, by, bz, 0, 0, 0)), Some(block))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            // Check which block the camera eye is inside (used for the
            // underwater post-process effect).
            let eye_pos = self.camera.eye_position();
            let eye_block = world.get_block(
                eye_pos.x.floor() as i32,
                eye_pos.y.floor() as i32,
                eye_pos.z.floor() as i32,
            );

            WorldSnapshot {
                missing_chunks,
                raycast_result,
                target_block,
                eye_block,
            }
        }; // Read lock released here.

        self.highlighted_block = snapshot
            .raycast_result
            .map(|(bx, by, bz, _, _, _)| (bx, by, bz));

        // Update the cached player chunk position after releasing the lock.
        if player_chunk_moved {
            self.last_gen_player_cx = player_cx;
            self.last_gen_player_cz = player_cz;
        }

        // --- 5. Chunk requests ---
        // Sort by ascending priority (smallest squared distance first) and cap
        // at twice the per-frame chunk limit to allow some look-ahead.
        let mut requests = snapshot.missing_chunks;
        requests.sort_by_key(|&(_, _, priority)| priority);
        for (cx, cz, priority) in requests.into_iter().take(MAX_CHUNKS_PER_FRAME * 2) {
            self.chunk_loader.request_chunk(cx, cz, priority);
        }

        // --- 6. Digging ---
        let mut write_ops = WorldWriteOps {
            completed_chunks: completed_chunks
                .into_iter()
                .map(|r| (r.cx, r.cz, r.chunk))
                .collect(),
            block_break: None,
            mark_dirty: Vec::new(),
        };

        if self.input.left_mouse {
            if let Some(target_block) = snapshot.target_block {
                if let Some((bx, by, bz, _, _, _)) = snapshot.raycast_result {
                    let target = (bx, by, bz);
                    let break_time = target_block.break_time();

                    if break_time.is_finite() && break_time > 0.0 {
                        if self.digging.target == Some(target) {
                            // Continue accumulating break progress on the same block.
                            self.digging.progress += dt;
                            if self.digging.progress >= break_time {
                                // Block fully broken — schedule removal.
                                write_ops.block_break = Some((bx, by, bz));
                                write_ops.mark_dirty.push((bx, by, bz));
                                self.digging.target = None;
                                self.digging.progress = 0.0;
                            }
                        } else {
                            // Player switched to a different block; reset progress.
                            self.digging.target = Some(target);
                            self.digging.progress = 0.0;
                            self.digging.break_time = break_time;
                        }
                    }
                }
            } else {
                // Mouse held but no block targeted (e.g. looking at sky).
                self.digging.target = None;
                self.digging.progress = 0.0;
            }
        } else {
            // Left mouse released — cancel any in-progress dig.
            self.digging.target = None;
            self.digging.progress = 0.0;
        }

        // --- 7. World write ---
        // Batch all mutations into a single write-lock window to minimize
        // contention with background generation and mesh threads.
        if !write_ops.completed_chunks.is_empty()
            || write_ops.block_break.is_some()
            || !write_ops.mark_dirty.is_empty()
        {
            let mut world = self.world.write();

            for (cx, cz, chunk) in write_ops.completed_chunks {
                world.chunks.insert((cx, cz), chunk);
            }

            if let Some((bx, by, bz)) = write_ops.block_break {
                world.set_block_player(bx, by, bz, BlockType::Air);
            }

            // Evict chunks that have moved outside the generation radius and
            // collect their identifiers so their GPU data can be freed below.
            let removed_chunks =
                world.update_chunks_around_player(self.camera.position.x, self.camera.position.z);

            drop(world); // Release the write lock before GPU work.

            self.remove_chunk_gpu_data(&removed_chunks);
        }

        // Dirty-marking runs outside the write-lock window above to avoid
        // holding the lock across the full neighbor scan.
        for (bx, by, bz) in write_ops.mark_dirty {
            self.mark_chunk_dirty(bx, by, bz);
        }

        // Update the underwater post-process uniform.
        self.is_underwater = if snapshot.eye_block == BlockType::Water {
            1.0
        } else {
            0.0
        };

        self.update_coords_ui();

        // --- 8. Mesh uploads ---
        // Drain completed mesh results up to the per-frame cap so a burst of
        // ready meshes doesn't cause a single-frame GPU upload spike.
        for _ in 0..MAX_MESH_BUILDS_PER_FRAME {
            if let Some(result) = self.mesh_loader.poll_result() {
                self.update_subchunk_mesh(result);
            } else {
                break;
            }
        }
    }

    /// Removes all GPU terrain and water mesh data for the given chunk columns.
    ///
    /// Iterates over every subchunk slot in each column and calls
    /// `remove_subchunk` on both indirect managers, zeroing the corresponding
    /// metadata slots so the GPU culling pass stops issuing draw calls for them.
    fn remove_chunk_gpu_data(&mut self, removed_chunks: &[(i32, i32)]) {
        for &(cx, cz) in removed_chunks {
            for sy in 0..NUM_SUBCHUNKS {
                let key = minerust::render::indirect::SubchunkKey {
                    chunk_x: cx,
                    chunk_z: cz,
                    subchunk_y: sy,
                };
                self.indirect_manager.remove_subchunk(&self.queue, key);
                self.water_indirect_manager.remove_subchunk(&self.queue, key);
            }
        }
    }

    /// Marks the subchunk containing block `(x, y, z)` and all six of its
    /// face-adjacent neighbors as dirty so their meshes are rebuilt.
    ///
    /// Neighbor dirtying is needed because a block on a chunk or subchunk
    /// boundary affects the visible faces of the adjacent chunk/subchunk.
    /// The six directions checked are:
    /// - **±X**: the chunk columns to the west and east.
    /// - **±Z**: the chunk columns to the north and south.
    /// - **±Y**: the subchunks directly below and above within the same column.
    ///
    /// Checks are bounds-guarded; out-of-range subchunk indices or absent
    /// chunks are silently skipped.
    pub fn mark_chunk_dirty(&mut self, x: i32, y: i32, z: i32) {
        let cx = (x as f32 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (z as f32 / CHUNK_SIZE as f32).floor() as i32;
        let sy = y / SUBCHUNK_HEIGHT;

        let mut world = self.world.write();

        // Mark the subchunk that owns this block.
        if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
            if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                chunk.subchunks[sy as usize].mesh_dirty = true;
            }
        }

        // Local coordinates within the chunk / subchunk — used to detect
        // whether the block lies on a boundary face.
        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);
        let ly = y.rem_euclid(SUBCHUNK_HEIGHT);

        // West neighbor (block is on the -X face of its chunk column).
        if lx == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx - 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        // East neighbor (block is on the +X face of its chunk column).
        if lx == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx + 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        // North neighbor (block is on the -Z face of its chunk column).
        if lz == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz - 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        // South neighbor (block is on the +Z face of its chunk column).
        if lz == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz + 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        // Subchunk below (block is on the bottom face of its subchunk).
        if ly == 0 && sy > 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy - 1) as usize].mesh_dirty = true;
            }
        }
        // Subchunk above (block is on the top face of its subchunk).
        if ly == SUBCHUNK_HEIGHT - 1 && sy < NUM_SUBCHUNKS - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy + 1) as usize].mesh_dirty = true;
            }
        }
    }

    /// Forwards all pending network events to the multiplayer subsystem.
    ///
    /// Sends the local player's current position, yaw, and pitch, processes
    /// incoming state updates from remote players, and handles lobby/game-state
    /// transitions.  Called at the very start of each frame so network state is
    /// fresh before any physics or world queries run.
    fn update_network_state(&mut self) {
        update_network(
            &mut self.my_player_id,
            &self.camera.position,
            self.camera.yaw,
            self.camera.pitch,
            &mut self.last_position_send,
            &self.network_tx,
            &mut self.network_rx,
            &mut self.remote_players,
            &mut self.game_state,
            &mut self.mouse_captured,
            &self.window,
        );
    }
}
