use std::time::Instant;

use render3d::{
    BlockType, CHUNK_SIZE, GENERATION_DISTANCE, MAX_CHUNKS_PER_FRAME, MAX_MESH_BUILDS_PER_FRAME,
    SUBCHUNK_HEIGHT, NUM_SUBCHUNKS,
};

use crate::ui;
use crate::multiplayer::network::update_network;

use super::state::{State, WorldSnapshot, WorldWriteOps};

impl State {
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

    pub fn update_subchunk_mesh(&mut self, result: render3d::mesh_loader::MeshResult) {
        let cx = result.cx;
        let cz = result.cz;
        let sy = result.sy;

        let aabb_copy = {
            let mut world = self.world.write();
            let chunk = match world.chunks.get_mut(&(cx, cz)) {
                Some(chunk) => chunk,
                None => return,
            };
            let subchunk = &mut chunk.subchunks[sy as usize];
            let aabb = subchunk.aabb;
            subchunk.num_indices = result.terrain.1.len() as u32;
            subchunk.num_water_indices = result.water.1.len() as u32;
            subchunk.mesh_dirty = false;
            aabb
        };

        let key = render3d::render::indirect::SubchunkKey {
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

        if !terrain_uploaded || !water_uploaded {
            let mut world = self.world.write();
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[sy as usize].mesh_dirty = true;
            }
        }
    }

    pub fn update(&mut self) {
        self.update_network_state();
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let completed_chunks = self.chunk_loader.poll_results(MAX_CHUNKS_PER_FRAME);

        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let player_chunk_moved =
            player_cx != self.last_gen_player_cx || player_cz != self.last_gen_player_cz;

        let snapshot = {
            let world = self.world.read();

            self.camera.update(&*world, dt, &self.input);

            let mut missing_chunks = Vec::new();
            if player_chunk_moved || self.chunk_loader.pending_count() < 32 {
                for cx in (player_cx - GENERATION_DISTANCE)..=(player_cx + GENERATION_DISTANCE) {
                    for cz in
                        (player_cz - GENERATION_DISTANCE)..=(player_cz + GENERATION_DISTANCE)
                    {
                        if !world.chunks.contains_key(&(cx, cz))
                            && !self.chunk_loader.is_pending(cx, cz)
                        {
                            let dx = cx - player_cx;
                            let dz = cz - player_cz;
                            let priority = dx * dx + dz * dz;
                            missing_chunks.push((cx, cz, priority));
                        }
                    }
                }
            }

            let (raycast_result, target_block) = if self.mouse_captured
                && (self.input.left_mouse || self.input.right_mouse) {
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
        };

        if player_chunk_moved {
            self.last_gen_player_cx = player_cx;
            self.last_gen_player_cz = player_cz;
        }

        let mut requests = snapshot.missing_chunks;
        requests.sort_by_key(|&(_, _, priority)| priority);
        for (cx, cz, priority) in requests.into_iter().take(MAX_CHUNKS_PER_FRAME * 2) {
            self.chunk_loader.request_chunk(cx, cz, priority);
        }

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
                            self.digging.progress += dt;
                            if self.digging.progress >= break_time {
                                write_ops.block_break = Some((bx, by, bz));
                                write_ops.mark_dirty.push((bx, by, bz));
                                self.digging.target = None;
                                self.digging.progress = 0.0;
                            }
                        } else {
                            self.digging.target = Some(target);
                            self.digging.progress = 0.0;
                            self.digging.break_time = break_time;
                        }
                    }
                }
            } else {
                self.digging.target = None;
                self.digging.progress = 0.0;
            }
        } else {
            self.digging.target = None;
            self.digging.progress = 0.0;
        }

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
            let removed_chunks =
                world.update_chunks_around_player(self.camera.position.x, self.camera.position.z);
            drop(world);
            self.remove_chunk_gpu_data(&removed_chunks);
        }
        for (bx, by, bz) in write_ops.mark_dirty {
            self.mark_chunk_dirty(bx, by, bz);
        }

        self.is_underwater = if snapshot.eye_block == BlockType::Water {
            1.0
        } else {
            0.0
        };

        self.update_coords_ui();

        // Limit mesh updates per frame to prevent FPS spikes
        let max_mesh_updates = MAX_MESH_BUILDS_PER_FRAME;
        for _ in 0..max_mesh_updates {
            if let Some(result) = self.mesh_loader.poll_result() {
                self.update_subchunk_mesh(result);
            } else {
                break;
            }
        }
    }

    fn remove_chunk_gpu_data(&mut self, removed_chunks: &[(i32, i32)]) {
        for &(cx, cz) in removed_chunks {
            for sy in 0..NUM_SUBCHUNKS {
                let key = render3d::render::indirect::SubchunkKey {
                    chunk_x: cx,
                    chunk_z: cz,
                    subchunk_y: sy,
                };
                self.indirect_manager.remove_subchunk(&self.queue, key);
                self.water_indirect_manager.remove_subchunk(&self.queue, key);
            }
        }
    }

    pub fn mark_chunk_dirty(&mut self, x: i32, y: i32, z: i32) {
        let cx = (x as f32 / CHUNK_SIZE as f32).floor() as i32;
        let cz = (z as f32 / CHUNK_SIZE as f32).floor() as i32;
        let sy = y / SUBCHUNK_HEIGHT;

        let mut world = self.world.write();

        if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
            if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                chunk.subchunks[sy as usize].mesh_dirty = true;
            }
        }

        let lx = x.rem_euclid(CHUNK_SIZE);
        let lz = z.rem_euclid(CHUNK_SIZE);
        let ly = y.rem_euclid(SUBCHUNK_HEIGHT);

        if lx == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx - 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lx == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx + 1, cz)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lz == 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz - 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if lz == CHUNK_SIZE - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz + 1)) {
                if sy >= 0 && (sy as usize) < chunk.subchunks.len() {
                    chunk.subchunks[sy as usize].mesh_dirty = true;
                }
            }
        }
        if ly == 0 && sy > 0 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy - 1) as usize].mesh_dirty = true;
            }
        }
        if ly == SUBCHUNK_HEIGHT - 1 && sy < NUM_SUBCHUNKS - 1 {
            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                chunk.subchunks[(sy + 1) as usize].mesh_dirty = true;
            }
        }
    }

    /// Internal wrapper for update_network to avoid name collision
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

