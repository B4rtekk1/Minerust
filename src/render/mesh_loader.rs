use crate::core::quad::PackedQuad;
use crate::world::World;
use crate::world::generator::ChunkGenerator;
use crate::world::terrain::SubchunkMeshSnapshot;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

/// A request to build the terrain and water meshes for one subchunk.
pub struct MeshRequest {
    /// X coordinate of the parent chunk column (in chunk units).
    pub cx: i32,
    /// Z coordinate of the parent chunk column (in chunk units).
    pub cz: i32,
    /// Vertical index of the subchunk within its chunk column.
    pub sy: i32,
}

/// The completed mesh data produced by a worker thread for one subchunk.
pub struct MeshResult {
    /// X coordinate of the parent chunk column (in chunk units).
    pub cx: i32,
    /// Z coordinate of the parent chunk column (in chunk units).
    pub cz: i32,
    /// Vertical index of the subchunk within its chunk column.
    pub sy: i32,
    /// Mesh-input revision captured with the world snapshot used for this mesh.
    pub mesh_version: u64,
    /// Terrain quads, expanded procedurally by the vertex shader.
    pub terrain: Vec<PackedQuad>,
    /// Water quads, expanded procedurally by the vertex shader.
    pub water: Vec<PackedQuad>,
    /// Padded voxel input for GPU face extraction.  Present only when all
    /// relevant blocks are full terrain cubes; custom geometry uses `None`.
    pub gpu_snapshot: Option<SubchunkMeshSnapshot>,
}

/// Asynchronous mesh-building system backed by a fixed pool of worker threads.
///
/// The main thread submits [`MeshRequest`]s via [`request_mesh`] and collects
/// finished [`MeshResult`]s by calling [`poll_result`] once per frame.  A
/// `pending` set prevents the same subchunk from being queued more than once
/// at a time.
///
/// # Channel capacities
/// Both the request and result channels are bounded to 256 entries.  If the
/// request channel is full, [`request_mesh`] silently drops the request; the
/// caller is expected to retry on a future frame.
pub struct MeshLoader {
    /// Sending half of the request channel shared with all worker threads.
    request_tx: Sender<MeshRequest>,
    /// Receiving half of the result channel; workers write completed meshes here.
    result_rx: Receiver<MeshResult>,
    /// Set of subchunk keys `(cx, cz, sy)` that have been queued but not yet
    /// collected, used to deduplicate in-flight requests.
    pending: HashSet<(i32, i32, i32)>,
}

impl MeshLoader {
    /// Creates a `MeshLoader` and spawns `worker_count` background mesh-builder threads.
    ///
    /// Each worker receives requests from a shared bounded channel, acquires a
    /// read lock on `world` to build the mesh, then sends the result back on a
    /// second bounded channel.  Workers exit cleanly when the request channel is
    /// dropped (i.e. when the `MeshLoader` itself is dropped).
    ///
    /// # Panics
    /// Panics if any worker thread cannot be spawned.
    pub fn new(world: Arc<parking_lot::RwLock<World>>, worker_count: usize) -> Self {
        let (request_tx, request_rx) = bounded::<MeshRequest>(256);
        let (result_tx, result_rx) = bounded::<MeshResult>(256);
        let seed = world.read().seed;

        for i in 0..worker_count {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let world = Arc::clone(&world);
            let generator = ChunkGenerator::new(seed);

            thread::Builder::new()
                .name(format!("mesh-worker-{}", i))
                .spawn(move || {
                    // Block until a request arrives; exit when the sender is dropped.
                    while let Ok(req) = rx.recv() {
                        let snapshot = {
                            // Hold the read lock only while copying the padded
                            // block cache needed for meshing.
                            let world_read = world.read();
                            world_read.snapshot_subchunk_mesh(req.cx, req.cz, req.sy)
                        };

                        let Some(snapshot) = snapshot else {
                            continue;
                        };

                        let gpu_snapshot =
                            crate::render::gpu_mesher::GpuFaceMesher::supports(&snapshot)
                                .then_some(snapshot.clone());
                        let meshes = if gpu_snapshot.is_some() {
                            (Vec::new(), Vec::new())
                        } else {
                            World::build_subchunk_mesh_from_snapshot(&generator, &snapshot)
                        };

                        if tx
                            .send(MeshResult {
                                cx: req.cx,
                                cz: req.cz,
                                sy: req.sy,
                                mesh_version: snapshot.mesh_version,
                                terrain: meshes.0,
                                water: meshes.1,
                                gpu_snapshot,
                            })
                            .is_err()
                        {
                            // The result receiver has been dropped — the
                            // MeshLoader is shutting down, so exit the loop.
                            break;
                        }
                    }
                })
                .expect("Failed to spawn mesh worker");
        }

        Self {
            request_tx,
            result_rx,
            pending: HashSet::new(),
        }
    }

    /// Enqueues a mesh-build request for the subchunk at `(cx, cz, sy)`.
    ///
    /// Does nothing if the subchunk is already in the pending set, preventing
    /// redundant in-flight work for the same subchunk.
    ///
    /// Returns `true` if the subchunk was queued or was already pending.
    /// Returns `false` if the request channel is currently full; callers should
    /// retry on a future frame.
    pub fn request_mesh(&mut self, cx: i32, cz: i32, sy: i32) -> bool {
        let key = (cx, cz, sy);
        if self.pending.contains(&key) {
            return true;
        }
        match self.request_tx.try_send(MeshRequest { cx, cz, sy }) {
            Ok(_) => {
                self.pending.insert(key);
                true
            }
            Err(_) => {
                // The request channel is full. The subchunk is intentionally
                // not inserted into `pending` here so the caller can retry it
                // on the next frame once the workers drain the backlog.
                //log(crate::logger::LogLevel::Warning, &format!("Mesh request channel full — dropping request for subchunk ({cx}, {cz}, {sy})"));
                false
            }
        }
    }

    /// Returns the next completed mesh result without blocking, or `None` if
    /// no results are currently available.
    ///
    /// Removes the corresponding entry from the pending set so the subchunk
    /// can be re-requested later if needed.
    pub fn poll_result(&mut self) -> Option<MeshResult> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.pending.remove(&(result.cx, result.cz, result.sy));
                Some(result)
            }
            Err(_) => None,
        }
    }

    /// Returns `true` if a mesh request for `(cx, cz, sy)` has been enqueued
    /// but its result has not yet been collected.
    pub fn is_pending(&self, cx: i32, cz: i32, sy: i32) -> bool {
        self.pending.contains(&(cx, cz, sy))
    }
}
