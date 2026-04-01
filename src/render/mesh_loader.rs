use crate::core::vertex::Vertex;
use crate::logger::{LogLevel, log};
use crate::world::World;
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
    /// Terrain mesh as `(vertices, indices)`.
    pub terrain: (Vec<Vertex>, Vec<u32>),
    /// Water mesh as `(vertices, indices)`.
    pub water: (Vec<Vertex>, Vec<u32>),
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

        for i in 0..worker_count {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let world = Arc::clone(&world);

            thread::Builder::new()
                .name(format!("mesh-worker-{}", i))
                .spawn(move || {
                    // Block until a request arrives; exit when the sender is dropped.
                    while let Ok(req) = rx.recv() {
                        let meshes = {
                            // Hold the read lock only for the duration of mesh
                            // building, then release it before sending the result.
                            let world_read = world.read();
                            world_read.build_subchunk_mesh(req.cx, req.cz, req.sy)
                        };

                        if tx
                            .send(MeshResult {
                                cx: req.cx,
                                cz: req.cz,
                                sy: req.sy,
                                terrain: meshes.0,
                                water: meshes.1,
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
    /// If the request channel is currently full the request is dropped and a
    /// warning is logged; the caller should retry on a future frame.  The
    /// subchunk is intentionally *not* added to `pending` in this case so that
    /// the next call for the same key can attempt to enqueue it again.
    pub fn request_mesh(&mut self, cx: i32, cz: i32, sy: i32) {
        let key = (cx, cz, sy);
        if self.pending.contains(&key) {
            return;
        }
        match self.request_tx.try_send(MeshRequest { cx, cz, sy }) {
            Ok(_) => {
                self.pending.insert(key);
            }
            Err(_) => {
                // The request channel is full. The subchunk is intentionally
                // not inserted into `pending` here so the caller can retry it
                // on the next frame once the workers drain the backlog.
                //log(crate::logger::LogLevel::Warning, &format!("Mesh request channel full — dropping request for subchunk ({cx}, {cz}, {sy})"));
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
