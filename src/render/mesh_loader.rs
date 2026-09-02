use crate::core::quad::PackedQuad;
use crate::world::World;
use crate::world::generator::ChunkGenerator;
use crate::world::terrain::SubchunkMeshSnapshot;
use crossbeam_channel::{Receiver, bounded};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

const MAX_QUEUED_MESH_REQUESTS: usize = 256;

/// A request to build the terrain and water meshes for one subchunk.
pub struct MeshRequest {
    /// X coordinate of the parent chunk column (in chunk units).
    pub cx: i32,
    /// Z coordinate of the parent chunk column (in chunk units).
    pub cz: i32,
    /// Vertical index of the subchunk within its chunk column.
    pub sy: i32,
    /// Lower values are processed first. This is the squared distance from
    /// the player's current subchunk, in subchunk units.
    pub priority: i64,
}

impl PartialEq for MeshRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.cx == other.cx
            && self.cz == other.cz
            && self.sy == other.sy
    }
}

impl Eq for MeshRequest {}

impl PartialOrd for MeshRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeshRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, therefore reverse the distance ordering.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.cx.cmp(&self.cx))
            .then_with(|| other.cz.cmp(&self.cz))
            .then_with(|| other.sy.cmp(&self.sy))
    }
}

struct MeshRequestQueue {
    requests: BinaryHeap<MeshRequest>,
    shutdown: bool,
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
/// Workers pull requests from a shared min-priority heap. This is important
/// while the player is moving: a newly visible subchunk may overtake older,
/// distant work that has not started yet. Completed results use a bounded
/// channel to apply backpressure to the workers. The heap is capped at 256;
/// when full, a closer request replaces its farthest queued entry.
pub struct MeshLoader {
    /// Shared priority queue consumed by all mesh workers.
    request_queue: Arc<(Mutex<MeshRequestQueue>, Condvar)>,
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
        let (result_tx, result_rx) = bounded::<MeshResult>(256);
        let request_queue = Arc::new((
            Mutex::new(MeshRequestQueue {
                requests: BinaryHeap::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let seed = world.read().seed;

        for i in 0..worker_count {
            let request_queue = Arc::clone(&request_queue);
            let tx = result_tx.clone();
            let world = Arc::clone(&world);
            let generator = ChunkGenerator::new(seed);

            thread::Builder::new()
                .name(format!("mesh-worker-{}", i))
                .spawn(move || {
                    loop {
                        let req = {
                            let (lock, wakeup) = &*request_queue;
                            let mut queue = lock.lock().expect("mesh request queue poisoned");
                            while queue.requests.is_empty() && !queue.shutdown {
                                queue = wakeup.wait(queue).expect("mesh request queue poisoned");
                            }
                            if queue.shutdown {
                                break;
                            }
                            queue.requests.pop().expect("non-empty mesh request queue")
                        };
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
            request_queue,
            result_rx,
            pending: HashSet::new(),
        }
    }

    /// Enqueues a mesh-build request for the subchunk at `(cx, cz, sy)`.
    ///
    /// Does nothing if the subchunk is already in the pending set, preventing
    /// redundant in-flight work for the same subchunk.
    ///
    /// `priority` is a squared distance score; lower values run first.
    /// Returns `true` if the subchunk was queued or was already pending, and
    /// `false` if it is no closer than the current bounded queue.
    pub fn request_mesh(&mut self, cx: i32, cz: i32, sy: i32, priority: i64) -> bool {
        let key = (cx, cz, sy);
        if self.pending.contains(&key) {
            return true;
        }
        let (lock, wakeup) = &*self.request_queue;
        let mut queue = lock.lock().expect("mesh request queue poisoned");
        if queue.requests.len() >= MAX_QUEUED_MESH_REQUESTS {
            let farthest = queue
                .requests
                .iter()
                .max_by_key(|request| request.priority)
                .expect("full mesh request queue has an entry");
            if priority >= farthest.priority {
                return false;
            }

            let evicted_key = (farthest.cx, farthest.cz, farthest.sy);
            queue
                .requests
                .retain(|request| (request.cx, request.cz, request.sy) != evicted_key);
            self.pending.remove(&evicted_key);
        }
        queue.requests.push(MeshRequest { cx, cz, sy, priority });
        self.pending.insert(key);
        wakeup.notify_one();
        true
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

impl Drop for MeshLoader {
    fn drop(&mut self) {
        let (lock, wakeup) = &*self.request_queue;
        if let Ok(mut queue) = lock.lock() {
            queue.shutdown = true;
            wakeup.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closest_request_is_popped_first() {
        let mut queue = BinaryHeap::new();
        queue.push(MeshRequest { cx: 4, cz: 0, sy: 0, priority: 16 });
        queue.push(MeshRequest { cx: 1, cz: 0, sy: 0, priority: 1 });
        queue.push(MeshRequest { cx: 2, cz: 0, sy: 0, priority: 4 });

        assert_eq!(queue.pop().unwrap().priority, 1);
        assert_eq!(queue.pop().unwrap().priority, 4);
        assert_eq!(queue.pop().unwrap().priority, 16);
    }
}
