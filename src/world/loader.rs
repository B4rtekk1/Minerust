use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crossbeam_channel::{Receiver, TryRecvError, bounded};

use crate::core::chunk::Chunk;
use crate::world::generator::ChunkGenerator;

// ─────────────────────────────────────────────────────────────────────────────
// Request / result types
// ─────────────────────────────────────────────────────────────────────────────

/// A request to generate the chunk at column `(cx, cz)`.
///
/// Requests are ordered by `priority` so that the caller can ensure
/// nearby chunks are generated before distant ones.  A **lower** raw
/// priority value means *higher* urgency — the `Ord` impl reverses the
/// comparison so that `BinaryHeap` (a max-heap) pops the most urgent
/// request first.
#[derive(Clone)]
pub struct ChunkGenRequest {
    /// Chunk column X coordinate (in chunks, not blocks).
    pub cx: i32,
    /// Chunk column Z coordinate (in chunks, not blocks).
    pub cz: i32,
    /// Urgency score.  Typically the squared chunk-distance from the camera:
    /// `dx² + dz²`, so closer chunks have a smaller (more urgent) value.
    pub priority: i32,
}

impl PartialEq for ChunkGenRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.cx == other.cx && self.cz == other.cz
    }
}

impl Eq for ChunkGenRequest {}

impl PartialOrd for ChunkGenRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkGenRequest {
    /// Reverses the natural integer ordering so that a **min-priority**
    /// (closest chunk) is treated as the *maximum* by `BinaryHeap`.
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the numeric priority for a min-priority heap, then use
        // coordinates as a deterministic tie-breaker.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.cx.cmp(&self.cx))
            .then_with(|| other.cz.cmp(&self.cz))
    }
}

struct RequestQueue {
    requests: BinaryHeap<ChunkGenRequest>,
    shutdown: bool,
}

/// The completed result of a chunk generation request.
pub struct ChunkGenResult {
    /// Chunk column X coordinate (mirrors the originating request).
    pub cx: i32,
    /// Chunk column Z coordinate (mirrors the originating request).
    pub cz: i32,
    /// The fully-generated chunk data, ready to be inserted into the world.
    pub chunk: Chunk,
}

// ─────────────────────────────────────────────────────────────────────────────
// ChunkLoader
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a pool of background threads that generate [`Chunk`] data
/// asynchronously from the main game loop.
///
/// # Architecture
///
/// ```text
///  Main thread                   Worker threads (N)
///  ───────────                   ──────────────────
///  request_chunk(cx, cz, prio)
///    → request_tx ──────────────→ request_rx → generate_chunk()
///                                              → result_tx ─────→ result_rx
///  poll_results()  ←──────────────────────────────────────────────┘
/// ```
///
/// A shared [`BinaryHeap`] gives requests to workers in priority order. The
/// result channel is bounded at 256 entries; workers send
///   [`ChunkGenResult`] values back; the main thread drains them each frame via
///   [`poll_results`].
///
/// A `HashSet<(i32, i32)>` called `pending` tracks which chunk columns have
/// been submitted but not yet returned.  This prevents duplicate requests and
/// lets callers query in-flight status without round-tripping through the channel.
///
/// # Worker lifecycle
///
/// Each worker owns its own [`ChunkGenerator`] (seeded identically from the
/// world seed), waits while the priority queue is empty, and exits when the
/// loader is dropped.
///
/// # Backpressure
///
/// The request heap is capped at 256 pending entries. Additional requests are
/// ignored until results are committed, keeping the game loop non-blocking.
pub struct ChunkLoader {
    /// Shared priority queue consumed by all generation workers.
    request_queue: Arc<(Mutex<RequestQueue>, Condvar)>,
    /// Receiver half of the result channel; polled each frame by the main thread.
    result_rx: Receiver<ChunkGenResult>,
    /// Set of chunk columns that have been submitted and not yet received.
    /// Used to deduplicate requests and answer `is_pending` queries cheaply.
    pending: HashSet<(i32, i32)>,
    /// Number of worker threads created at construction time.
    worker_count: usize,
}

impl ChunkLoader {
    /// Creates a loader with the default worker count from [`get_chunk_worker_count`].
    ///
    /// The worker count is typically `num_physical_cpus - 1` (clamped to a
    /// sensible range) so the main render thread retains at least one core.
    pub fn new(seed: u32) -> Self {
        Self::with_worker_count(crate::constants::get_chunk_worker_count(), seed)
    }

    /// Creates a loader with exactly `num_workers` background threads.
    ///
    /// Each worker thread:
    /// 1. Waits on the shared priority queue and pops its closest request.
    /// 2. Constructs an independent [`ChunkGenerator`] from `seed` so no
    ///    generator state is shared between threads.
    /// 3. Generates chunks on demand and sends results back via `result_tx`.
    /// 4. Exits when `ChunkLoader` signals shutdown.
    ///
    /// # Panics
    /// Panics if any worker thread cannot be spawned.
    pub fn with_worker_count(num_workers: usize, seed: u32) -> Self {
        let (result_tx, result_rx) = bounded::<ChunkGenResult>(256);
        let request_queue = Arc::new((
            Mutex::new(RequestQueue {
                requests: BinaryHeap::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));

        for worker_id in 0..num_workers {
            let request_queue = Arc::clone(&request_queue);
            let tx = result_tx.clone();
            // Each worker owns its own generator — no mutex needed.
            let generator = ChunkGenerator::new(seed);

            thread::Builder::new()
                .name(format!("chunk-gen-{}", worker_id))
                .spawn(move || {
                    loop {
                        let req = {
                            let (lock, wakeup) = &*request_queue;
                            let mut queue = lock.lock().expect("chunk request queue poisoned");
                            while queue.requests.is_empty() && !queue.shutdown {
                                queue = wakeup.wait(queue).expect("chunk request queue poisoned");
                            }
                            if queue.shutdown {
                                break;
                            }
                            queue.requests.pop().expect("non-empty chunk request queue")
                        };

                        let chunk = generator.generate_chunk(req.cx, req.cz);
                        // If the result channel is disconnected (main thread
                        // dropped ChunkLoader), exit cleanly.
                        if tx
                            .send(ChunkGenResult {
                                cx: req.cx,
                                cz: req.cz,
                                chunk,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .expect("Failed to spawn chunk generation worker");
        }

        ChunkLoader {
            request_queue,
            result_rx,
            pending: HashSet::new(),
            worker_count: num_workers,
        }
    }

    // ── Request submission ────────────────────────────────────────────────── //

    /// Submits a request to generate the chunk at `(cx, cz)` with the given
    /// `priority`.
    ///
    /// The request is silently ignored if:
    /// - `(cx, cz)` is already in the `pending` set (deduplication).
    /// - The pending request cap has been reached.
    pub fn request_chunk(&mut self, cx: i32, cz: i32, priority: i32) {
        if self.pending.contains(&(cx, cz)) {
            return; // already in flight
        }

        if self.pending.len() >= 256 {
            return;
        }
        self.pending.insert((cx, cz));
        let (lock, wakeup) = &*self.request_queue;
        let mut queue = lock.lock().expect("chunk request queue poisoned");
        queue.requests.push(ChunkGenRequest { cx, cz, priority });
        wakeup.notify_one();
    }

    /// Submits multiple chunk requests in a single call, sorted by priority
    /// before insertion so the most urgent chunks enter the channel first.
    ///
    /// Requests for chunks already in `pending` are filtered out before
    /// sorting.  Submission stops early when the `pending` set reaches 256
    /// entries to prevent unbounded memory growth (the channel capacity is
    /// also 256, so additional entries would be dropped by `try_send` anyway).
    ///
    /// # Parameters
    /// - `requests` – Slice of `(cx, cz, priority)` tuples.
    pub fn request_chunks(&mut self, requests: &[(i32, i32, i32)]) {
        // Filter duplicates and sort ascending by priority (lowest = most urgent).
        let mut sorted: Vec<_> = requests
            .iter()
            .filter(|(cx, cz, _)| !self.pending.contains(&(*cx, *cz)))
            .collect();
        sorted.sort_by_key(|(_, _, priority)| *priority);

        for (cx, cz, priority) in sorted {
            // Hard cap at 256 pending to match the channel capacity.
            if self.pending.len() >= 256 {
                break;
            }
            self.request_chunk(*cx, *cz, *priority);
        }
    }

    // ── Status queries ────────────────────────────────────────────────────── //

    /// Returns `true` if a generation request for `(cx, cz)` has been
    /// submitted but the result has not yet been polled.
    ///
    /// This is used by the render loop to avoid re-submitting requests for
    /// chunks that are already being generated on a worker thread.
    pub fn is_pending(&self, cx: i32, cz: i32) -> bool {
        self.pending.contains(&(cx, cz))
    }

    /// Returns the number of chunk columns currently in flight (submitted but
    /// not yet polled).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // ── Result collection ─────────────────────────────────────────────────── //

    /// Drains up to `max_results` completed chunks from the result channel
    /// without blocking.
    ///
    /// Each successful receive removes the corresponding `(cx, cz)` from
    /// `pending` so future calls to `is_pending` return `false`.  The loop
    /// exits early on either `Empty` (no more results ready) or `Disconnected`
    /// (all workers have exited).
    ///
    /// # Returns
    /// A `Vec` of up to `max_results` [`ChunkGenResult`] values, in the order
    /// they were completed by the workers (which may differ from submission
    /// order if workers process chunks at different speeds).
    pub fn poll_results(&mut self, max_results: usize) -> Vec<ChunkGenResult> {
        let mut results = Vec::with_capacity(max_results);

        for _ in 0..max_results {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    self.pending.remove(&(result.cx, result.cz));
                    results.push(result);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        results
    }

    /// Convenience wrapper that drains up to 64 results per call.
    ///
    /// 64 is chosen to limit the amount of mesh-upload work done in a single
    /// frame while still clearing the result backlog quickly during fast travel
    /// or initial world load.
    pub fn poll_all_results(&mut self) -> Vec<ChunkGenResult> {
        self.poll_results(64)
    }

    // ── Cancellation ─────────────────────────────────────────────────────── //

    /// Removes `(cx, cz)` from the `pending` set without cancelling the
    /// in-flight request.
    ///
    /// The worker will still generate the chunk and send the result; the caller
    /// simply stops tracking it.  The result will be received by the next
    /// `poll_results` call and can be discarded at that point if no longer needed.
    ///
    /// True mid-flight cancellation is not supported because the bounded
    /// channel does not allow removing arbitrary elements once enqueued.
    pub fn cancel(&mut self, cx: i32, cz: i32) {
        self.pending.remove(&(cx, cz));
    }

    /// Clears the entire `pending` set.
    ///
    /// Like [`cancel`], this does not prevent already-enqueued requests from
    /// being processed by the workers.  Any results that arrive after this
    /// call will be received by [`poll_results`] but the caller is responsible
    /// for deciding whether to use or discard them.
    ///
    /// Typically called when the player teleports or the world is reloaded and
    /// all in-flight generation is no longer relevant.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    // ── Introspection ─────────────────────────────────────────────────────── //

    /// Returns the number of worker threads managed by this loader.
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }
}

impl Drop for ChunkLoader {
    fn drop(&mut self) {
        let (lock, wakeup) = &*self.request_queue;
        if let Ok(mut queue) = lock.lock() {
            queue.shutdown = true;
            wakeup.notify_all();
        }
    }
}
