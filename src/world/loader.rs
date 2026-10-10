use crate::constants::{GENERATION_DISTANCE, MAX_PENDING_CHUNKS};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
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

// Reuse one deterministic near-to-far order across frames and loader instances.
static GENERATION_OFFSETS: LazyLock<Vec<(i32, i32, i32)>> = LazyLock::new(|| {
    let mut offsets: Vec<_> = (-GENERATION_DISTANCE..=GENERATION_DISTANCE)
        .flat_map(|dx| {
            (-GENERATION_DISTANCE..=GENERATION_DISTANCE).map(move |dz| (dx, dz, dx * dx + dz * dz))
        })
        .collect();
    offsets.sort_unstable_by_key(|&(dx, dz, priority)| (priority, dx, dz));
    offsets
});

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
    scan_center: Option<(i32, i32)>,
    scan_cursor: usize,
}

impl ChunkLoader {
    /// Creates a loader with the default worker count from [`get_chunk_worker_count`].
    ///
    /// Generation and meshing share a bounded CPU budget so foreground
    /// rendering retains CPU time.
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
        let (result_tx, result_rx) = bounded::<ChunkGenResult>(MAX_PENDING_CHUNKS);
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
            scan_center: None,
            scan_cursor: 0,
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

        if self.pending.len() >= MAX_PENDING_CHUNKS {
            return;
        }
        self.pending.insert((cx, cz));
        let (lock, wakeup) = &*self.request_queue;
        let mut queue = lock.lock().expect("chunk request queue poisoned");
        queue.requests.push(ChunkGenRequest { cx, cz, priority });
        wakeup.notify_one();
    }

    /// Incrementally visits the generation square in nearest-first order.
    /// Each position is checked once per player chunk, rather than rescanning
    /// the full square each frame. `is_loaded` must also include results polled
    /// this frame but not yet inserted into the world.
    ///
    /// Replace the loader when replacing the world. For an external removal
    /// inside the generation square, call `clear_pending` to restart the scan.
    pub fn request_missing_chunks(
        &mut self,
        center: (i32, i32),
        max_requests: usize,
        mut is_loaded: impl FnMut(i32, i32) -> bool,
    ) {
        if self.scan_center != Some(center) {
            self.scan_center = Some(center);
            self.scan_cursor = 0;
        }
        let budget = max_requests.min(MAX_PENDING_CHUNKS.saturating_sub(self.pending.len()));
        let mut submitted = 0;
        while submitted < budget && self.scan_cursor < GENERATION_OFFSETS.len() {
            let (dx, dz, priority) = GENERATION_OFFSETS[self.scan_cursor];
            self.scan_cursor += 1;
            let (cx, cz) = (center.0 + dx, center.1 + dz);
            if self.is_pending(cx, cz) || is_loaded(cx, cz) {
                continue;
            }
            self.request_chunk(cx, cz, priority);
            submitted += 1;
        }
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
            if self.pending.len() >= MAX_PENDING_CHUNKS {
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
        self.scan_center = None;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn queued(loader: &ChunkLoader) -> Vec<(i32, i32, i32)> {
        let (lock, _) = &*loader.request_queue;
        let mut heap = lock.lock().unwrap().requests.clone();
        std::iter::from_fn(|| heap.pop())
            .map(|r| (r.cx, r.cz, r.priority))
            .collect()
    }

    #[test]
    fn incremental_batches_match_full_nearest_first_scan() {
        let center = (-17, 23);
        let loaded: HashSet<_> = [center, (-16, 23), (-17, 22)].into_iter().collect();
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        let mut expected: Vec<_> = (-GENERATION_DISTANCE..=GENERATION_DISTANCE)
            .flat_map(|dx| {
                (-GENERATION_DISTANCE..=GENERATION_DISTANCE)
                    .map(move |dz| (center.0 + dx, center.1 + dz, dx * dx + dz * dz))
            })
            .filter(|&(cx, cz, _)| !loaded.contains(&(cx, cz)))
            .collect();
        expected.sort_unstable_by_key(|&(cx, cz, priority)| (priority, cx, cz));
        // Include already pending work from before the scan was initialized.
        let first = expected[0];
        loader.request_chunk(first.0, first.1, first.2);
        for batch in 1..=8 {
            loader.request_missing_chunks(center, 8, |cx, cz| loaded.contains(&(cx, cz)));
            assert_eq!(queued(&loader), expected[..1 + batch * 8]);
        }
    }

    #[test]
    fn loaded_square_is_checked_once_until_center_changes() {
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        let checks = Cell::new(0);
        for _ in 0..100 {
            loader.request_missing_chunks((0, 0), 8, |_, _| {
                checks.set(checks.get() + 1);
                true
            });
        }
        assert_eq!(checks.get(), GENERATION_OFFSETS.len());
        assert_eq!(loader.pending_count(), 0);
        loader.request_missing_chunks((-1, 1), 8, |_, _| {
            checks.set(checks.get() + 1);
            true
        });
        assert_eq!(checks.get(), 2 * GENERATION_OFFSETS.len());
    }

    #[test]
    fn full_queue_and_zero_budget_do_not_skip_unscheduled_positions() {
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        loader.request_missing_chunks((0, 0), 0, |_, _| panic!("zero budget"));
        assert_eq!(loader.scan_cursor, 0);
        for cx in 1000..1000 + MAX_PENDING_CHUNKS as i32 {
            loader.request_chunk(cx, 1000, 0);
        }
        loader.request_missing_chunks((0, 0), 8, |_, _| panic!("full queue"));
        assert_eq!(loader.scan_cursor, 0);
        // Model receiving a completed request, freeing a single slot.
        loader.pending.remove(&(1000, 1000));
        loader.request_missing_chunks((0, 0), 8, |_, _| false);
        assert!(loader.is_pending(0, 0));
        assert_eq!(loader.pending_count(), MAX_PENDING_CHUNKS);
        assert_eq!(loader.scan_cursor, 1);
    }

    #[test]
    fn moving_center_and_polled_results_preserve_missing_requests() {
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        loader.request_missing_chunks((0, 0), 8, |_, _| false);
        let polled = (0, 0);
        loader.pending.remove(&polled);
        // A completed result is not in World yet. The occupancy callback
        // must prevent requesting it again when moving resets the scan.
        loader.request_missing_chunks((-1, 0), 8, |cx, cz| (cx, cz) == polled);
        assert!(!loader.is_pending(polled.0, polled.1));
        assert_eq!(loader.pending_count(), 15);
        assert_eq!(
            queued(&loader)
                .iter()
                .filter(|r| (r.0, r.1) == polled)
                .count(),
            1
        );
    }

    #[test]
    fn clearing_pending_restarts_an_exhausted_scan() {
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        loader.request_missing_chunks((0, 0), 8, |_, _| true);
        loader.clear_pending();
        loader.request_missing_chunks((0, 0), 1, |_, _| false);
        assert!(loader.is_pending(0, 0));
    }

    #[test]
    #[ignore = "manual CPU generation-planning benchmark; use an optimized test profile"]
    fn benchmark_loaded_generation_scan() {
        use std::hint::black_box;
        use std::time::Instant;
        let loaded: HashSet<_> = GENERATION_OFFSETS.iter().map(|&(x, z, _)| (x, z)).collect();
        let iterations = 20_000;
        let start = Instant::now();
        for _ in 0..iterations {
            let mut missing = Vec::new();
            for cx in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
                for cz in -GENERATION_DISTANCE..=GENERATION_DISTANCE {
                    if !black_box(&loaded).contains(&(cx, cz)) {
                        missing.push((cx, cz));
                    }
                }
            }
            black_box(missing);
        }
        let previous = start.elapsed();
        let mut loader = ChunkLoader::with_worker_count(0, 1);
        let start = Instant::now();
        for _ in 0..iterations {
            loader.request_missing_chunks(black_box((0, 0)), 8, |cx, cz| {
                black_box(&loaded).contains(&(cx, cz))
            });
            black_box(&loader);
        }
        let current = start.elapsed();
        println!("{iterations} loaded frames: full scan={previous:?}, incremental={current:?}");
    }
}
