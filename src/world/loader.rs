//! Async chunk generation system with priority queue
//!
//! This module provides background chunk generation to avoid blocking
//! the main thread during world exploration. Uses crossbeam channels
//! for efficient inter-thread communication.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::thread;

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};

use crate::core::chunk::Chunk;
use crate::world::generator::ChunkGenerator;

/// Request for chunk generation with priority
#[derive(Clone)]
pub struct ChunkGenRequest {
    pub cx: i32,
    pub cz: i32,
    pub priority: i32, // Lower = higher priority (distance squared)
}

// Ordering for priority queue (min-heap by priority)
impl PartialEq for ChunkGenRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ChunkGenRequest {}

impl PartialOrd for ChunkGenRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ChunkGenRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower priority value = higher priority)
        other.priority.cmp(&self.priority)
    }
}

/// Result of background chunk generation
pub struct ChunkGenResult {
    pub cx: i32,
    pub cz: i32,
    pub chunk: Chunk,
}

/// Manages background chunk generation with worker threads
pub struct ChunkLoader {
    request_tx: Sender<ChunkGenRequest>,
    result_rx: Receiver<ChunkGenResult>,
    pending: HashSet<(i32, i32)>,
    worker_count: usize,
}

impl ChunkLoader {
    /// Create a new ChunkLoader with worker threads
    pub fn new(seed: u32) -> Self {
        Self::with_worker_count(crate::constants::get_chunk_worker_count(), seed)
    }

    /// Create a ChunkLoader with a specific number of workers
    pub fn with_worker_count(num_workers: usize, seed: u32) -> Self {
        // Bounded channels prevent unbounded memory growth
        let (request_tx, request_rx) = bounded::<ChunkGenRequest>(256);
        let (result_tx, result_rx) = bounded::<ChunkGenResult>(64);

        // Spawn worker threads, each with their own ChunkGenerator
        for worker_id in 0..num_workers {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let generator = ChunkGenerator::new(seed);

            thread::Builder::new()
                .name(format!("chunk-gen-{}", worker_id))
                .spawn(move || {
                    loop {
                        match rx.recv() {
                            Ok(req) => {
                                // Generate the chunk using FastNoiseLite
                                let chunk = generator.generate_chunk(req.cx, req.cz);

                                // Send result back to main thread
                                if tx
                                    .send(ChunkGenResult {
                                        cx: req.cx,
                                        cz: req.cz,
                                        chunk,
                                    })
                                    .is_err()
                                {
                                    // Main thread has closed, exit
                                    break;
                                }
                            }
                            Err(_) => {
                                // Channel closed, exit worker
                                break;
                            }
                        }
                    }
                })
                .expect("Failed to spawn chunk generation worker");
        }

        ChunkLoader {
            request_tx,
            result_rx,
            pending: HashSet::new(),
            worker_count: num_workers,
        }
    }

    /// Request a chunk to be generated with a priority
    /// Lower priority values are processed first (use distance squared)
    pub fn request_chunk(&mut self, cx: i32, cz: i32, priority: i32) {
        if self.pending.contains(&(cx, cz)) {
            return; // Already requested
        }

        self.pending.insert((cx, cz));

        // Non-blocking send - if the queue is full, skip this request for now
        if self
            .request_tx
            .try_send(ChunkGenRequest { cx, cz, priority })
            .is_err()
        {
            // Keep pending consistent with actual queued work.
            self.pending.remove(&(cx, cz));
        }
    }

    /// Request multiple chunks sorted by priority
    pub fn request_chunks(&mut self, requests: &[(i32, i32, i32)]) {
        // Sort by priority (lowest first)
        let mut sorted: Vec<_> = requests
            .iter()
            .filter(|(cx, cz, _)| !self.pending.contains(&(*cx, *cz)))
            .collect();
        sorted.sort_by_key(|(_, _, priority)| *priority);

        for (cx, cz, priority) in sorted {
            if self.pending.len() >= 256 {
                break; // Don't overwhelm the queue
            }
            self.pending.insert((*cx, *cz));
            if self
                .request_tx
                .try_send(ChunkGenRequest {
                    cx: *cx,
                    cz: *cz,
                    priority: *priority,
                })
                .is_err()
            {
                // Retry in later frames if queue is currently full.
                self.pending.remove(&(*cx, *cz));
            }
        }
    }

    /// Check if a chunk is pending generation
    pub fn is_pending(&self, cx: i32, cz: i32) -> bool {
        self.pending.contains(&(cx, cz))
    }

    /// Get the number of pending chunks
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Poll for completed chunks (non-blocking)
    /// Returns up to max_results completed chunks
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

    /// Poll all available results (non-blocking)
    pub fn poll_all_results(&mut self) -> Vec<ChunkGenResult> {
        self.poll_results(64)
    }

    /// Cancel a pending chunk request (removes from pending set)
    pub fn cancel(&mut self, cx: i32, cz: i32) {
        self.pending.remove(&(cx, cz));
    }

    /// Clear all pending requests (for example when teleporting)
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Get worker count
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }
}
