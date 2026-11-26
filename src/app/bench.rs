//! Automated CPU frame-time benchmark.
//!
//! ```text
//! MINERUST_AUTOSTART=12345 MINERUST_BENCH=30 cargo run --release
//! ```
//!
//! With `MINERUST_BENCH=<seconds>` set (and a world entered, e.g. through
//! `MINERUST_AUTOSTART`), the camera flies a fixed curving path over the
//! terrain at `MINERUST_BENCH_SPEED` blocks per second (default 20, faster
//! than sprinting), which keeps chunk generation, meshing, uploads and
//! unloading busy. When the time is up the game prints per-section frame
//! statistics and the worst frames broken down by section, then exits.
//!
//! `MINERUST_BENCH_VIEW=horizon` instead hovers high above the spawn and
//! slowly turns a full circle looking at the horizon (open-landscape
//! rendering cost, no streaming once loaded). `MINERUST_BENCH_SHADOWS=off`
//! disables sun shadows for the run (not saved). GPU pass times are
//! reported when the adapter supports timestamp queries.

use std::time::Instant;

use super::state::{FrameProfile, GpuFrameProfile, State};
use crate::logger::{LogLevel, log};

/// Frames slower than this (CPU work, excluding the swap-chain wait) count
/// as spikes.
const SPIKE_MS: f32 = 4.0;
/// Seconds after the start that are not recorded (initial spawn loading).
const WARMUP_SECONDS: f32 = 2.0;
/// The horizon view waits for the whole render distance to load.
const HORIZON_WARMUP_SECONDS: f32 = 10.0;

pub struct Bench {
    duration: f32,
    speed: f32,
    horizon: bool,
    shadows: Option<bool>,
    start: Option<(Instant, glam::Vec3)>,
    samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
struct Sample {
    time: f32,
    wall_ms: f32,
    profile: FrameProfile,
    gpu: Option<GpuFrameProfile>,
}

impl Sample {
    /// CPU work of the frame: update plus render, minus waiting for the
    /// swap-chain image.
    fn cpu_ms(&self) -> f32 {
        self.profile.update_ms + self.profile.render_ms - self.profile.acquire_ms
    }
}

/// Named sections of a frame, in frame order.
fn sections(p: &FrameProfile) -> [(&'static str, f32); 16] {
    [
        ("update", p.update_ms),
        ("  network", p.network_ms),
        ("  chunk poll", p.chunk_poll_ms),
        ("  physics/snapshot", p.physics_snapshot_ms),
        ("  chunk requests", p.chunk_requests_ms),
        ("  chunk commit", p.chunk_commit_ms),
        ("  mesh commit", p.mesh_commit_ms),
        ("render", p.render_ms),
        ("  acquire (wait)", p.acquire_ms),
        ("  preparation", p.frame_preparation_ms),
        ("    visible cache", p.visible_cache_ms),
        ("    shadow grid", p.shadow_grid_ms),
        ("    mesh requests", p.render_chunk_scan_ms),
        ("  encode passes", p.encode_ms),
        ("  text", p.text_ms),
        ("  submit+present", p.submit_ms),
    ]
}

fn percentile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * q).round() as usize]
}

impl Bench {
    pub fn from_env() -> Option<Self> {
        let duration = std::env::var("MINERUST_BENCH").ok()?.parse::<f32>().ok()?;
        let speed = std::env::var("MINERUST_BENCH_SPEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0);
        let horizon = std::env::var("MINERUST_BENCH_VIEW").is_ok_and(|v| v == "horizon");
        let shadows = std::env::var("MINERUST_BENCH_SHADOWS").ok().map(|v| v != "off");
        Some(Self { duration, speed, horizon, shadows, start: None, samples: Vec::new() })
    }

    fn warmup(&self) -> f32 {
        if self.horizon { HORIZON_WARMUP_SECONDS } else { WARMUP_SECONDS }
    }

    fn report(&self) -> String {
        let mut out = String::new();
        let n = self.samples.len().max(1) as f32;
        let mut cpu: Vec<f32> = self.samples.iter().map(Sample::cpu_ms).collect();
        cpu.sort_by(f32::total_cmp);
        let mut wall: Vec<f32> = self.samples.iter().map(|s| s.wall_ms).collect();
        wall.sort_by(f32::total_cmp);
        let spikes = cpu.iter().filter(|&&c| c > SPIKE_MS).count();
        let scenario = if self.horizon {
            "horizon view".to_string()
        } else {
            format!("flight at {} blocks/s", self.speed)
        };
        out += &format!(
            "\n=== Frame benchmark: {} frames, {:.0} s, {scenario}{} ===\n",
            self.samples.len(),
            self.duration - self.warmup(),
            match self.shadows {
                Some(false) => ", shadows off",
                _ => "",
            }
        );
        out += &format!(
            "frame wall   avg {:6.2}  p50 {:6.2}  p99 {:6.2}  max {:6.2} ms\n",
            wall.iter().sum::<f32>() / n,
            percentile(&wall, 0.5),
            percentile(&wall, 0.99),
            percentile(&wall, 1.0)
        );
        out += &format!(
            "frame CPU    avg {:6.2}  p50 {:6.2}  p99 {:6.2}  max {:6.2} ms   spikes > {SPIKE_MS} ms: {spikes}\n\n",
            cpu.iter().sum::<f32>() / n,
            percentile(&cpu, 0.5),
            percentile(&cpu, 0.99),
            percentile(&cpu, 1.0)
        );
        out += &format!("{:<22}{:>8}{:>8}{:>8}{:>8}\n", "section (ms)", "avg", "p99", "max", "sum s");
        for (i, (name, _)) in sections(&FrameProfile::default()).iter().enumerate() {
            let mut values: Vec<f32> = self.samples.iter().map(|s| sections(&s.profile)[i].1).collect();
            let sum: f32 = values.iter().sum();
            values.sort_by(f32::total_cmp);
            out += &format!(
                "{name:<22}{:>8.3}{:>8.3}{:>8.3}{:>8.2}\n",
                sum / n,
                percentile(&values, 0.99),
                percentile(&values, 1.0),
                sum / 1000.0
            );
        }
        let gpu: Vec<GpuFrameProfile> = self.samples.iter().filter_map(|s| s.gpu).collect();
        if !gpu.is_empty() {
            let g = gpu.len() as f32;
            let avg = |f: fn(&GpuFrameProfile) -> f32| gpu.iter().map(f).sum::<f32>() / g;
            let mut frame: Vec<f32> = gpu.iter().map(|p| p.frame_ms).collect();
            frame.sort_by(f32::total_cmp);
            out += &format!(
                "\nGPU (ms, avg)  frame {:.2} (p99 {:.2})  cull {:.2}  opaque {:.2}  depth resolve {:.2}  hi-z {:.2}  water {:.2}  composite {:.2}  ui {:.2}  text {:.2}\n",
                avg(|p| p.frame_ms),
                percentile(&frame, 0.99),
                avg(|p| p.cull_ms),
                avg(|p| p.opaque_ms),
                avg(|p| p.depth_resolve_ms),
                avg(|p| p.hiz_ms),
                avg(|p| p.water_ms),
                avg(|p| p.composite_ms),
                avg(|p| p.ui_ms),
                avg(|p| p.text_ms),
            );
        }
        let mut worst: Vec<&Sample> = self.samples.iter().collect();
        worst.sort_by(|a, b| b.cpu_ms().total_cmp(&a.cpu_ms()));
        out += "\nworst frames (CPU ms; largest sections):\n";
        for sample in worst.iter().take(10) {
            let mut parts: Vec<(&str, f32)> = sections(&sample.profile)
                .into_iter()
                .filter(|(name, _)| name.starts_with("  ") && *name != "  acquire (wait)")
                .collect();
            parts.sort_by(|a, b| b.1.total_cmp(&a.1));
            let parts: Vec<String> = parts
                .iter()
                .take(4)
                .map(|(name, ms)| format!("{} {:.2}", name.trim(), ms))
                .collect();
            out += &format!(
                "  t={:5.1}s  cpu {:6.2}  wall {:6.2}  | {}\n",
                sample.time,
                sample.cpu_ms(),
                sample.wall_ms,
                parts.join(", ")
            );
        }
        out
    }
}

impl State {
    /// Moves the camera along the benchmark path. Call before `update`.
    pub(super) fn bench_before_frame(&mut self) {
        let Some(bench) = self.bench.as_mut() else {
            return;
        };
        if !self.has_entered_world {
            return;
        }
        if bench.start.is_none() {
            if let Some(shadows) = bench.shadows {
                self.graphics.shadows = shadows;
            }
        }
        let (start, origin) = *bench.start.get_or_insert((Instant::now(), self.camera.position));
        let t = start.elapsed().as_secs_f32();
        if bench.horizon {
            // Hover high above the spawn and turn a full circle during the
            // measured part, looking slightly below the horizon.
            let ground = self.world.read().get_terrain_height(origin.x as i32, origin.z as i32);
            self.camera.position =
                glam::Vec3::new(origin.x, ground.max(minerust::SEA_LEVEL) as f32 + 35.0, origin.z);
            self.camera.velocity = glam::Vec3::ZERO;
            let turn = ((t - HORIZON_WARMUP_SECONDS).max(0.0)
                / (bench.duration - HORIZON_WARMUP_SECONDS).max(1.0))
                * std::f32::consts::TAU;
            self.camera.yaw = turn;
            self.camera.pitch = -0.12;
            self.mouse_captured = true;
            return;
        }
        // A gently curving path so the view direction and the set of loaded
        // chunks keep changing.
        let x = origin.x + bench.speed * t;
        let z = origin.z + (t * 0.25).sin() * 60.0;
        let ground = self.world.read().get_terrain_height(x.floor() as i32, z.floor() as i32);
        self.camera.position = glam::Vec3::new(x, ground.max(minerust::SEA_LEVEL) as f32 + 6.0, z);
        self.camera.velocity = glam::Vec3::ZERO;
        self.camera.yaw = (t * 0.25).cos() * 0.6;
        self.camera.pitch = -0.2;
        self.mouse_captured = true;
    }

    /// Records the finished frame; returns `true` when the run is over.
    pub(super) fn bench_after_frame(&mut self) -> bool {
        let wall_ms = self.frame_time_ms;
        let profile = self.frame_profile;
        let gpu = self.gpu_frame_profile;
        let Some(bench) = self.bench.as_mut() else {
            return false;
        };
        let Some((start, _)) = bench.start else {
            return false;
        };
        let time = start.elapsed().as_secs_f32();
        if time >= bench.warmup() {
            bench.samples.push(Sample { time, wall_ms, profile, gpu });
        }
        if time < bench.duration {
            return false;
        }
        let report = bench.report();
        println!("{report}");
        log(LogLevel::Info, &report);
        true
    }
}
