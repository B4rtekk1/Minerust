//! Renders a shaded top-down map of the world generator to a PNG.
//!
//! Lets terrain changes be judged without starting the game:
//!
//! ```text
//! cargo run --release --example world_map -- [seed] [radius_chunks] [out.png] [center_x] [center_z]
//! ```
//!
//! Each pixel is one block column: the colour of the highest visible block
//! (taken from the real texture atlas, with biome tint applied), lit by a
//! simple north-west hillshade, with water darkened by depth.

use std::sync::Mutex;

use minerust::{BlockType, CHUNK_SIZE, ChunkGenerator, SEA_LEVEL, WORLD_HEIGHT};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12345);
    let radius: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let out = args.get(3).cloned().unwrap_or_else(|| "world_map.png".into());
    let center_x: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let center_z: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let tile_colors = atlas_tile_colors();
    let generator = ChunkGenerator::new(seed);
    let span = (radius * 2) as usize;
    let size = span * CHUNK_SIZE as usize;
    let cx0 = center_x.div_euclid(CHUNK_SIZE) - radius;
    let cz0 = center_z.div_euclid(CHUNK_SIZE) - radius;

    // Per column: (surface colour, terrain height, water depth).
    let columns = Mutex::new(vec![([0f32; 3], 0i32, 0i32); size * size]);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let start = std::time::Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= span * span {
                        break;
                    }
                    let (ix, iz) = (i % span, i / span);
                    let chunk = generator.generate_chunk(cx0 + ix as i32, cz0 + iz as i32);
                    let mut local = Vec::with_capacity(256);
                    for lx in 0..CHUNK_SIZE {
                        for lz in 0..CHUNK_SIZE {
                            local.push((lx, lz, column_sample(&chunk, lx, lz, &tile_colors)));
                        }
                    }
                    let mut columns = columns.lock().unwrap();
                    for (lx, lz, sample) in local {
                        let px = ix * CHUNK_SIZE as usize + lx as usize;
                        let pz = iz * CHUNK_SIZE as usize + lz as usize;
                        columns[pz * size + px] = sample;
                    }
                }
            });
        }
    });
    let elapsed = start.elapsed();
    let columns = columns.into_inner().unwrap();

    let mut image = image::RgbImage::new(size as u32, size as u32);
    let height_at = |x: usize, z: usize| columns[z.min(size - 1) * size + x.min(size - 1)].1 as f32;
    for z in 0..size {
        for x in 0..size {
            let (color, _, water_depth) = columns[z * size + x];
            let dx = height_at(x + 1, z) - height_at(x.saturating_sub(1), z);
            let dz = height_at(x, z + 1) - height_at(x, z.saturating_sub(1));
            let shade = if water_depth > 0 {
                1.0 - (water_depth as f32 / 40.0).min(0.55)
            } else {
                (1.0 - (dx + dz) * 0.09).clamp(0.45, 1.35)
            };
            let px = color.map(|c| ((c * shade).clamp(0.0, 1.0) * 255.0) as u8);
            image.put_pixel(x as u32, z as u32, image::Rgb(px));
        }
    }
    image.save(&out).expect("write map image");
    println!(
        "seed {seed}: {}x{} chunks generated in {:.2?} ({:.2} ms/chunk) -> {out}",
        span,
        span,
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / (span * span) as f64
    );
}

fn column_sample(
    chunk: &minerust::Chunk,
    lx: i32,
    lz: i32,
    tiles: &[[f32; 3]; 16],
) -> ([f32; 3], i32, i32) {
    let tint = chunk.column_tint(lx, lz);
    let mut water_top = None;
    for y in (0..WORLD_HEIGHT).rev() {
        let block = chunk.get_block(lx, y, lz);
        match block {
            BlockType::Air => continue,
            BlockType::Water | BlockType::Ice if block == BlockType::Water => {
                water_top.get_or_insert(y);
                continue;
            }
            _ => {}
        }
        let mut color = tiles[block.tex_top() as usize];
        if minerust::biome::tint_applies(block.tex_top() as u8) {
            color = minerust::biome::apply_tint(color, tint);
        }
        if let Some(top) = water_top {
            let water = tiles[BlockType::Water.tex_top() as usize];
            let depth = top - y;
            let t = (0.45 + depth as f32 * 0.05).min(0.9);
            let mixed = [0, 1, 2].map(|i| color[i] * (1.0 - t) + water[i] * t);
            return (mixed, top, depth.max(1));
        }
        return (color, y, 0);
    }
    (tiles[BlockType::Water.tex_top() as usize], SEA_LEVEL, 1)
}

/// Average colour of each 4x4 atlas tile, in linear-ish display space.
fn atlas_tile_colors() -> [[f32; 3]; 16] {
    let atlas = image::open(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/textures.png"))
        .expect("read texture atlas")
        .to_rgba8();
    let tile = atlas.width() / 4;
    std::array::from_fn(|i| {
        let (tx, ty) = ((i as u32 % 4) * tile, (i as u32 / 4) * tile);
        let mut sum = [0f64; 3];
        let mut n = 0f64;
        for y in ty..ty + tile {
            for x in tx..tx + tile {
                let p = atlas.get_pixel(x, y);
                // Skip the transparent / black cut-out background (dead bush, leaves).
                if p[3] < 8 || (p[0] as u32 + p[1] as u32 + p[2] as u32) < 30 {
                    continue;
                }
                for c in 0..3 {
                    sum[c] += p[c] as f64;
                }
                n += 1.0;
            }
        }
        sum.map(|s| (s / n.max(1.0) / 255.0) as f32)
    })
}
