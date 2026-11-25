//! Renders cross-sections of the generated world, to judge caves.
//!
//! ```text
//! cargo run --release --example cave_slice -- [seed] [out_prefix] [center_x] [center_z] [y]
//! ```
//!
//! Writes `<prefix>_vertical.png` (an X/Y cut through `center_z`, 512 blocks
//! wide) and `<prefix>_horizontal.png` (a 512×512 X/Z cut at height `y`).

use minerust::{BlockType, CHUNK_SIZE, Chunk, ChunkGenerator};
use std::collections::HashMap;

fn color(block: BlockType, y: i32) -> [u8; 3] {
    match block {
        BlockType::Air if y >= 64 => [170, 205, 240],
        BlockType::Air => [20, 18, 24],
        BlockType::Water => [40, 80, 200],
        BlockType::Ice => [170, 210, 240],
        BlockType::Stone => [120, 120, 124],
        BlockType::Gravel => [150, 140, 135],
        BlockType::Dirt => [125, 90, 60],
        BlockType::Grass => [90, 160, 60],
        BlockType::Sand => [220, 205, 150],
        BlockType::Clay => [160, 165, 185],
        BlockType::Snow => [245, 245, 250],
        BlockType::Bedrock => [50, 50, 50],
        BlockType::Leaves => [50, 120, 40],
        BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ => [110, 80, 45],
        _ => [200, 60, 200],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12345);
    let prefix = args.get(2).cloned().unwrap_or_else(|| "caves".into());
    let cx: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cz: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let slice_y: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(30);
    let generator = ChunkGenerator::new(seed);
    let mut chunks: HashMap<(i32, i32), Chunk> = HashMap::new();
    let mut block = |x: i32, y: i32, z: i32| {
        let key = (x.div_euclid(CHUNK_SIZE), z.div_euclid(CHUNK_SIZE));
        chunks
            .entry(key)
            .or_insert_with(|| generator.generate_chunk(key.0, key.1))
            .get_block(x.rem_euclid(CHUNK_SIZE), y, z.rem_euclid(CHUNK_SIZE))
    };

    let (w, h) = (512u32, 180u32);
    let mut vertical = image::RgbImage::new(w, h);
    for px in 0..w {
        for py in 0..h {
            let (x, y) = (cx - 256 + px as i32, (h - 1 - py) as i32);
            vertical.put_pixel(px, py, image::Rgb(color(block(x, y, cz), y)));
        }
    }
    vertical.save(format!("{prefix}_vertical.png")).unwrap();

    let mut horizontal = image::RgbImage::new(512, 512);
    for px in 0..512u32 {
        for pz in 0..512u32 {
            let (x, z) = (cx - 256 + px as i32, cz - 256 + pz as i32);
            horizontal.put_pixel(px, pz, image::Rgb(color(block(x, slice_y, z), 0)));
        }
    }
    horizontal.save(format!("{prefix}_horizontal.png")).unwrap();

    // X-ray: darker where a column holds more cave space below its surface;
    // blue where caves are flooded.
    let mut xray = image::RgbImage::new(512, 512);
    let mut openings = 0usize;
    for px in 0..512u32 {
        for pz in 0..512u32 {
            let (x, z) = (cx - 256 + px as i32, cz - 256 + pz as i32);
            let top = (0..200)
                .rev()
                .find(|&y| {
                    matches!(
                        block(x, y, z),
                        BlockType::Stone | BlockType::Dirt | BlockType::Grass | BlockType::Sand
                            | BlockType::Snow | BlockType::Gravel | BlockType::Clay
                    )
                })
                .unwrap_or(0);
            let (mut air, mut water) = (0, 0);
            for y in 1..top {
                match block(x, y, z) {
                    BlockType::Air => air += 1,
                    BlockType::Water => water += 1,
                    _ => {}
                }
            }
            let v = (235 - (air * 9).min(220)) as u8;
            // Red: a cave or ravine opens to the sky here.
            let opening = top < generator.get_terrain_height_pub(x, z) - 1;
            openings += usize::from(opening);
            let px_color = if opening {
                [220, 40, 40]
            } else if water > 0 && air < 4 {
                [70, 110, 220]
            } else {
                [v, v, v.saturating_add(8)]
            };
            xray.put_pixel(px, pz, image::Rgb(px_color));
        }
    }
    xray.save(format!("{prefix}_xray.png")).unwrap();
    println!("wrote {prefix}_vertical.png, {prefix}_horizontal.png and {prefix}_xray.png; {openings} open columns");
}
