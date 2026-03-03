use image::GenericImageView;
use std::path::Path;

use crate::constants::{ATLAS_SIZE, TEXTURE_SIZE};

pub fn load_texture_atlas_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::open(path).map_err(|e| format!("Failed to load texture: {}", e))?;
    let rgba = img.to_rgba8();
    let (width, height) = img.dimensions();

    if width % 4 != 0 || height % 4 != 0 {
        return Err(format!(
            "Texture atlas dimensions {}x{} not divisible by 4",
            width, height
        ));
    }

    let tile_w = width / 4;
    let tile_h = height / 4;

    if tile_w != tile_h {
        return Err(format!(
            "Texture atlas tiles are not square: {}x{}",
            tile_w, tile_h
        ));
    }

    let mut layers = Vec::with_capacity((width * height * 4) as usize);

    for i in 0..16 {
        let col = i % 4;
        let row = i / 4;
        let start_x = col * tile_w;
        let start_y = row * tile_h;

        for y in 0..tile_h {
            for x in 0..tile_w {
                let pixel = rgba.get_pixel(start_x + x, start_y + y);
                layers.extend_from_slice(&pixel.0);
            }
        }
    }

    Ok((layers, tile_w, tile_h))
}

pub fn generate_texture_atlas() -> Vec<u8> {
    let total_pixels = (TEXTURE_SIZE * TEXTURE_SIZE * ATLAS_SIZE * ATLAS_SIZE) as usize;
    let mut data = vec![0u8; total_pixels * 4];

    let set_pixel = |data: &mut [u8], tex_idx: u32, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8| {
        let layer_size = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
        let layer_offset = (tex_idx as usize) * layer_size;
        let pixel_offset = ((y * TEXTURE_SIZE + x) * 4) as usize;
        let idx = layer_offset + pixel_offset;

        if idx + 3 < data.len() {
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = a;
        }
    };

    let hash = |x: u32, y: u32, seed: u32| -> u8 {
        let n = x
            .wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(seed);
        let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
        ((n ^ (n >> 16)) & 0xFF) as u8
    };

    for tex_idx in 0..16u32 {
        for y in 0..TEXTURE_SIZE {
            for x in 0..TEXTURE_SIZE {
                let (r, g, b, a) = match tex_idx {
                    0 => {
                        let noise = hash(x, y, 0) as i32 - 128;
                        let g_val = (100 + noise / 8).clamp(60, 140) as u8;
                        (50, g_val, 30, 255)
                    }
                    1 => {
                        // Grass side - dirt with green strip
                        if y < 3 {
                            let noise = hash(x, y, 1) as i32 - 128;
                            let g_val = (100 + noise / 8).clamp(60, 140) as u8;
                            (50, g_val, 30, 255)
                        } else {
                            let noise = hash(x, y, 2) as i32 - 128;
                            let base = 139 + noise / 10;
                            (
                                base.clamp(100, 160) as u8,
                                (base - 40).clamp(60, 120) as u8,
                                (base - 80).clamp(20, 60) as u8,
                                255,
                            )
                        }
                    }
                    2 => {
                        // Dirt
                        let noise = hash(x, y, 3) as i32 - 128;
                        let base = 139 + noise / 8;
                        (
                            base.clamp(100, 170) as u8,
                            (base - 40).clamp(60, 130) as u8,
                            (base - 80).clamp(20, 70) as u8,
                            255,
                        )
                    }
                    3 => {
                        // Stone
                        let noise = hash(x, y, 4) as i32 - 128;
                        let base = 128 + noise / 6;
                        let v = base.clamp(90, 160) as u8;
                        (v, v, v, 255)
                    }
                    4 => {
                        // Sand
                        let noise = hash(x, y, 5) as i32 - 128;
                        let base = 220 + noise / 12;
                        (
                            base.clamp(180, 240) as u8,
                            (base - 20).clamp(160, 220) as u8,
                            (base - 80).clamp(100, 160) as u8,
                            255,
                        )
                    }
                    5 => {
                        // Water
                        let noise = hash(x, y, 6) as i32 - 128;
                        let b_val = 180 + noise / 10;
                        (
                            30,
                            100 + (noise / 15) as u8,
                            b_val.clamp(150, 220) as u8,
                            200,
                        )
                    }
                    6 => {
                        // Wood side (bark)
                        let stripe = if x % 4 == 0 || x % 4 == 3 { 10i32 } else { 0 };
                        let noise = hash(x, y, 7) as i32 - 128;
                        let base = 100 + noise / 12 + stripe;
                        (
                            (base + 30).clamp(80, 150) as u8,
                            base.clamp(50, 120) as u8,
                            (base - 30).clamp(20, 70) as u8,
                            255,
                        )
                    }
                    7 => {
                        // Wood top (rings)
                        let cx = x as i32 - 8;
                        let cy = y as i32 - 8;
                        let dist = ((cx * cx + cy * cy) as f32).sqrt() as i32;
                        let ring = if dist % 3 == 0 { 20i32 } else { 0 };
                        let noise = hash(x, y, 8) as i32 - 128;
                        let base = 150 + noise / 15 - ring;
                        (
                            base.clamp(100, 180) as u8,
                            (base - 40).clamp(60, 140) as u8,
                            (base - 80).clamp(20, 80) as u8,
                            255,
                        )
                    }
                    8 => {
                        // Leaves
                        let noise = hash(x, y, 9);
                        if noise > 180 {
                            (0, 0, 0, 0) // Transparent holes
                        } else {
                            let g_val = 80 + (noise / 4);
                            (30, g_val, 20, 240)
                        }
                    }
                    9 => {
                        // Bedrock
                        let noise = hash(x, y, 10) as i32 - 128;
                        let base = 50 + noise / 8;
                        let v = base.clamp(30, 80) as u8;
                        (v, v, v, 255)
                    }
                    10 => {
                        // Snow
                        let noise = hash(x, y, 11) as i32 - 128;
                        let base = 245 + noise / 20;
                        let v = base.clamp(230, 255) as u8;
                        (v, v, (v as i32 + 5).min(255) as u8, 255)
                    }
                    11 => {
                        // Gravel
                        let noise = hash(x, y, 12);
                        let pebble = if (noise / 40) % 3 == 0 { 30i32 } else { 0 };
                        let base = 120 + (noise as i32 / 10) - pebble;
                        let v = base.clamp(80, 150) as u8;
                        (v, v, v, 255)
                    }
                    12 => {
                        // Clay
                        let noise = hash(x, y, 13) as i32 - 128;
                        let base = 150 + noise / 12;
                        (
                            base.clamp(120, 170) as u8,
                            (base - 20).clamp(100, 150) as u8,
                            (base - 10).clamp(110, 160) as u8,
                            255,
                        )
                    }
                    13 => {
                        // Ice
                        let noise = hash(x, y, 14) as i32 - 128;
                        let base = 200 + noise / 15;
                        (
                            base.clamp(170, 230) as u8,
                            (base + 20).clamp(190, 250) as u8,
                            255,
                            220,
                        )
                    }
                    14 => {
                        // Cactus
                        let edge = x == 0 || x == 15 || y == 0 || y == 15;
                        let noise = hash(x, y, 15) as i32 - 128;
                        if edge {
                            (30, 80, 20, 255) // Darker edge
                        } else {
                            let g_val = 120 + noise / 10;
                            (40, g_val.clamp(100, 150) as u8, 30, 255)
                        }
                    }
                    15 => {
                        // Dead bush (mostly transparent)
                        let noise = hash(x, y, 16);
                        let is_branch = (x + y) % 5 == 0 && noise > 100;
                        if is_branch {
                            (100, 70, 40, 255)
                        } else {
                            (0, 0, 0, 0)
                        }
                    }
                    _ => (255, 0, 255, 255),
                };
                set_pixel(&mut data, tex_idx, x, y, r, g, b, a);
            }
        }
    }

    data
}
