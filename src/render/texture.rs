use image::GenericImageView;
use std::path::Path;

use crate::constants::{ATLAS_COLUMNS, TEXTURE_LAYERS, TEXTURE_SIZE};

/// Loads the block texture atlas from disk and extracts its tiles into a
/// flat, layer-ordered byte array suitable for upload as a `Texture2DArray`.
///
/// The atlas is a grid of square tiles, [`ATLAS_COLUMNS`] wide, holding
/// [`TEXTURE_LAYERS`] tiles. Tiles are read in row-major order
/// (left-to-right, top-to-bottom) and concatenated so that layer `i`
/// occupies bytes `[i * tile² * 4 .. (i+1) * tile² * 4]`.
///
/// # Returns
/// A tuple `(data, tile_width, tile_height)` where `data` is the raw RGBA8
/// pixel data in layer order.
///
/// # Errors
/// Returns a descriptive `String` if the file cannot be opened or does not
/// hold a full grid of [`TEXTURE_LAYERS`] square tiles.
pub fn load_texture_atlas_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::open(path).map_err(|e| format!("Failed to load texture: {}", e))?;
    let rgba = img.to_rgba8();
    let (width, height) = img.dimensions();

    let rows = TEXTURE_LAYERS.div_ceil(ATLAS_COLUMNS);
    if width % ATLAS_COLUMNS != 0 || height % rows != 0 {
        return Err(format!(
            "Texture atlas {}x{} is not a {}x{} tile grid (re-run tools/generate_block_textures.py)",
            width, height, ATLAS_COLUMNS, rows
        ));
    }

    let tile_w = width / ATLAS_COLUMNS;
    let tile_h = height / rows;

    if tile_w != tile_h {
        return Err(format!(
            "Texture atlas tiles are not square: {}x{}",
            tile_w, tile_h
        ));
    }

    let mut layers = Vec::with_capacity((tile_w * tile_h * 4 * TEXTURE_LAYERS) as usize);

    // Extract each tile in row-major order and append it as a contiguous layer.
    for i in 0..TEXTURE_LAYERS {
        let col = i % ATLAS_COLUMNS;
        let row = i / ATLAS_COLUMNS;
        let start_x = col * tile_w;
        let start_y = row * tile_h;

        // Only cutout materials (leaves, dead bushes) are alpha-tested.
        // Stray transparent texels elsewhere would show through the opaque
        // terrain pipeline (which skips `discard` for early depth testing),
        // so seal them.
        let cutout = crate::render::quad::is_cutout_material(i as u8);
        for y in 0..tile_h {
            for x in 0..tile_w {
                let mut pixel = rgba.get_pixel(start_x + x, start_y + y).0;
                if !cutout && i != crate::constants::TEX_WATER as u32 {
                    pixel[3] = 255;
                }
                layers.extend_from_slice(&pixel);
            }
        }
    }

    Ok((layers, tile_w, tile_h))
}

/// Procedurally generates a [`TEXTURE_LAYERS`]-layer RGBA8 texture array at runtime.
///
/// Each layer is a `TEXTURE_SIZE × TEXTURE_SIZE` tile whose appearance is
/// driven by a fast integer hash function, avoiding any file I/O.  This is
/// useful as a fallback when no atlas file is present, or during development.
///
/// The returned `Vec` is packed in layer-major order: layer `i` occupies
/// bytes `[i * TEXTURE_SIZE² * 4 .. (i+1) * TEXTURE_SIZE² * 4]`, matching
/// the layout expected by a `Texture2DArray` upload.
///
/// # Texture index mapping
/// | Index | Block         | Notes                                      |
/// |-------|---------------|--------------------------------------------|
/// | 0     | Grass top     | Green with subtle noise                    |
/// | 1     | Grass side    | Green strip (top 3 px) over dirt           |
/// | 2     | Dirt          | Brown with noise                           |
/// | 3     | Stone         | Grey with noise                            |
/// | 4     | Sand          | Light tan with noise                       |
/// | 5     | Water         | Semi-transparent blue (alpha 200)          |
/// | 6     | Wood side     | Bark with vertical stripe pattern          |
/// | 7     | Wood top      | Concentric ring pattern                    |
/// | 8     | Leaves        | Green with transparent holes (alpha 0/240) |
/// | 9     | Bedrock       | Very dark gray with noise                  |
/// | 10    | Snow          | Near-white with slight blue tint           |
/// | 11    | Gravel        | Grey with pebble pattern                   |
/// | 12    | Clay          | Muted gray-pink with noise                 |
/// | 13    | Ice           | Light blue, semi-transparent (alpha 220)   |
/// | 14    | Cactus        | Green with darker border                   |
/// | 15    | Dead bush     | Sparse brown branches, mostly transparent  |
pub fn generate_texture_atlas() -> Vec<u8> {
    let total_pixels = (TEXTURE_SIZE * TEXTURE_SIZE * TEXTURE_LAYERS) as usize;
    let mut data = vec![0u8; total_pixels * 4];

    /// Writes one RGBA pixel into `data` at the given layer and pixel coordinates.
    ///
    /// Silently does nothing if the computed index would exceed the buffer,
    /// guarding against accidental out-of-bounds writes during development.
    let set_pixel = |data: &mut [u8], tex_idx: u32, x: u32, y: u32, r: u8, g: u8, b: u8, a: u8| {
        let layer_size = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
        let layer_offset = tex_idx as usize * layer_size;
        let pixel_offset = ((y * TEXTURE_SIZE + x) * 4) as usize;
        let idx = layer_offset + pixel_offset;

        if idx + 3 < data.len() {
            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = a;
        }
    };

    /// A fast, seed-able integer hash that maps `(x, y, seed)` to a pseudo-random
    /// byte in `[0, 255]`.  Used to add per-pixel noise to every texture.
    ///
    /// The constants are from well-known hash multiplier families (Knuth,
    /// Murmur-style) and produce good avalanche with minimal arithmetic.
    let hash = |x: u32, y: u32, seed: u32| -> u8 {
        let n = x
            .wrapping_mul(374_761_393)
            .wrapping_add(y.wrapping_mul(668_265_263))
            .wrapping_add(seed);
        let n = (n ^ (n >> 13)).wrapping_mul(1_274_126_177);
        ((n ^ (n >> 16)) & 0xFF) as u8
    };

    for tex_idx in 0..TEXTURE_LAYERS {
        for y in 0..TEXTURE_SIZE {
            for x in 0..TEXTURE_SIZE {
                let (r, g, b, a) = match tex_idx {
                    // --- Grass top: solid green with per-pixel brightness noise ---
                    0 => {
                        let noise = hash(x, y, 0) as i32 - 128;
                        let g_val = (100 + noise / 8).clamp(60, 140) as u8;
                        (50, g_val, 30, 255)
                    }

                    // --- Grass side: green cap (top 3 rows) blending into dirt ---
                    1 => {
                        if y < 3 {
                            // Green cap strip.
                            let noise = hash(x, y, 1) as i32 - 128;
                            let g_val = (100 + noise / 8).clamp(60, 140) as u8;
                            (50, g_val, 30, 255)
                        } else {
                            // Dirt body.
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

                    // --- Dirt: noisy brown ---
                    2 => {
                        let noise = hash(x, y, 3) as i32 - 128;
                        let base = 139 + noise / 8;
                        (
                            base.clamp(100, 170) as u8,
                            (base - 40).clamp(60, 130) as u8,
                            (base - 80).clamp(20, 70) as u8,
                            255,
                        )
                    }

                    // --- Stone: noisy mid-grey ---
                    3 => {
                        let noise = hash(x, y, 4) as i32 - 128;
                        let base = 128 + noise / 6;
                        let v = base.clamp(90, 160) as u8;
                        (v, v, v, 255)
                    }

                    // --- Sand: warm tan with gentle noise ---
                    4 => {
                        let noise = hash(x, y, 5) as i32 - 128;
                        let base = 220 + noise / 12;
                        (
                            base.clamp(180, 240) as u8,
                            (base - 20).clamp(160, 220) as u8,
                            (base - 80).clamp(100, 160) as u8,
                            255,
                        )
                    }

                    // --- Water: semi-transparent blue with ripple noise ---
                    5 => {
                        let noise = hash(x, y, 6) as i32 - 128;
                        let b_val = 180 + noise / 10;
                        (
                            30,
                            (100 + noise / 15) as u8,
                            b_val.clamp(150, 220) as u8,
                            200, // Semi-transparent so underlying geometry shows through.
                        )
                    }

                    // --- Wood side (bark): vertical stripe grain pattern ---
                    6 => {
                        // Brighten pixels at the edges of each 4-px stripe column.
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

                    // --- Wood top (end grain): concentric ring pattern ---
                    7 => {
                        let cx = x as i32 - 8;
                        let cy = y as i32 - 8;
                        // Darken every third distance step to simulate growth rings.
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

                    // --- Leaves: green with random transparent holes ---
                    8 => {
                        let noise = hash(x, y, 9);
                        if noise > 180 {
                            (0, 0, 0, 0) // Punch a transparent hole through the leaf cluster.
                        } else {
                            let g_val = 80 + noise / 4;
                            (30, g_val, 20, 240)
                        }
                    }

                    // --- Bedrock: very dark noisy grey ---
                    9 => {
                        let noise = hash(x, y, 10) as i32 - 128;
                        let base = 50 + noise / 8;
                        let v = base.clamp(30, 80) as u8;
                        (v, v, v, 255)
                    }

                    // --- Snow: near-white with a slight cool blue tint ---
                    10 => {
                        let noise = hash(x, y, 11) as i32 - 128;
                        let base = 245 + noise / 20;
                        let v = base.clamp(230, 255) as u8;
                        // Blue channel is nudged +5 for a cool tint.
                        (v, v, (v as i32 + 5).min(255) as u8, 255)
                    }

                    // --- Gravel: gray with scattered darker pebble patches ---
                    11 => {
                        let noise = hash(x, y, 12);
                        // Every third bucket of the hash value creates a darker pebble.
                        let pebble = if (noise / 40) % 3 == 0 { 30i32 } else { 0 };
                        let base = 120 + (noise as i32 / 10) - pebble;
                        let v = base.clamp(80, 150) as u8;
                        (v, v, v, 255)
                    }

                    // --- Clay: smooth muted gray-pink ---
                    12 => {
                        let noise = hash(x, y, 13) as i32 - 128;
                        let base = 150 + noise / 12;
                        (
                            base.clamp(120, 170) as u8,
                            (base - 20).clamp(100, 150) as u8,
                            (base - 10).clamp(110, 160) as u8,
                            255,
                        )
                    }

                    // --- Ice: light blue, semi-transparent ---
                    13 => {
                        let noise = hash(x, y, 14) as i32 - 128;
                        let base = 200 + noise / 15;
                        (
                            base.clamp(170, 230) as u8,
                            (base + 20).clamp(190, 250) as u8,
                            255,
                            220, // Partially transparent to show water or blocks beneath.
                        )
                    }

                    // --- Cactus side: green with darker border pixels ---
                    14 => {
                        let edge =
                            x == 0 || x == TEXTURE_SIZE - 1 || y == 0 || y == TEXTURE_SIZE - 1;
                        let noise = hash(x, y, 15) as i32 - 128;
                        if edge {
                            (30, 80, 20, 255) // Darker outline simulates the cactus ridge.
                        } else {
                            let g_val = 120 + noise / 10;
                            (40, g_val.clamp(100, 150) as u8, 30, 255)
                        }
                    }

                    // --- Dead bush: sparse brown branch pixels, rest transparent ---
                    15 => {
                        let noise = hash(x, y, 16);
                        // A diagonal pattern filtered by noise produces sparse twigs.
                        let is_branch = (x + y) % 5 == 0 && noise > 100;
                        if is_branch {
                            (100, 70, 40, 255)
                        } else {
                            (0, 0, 0, 0)
                        }
                    }

                    // Newer layers: a noisy fill of the block's base colour.
                    _ => {
                        let [r, g, b] = fallback_layer_color(tex_idx);
                        let noise = hash(x / 16, y / 16, tex_idx) as i32 - 128;
                        let shade = |c: u8| (c as i32 + noise / 6).clamp(0, 255) as u8;
                        (shade(r), shade(g), shade(b), 255)
                    }
                };
                set_pixel(&mut data, tex_idx, x, y, r, g, b, a);
            }
        }
    }

    data
}

/// Base colour of the procedural fallback for layers added after the
/// original sixteen (only used when `assets/textures.png` is missing).
fn fallback_layer_color(layer: u32) -> [u8; 3] {
    match layer {
        16 => [90, 90, 92],    // coal ore
        17 => [150, 130, 115], // iron ore
        18 => [110, 100, 60],  // gold ore
        19 => [70, 120, 120],  // diamond ore
        20 => [64, 64, 72],    // deepslate
        21 => [190, 100, 40],  // red sand
        22 => [158, 94, 66],   // terracotta
        23 => [146, 62, 46],   // red terracotta
        24 => [212, 180, 163], // white terracotta
        25 => [94, 64, 28],    // podzol top
        26 => [110, 76, 46],   // podzol side
        27 => [84, 108, 40],   // moss
        28 => [100, 116, 80],  // mossy stone
        29 => [112, 98, 110],  // mycelium top
        30 => [116, 90, 76],   // mycelium side
        31 => [206, 200, 188], // mushroom stem
        32 => [196, 44, 42],   // red mushroom
        33 => [146, 110, 80],  // brown mushroom
        34 => [214, 196, 140], // sandstone
        35 => [224, 222, 214], // birch log
        36 => [236, 168, 196], // cherry leaves
        37 => [200, 80, 20],   // magma
        38 => [134, 106, 90],  // dripstone
        39 => [130, 96, 190],  // amethyst
        40 => [214, 216, 212], // calcite
        41 => [60, 38, 48],    // cherry log
        _ => [128, 128, 128],
    }
}
