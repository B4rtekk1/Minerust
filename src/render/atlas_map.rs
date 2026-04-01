/// A packed texture atlas containing multiple equal-sized tiles arranged in a
/// square grid.
///
/// Tiles are laid out left-to-right, top-to-bottom.  The grid dimensions are
/// chosen to be the smallest square that fits all tiles (i.e. `ceil(sqrt(n))`
/// tiles per side).
struct Atlas {
    /// Total width of the atlas image in pixels.
    width: u32,
    /// Total height of the atlas image in pixels.
    height: u32,
    /// Number of tiles along each row (and column) of the grid.
    tiles_per_row: u32,
    /// Raw RGBA pixel data, row-major, 4 bytes per pixel.
    data: Vec<u8>,
}

/// Loads a list of image files from disk and returns their raw RGBA pixel data.
///
/// Every image must be exactly `tile_size × tile_size` pixels; the function
/// panics otherwise.  The returned `Vec` preserves the same order as `paths`,
/// so index `i` in the output corresponds to `paths[i]`.
///
/// # Arguments
/// * `paths`     – Paths to the image files to load.
/// * `tile_size` – Expected width and height of each tile in pixels.
///
/// # Panics
/// Panics if any file cannot be opened or if any image dimensions differ from
/// `tile_size × tile_size`.
fn load_textures(paths: &[&str], tile_size: u32) -> Vec<Vec<u8>> {
    let mut textures = Vec::with_capacity(paths.len());

    for path in paths {
        let img = image::open(path).expect("failed to open image").to_rgba8();
        assert_eq!(img.width(), tile_size, "tile width mismatch in {path}");
        assert_eq!(img.height(), tile_size, "tile height mismatch in {path}");

        textures.push(img.into_raw());
    }
    textures
}

/// Packs a slice of equal-sized RGBA tiles into a single square texture atlas.
///
/// Tiles are placed left-to-right, top-to-bottom into a grid whose side length
/// is `ceil(sqrt(n))` tiles, giving the smallest square that holds all `n`
/// tiles.  Any unused slots in the bottom-right corner are left as transparent
/// black (`0x00000000`).
///
/// # Arguments
/// * `tile_size` – Width and height of each individual tile in pixels.
/// * `textures`  – Raw RGBA pixel data for each tile, as returned by
///   [`load_textures`].  Each element must contain exactly
///   `tile_size * tile_size * 4` bytes.
///
/// # Returns
/// An [`Atlas`] whose pixel dimensions are `(tiles_per_row * tile_size)²`.
fn create_atlas(tile_size: u32, textures: &[Vec<u8>]) -> Atlas {
    // Choose the smallest square grid that fits all tiles.
    let tiles_per_row = (textures.len() as f32).sqrt().ceil() as u32;
    let atlas_size = tiles_per_row * tile_size;

    // Initialize to transparent black; unused grid slots stay empty.
    let mut data = vec![0u8; (atlas_size * atlas_size * 4) as usize];

    for (i, tex) in textures.iter().enumerate() {
        let tile_col = i as u32 % tiles_per_row;
        let tile_row = i as u32 / tiles_per_row;

        // Top-left pixel of this tile inside the atlas.
        let x = tile_col * tile_size;
        let y = tile_row * tile_size;

        // Copy the tile row by row to account for the atlas stride.
        for row in 0..tile_size {
            let dst_start = ((y + row) * atlas_size + x) as usize * 4;
            let src_start = (row * tile_size) as usize * 4;
            let len = (tile_size * 4) as usize;
            data[dst_start..dst_start + len].copy_from_slice(&tex[src_start..src_start + len]);
        }
    }

    Atlas {
        width: atlas_size,
        height: atlas_size,
        tiles_per_row,
        data,
    }
}
