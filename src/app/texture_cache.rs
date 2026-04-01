use std::fs;
use std::path::Path;

use crate::logger::{LogLevel, log};
use minerust::{TEXTURE_SIZE, generate_texture_atlas, load_texture_atlas_from_file};

/// Manages a file-based cache for the texture atlas binary data.
///
/// The cache stores raw RGBA texture atlas bytes on disk to avoid
/// regenerating or re-loading the atlas from source assets on every run.
struct TextureAtlasCache {
    /// Filesystem path to the cache file (e.g. `"assets/texture_atlas.cache"`).
    cache_path: String,
}

impl TextureAtlasCache {
    /// Creates a new [`TextureAtlasCache`] pointing at the given path.
    ///
    /// # Arguments
    ///
    /// * `cache_path` - Path to the cache file. The file does not need to exist yet.
    fn new(cache_path: &str) -> Self {
        Self {
            cache_path: cache_path.to_string(),
        }
    }

    /// Returns `true` if the cache file exists on disk.
    fn exists(&self) -> bool {
        Path::new(&self.cache_path).exists()
    }

    /// Attempts to read and return the raw bytes stored in the cache file.
    ///
    /// Returns `Some(bytes)` on success, or `None` if the file cannot be read.
    fn load(&self) -> Option<Vec<u8>> {
        fs::read(&self.cache_path).ok()
    }
}

/// Generates a full mipmap chain from a 2D-array texture atlas.
///
/// Each mip level is half the size of the previous level in both dimensions
/// (clamped to a minimum of 1×1). The input atlas is assumed to consist of
/// `16` array layers packed contiguously in memory (RGBA8, 4 bytes per texel).
/// Downsampling uses a bilinear (Triangle) filter.
///
/// # Arguments
///
/// * `atlas_data`   - Raw RGBA8 pixel data for all 16 layers at mip level 0.
/// * `atlas_width`  - Width of a single layer in texels.
/// * `atlas_height` - Height of a single layer in texels.
///
/// # Returns
///
/// A `Vec` where index `i` holds the raw RGBA8 bytes for mip level `i`.
/// Level 0 is the original full-resolution data; subsequent levels are
/// progressively downsampled.
pub fn generate_texture_atlas_with_mipmaps(
    atlas_data: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) -> Vec<Vec<u8>> {
    let mip_level_count = (atlas_width.max(atlas_height) as f32).log2().floor() as u32 + 1;
    let mut mip_levels = Vec::with_capacity(mip_level_count as usize);

    // Level 0 is the unmodified source data.
    mip_levels.push(atlas_data.to_vec());

    for level in 1..mip_level_count {
        let src_level = level - 1;
        let src_width = (atlas_width >> src_level).max(1);
        let src_height = (atlas_height >> src_level).max(1);
        let dst_width = (atlas_width >> level).max(1);
        let dst_height = (atlas_height >> level).max(1);

        let mut level_data = Vec::with_capacity((dst_width * dst_height * 4 * 16) as usize);

        // Downsample each of the 16 array layers independently.
        for layer in 0..16 {
            let layer_size = (src_width * src_height * 4) as usize;
            let layer_offset = layer * layer_size;
            let src_data = &mip_levels[src_level as usize];
            let layer_pixels = &src_data[layer_offset..layer_offset + layer_size];

            let img = image::RgbaImage::from_raw(src_width, src_height, layer_pixels.to_vec())
                .expect("Failed to create image from mipmap level");

            let resized = image::imageops::resize(
                &img,
                dst_width,
                dst_height,
                image::imageops::FilterType::Triangle,
            );
            level_data.extend_from_slice(&resized.into_raw());
        }

        mip_levels.push(level_data);
    }

    mip_levels
}

/// Uploads a texture atlas (with auto-generated mipmaps) to the GPU.
///
/// Creates a [`wgpu::Texture`] with format [`wgpu::TextureFormat::Rgba8UnormSrgb`],
/// `16` array layers, and a full mipmap chain. All mip levels are written to the
/// GPU via [`wgpu::Queue::write_texture`].
///
/// # Arguments
///
/// * `device`       - The wgpu device used to allocate the texture.
/// * `queue`        - The wgpu queue used to upload pixel data.
/// * `atlas_data`   - Raw RGBA8 pixel data for all 16 layers at mip level 0.
/// * `atlas_width`  - Width of the atlas in texels.
/// * `atlas_height` - Height of the atlas in texels.
///
/// # Returns
///
/// A tuple of `(texture, view)` where `view` is a [`wgpu::TextureViewDimension::D2Array`]
/// view suitable for use in shaders as a `texture2d_array`.
pub fn create_texture_atlas_optimized(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas_data: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mip_level_count = (atlas_width.max(atlas_height) as f32).log2().floor() as u32 + 1;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture Atlas"),
        size: wgpu::Extent3d {
            width: atlas_width,
            height: atlas_height,
            depth_or_array_layers: 16,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mip_levels = generate_texture_atlas_with_mipmaps(atlas_data, atlas_width, atlas_height);

    // Upload each mip level. All 16 layers are packed in a single write_texture
    // call per level by setting depth_or_array_layers to 16.
    for (level, level_data) in mip_levels.iter().enumerate() {
        let mip_width = (atlas_width >> level).max(1);
        let mip_height = (atlas_height >> level).max(1);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            level_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * mip_width),
                rows_per_image: Some(mip_height),
            },
            wgpu::Extent3d {
                width: mip_width,
                height: mip_height,
                depth_or_array_layers: 16,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Texture Atlas View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

/// Loads or generates the texture atlas, then uploads it to the GPU.
///
/// Resolution order for the atlas source data:
///
/// 1. **Disk cache** (`assets/texture_atlas.cache`) — raw bytes written by a
///    previous run; fastest path, skips all image decoding.
/// 2. **PNG file** (`assets/textures.png`) — decoded on first run and used
///    directly; dimensions are read from the file.
/// 3. **Procedural generation** — fallback when neither asset is available;
///    produces a [`TEXTURE_SIZE`]×[`TEXTURE_SIZE`] atlas via [`generate_texture_atlas`].
///
/// After acquiring the raw pixel data the function calls
/// [`create_texture_atlas_optimized`] to build the GPU texture with mipmaps.
///
/// # Arguments
///
/// * `device` - The wgpu device used to allocate GPU resources.
/// * `queue`  - The wgpu queue used to upload pixel data.
///
/// # Returns
///
/// A tuple of `(texture, view, width, height)`:
/// - `texture` — the allocated GPU texture.
/// - `view`    — a `D2Array` texture view ready for binding in shaders.
/// - `width`   — atlas width in texels.
/// - `height`  — atlas height in texels.
pub fn load_or_generate_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, u32, u32) {
    let cache = TextureAtlasCache::new("assets/texture_atlas.cache");

    let (atlas_data, atlas_width, atlas_height) = if cache.exists() {
        match cache.load() {
            Some(cached_data) => {
                log(
                    LogLevel::Info,
                    &format!(
                        "Loaded texture atlas from cache ({} bytes)",
                        cached_data.len()
                    ),
                );
                (cached_data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
            None => {
                // Cache file exists but could not be read; fall back to generation.
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    } else {
        match load_texture_atlas_from_file("assets/textures.png") {
            Ok((data, width, height)) => {
                log(
                    LogLevel::Info,
                    &format!(
                        "Loaded texture atlas from PNG ({} bytes, {}x{})",
                        data.len(),
                        width,
                        height
                    ),
                );
                (data, width, height)
            }
            Err(e) => {
                log(
                    LogLevel::Warning,
                    &format!(
                        "Failed to load texture atlas from PNG: {}; falling back to procedural generation",
                        e
                    ),
                );
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    };

    let (texture, view) =
        create_texture_atlas_optimized(device, queue, &atlas_data, atlas_width, atlas_height);

    (texture, view, atlas_width, atlas_height)
}
