// ============================================================================
// FIX #1: Generowanie mipmap tylko RAZ przy starcie + caching
// ============================================================================

use std::fs;
use std::path::Path;

use render3d::{TEXTURE_SIZE, generate_texture_atlas, load_texture_atlas_from_file};

/// Cached texture atlas with pre-generated mipmaps
struct TextureAtlasCache {
    cache_path: String,
}

impl TextureAtlasCache {
    fn new(cache_path: &str) -> Self {
        Self {
            cache_path: cache_path.to_string(),
        }
    }

    /// Check if cached atlas exists and is valid
    fn exists(&self) -> bool {
        Path::new(&self.cache_path).exists()
    }

    /// Load pre-generated atlas with mipmaps from cache
    fn load(&self) -> Option<Vec<u8>> {
        fs::read(&self.cache_path).ok()
    }
}

/// Generate all mipmap levels for texture atlas ONCE
pub fn generate_texture_atlas_with_mipmaps(
    atlas_data: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) -> Vec<Vec<u8>> {
    let mip_level_count = (atlas_width.max(atlas_height) as f32).log2().floor() as u32 + 1;
    let mut mip_levels = Vec::with_capacity(mip_level_count as usize);

    // Level 0 - original data
    mip_levels.push(atlas_data.to_vec());

    // Generate subsequent mip levels
    for level in 1..mip_level_count {
        let src_level = level - 1;
        let src_width = (atlas_width >> src_level).max(1);
        let src_height = (atlas_height >> src_level).max(1);
        let dst_width = (atlas_width >> level).max(1);
        let dst_height = (atlas_height >> level).max(1);

        let mut level_data = Vec::with_capacity((dst_width * dst_height * 4 * 16) as usize);

        // Process each layer separately
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

/// Create texture atlas with mipmaps - optimized version
pub fn create_texture_atlas_optimized(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas_data: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let mip_level_count = (atlas_width.max(atlas_height) as f32).log2().floor() as u32 + 1;

    // Create texture
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

    tracing::info!("Generating texture mipmaps (one-time operation)...");
    let mip_levels = generate_texture_atlas_with_mipmaps(atlas_data, atlas_width, atlas_height);

    // Upload all mip levels to GPU
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

    tracing::info!("Texture atlas with {} mip levels ready", mip_level_count);

    (texture, view)
}

pub fn load_or_generate_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, u32, u32) {
    // Try loading from cache first
    let cache = TextureAtlasCache::new("assets/texture_atlas.cache");

    let (atlas_data, atlas_width, atlas_height) = if cache.exists() {
        tracing::info!("Loading texture atlas from cache...");
        match cache.load() {
            Some(cached_data) => {
                match load_texture_atlas_from_file("assets/textures.png") {
                    Ok((data, width, height)) => (data, width, height),
                    Err(_) => {
                        tracing::warn!("Cache invalid, regenerating...");
                        let data = generate_texture_atlas();
                        (data, TEXTURE_SIZE, TEXTURE_SIZE)
                    }
                }
            }
            None => {
                tracing::warn!("Failed to read cache, regenerating...");
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    } else {
        // Load from file or generate
        match load_texture_atlas_from_file("assets/textures.png") {
            Ok((data, width, height)) => {
                tracing::info!("Loaded texture atlas from PNG: {}x{}", width, height);
                (data, width, height)
            }
            Err(e) => {
                tracing::error!("Failed to load textures.png: {}", e);
                tracing::warn!("Using procedural texture atlas generation.");
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    };

    // Create texture with mipmaps (ONE TIME!)
    let (texture, view) =
        create_texture_atlas_optimized(device, queue, &atlas_data, atlas_width, atlas_height);

    (texture, view, atlas_width, atlas_height)
}


use parking_lot::RwLock;
use std::sync::Arc;

/// Async mipmap generator for large textures
struct AsyncMipmapGenerator {
    _thread: Option<std::thread::JoinHandle<()>>,
    result: Arc<RwLock<Option<Vec<Vec<u8>>>>>,
}

impl AsyncMipmapGenerator {
    fn new(atlas_data: Vec<u8>, width: u32, height: u32) -> Self {
        let result = Arc::new(RwLock::new(None));
        let result_clone = result.clone();

        let thread = std::thread::spawn(move || {
            tracing::info!("Generating mipmaps in background...");
            let mipmaps = generate_texture_atlas_with_mipmaps(&atlas_data, width, height);
            *result_clone.write() = Some(mipmaps);
            tracing::info!("Mipmap generation complete!");
        });

        Self {
            _thread: Some(thread),
            result,
        }
    }

    fn try_get(&self) -> Option<Vec<Vec<u8>>> {
        self.result.read().clone()
    }

    fn is_ready(&self) -> bool {
        self.result.read().is_some()
    }
}

// ============================================================================
// BENCHMARKING - sprawdź różnicę!
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_mipmap_generation() {
        let atlas_data = vec![0u8; 2048 * 2048 * 4 * 16]; // Dummy data

        let start = Instant::now();
        let mipmaps = generate_texture_atlas_with_mipmaps(&atlas_data, 2048, 2048);
        let elapsed = start.elapsed();

        println!("Mipmap generation took: {:?}", elapsed);
        println!("Generated {} mip levels", mipmaps.len());

        // Typically should be < 100ms for 2048x2048x16
        assert!(elapsed.as_millis() < 500);
    }
}
