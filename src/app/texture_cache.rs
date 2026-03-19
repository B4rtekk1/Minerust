use std::fs;
use std::path::Path;

use minerust::{TEXTURE_SIZE, generate_texture_atlas, load_texture_atlas_from_file};

struct TextureAtlasCache {
    cache_path: String,
}

impl TextureAtlasCache {
    fn new(cache_path: &str) -> Self {
        Self {
            cache_path: cache_path.to_string(),
        }
    }

    fn exists(&self) -> bool {
        Path::new(&self.cache_path).exists()
    }

    fn load(&self) -> Option<Vec<u8>> {
        fs::read(&self.cache_path).ok()
    }
}

pub fn generate_texture_atlas_with_mipmaps(
    atlas_data: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) -> Vec<Vec<u8>> {
    let mip_level_count = (atlas_width.max(atlas_height) as f32).log2().floor() as u32 + 1;
    let mut mip_levels = Vec::with_capacity(mip_level_count as usize);

    mip_levels.push(atlas_data.to_vec());

    for level in 1..mip_level_count {
        let src_level = level - 1;
        let src_width = (atlas_width >> src_level).max(1);
        let src_height = (atlas_height >> src_level).max(1);
        let dst_width = (atlas_width >> level).max(1);
        let dst_height = (atlas_height >> level).max(1);

        let mut level_data = Vec::with_capacity((dst_width * dst_height * 4 * 16) as usize);

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

pub fn load_or_generate_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, u32, u32) {
    let cache = TextureAtlasCache::new("assets/texture_atlas.cache");

    let (atlas_data, atlas_width, atlas_height) = if cache.exists() {
        match cache.load() {
            Some(cached_data) => {
                tracing::info!(
                    "Loaded texture atlas from cache ({} bytes)",
                    cached_data.len()
                );
                (cached_data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
            None => {
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    } else {
        match load_texture_atlas_from_file("assets/textures.png") {
            Ok((data, width, height)) => {
                tracing::info!("Loaded texture atlas from PNG: {}x{}", width, height);
                (data, width, height)
            }
            Err(e) => {
                tracing::warn!("Failed to load texture atlas from PNG: {}", e);
                let data = generate_texture_atlas();
                (data, TEXTURE_SIZE, TEXTURE_SIZE)
            }
        }
    };

    let (texture, view) =
        create_texture_atlas_optimized(device, queue, &atlas_data, atlas_width, atlas_height);

    (texture, view, atlas_width, atlas_height)
}

