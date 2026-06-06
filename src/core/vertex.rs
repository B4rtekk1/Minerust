use bytemuck::{Pod, Zeroable};

/// A GPU-ready packed vertex for voxel, terrain, and simple UI geometry.
///
/// The layout is tightly packed to 16 bytes and matches the wgpu attribute layout
/// returned by [`Vertex::desc`]. Implements [`Pod`] and [`Zeroable`] for safe
/// direct casting to/from byte slices.
///
/// This format reduces memory bandwidth and VRAM usage by packing normals,
/// colors, and UV metadata into a single 32-bit field.
///
/// # Memory layout
/// | Field       | Offset | Size | Format        |
/// |-------------|--------|------|---------------|
/// | `position`  | 0      | 12 B | `Float32x3`   |
/// | `packed`    | 12     | 4 B  | `Uint32`      |
///
/// # Packed Data Bits (32 bits total)
/// | Bits  | Purpose        | Range         |
/// |-------|----------------|---------------|
/// | 0-2   | Normal Index   | 0-5 (cardinal)|
/// | 3-10  | Texture Index  | 0-255         |
/// | 11-12 | UV Corner      | 0-3           |
/// | 13-16 | Width - 1      | 0-15          |
/// | 17-20 | Height - 1     | 0-15          |
/// | 21-22 | Voxel AO       | 0-3           |
/// | 23-25 | Color R (3-bit)| 0-7           |
/// | 26-28 | Color G (3-bit)| 0-7           |
/// | 29-31 | Color B (3-bit)| 0-7           |
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub packed: u32,
}

impl Vertex {
    /// Packs normal, color, texture, corner, and dimensions into the 32-bit `packed` field.
    ///
    /// Terrain vertices default to fully open voxel AO (`3`). Use
    /// [`Self::pack_with_ao`] when building world meshes with per-corner AO.
    pub fn pack(
        normal_idx: u8,  // 0-5 (3 bits)
        color: [f32; 3], // 0.0-1.0 (9 bits: 3R, 3G, 3B)
        tex_index: u8,   // 0-255 (8 bits)
        corner_idx: u8,  // 0-3 (2 bits)
        width: u8,       // 1-16 (4 bits)
        height: u8,      // 1-16 (4 bits)
    ) -> u32 {
        Self::pack_with_ao(normal_idx, color, tex_index, corner_idx, width, height, 3)
    }

    /// Packs a terrain/world vertex with explicit 2-bit voxel ambient occlusion.
    ///
    /// `ao` uses `0 = darkest` and `3 = fully open`.
    pub fn pack_with_ao(
        normal_idx: u8,  // 0-5 (3 bits)
        color: [f32; 3], // 0.0-1.0 (9 bits: 3R, 3G, 3B)
        tex_index: u8,   // 0-255 (8 bits)
        corner_idx: u8,  // 0-3 (2 bits)
        width: u8,       // 1-16 (4 bits)
        height: u8,      // 1-16 (4 bits)
        ao: u8,          // 0-3 (2 bits)
    ) -> u32 {
        let n = (normal_idx as u32) & 0x7;
        let t = (tex_index as u32) & 0xFF;
        let uv = (corner_idx as u32) & 0x3;
        let w = ((width.saturating_sub(1)) as u32) & 0xF;
        let h = ((height.saturating_sub(1)) as u32) & 0xF;
        let ao = (ao.min(3) as u32) & 0x3;

        // 9-bit color is enough for the coarse GI tint and leaves room for AO.
        let r = ((color[0].clamp(0.0, 1.0) * 7.0) as u32) & 0x7;
        let g = ((color[1].clamp(0.0, 1.0) * 7.0) as u32) & 0x7;
        let b = ((color[2].clamp(0.0, 1.0) * 7.0) as u32) & 0x7;

        n | (t << 3)
            | (uv << 11)
            | (w << 13)
            | (h << 17)
            | (ao << 21)
            | (r << 23)
            | (g << 26)
            | (b << 29)
    }

    /// Packs a screen-space/UI vertex and stores alpha in the width/height bits.
    ///
    /// UI uses its own shader-specific layout for color so terrain AO packing
    /// does not reduce HUD color precision.
    pub fn pack_ui(normal_idx: u8, color: [f32; 4], tex_index: u8, corner_idx: u8) -> u32 {
        let alpha = ((color[3].clamp(0.0, 1.0) * 255.0).round() as u32) & 0xFF;
        let n = (normal_idx as u32) & 0x7;
        let t = (tex_index as u32) & 0xFF;
        let uv = (corner_idx as u32) & 0x3;
        let r = ((color[0].clamp(0.0, 1.0) * 15.0) as u32) & 0xF;
        let g = ((color[1].clamp(0.0, 1.0) * 15.0) as u32) & 0xF;
        let b = ((color[2].clamp(0.0, 1.0) * 7.0) as u32) & 0x7;

        n | (t << 3)
            | (uv << 11)
            | ((alpha & 0x0F) << 13)
            | (((alpha >> 4) & 0x0F) << 17)
            | (r << 21)
            | (g << 25)
            | (b << 29)
    }

    /// Legacy pack helpers (unused but kept for compatibility during refactor if needed)
    #[inline]
    pub fn pack_normal(n: [f32; 3]) -> u8 {
        if n[0] > 0.5 {
            1
        } else if n[0] < -0.5 {
            0
        } else if n[1] > 0.5 {
            3
        } else if n[1] < -0.5 {
            2
        } else if n[2] > 0.5 {
            5
        } else {
            4
        }
    }

    #[inline]
    pub fn pack_color(c: [f32; 3]) -> [u8; 3] {
        [
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
        ]
    }

    #[inline]
    pub fn pack_color_rgba(c: [f32; 4]) -> [u8; 4] {
        [
            (c[0] * 255.0) as u8,
            (c[1] * 255.0) as u8,
            (c[2] * 255.0) as u8,
            (c[3] * 255.0) as u8,
        ]
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

/// A dedicated vertex format for the block outline shader.
///
/// The outline shader reuses the `normal` slot to store the opposite endpoint
/// of the line segment plus a `side` flag in `.w`. Unlike [`Vertex`], this
/// attribute must remain full `f32` precision because it carries world-space
/// positions.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct OutlineVertex {
    /// World-space position of this segment endpoint.
    pub position: [f32; 3],
    /// Opposite endpoint of the segment in `.xyz`, and `side` in `.w`.
    pub other: [f32; 4],
    /// Packed RGBA outline color.
    pub color: [u8; 4],
    /// `uv.x` stores the half-width in pixels; `uv.y` is unused.
    pub uv: [f32; 2],
    /// Unused by the shader, kept for layout parity.
    pub tex_index: f32,
}

impl OutlineVertex {
    /// Returns the wgpu vertex layout used by the outline pipeline.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<OutlineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}
