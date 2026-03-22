use bytemuck::{Pod, Zeroable};

/// A GPU-ready vertex with position, normal, color, UV coordinates, and texture index.
///
/// The layout is tightly packed to 32 bytes and matches the wgpu attribute layout
/// returned by [`Vertex::desc`]. Implements [`Pod`] and [`Zeroable`] for safe
/// direct casting to/from byte slices.
///
/// # Memory layout
/// | Field       | Offset | Size | Format        |
/// |-------------|--------|------|---------------|
/// | `position`  | 0      | 12 B | `Float32x3`   |
/// | `normal`    | 12     | 4 B  | `Snorm8x4`    |
/// | `color`     | 16     | 4 B  | `Unorm8x4`    |
/// | `uv`        | 20     | 8 B  | `Float32x2`   |
/// | `tex_index` | 28     | 4 B  | `Float32`     |
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// World-space position `[x, y, z]` in floating-point coordinates.
    pub position: [f32; 3],
    /// Packed surface normal. Use [`Vertex::pack_normal`] to convert from `[f32; 3]`.
    /// The fourth byte is padding and should be `0`.
    pub normal: [i8; 4],
    /// RGBA color packed as `u8` per channel. Use [`Vertex::pack_color`] or
    /// [`Vertex::pack_color_rgba`] to convert from normalized `f32` values.
    pub color: [u8; 4],
    /// Texture UV coordinates `[u, v]` in normalized `[0.0, 1.0]` space.
    pub uv: [f32; 2],
    /// Index of the texture to sample, stored as `f32` for shader compatibility.
    pub tex_index: f32,
}

impl Vertex {
    /// Packs a floating-point RGB normal into a signed 8-bit `[i8; 4]` representation.
    ///
    /// Each component is clamped to `[-1.0, 1.0]` and scaled to the `[-127, 127]` range.
    /// The fourth element is always `0` (padding).
    ///
    /// # Parameters
    /// - `n`: A unit-length normal vector `[x, y, z]` with components in `[-1.0, 1.0]`.
    ///
    /// # Returns
    /// A 4-byte packed normal suitable for [`Vertex::normal`].
    #[inline]
    pub fn pack_normal(n: [f32; 3]) -> [i8; 4] {
        [
            (n[0].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[1].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[2].clamp(-1.0, 1.0) * 127.0) as i8,
            0,
        ]
    }

    /// Packs a floating-point RGB color into an unsigned 8-bit `[u8; 4]` representation.
    ///
    /// Each component is clamped to `[0.0, 1.0]` and scaled to `[0, 255]`.
    /// Alpha is set to `255` (fully opaque).
    ///
    /// # Parameters
    /// - `c`: An RGB color `[r, g, b]` with components in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// A 4-byte packed RGBA color suitable for [`Vertex::color`].
    #[inline]
    pub fn pack_color(c: [f32; 3]) -> [u8; 4] {
        [
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    }

    /// Packs a floating-point RGBA color into an unsigned 8-bit `[u8; 4]` representation.
    ///
    /// Each component is clamped to `[0.0, 1.0]` and scaled to `[0, 255]`.
    ///
    /// # Parameters
    /// - `c`: An RGBA color `[r, g, b, a]` with components in `[0.0, 1.0]`.
    ///
    /// # Returns
    /// A 4-byte packed RGBA color suitable for [`Vertex::color`].
    #[inline]
    pub fn pack_color_rgba(c: [f32; 4]) -> [u8; 4] {
        [
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            (c[3].clamp(0.0, 1.0) * 255.0) as u8,
        ]
    }

    /// Returns the wgpu vertex buffer layout descriptor for this vertex type.
    ///
    /// Describes the stride, step mode, and all five shader attributes so wgpu
    /// knows how to interpret a buffer of [`Vertex`] values.
    ///
    /// # Shader locations
    /// | Location | Field       | Format      |
    /// |----------|-------------|-------------|
    /// | 0        | `position`  | `Float32x3` |
    /// | 1        | `normal`    | `Snorm8x4`  |
    /// | 2        | `color`     | `Unorm8x4`  |
    /// | 3        | `uv`        | `Float32x2` |
    /// | 4        | `tex_index` | `Float32`   |

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
                    format: wgpu::VertexFormat::Snorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
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
