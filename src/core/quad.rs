use bytemuck::{Pod, Zeroable};

use super::vertex::Vertex;

/// One axis-aligned terrain rectangle, expanded into six vertices by the GPU.
/// Origins and extents use half-block units relative to the owning subchunk.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PackedQuad {
    pub origin_and_face: u32,
    pub size_material_ao: u32,
    pub color_flags: u32,
    pub _reserved: u32,
}

impl PackedQuad {
    /// Converts a legacy four-vertex axis-aligned quad into its compact descriptor.
    pub fn from_vertices(vertices: &[Vertex], subchunk_origin: [i32; 3]) -> Self {
        debug_assert_eq!(vertices.len(), 4);
        let packed = vertices[0].packed;
        let face = packed & 0x7;
        let min = [
            vertices
                .iter()
                .map(|v| v.position[0])
                .fold(f32::INFINITY, f32::min),
            vertices
                .iter()
                .map(|v| v.position[1])
                .fold(f32::INFINITY, f32::min),
            vertices
                .iter()
                .map(|v| v.position[2])
                .fold(f32::INFINITY, f32::min),
        ];
        let max = [
            vertices
                .iter()
                .map(|v| v.position[0])
                .fold(f32::NEG_INFINITY, f32::max),
            vertices
                .iter()
                .map(|v| v.position[1])
                .fold(f32::NEG_INFINITY, f32::max),
            vertices
                .iter()
                .map(|v| v.position[2])
                .fold(f32::NEG_INFINITY, f32::max),
        ];
        let local_half =
            |axis: usize| ((min[axis] - subchunk_origin[axis] as f32) * 2.0).round() as u32;
        let extent_half = |axis: usize| ((max[axis] - min[axis]) * 2.0).round().max(1.0) as u32;
        let (width, height) = match face {
            0 | 1 => (extent_half(2), extent_half(1)),
            2 | 3 => (extent_half(2), extent_half(0)),
            4 | 5 => (extent_half(0), extent_half(1)),
            _ => unreachable!("terrain quads have a cardinal face"),
        };
        let ao = vertices
            .iter()
            .enumerate()
            .fold(0u32, |value, (i, vertex)| {
                value | (((vertex.packed >> 21) & 0x3) << (i * 2))
            });
        let color = (packed >> 23) & 0x1ff;
        let diagonal = u32::from(
            ((vertices[0].packed >> 21) & 0x3) + ((vertices[2].packed >> 21) & 0x3)
                > ((vertices[1].packed >> 21) & 0x3) + ((vertices[3].packed >> 21) & 0x3),
        );
        Self {
            origin_and_face: local_half(0)
                | (local_half(1) << 5)
                | (local_half(2) << 10)
                | (face << 15),
            size_material_ao: ((width - 1) & 0x1f)
                | (((height - 1) & 0x1f) << 5)
                | (((packed >> 3) & 0xff) << 10)
                | (ao << 18),
            color_flags: color | (diagonal << 9),
            _reserved: 0,
        }
    }
}
