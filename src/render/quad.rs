use bytemuck::{Pod, Zeroable};

use crate::core::vertex::Vertex;

/// Compact, procedurally expanded terrain face. Coordinates and dimensions are
/// expressed in half-block units relative to the owning subchunk.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct PackedQuad {
    /// x(6), y(6), z(6), face(3), RGB tint(9), AO-diagonal selector(1).
    pub origin_and_face: u32,
    /// width(6), height(6), material(8), four 2-bit AO values(8).
    pub size_material_ao: u32,
}

impl PackedQuad {
    /// Builds a terrain descriptor directly, without transient vertex or index
    /// buffers. `origin` and dimensions are in half-block units relative to
    /// the owning subchunk.
    #[inline]
    pub fn terrain(
        origin: [u32; 3],
        face: u8,
        width: u32,
        height: u32,
        material: u8,
        color: [u8; 3],
        ao: [u8; 4],
    ) -> Self {
        let diagonal =
            u32::from(u16::from(ao[0]) + u16::from(ao[2]) > u16::from(ao[1]) + u16::from(ao[3]));
        let ao = ao.iter().enumerate().fold(0u32, |bits, (i, value)| {
            bits | (u32::from((*value).min(3)) << (i * 2))
        });
        let color = u32::from(color[0].min(7))
            | (u32::from(color[1].min(7)) << 3)
            | (u32::from(color[2].min(7)) << 6);

        Self {
            origin_and_face: (origin[0].min(63))
                | (origin[1].min(63) << 6)
                | (origin[2].min(63) << 12)
                | (u32::from(face & 0x7) << 18)
                | (color << 21)
                | (diagonal << 30),
            size_material_ao: width.clamp(1, 63)
                | (height.clamp(1, 63) << 6)
                | (u32::from(material) << 12)
                | (ao << 20),
        }
    }

    /// Converts one independently emitted legacy quad into its compact form.
    /// `vertices` must be in the source quad's v0..v3 order.
    pub fn from_vertices(vertices: &[Vertex], indices: &[u32], subchunk_origin: [i32; 3]) -> Self {
        debug_assert_eq!(vertices.len(), 4);
        debug_assert_eq!(indices.len(), 6);
        let packed = vertices[0].packed;
        let face = packed & 0x7;
        let color = (packed >> 23) & 0x1ff;
        let ao = vertices.iter().enumerate().fold(0u32, |bits, (i, vertex)| {
            bits | (((vertex.packed >> 21) & 0x3) << (i * 2))
        });
        let p0 = vertices[0].position;
        let half = |value: f32, origin: i32| -> u32 {
            let result = ((value - origin as f32) * 2.0).round() as i32;
            debug_assert!((0..64).contains(&result));
            result.clamp(0, 63) as u32
        };
        let x = half(p0[0], subchunk_origin[0]);
        let y = half(p0[1], subchunk_origin[1]);
        let z = half(p0[2], subchunk_origin[2]);
        let edge_len = |a: [f32; 3], b: [f32; 3]| -> u32 {
            let distance = (a[0] - b[0])
                .abs()
                .max((a[1] - b[1]).abs())
                .max((a[2] - b[2]).abs());
            (distance * 2.0).round().clamp(1.0, 63.0) as u32
        };
        let width = edge_len(vertices[0].position, vertices[1].position);
        let height = edge_len(vertices[1].position, vertices[2].position);
        let material = (packed >> 3) & 0xff;
        // The source index order selects the smoother AO diagonal.
        let diagonal = u32::from(indices[2] == indices[0] + 3);

        Self {
            origin_and_face: x
                | (y << 6)
                | (z << 12)
                | (face << 18)
                | (color << 21)
                | (diagonal << 30),
            size_material_ao: width | (height << 6) | (material << 12) | (ao << 20),
        }
    }
}

/// Appends one terrain quad directly to the compact descriptor stream.
/// Coordinates are world-space block units; dimensions are half-block units.
#[inline]
pub fn emit_packed_quad(
    quads: &mut Vec<PackedQuad>,
    subchunk_origin: [i32; 3],
    origin: [f32; 3],
    face: u8,
    width: u32,
    height: u32,
    material: u8,
    color: [u8; 3],
    ao: [u8; 4],
) {
    let half = |value: f32, axis: usize| {
        let value = ((value - subchunk_origin[axis] as f32) * 2.0).round() as i32;
        debug_assert!((0..64).contains(&value));
        value.clamp(0, 63) as u32
    };
    quads.push(PackedQuad::terrain(
        [half(origin[0], 0), half(origin[1], 1), half(origin[2], 2)],
        face,
        width,
        height,
        material,
        color,
        ao,
    ));
}

/// Converts the legacy independently-emitted quad stream to descriptors.
pub fn pack_quad_stream(
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    subchunk_origin: [i32; 3],
) -> Vec<PackedQuad> {
    debug_assert_eq!(vertices.len() % 4, 0);
    debug_assert_eq!(indices.len() % 6, 0);
    vertices
        .chunks_exact(4)
        .zip(indices.chunks_exact(6))
        .map(|(quad_vertices, quad_indices)| {
            PackedQuad::from_vertices(quad_vertices, quad_indices, subchunk_origin)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_eight_bytes_and_preserves_half_block_geometry() {
        assert_eq!(std::mem::size_of::<PackedQuad>(), 8);
        let packed = Vertex::pack_with_ao(3, [1.0, 0.5, 0.0], 42, 1, 1, 1, 0);
        let vertices = [
            Vertex {
                position: [1.0, 2.5, 3.0],
                packed,
            },
            Vertex {
                position: [1.5, 2.5, 3.0],
                packed: (packed & !(3 << 21)) | (1 << 21),
            },
            Vertex {
                position: [1.5, 3.5, 3.0],
                packed: (packed & !(3 << 21)) | (2 << 21),
            },
            Vertex {
                position: [1.0, 3.5, 3.0],
                packed: (packed & !(3 << 21)) | (3 << 21),
            },
        ];
        let quad = PackedQuad::from_vertices(&vertices, &[0, 1, 3, 1, 2, 3], [0, 0, 0]);
        let legacy_default = PackedQuad::from_vertices(&vertices, &[0, 1, 2, 0, 2, 3], [0, 0, 0]);
        let direct = PackedQuad::terrain([2, 5, 6], 3, 1, 2, 42, [7, 3, 0], [0, 1, 2, 3]);
        assert_eq!(legacy_default.origin_and_face, direct.origin_and_face);
        assert_eq!(legacy_default.size_material_ao, direct.size_material_ao);
        assert_eq!(quad.origin_and_face & 0x3ffff, 2 | (5 << 6) | (6 << 12));
        assert_eq!(
            (
                quad.size_material_ao & 0x3f,
                (quad.size_material_ao >> 6) & 0x3f
            ),
            (1, 2)
        );
        assert_eq!((quad.size_material_ao >> 12) & 0xff, 42);
        assert_eq!((quad.size_material_ao >> 20) & 0xff, 0b11_10_01_00);
        assert_eq!((quad.origin_and_face >> 30) & 1, 1);
    }

    #[test]
    fn vertex_pulling_shaders_parse_and_validate() {
        for source in [
            include_str!("../shaders/terrain.wgsl"),
            include_str!("../shaders/water.wgsl"),
            include_str!("../shaders/cull.wgsl"),
        ] {
            let module =
                wgpu::naga::front::wgsl::parse_str(source).expect("vertex-pulling WGSL must parse");
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("vertex-pulling WGSL must validate");
        }
    }
}
