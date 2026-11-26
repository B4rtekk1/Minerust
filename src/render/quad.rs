use bytemuck::{Pod, Zeroable};

use crate::core::vertex::Vertex;

/// Compact, procedurally expanded terrain face. Coordinates and dimensions are
/// expressed in half-block units relative to the owning subchunk.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct PackedQuad {
    /// x(6), y(6), z(6), face(3), RGB tint(9), AO-diagonal selector(1).
    pub origin_and_face: u32,
    /// width(6), height(6), material(8), four 2-bit AO values(8),
    /// climate tint palette index(4).
    pub size_material_ao: u32,
    /// Four 4-bit sky and block light levels, one per corner per channel.
    pub light: u32,
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
        light: [u8; 4],
        block_light: [u8; 4],
        tint: u8,
    ) -> Self {
        // Split the quad along the diagonal whose corners are darker, so the
        // interpolated shading follows the occlusion instead of streaking.
        let shade = |i: usize| u16::from(ao[i]) * 5 + u16::from(light[i].min(15));
        let diagonal = u32::from(shade(0) + shade(2) > shade(1) + shade(3));
        let packed_light = light.iter().enumerate().fold(0u32, |bits, (i, &l)| {
            bits | (u32::from(l.min(15)) << (i * 4))
        });
        let packed_block_light = block_light.iter().enumerate().fold(0u32, |bits, (i, &l)| {
            bits | (u32::from(l.min(15)) << (16 + i * 4))
        });
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
                | (ao << 20)
                | (u32::from(tint & 0xf) << 28),
            light: packed_light | packed_block_light,
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
            light: 0xffff,
        }
    }
}

/// Terrain draw groups per subchunk: six face directions of opaque geometry
/// (indexed like the mesher's `face_dir`) followed by alpha-tested geometry.
pub const TERRAIN_GROUPS: usize = 7;
/// Index of the alpha-tested (cutout) group.
pub const CUTOUT_GROUP: usize = 6;

/// Atlas layers drawn with alpha testing (leaves, dead bush, cherry leaves).
pub fn is_cutout_material(tex_index: u8) -> bool {
    matches!(tex_index, 8 | 15 | 36)
}

impl PackedQuad {
    /// Face direction (0..6) of the quad.
    pub fn face(&self) -> usize {
        ((self.origin_and_face >> 18) & 0x7) as usize
    }

    /// Atlas layer of the quad.
    pub fn material(&self) -> u8 {
        ((self.size_material_ao >> 12) & 0xff) as u8
    }

    /// Draw group of a terrain quad (see [`TERRAIN_GROUPS`]).
    pub fn terrain_group(&self) -> usize {
        if is_cutout_material(self.material()) {
            CUTOUT_GROUP
        } else {
            self.face().min(5)
        }
    }
}

/// Reorders terrain quads into contiguous draw groups (stable counting
/// sort) and returns the quad count of each group.
pub fn group_terrain_quads(quads: &[PackedQuad]) -> (Vec<PackedQuad>, [u32; TERRAIN_GROUPS]) {
    let mut counts = [0u32; TERRAIN_GROUPS];
    for quad in quads {
        counts[quad.terrain_group()] += 1;
    }
    let mut starts = [0usize; TERRAIN_GROUPS];
    for group in 1..TERRAIN_GROUPS {
        starts[group] = starts[group - 1] + counts[group - 1] as usize;
    }
    debug_assert!(
        counts
            .iter()
            .all(|&c| c as usize <= crate::render::indirect::MAX_QUADS_PER_DRAW),
        "a terrain draw group exceeds the shared index pattern"
    );
    let mut grouped = vec![PackedQuad::default(); quads.len()];
    for quad in quads {
        let group = quad.terrain_group();
        grouped[starts[group]] = *quad;
        starts[group] += 1;
    }
    (grouped, counts)
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
    light: [u8; 4],
    block_light: [u8; 4],
    tint: u8,
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
        light,
        block_light,
        tint,
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
    fn descriptor_is_twelve_bytes_and_preserves_half_block_geometry() {
        assert_eq!(std::mem::size_of::<PackedQuad>(), 12);
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
        let direct = PackedQuad::terrain(
            [2, 5, 6],
            3,
            1,
            2,
            42,
            [7, 3, 0],
            [0, 1, 2, 3],
            [15; 4],
            [0; 4],
            0,
        );
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
    fn packed_quad_preserves_per_corner_ao_and_light() {
        let ao = [0, 1, 2, 3];
        let light = [1, 5, 10, 15];
        let block_light = [15, 11, 7, 3];
        let quad = PackedQuad::terrain([0; 3], 3, 2, 2, 0, [7; 3], ao, light, block_light, 0);

        let unpacked_ao = std::array::from_fn(|corner| {
            ((quad.size_material_ao >> (20 + corner * 2)) & 0x3) as u8
        });
        let unpacked_light =
            std::array::from_fn(|corner| ((quad.light >> (corner * 4)) & 0xf) as u8);
        let unpacked_block_light =
            std::array::from_fn(|corner| ((quad.light >> (16 + corner * 4)) & 0xf) as u8);

        assert_eq!(unpacked_ao, ao);
        assert_eq!(unpacked_light, light);
        assert_eq!(unpacked_block_light, block_light);
    }

    #[test]
    fn terrain_quads_are_grouped_by_direction_with_cutout_last() {
        let quad = |face: u8, material: u8| {
            PackedQuad::terrain(
                [0, 0, 0],
                face,
                2,
                2,
                material,
                [7; 3],
                [3; 4],
                [15; 4],
                [0; 4],
                0,
            )
        };
        let input = [
            quad(3, 0),
            quad(8, 8),
            quad(0, 3),
            quad(3, 2),
            quad(5, 15),
            quad(1, 3),
        ];
        let (grouped, counts) = group_terrain_quads(&input);
        assert_eq!(counts, [1, 1, 0, 2, 0, 0, 2]);
        let groups: Vec<usize> = grouped.iter().map(PackedQuad::terrain_group).collect();
        assert_eq!(groups, [0, 1, 3, 3, CUTOUT_GROUP, CUTOUT_GROUP]);
        // Stable: original order kept within a group.
        assert_eq!(grouped[2].material(), 0);
        assert_eq!(grouped[3].material(), 2);
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
