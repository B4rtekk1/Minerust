use crate::core::vertex::{OutlineVertex, Vertex};

/// Adds a single quad (two triangles) to the vertex and index buffers.
///
/// The quad is defined by four corner positions in counter-clockwise order.
/// UV coordinates are assigned as unit square (0.0–1.0 on both axes).
///
/// # Arguments
/// * `vertices` - Mutable reference to the vertex buffer to append to.
/// * `indices` - Mutable reference to the index buffer to append to.
/// * `v0..v3` - World-space positions of the four corners.
/// * `normal` - Surface normal vector for all four vertices.
/// * `color` - RGB color applied to all four vertices.
/// * `tex_index` - Index into the texture array sampler.
/// * `_roughness` - Reserved for PBR roughness (currently unused).
/// * `_metallic` - Reserved for PBR metallic factor (currently unused).
pub fn add_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    tex_index: f32,
    _roughness: f32,
    _metallic: f32,
) {
    let n_idx = Vertex::pack_normal(normal);
    let base_idx = vertices.len() as u32;

    vertices.push(Vertex {
        position: v0,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 1, 1, 1), // Corner 1 (0, 1)
    });
    vertices.push(Vertex {
        position: v1,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 2, 1, 1), // Corner 2 (1, 1)
    });
    vertices.push(Vertex {
        position: v2,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 3, 1, 1), // Corner 3 (1, 0)
    });
    vertices.push(Vertex {
        position: v3,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 0, 1, 1), // Corner 0 (0, 0)
    });
    indices.extend_from_slice(&[
        base_idx,
        base_idx + 1,
        base_idx + 2,
        base_idx,
        base_idx + 2,
        base_idx + 3,
    ]);
}

/// Adds a greedy-meshed quad to the vertex and index buffers.
///
/// Similar to [`add_quad`], but UV coordinates are scaled by `width` and `height`
/// to support texture tiling across merged voxel faces produced by greedy meshing.
///
/// # Arguments
/// * `vertices` - Mutable reference to the vertex buffer to append to.
/// * `indices` - Mutable reference to the index buffer to append to.
/// * `v0..v3` - World-space positions of the four corners.
/// * `normal` - Surface normal vector for all four vertices.
/// * `color` - RGB color applied to all four vertices.
/// * `tex_index` - Index into the texture array sampler.
/// * `_roughness` - Reserved for PBR roughness (currently unused).
/// * `_metallic` - Reserved for PBR metallic factor (currently unused).
/// * `width` - Number of voxels this quad spans along the horizontal axis (U scale).
/// * `height` - Number of voxels this quad spans along the vertical axis (V scale).
pub fn add_greedy_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    tex_index: f32,
    _roughness: f32,
    _metallic: f32,
    width: f32,
    height: f32,
) {
    let n_idx = Vertex::pack_normal(normal);
    let base_idx = vertices.len() as u32;

    // For greedy quads, we are still using unit UV corners in the vertex,
    // but the shader will multiply them by width/height?
    // Wait, width and height are not in my current 16-byte pack.
    // I should add them or pass them differently.

    // Actually, I can put width/height into the packed data for greedy quads!
    // I need to update Vertex::pack to include width/height if it fits.

    let w = width as u8;
    let h = height as u8;

    vertices.push(Vertex {
        position: v0,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 1, w, h),
    });
    vertices.push(Vertex {
        position: v1,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 2, w, h),
    });
    vertices.push(Vertex {
        position: v2,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 3, w, h),
    });
    vertices.push(Vertex {
        position: v3,
        packed: Vertex::pack(n_idx, color, tex_index as u8, 0, w, h),
    });
    indices.extend_from_slice(&[
        base_idx,
        base_idx + 1,
        base_idx + 2,
        base_idx,
        base_idx + 2,
        base_idx + 3,
    ]);
}

/// Builds the geometry for a screen-space crosshair overlay.
///
/// Produces two orthogonal rectangles (a horizontal bar and a vertical bar)
/// centered at the origin in normalized device coordinates. The aspect ratio
/// correction uses a hardcoded 16:9 ratio so the crosshair appears square
/// on widescreen displays.
///
/// Returns a tuple of `(vertices, indices)` ready to be uploaded to the GPU.
pub fn build_crosshair() -> (Vec<Vertex>, Vec<u32>) {
    let size = 0.02;
    let thickness = 0.001;
    let n_idx = Vertex::pack_normal([0.0, 0.0, 1.0]);

    // Compensate for widescreen aspect ratio so the crosshair isn't stretched.
    let aspect = 16.0 / 9.0;
    let size_x = size / aspect;
    let thickness_x = thickness / aspect;

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    // Horizontal bar.
    vertices.push(Vertex {
        position: [-size_x, -thickness, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 0),
    });
    vertices.push(Vertex {
        position: [size_x, -thickness, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 3),
    });
    vertices.push(Vertex {
        position: [size_x, thickness, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 2),
    });
    vertices.push(Vertex {
        position: [-size_x, thickness, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 1),
    });
    indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);

    // Vertical bar.
    vertices.push(Vertex {
        position: [-thickness_x, -size, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 0),
    });
    vertices.push(Vertex {
        position: [thickness_x, -size, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 3),
    });
    vertices.push(Vertex {
        position: [thickness_x, size, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 2),
    });
    vertices.push(Vertex {
        position: [-thickness_x, size, 0.0],
        packed: Vertex::pack_ui(n_idx, [1.0, 1.0, 1.0, 1.0], 0, 1),
    });
    indices.extend_from_slice(&[4, 5, 6, 4, 6, 7]);

    (vertices, indices)
}

/// Builds a screen-space thick outline for a single block at `(x, y, z)`.
///
/// Only the exposed (visible) faces are outlined — faces that have no
/// solid neighbor in the given direction. Each edge is emitted as a small
/// quad so the shader can expand it into a thick line in screen space.
///
/// # Arguments
/// * `x`, `y`, `z` - Block grid position.
/// * `visible_faces` - A bitmask or six booleans indicating which of the
///   six faces (+X, -X, +Y, -Y, +Z, -Z) have no solid neighbor.
pub fn build_block_outline(
    x: i32,
    y: i32,
    z: i32,
    visible_faces: [bool; 6], // [+X, -X, +Y, -Y, +Z, -Z]
) -> (Vec<OutlineVertex>, Vec<u32>) {
    let pad = 0.005; // smaller pad for face outlines
    let min_x = x as f32 - pad;
    let min_y = y as f32 - pad;
    let min_z = z as f32 - pad;
    let max_x = x as f32 + 1.0 + pad;
    let max_y = y as f32 + 1.0 + pad;
    let max_z = z as f32 + 1.0 + pad;

    let packed_color = Vertex::pack_color_rgba([1.0, 0.9, 0.2, 0.95]);
    let half_width_px = 1.5;

    // Each face: 4 corners + 4 edges forming a square outline.
    // Order: [+X, -X, +Y, -Y, +Z, -Z]
    let face_corners: [[[f32; 3]; 4]; 6] = [
        // +X face
        [
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
            [max_x, min_y, max_z],
        ],
        // -X face
        [
            [min_x, min_y, max_z],
            [min_x, max_y, max_z],
            [min_x, max_y, min_z],
            [min_x, min_y, min_z],
        ],
        // +Y face
        [
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, max_y, max_z],
            [max_x, max_y, min_z],
        ],
        // -Y face
        [
            [min_x, min_y, max_z],
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
        ],
        // +Z face
        [
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        // -Z face
        [
            [max_x, min_y, min_z],
            [min_x, min_y, min_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
        ],
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let mut push_segment = |a: [f32; 3], b: [f32; 3]| {
        let base = vertices.len() as u32;
        vertices.push(OutlineVertex {
            position: a,
            other: [b[0], b[1], b[2], -1.0],
            color: packed_color,
            uv: [half_width_px, 0.0],
            tex_index: 0.0,
        });
        vertices.push(OutlineVertex {
            position: a,
            other: [b[0], b[1], b[2], 1.0],
            color: packed_color,
            uv: [half_width_px, 0.0],
            tex_index: 0.0,
        });
        vertices.push(OutlineVertex {
            position: b,
            other: [a[0], a[1], a[2], -1.0],
            color: packed_color,
            uv: [half_width_px, 0.0],
            tex_index: 0.0,
        });
        vertices.push(OutlineVertex {
            position: b,
            other: [a[0], a[1], a[2], 1.0],
            color: packed_color,
            uv: [half_width_px, 0.0],
            tex_index: 0.0,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    for (face_idx, &visible) in visible_faces.iter().enumerate() {
        if !visible {
            continue;
        }
        let corners = &face_corners[face_idx];
        push_segment(corners[0], corners[1]);
        push_segment(corners[1], corners[2]);
        push_segment(corners[2], corners[3]);
        push_segment(corners[3], corners[0]);
    }

    (vertices, indices)
}
/// Builds a simple block-based player model at the given world position and yaw.
///
/// The model consists of eight axis-aligned boxes (head, torso, two arms, two
/// upper legs, and two lower legs/shoes) that are rotated around the Y-axis by
/// `yaw` before being placed in world space.
///
/// All geometry uses `tex_index = -1.0` to signal that no texture should be
/// sampled; shading relies purely on vertex colors.
///
/// # Arguments
/// * `x`, `y`, `z` - World-space origin at the player's feet.
/// * `yaw` - Rotation around the Y-axis in radians (0 = facing +Z).
///
/// # Returns
/// A tuple of `(vertices, indices)` ready to be uploaded to the GPU.
pub fn build_player_model(x: f32, y: f32, z: f32, yaw: f32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(2000);
    let mut indices = Vec::with_capacity(1000);

    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();

    // Rotates a 2-D offset `(dx, dz)` around the Y-axis by the outer `yaw`.
    let rotate = |dx: f32, dz: f32| -> (f32, f32) {
        (dx * cos_yaw - dz * sin_yaw, dx * sin_yaw + dz * cos_yaw)
    };

    // Appends a yaw-rotated, axis-aligned box to `vertices` and `indices`.
    //
    // The box is defined by its center offset from the player origin
    // `(cx, cy, cz)` and half-extents `(hw, hh, hd)`. All six faces are
    // emitted with correct outward normals and a flat `color`.
    let add_box = |vertices: &mut Vec<Vertex>,
                   indices: &mut Vec<u32>,
                   cx: f32,
                   cy: f32,
                   cz: f32,
                   hw: f32,
                   hh: f32,
                   hd: f32,
                   color: [f32; 3]| {
        // Eight corners of the un-rotated box.
        let corners = [
            (-hw, -hh, -hd),
            (hw, -hh, -hd),
            (hw, hh, -hd),
            (-hw, hh, -hd),
            (-hw, -hh, hd),
            (hw, -hh, hd),
            (hw, hh, hd),
            (-hw, hh, hd),
        ];

        // Apply yaw rotation and translate to world space.
        let transformed: Vec<[f32; 3]> = corners
            .iter()
            .map(|&(dx, dy, dz)| {
                let (rx, rz) = rotate(cx + dx, cz + dz);
                [x + rx, y + cy + dy, z + rz]
            })
            .collect();

        // Each face is a list of corner indices and its outward normal.
        let faces = [
            ([4, 5, 6, 7], [0.0_f32, 0.0, 1.0]), // Front  (+Z)
            ([1, 0, 3, 2], [0.0, 0.0, -1.0]),    // Back   (-Z)
            ([5, 1, 2, 6], [1.0, 0.0, 0.0]),     // Right  (+X)
            ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),    // Left   (-X)
            ([7, 6, 2, 3], [0.0, 1.0, 0.0]),     // Top    (+Y)
            ([0, 1, 5, 4], [0.0, -1.0, 0.0]),    // Bottom (-Y)
        ];

        for (face_indices, normal) in faces {
            let n_idx = Vertex::pack_normal(normal);
            let base_idx = vertices.len() as u32;
            for (i, &idx) in face_indices.iter().enumerate() {
                vertices.push(Vertex {
                    position: transformed[idx],
                    packed: Vertex::pack(n_idx, color, 255, i as u8, 1, 1),
                });
            }
            indices.extend_from_slice(&[
                base_idx,
                base_idx + 1,
                base_idx + 2,
                base_idx,
                base_idx + 2,
                base_idx + 3,
            ]);
        }
    };

    let skin_color = [0.9, 0.75, 0.6]; // Light skin tone.
    let shirt_color = [0.2, 0.5, 0.9]; // Blue shirt / sleeves.
    let pants_color = [0.3, 0.25, 0.2]; // Brown trousers.
    let shoes_color = [0.15, 0.15, 0.15]; // Dark shoes.

    // Head – centered 1.75 units above the feet.
    add_box(
        &mut vertices,
        &mut indices,
        0.0,
        1.75,
        0.0,
        0.25,
        0.25,
        0.25,
        skin_color,
    );

    // Torso.
    add_box(
        &mut vertices,
        &mut indices,
        0.0,
        1.125,
        0.0,
        0.25,
        0.375,
        0.125,
        shirt_color,
    );

    // Right arm.
    add_box(
        &mut vertices,
        &mut indices,
        -0.375,
        1.125,
        0.0,
        0.125,
        0.375,
        0.125,
        shirt_color,
    );

    // Left arm.
    add_box(
        &mut vertices,
        &mut indices,
        0.375,
        1.125,
        0.0,
        0.125,
        0.375,
        0.125,
        shirt_color,
    );

    // Right upper leg (trousers).
    add_box(
        &mut vertices,
        &mut indices,
        -0.125,
        0.5,
        0.0,
        0.125,
        0.25,
        0.125,
        pants_color,
    );

    // Left upper leg (trousers).
    add_box(
        &mut vertices,
        &mut indices,
        0.125,
        0.5,
        0.0,
        0.125,
        0.25,
        0.125,
        pants_color,
    );

    // Right lower leg (shoe).
    add_box(
        &mut vertices,
        &mut indices,
        -0.125,
        0.125,
        0.0,
        0.125,
        0.125,
        0.125,
        shoes_color,
    );

    // Left lower leg (shoe).
    add_box(
        &mut vertices,
        &mut indices,
        0.125,
        0.125,
        0.0,
        0.125,
        0.125,
        0.125,
        shoes_color,
    );

    (vertices, indices)
}
