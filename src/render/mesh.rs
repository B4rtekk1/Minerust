use crate::core::vertex::Vertex;

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
    let packed_normal = Vertex::pack_normal(normal);
    let packed_color = Vertex::pack_color(color);
    let base_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: v0,
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 1.0],
        tex_index,
    });
    vertices.push(Vertex {
        position: v1,
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 1.0],
        tex_index,
    });
    vertices.push(Vertex {
        position: v2,
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 0.0],
        tex_index,
    });
    vertices.push(Vertex {
        position: v3,
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 0.0],
        tex_index,
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

/// Add a quad with tiled UVs for greedy meshing
/// width and height specify how many blocks the quad spans for UV tiling
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
    let packed_normal = Vertex::pack_normal(normal);
    let packed_color = Vertex::pack_color(color);
    let base_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: v0,
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, height],
        tex_index,
    });
    vertices.push(Vertex {
        position: v1,
        normal: packed_normal,
        color: packed_color,
        uv: [width, height],
        tex_index,
    });
    vertices.push(Vertex {
        position: v2,
        normal: packed_normal,
        color: packed_color,
        uv: [width, 0.0],
        tex_index,
    });
    vertices.push(Vertex {
        position: v3,
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 0.0],
        tex_index,
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

pub fn build_crosshair() -> (Vec<Vertex>, Vec<u32>) {
    let size = 0.02;
    let thickness = 0.001;
    let packed_color = Vertex::pack_color([1.0, 1.0, 1.0]);
    let packed_normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    // Aspect ratio correction for 16:9 screens
    // The horizontal line needs to be shorter in X to appear same length as vertical
    let aspect = 16.0 / 9.0;
    let size_x = size / aspect;
    let thickness_x = thickness / aspect;

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    // Horizontal bar (corrected for aspect ratio)
    vertices.push(Vertex {
        position: [-size_x, -thickness, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 0.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [size_x, -thickness, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 0.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [size_x, thickness, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 1.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [-size_x, thickness, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 1.0],
        tex_index: 0.0,
    });
    indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    // Vertical bar (use thickness_x for correct aspect ratio)
    vertices.push(Vertex {
        position: [-thickness_x, -size, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 0.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [thickness_x, -size, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 0.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [thickness_x, size, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [1.0, 1.0],
        tex_index: 0.0,
    });
    vertices.push(Vertex {
        position: [-thickness_x, size, 0.0],
        normal: packed_normal,
        color: packed_color,
        uv: [0.0, 1.0],
        tex_index: 0.0,
    });
    indices.extend_from_slice(&[4, 5, 6, 4, 6, 7]);

    (vertices, indices)
}

/// Build a simple Minecraft-style player model (head, body, arms, legs)
/// Position (x, y, z) is at the player's feet, yaw is the horizontal rotation in radians
pub fn build_player_model(x: f32, y: f32, z: f32, yaw: f32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(2000);
    let mut indices = Vec::with_capacity(1000);

    let cos_yaw = yaw.cos();
    let sin_yaw = yaw.sin();

    // Helper to rotate a point around the Y axis at origin (x, z)
    let rotate = |dx: f32, dz: f32| -> (f32, f32) {
        (dx * cos_yaw - dz * sin_yaw, dx * sin_yaw + dz * cos_yaw)
    };

    // Helper to add a box (6 faces)
    let add_box = |vertices: &mut Vec<Vertex>,
                   indices: &mut Vec<u32>,
                   cx: f32, // center x offset
                   cy: f32, // center y offset (from feet)
                   cz: f32, // center z offset
                   hw: f32, // half width (x)
                   hh: f32, // half height (y)
                   hd: f32, // half depth (z)
                   color: [f32; 3]| {
        // 8 corners of the box before rotation
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

        // Transform corners: rotate around Y, then translate
        let transformed: Vec<[f32; 3]> = corners
            .iter()
            .map(|&(dx, dy, dz)| {
                let (rx, rz) = rotate(cx + dx, cz + dz);
                [x + rx, y + cy + dy, z + rz]
            })
            .collect();

        // Each face as 4 vertices (for proper normals)
        let faces = [
            // Front (+Z)
            ([4, 5, 6, 7], [0.0, 0.0, 1.0]),
            // Back (-Z)
            ([1, 0, 3, 2], [0.0, 0.0, -1.0]),
            // Right (+X)
            ([5, 1, 2, 6], [1.0, 0.0, 0.0]),
            // Left (-X)
            ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),
            // Top (+Y)
            ([7, 6, 2, 3], [0.0, 1.0, 0.0]),
            // Bottom (-Y)
            ([0, 1, 5, 4], [0.0, -1.0, 0.0]),
        ];

        for (face_indices, normal) in faces {
            let packed_normal = Vertex::pack_normal(normal);
            let packed_color = Vertex::pack_color(color);
            let base_idx = vertices.len() as u32;
            for &idx in &face_indices {
                vertices.push(Vertex {
                    position: transformed[idx],
                    normal: packed_normal,
                    color: packed_color,
                    uv: [0.0, 0.0],
                    tex_index: -1.0, // No texture, just use color
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

    // Colors
    let skin_color = [0.9, 0.75, 0.6]; // Light skin
    let shirt_color = [0.2, 0.5, 0.9]; // Blue shirt
    let pants_color = [0.3, 0.25, 0.2]; // Brown pants
    let shoes_color = [0.15, 0.15, 0.15]; // Dark shoes

    // Head (8x8x8 pixels in Minecraft = 0.5 blocks)
    add_box(
        &mut vertices,
        &mut indices,
        0.0,
        1.75,
        0.0, // center position
        0.25,
        0.25,
        0.25, // half dimensions
        skin_color,
    );

    // Body (8x12x4 pixels = 0.5 x 0.75 x 0.25 blocks)
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

    // Right arm
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

    // Left arm
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

    // Right leg (upper part - pants)
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

    // Left leg (upper part - pants)
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

    // Right leg (lower part - shoes)
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

    // Left leg (lower part - shoes)
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
