use minerust::{BlockType, Vertex};
use wgpu::util::DeviceExt;

/// The fixed set of block types assigned to hotbar slots 0–8, left to right.
///
/// The index of a block in this array corresponds directly to its hotbar slot
/// number. Use [`block_type_to_index`] to perform the reverse lookup.
pub const HOTBAR_SLOTS: [BlockType; 9] = [
    BlockType::Grass,
    BlockType::Dirt,
    BlockType::Stone,
    BlockType::Sand,
    BlockType::Wood,
    BlockType::Leaves,
    BlockType::Gravel,
    BlockType::Clay,
    BlockType::Ice,
];

/// Returns the hotbar slot index of `block` as an `f32`, or `None` if the
/// block is not present in [`HOTBAR_SLOTS`].
///
/// The result is `f32` so it can be passed directly to shader uniforms or
/// stored in vertex data without an extra cast at the call site.
pub fn block_type_to_index(block: BlockType) -> Option<f32> {
    HOTBAR_SLOTS.iter().position(|&b| b == block).map(|i| i as f32)
}

/// Builds GPU vertex and index buffers for the HUD hotbar.
///
/// Generates a row of nine block-preview slots centred horizontally at the
/// bottom of the screen. Each slot is made up of three layered quads:
///
/// 1. **Border quad** — white for the selected slot, dark grey otherwise.
/// 2. **Background quad** — slightly lighter grey for the selected slot.
/// 3. **Block-colour quad** — filled with the block's representative colour,
///    inset by a fixed padding fraction of the slot size.
///
/// All coordinates are in normalised device coordinates (NDC): X and Y both
/// range from `-1.0` (left / bottom) to `+1.0` (right / top). The `aspect`
/// ratio is applied to vertical measurements so slots appear square regardless
/// of window dimensions.
///
/// # Arguments
///
/// * `device`        - wgpu device used to allocate the GPU buffers.
/// * `selected_slot` - Index (0–8) of the currently active hotbar slot.
/// * `aspect`        - Viewport height divided by width (`h / w`). Multiplied
///                     into all Y-axis sizes to maintain square slots.
///
/// # Returns
///
/// A tuple of `(vertex_buffer, index_buffer, index_count)` ready to be bound
/// and drawn with `draw_indexed`.
pub fn build_hotbar(
    device: &wgpu::Device,
    selected_slot: usize,
    aspect: f32,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let slot_count = HOTBAR_SLOTS.len() as f32;
    let slot_size = 0.08_f32;
    let slot_h = slot_size * aspect;
    let gap = 0.004_f32;
    let total_w = slot_count * slot_size + (slot_count - 1.0) * gap;
    let start_x = -total_w * 0.5;
    let bottom_y = -0.95_f32;

    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Appends a screen-aligned quad spanning (x0, y0)–(x1, y1) with a solid
    // `color`. Vertices are wound counter-clockwise: BL, BR, TR, TL.
    let mut add_quad = |x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 4]| {
        let base = vertices.len() as u32;
        for (px, py) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
            vertices.push(Vertex {
                position: [px, py, 0.0],
                normal,
                color,
                uv: [0.0, 0.0],
                tex_index: 0.0,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    for i in 0..HOTBAR_SLOTS.len() {
        let x0 = start_x + i as f32 * (slot_size + gap);
        let x1 = x0 + slot_size;
        let y0 = bottom_y;
        let y1 = y0 + slot_h;

        // Layer 1: border — bright for the selected slot, dim otherwise.
        let border_color = if i == selected_slot {
            Vertex::pack_color([1.0, 1.0, 1.0])
        } else {
            Vertex::pack_color([0.4, 0.4, 0.4])
        };
        let border = 0.004;
        add_quad(x0, y0, x1, y1, border_color);

        // Layer 2: background — inset by `border` on all sides.
        let bg_color = if i == selected_slot {
            Vertex::pack_color([0.25, 0.25, 0.25])
        } else {
            Vertex::pack_color([0.12, 0.12, 0.12])
        };
        add_quad(
            x0 + border,
            y0 + border * aspect,
            x1 - border,
            y1 - border * aspect,
            bg_color,
        );

        // Layer 3: block colour swatch — inset by 18% of slot size on all sides.
        let block = HOTBAR_SLOTS[i];
        let [r, g, b] = block.color();
        let block_color = Vertex::pack_color([r, g, b]);
        let pad = slot_size * 0.18;
        let pad_h = pad * aspect;
        add_quad(x0 + pad, y0 + pad_h, x1 - pad, y1 - pad_h, block_color);
    }

    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Hotbar VB"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Hotbar IB"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vb, ib, indices.len() as u32)
}

/// Rebuilds the coordinate overlay GPU buffers when the camera position
/// changes, returning `None` if the integer-truncated position is unchanged.
///
/// Renders the player's current world coordinates as a white seven-segment
/// style text label (e.g. `"X:128 Y:64 Z:-32"`) in the top-right corner of
/// the screen. Characters are drawn as a series of thick line segments
/// (quads) using [`get_char_segments`] for the segment layout.
///
/// # Change detection
///
/// Coordinates are compared at integer granularity. `last_coords_position` is
/// updated in-place when a change is detected and left untouched otherwise,
/// allowing the caller to reuse the previous buffers without re-uploading.
///
/// # Arguments
///
/// * `device`               - wgpu device used to allocate new GPU buffers.
/// * `camera_pos`           - Current camera position in world space.
/// * `last_coords_position` - Mutable cache of the last rendered `(x, y, z)`
///                            as integers. Updated on every rebuild.
///
/// # Returns
///
/// `Some((vertex_buffer, index_buffer, index_count))` when the buffers were
/// rebuilt, or `None` when the position has not changed since the last call.
pub fn update_coords_ui(
    device: &wgpu::Device,
    camera_pos: cgmath::Point3<f32>,
    last_coords_position: &mut (i32, i32, i32),
) -> Option<(wgpu::Buffer, wgpu::Buffer, u32)> {
    let x = camera_pos.x;
    let y = camera_pos.y;
    let z = camera_pos.z;

    let current_pos = (x as i32, y as i32, z as i32);
    if current_pos == *last_coords_position {
        return None;
    }
    *last_coords_position = current_pos;

    let text = format!("X:{:.0} Y:{:.0} Z:{:.0}", x, y, z);

    let mut vertices = Vec::with_capacity(500);
    let mut indices = Vec::with_capacity(250);

    // Visual metrics for the stroke-based font.
    let char_width = 0.018;
    let char_height = 0.032;
    let line_thickness = 0.004;
    let char_spacing = char_width * 0.6; // advance for a space character
    let gap_spacing = char_width + 0.005; // advance for a normal character

    // Pre-compute total text width so the label can be right-aligned.
    let mut total_width = 0.0;
    for ch in text.chars() {
        if ch == ' ' {
            total_width += char_spacing;
        } else {
            total_width += gap_spacing;
        }
    }

    // Anchor the label 0.02 NDC units from the right edge, near the top.
    let start_x = 0.98 - total_width;
    let start_y = 0.95;

    let mut cursor_x = start_x;
    let cursor_y = start_y;
    let color = Vertex::pack_color([1.0, 1.0, 1.0]);
    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    // Appends a screen-space line segment as a quad with width `line_thickness`.
    // The quad is extruded perpendicular to the segment direction so it always
    // appears as a constant-width stroke regardless of angle.
    // Segments shorter than 0.001 NDC units are skipped to avoid divide-by-zero.
    let add_segment =
        |x1: f32, y1: f32, x2: f32, y2: f32, verts: &mut Vec<Vertex>, inds: &mut Vec<u32>| {
            let base_idx = verts.len() as u32;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 0.001 {
                return;
            }
            // Perpendicular offset vector, scaled to half the desired thickness.
            let nx = -dy / len * line_thickness * 0.5;
            let ny = dx / len * line_thickness * 0.5;

            verts.push(Vertex {
                position: [x1 - nx, y1 - ny, 0.0],
                normal,
                color,
                uv: [0.0, 0.0],
                tex_index: 0.0,
            });
            verts.push(Vertex {
                position: [x2 - nx, y2 - ny, 0.0],
                normal,
                color,
                uv: [1.0, 0.0],
                tex_index: 0.0,
            });
            verts.push(Vertex {
                position: [x2 + nx, y2 + ny, 0.0],
                normal,
                color,
                uv: [1.0, 1.0],
                tex_index: 0.0,
            });
            verts.push(Vertex {
                position: [x1 + nx, y1 + ny, 0.0],
                normal,
                color,
                uv: [0.0, 1.0],
                tex_index: 0.0,
            });
            inds.extend_from_slice(&[
                base_idx,
                base_idx + 1,
                base_idx + 2,
                base_idx,
                base_idx + 2,
                base_idx + 3,
            ]);
        };

    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += char_spacing;
            continue;
        }

        // Scale each abstract segment coordinate into screen space and emit.
        let segments = get_char_segments(ch);
        for (x1, y1, x2, y2) in segments {
            let px1 = cursor_x + x1 * char_width;
            let py1 = cursor_y - char_height + y1 * char_height;
            let px2 = cursor_x + x2 * char_width;
            let py2 = cursor_y - char_height + y2 * char_height;
            add_segment(px1, py1, px2, py2, &mut vertices, &mut indices);
        }

        cursor_x += gap_spacing;
    }

    if vertices.is_empty() {
        return None;
    }

    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Coords Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Coords Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Some((vb, ib, indices.len() as u32))
}

/// Returns the stroke segments that define `ch` in a simple seven-segment
/// style font.
///
/// Each segment is a tuple `(x1, y1, x2, y2)` in a normalised `[0, 1]²`
/// glyph cell where `(0, 0)` is the bottom-left corner and `(1, 1)` is the
/// top-right. The caller is responsible for scaling these coordinates into
/// screen space.
///
/// Supported characters: `0`–`9`, `X`, `Y`, `Z`, `:`, `.`, `-`.
/// Any unrecognised character returns an empty `Vec`, producing no visible
/// output (effectively a blank glyph).
fn get_char_segments(ch: char) -> Vec<(f32, f32, f32, f32)> {
    // Named aliases for the seven standard segment positions.
    let seg_top = (0.0, 1.0, 1.0, 1.0); // top horizontal
    let seg_tr = (1.0, 1.0, 1.0, 0.5);  // top-right vertical
    let seg_br = (1.0, 0.5, 1.0, 0.0);  // bottom-right vertical
    let seg_bot = (0.0, 0.0, 1.0, 0.0); // bottom horizontal
    let seg_bl = (0.0, 0.5, 0.0, 0.0);  // bottom-left vertical
    let seg_tl = (0.0, 1.0, 0.0, 0.5);  // top-left vertical
    let seg_mid = (0.0, 0.5, 1.0, 0.5); // middle horizontal

    match ch {
        '0' => vec![seg_top, seg_tr, seg_br, seg_bot, seg_bl, seg_tl],
        '1' => vec![seg_tr, seg_br],
        '2' => vec![seg_top, seg_tr, seg_mid, seg_bl, seg_bot],
        '3' => vec![seg_top, seg_tr, seg_mid, seg_br, seg_bot],
        '4' => vec![seg_tl, seg_mid, seg_tr, seg_br],
        '5' => vec![seg_top, seg_tl, seg_mid, seg_br, seg_bot],
        '6' => vec![seg_top, seg_tl, seg_mid, seg_br, seg_bot, seg_bl],
        '7' => vec![seg_top, seg_tr, seg_br],
        '8' => vec![seg_top, seg_tr, seg_br, seg_bot, seg_bl, seg_tl, seg_mid],
        '9' => vec![seg_top, seg_tr, seg_br, seg_bot, seg_tl, seg_mid],
        'X' => vec![(0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0)],
        'Y' => vec![
            (0.0, 1.0, 0.5, 0.5),
            (1.0, 1.0, 0.5, 0.5),
            (0.5, 0.5, 0.5, 0.0),
        ],
        'Z' => vec![seg_top, (1.0, 1.0, 0.0, 0.0), seg_bot],
        ':' => vec![(0.4, 0.7, 0.6, 0.7), (0.4, 0.3, 0.6, 0.3)],
        '.' => vec![(0.4, 0.1, 0.6, 0.1)],
        '-' => vec![seg_mid],
        _ => vec![],
    }
}