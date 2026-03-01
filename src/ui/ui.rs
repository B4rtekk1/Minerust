//! In-game HUD and specialized vector-based text rendering.
//!
//! This module provides the logic for drawing overlay elements like coordinates
//! using a custom segment-based font rendering system. This allows for high-performance
//! text rendering without requiring large font atlas textures or external glyph crates.

// TODO: Add more UI elements

use render3d::Vertex;
use wgpu::util::DeviceExt;

/// Updates the coordinate display UI.
///
/// This function generates vertex and index buffers for rendering the current camera coordinates
/// on the screen. It uses a custom segment-based font rendering system to avoid external dependencies.
///
/// # Arguments
/// * `device` - The WGPU device used to create buffers.
/// * `camera_pos` - Current position of the camera to display.
/// * `last_coords_position` - Cached last position to avoid redundant buffer updates.
///
/// # Returns
/// Returns `Some((vertex_buffer, index_buffer, index_count))` if the coordinates have changed
/// and the buffers were successfully recreated, or `None` if no update is needed or possible.
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

    let char_width = 0.018;
    let char_height = 0.032;
    let line_thickness = 0.004;
    let char_spacing = char_width * 0.6;
    let gap_spacing = char_width + 0.005;

    let mut total_width = 0.0;
    for ch in text.chars() {
        if ch == ' ' {
            total_width += char_spacing;
        } else {
            total_width += gap_spacing;
        }
    }

    let start_x = 0.98 - total_width;
    let start_y = 0.95;

    let mut cursor_x = start_x;
    let cursor_y = start_y;
    let color = Vertex::pack_color([1.0, 1.0, 1.0]);
    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    let add_segment =
        |x1: f32, y1: f32, x2: f32, y2: f32, verts: &mut Vec<Vertex>, inds: &mut Vec<u32>| {
            let base_idx = verts.len() as u32;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len < 0.001 {
                return;
            }
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

/// Returns the geometric segments required to draw a specific character.
///
/// Each segment is represented as a tuple of `(x1, y1, x2, y2)` in a specialized coordinate system
/// where (0,0) is the bottom-left of the character and (1,1) is the top-right.
fn get_char_segments(ch: char) -> Vec<(f32, f32, f32, f32)> {
    let seg_top = (0.0, 1.0, 1.0, 1.0);
    let seg_tr = (1.0, 1.0, 1.0, 0.5);
    let seg_br = (1.0, 0.5, 1.0, 0.0);
    let seg_bot = (0.0, 0.0, 1.0, 0.0);
    let seg_bl = (0.0, 0.5, 0.0, 0.0);
    let seg_tl = (0.0, 1.0, 0.0, 0.5);
    let seg_mid = (0.0, 0.5, 1.0, 0.5);

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
