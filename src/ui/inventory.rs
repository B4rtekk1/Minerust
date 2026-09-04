use minerust::{Inventory, Vertex, block_for_item};
use wgpu::util::DeviceExt;

/// Large, readable slots for the full-screen inventory overlay.
pub const SLOT: f32 = 64.0;
const GAP: f32 = 8.0;

fn add_quantity(
    vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, quantity: u32, x: f32, y: f32,
    screen_w: u32, screen_h: u32,
) {
    // Seven-segment digits keep stack counts in the same GPU-only UI pass.
    const SEGMENTS: [[bool; 7]; 10] = [
        [true,true,true,true,true,true,false], [false,true,true,false,false,false,false],
        [true,true,false,true,true,false,true], [true,true,true,true,false,false,true],
        [false,true,true,false,false,true,true], [true,false,true,true,false,true,true],
        [true,false,true,true,true,true,true], [true,true,true,false,false,false,false],
        [true,true,true,true,true,true,true], [true,true,true,true,false,true,true],
    ];
    let text = quantity.to_string();
    for (digit_index, byte) in text.bytes().enumerate() {
        let digit = (byte - b'0') as usize;
        for (segment, enabled) in SEGMENTS[digit].iter().enumerate() {
            if !enabled { continue; }
            let (dx, dy, w, h) = match segment {
                0 => (2.0, 0.0, 8.0, 2.0), 1 => (10.0, 2.0, 2.0, 7.0),
                2 => (10.0, 11.0, 2.0, 7.0), 3 => (2.0, 18.0, 8.0, 2.0),
                4 => (0.0, 11.0, 2.0, 7.0), 5 => (0.0, 2.0, 2.0, 7.0),
                _ => (2.0, 9.0, 8.0, 2.0),
            };
            let px = x + digit_index as f32 * 13.0 + dx;
            let py = y + dy;
            let base = vertices.len() as u32;
            let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);
            for (corner, (vx, vy)) in [(px,py),(px+w,py),(px+w,py+h),(px,py+h)].into_iter().enumerate() {
                vertices.push(Vertex { position: [vx / screen_w as f32 * 2.0 - 1.0, 1.0 - vy / screen_h as f32 * 2.0, 0.0], packed: Vertex::pack_ui(normal, [1.0, 1.0, 1.0, 1.0], 0, corner as u8) });
            }
            indices.extend_from_slice(&[base,base+1,base+2,base,base+2,base+3]);
        }
    }
}

#[derive(Clone, Copy)]
pub struct InventoryLayout { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }

impl InventoryLayout {
    pub fn new(width: u32, height: u32) -> Self {
        let w = SLOT * 9.0 + GAP * 10.0;
        let h = 440.0;
        Self { x: (width as f32 - w) * 0.5, y: (height as f32 - h) * 0.5, w, h }
    }
    pub fn slot_rect(&self, slot: usize) -> (f32, f32, f32, f32) {
        let (row, column, top) = if slot < 27 { (slot / 9, slot % 9, self.y + 92.0) }
        else { (0, slot - 27, self.y + 324.0) };
        let x = self.x + GAP + column as f32 * (SLOT + GAP);
        let y = top + row as f32 * (SLOT + GAP);
        (x, y, SLOT, SLOT)
    }
    pub fn slot_at(&self, x: f32, y: f32) -> Option<usize> {
        (0..36).find(|&slot| { let (sx, sy, w, h) = self.slot_rect(slot); x >= sx && x < sx+w && y >= sy && y < sy+h })
    }
}

pub fn build(device: &wgpu::Device, inventory: &Inventory, selected_hotbar: usize, cursor_position: Option<(f32, f32)>, width: u32, height: u32) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let layout = InventoryLayout::new(width, height);
    let mut vertices = Vec::with_capacity(36 * 12 + 8);
    let mut indices = Vec::with_capacity(36 * 18 + 12);
    let add = |vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, x: f32, y: f32, w: f32, h: f32, color: [f32;3]| {
        let base = vertices.len() as u32;
        let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);
        for (i, (px, py)) in [(x,y),(x+w,y),(x+w,y+h),(x,y+h)].into_iter().enumerate() {
            // Pixel coordinates converted to NDC; invert Y for screen origin.
            vertices.push(Vertex { position: [px / width as f32 * 2.0 - 1.0, 1.0 - py / height as f32 * 2.0, 0.0], packed: Vertex::pack_ui(normal, [color[0], color[1], color[2], 0.92], 0, i as u8) });
        }
        indices.extend_from_slice(&[base,base+1,base+2,base,base+2,base+3]);
    };
    add(&mut vertices, &mut indices, layout.x, layout.y, layout.w, layout.h, [0.06,0.07,0.09]);
    for slot in 0..36 {
        let (x,y,w,h) = layout.slot_rect(slot);
        let selected = slot == 27 + selected_hotbar;
        add(&mut vertices, &mut indices, x, y, w, h, if selected {[0.95,0.78,0.24]} else {[0.36,0.40,0.46]});
        add(&mut vertices, &mut indices, x+3.0, y+3.0, w-6.0, h-6.0, [0.13,0.15,0.18]);
        if let Some(stack) = &inventory.slots[slot] {
            let color = block_for_item(stack.item.id).map(|block| block.color()).unwrap_or([0.8,0.8,0.8]);
            add(&mut vertices, &mut indices, x+14.0, y+14.0, w-28.0, h-28.0, color);
            add_quantity(&mut vertices, &mut indices, stack.quantity, x + w - 30.0, y + h - 24.0, width, height);
        }
    }
    if let (Some(cursor), Some((x, y))) = (&inventory.cursor_item, cursor_position) {
        let color = block_for_item(cursor.item.id).map(|block| block.color()).unwrap_or([0.8, 0.8, 0.8]);
        add(&mut vertices, &mut indices, x + 10.0, y + 10.0, 38.0, 38.0, color);
    }
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Inventory UI VB"), contents: bytemuck::cast_slice(&vertices), usage: wgpu::BufferUsages::VERTEX });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Inventory UI IB"), contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX });
    (vb, ib, indices.len() as u32)
}
