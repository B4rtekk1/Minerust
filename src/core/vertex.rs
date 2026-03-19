use bytemuck::{Pod, Zeroable};
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3], // 12 bytes
    pub normal: [i8; 4],    // 4 bytes
    pub color: [u8; 4],     // 4 bytes
    pub uv: [f32; 2],       // 8 bytes
    pub tex_index: f32,     // 4 bytes
}

impl Vertex {
    #[inline]
    pub fn pack_normal(n: [f32; 3]) -> [i8; 4] {
        [
            (n[0].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[1].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[2].clamp(-1.0, 1.0) * 127.0) as i8,
            0,
        ]
    }

    #[inline]
    pub fn pack_color(c: [f32; 3]) -> [u8; 4] {
        [
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    }

    #[inline]
    pub fn pack_color_rgba(c: [f32; 4]) -> [u8; 4] {
        [
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            (c[3].clamp(0.0, 1.0) * 255.0) as u8,
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
