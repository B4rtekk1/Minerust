//! GPU face extraction for ordinary cube-shaped terrain blocks.
//!
//! A job uploads one padded 18³ voxel cache, reserves its output range in the
//! existing indirect manager, and emits one `PackedQuad` per exposed face.
//! The generated descriptors remain resident until the subchunk becomes dirty
//! again; this is intentionally not a per-frame meshing pass.

use bytemuck::{Pod, Zeroable};

use crate::core::block::BlockType;
use crate::render::indirect::{GpuQuadAllocation, IndirectManager};
use crate::world::terrain::SubchunkMeshSnapshot;

const PADDED_SIDE: usize = 18;
const PADDED_VOXEL_COUNT: usize = PADDED_SIDE * PADDED_SIDE * PADDED_SIDE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FaceExtractUniforms {
    output_quad_offset: u32,
    quad_capacity: u32,
    subchunk_slot: u32,
    _padding: u32,
}

/// Owns the compute pipeline and scratch buffers for persistent face meshes.
pub struct GpuFaceMesher {
    voxel_buffer: wgpu::Buffer,
    face_counter: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    extract_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
}

impl GpuFaceMesher {
    pub fn new(device: &wgpu::Device) -> Self {
        let voxel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Face Mesher Voxel Cache"),
            size: (PADDED_VOXEL_COUNT * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let face_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Face Mesher Face Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Face Mesher Uniforms"),
            size: size_of::<FaceExtractUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Face Mesher Bind Group Layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU Face Extraction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/face_extract.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GPU Face Mesher Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Face Extraction Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("extract_faces"),
            compilation_options: Default::default(),
            cache: None,
        });
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GPU Face Extraction Finalize Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("finalize_faces"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            voxel_buffer,
            face_counter,
            uniforms,
            bind_group_layout,
            extract_pipeline,
            finalize_pipeline,
        }
    }

    /// Returns true only for subchunks represented entirely by full terrain
    /// cubes. Water and the two custom meshes retain their CPU implementation.
    pub fn supports(snapshot: &SubchunkMeshSnapshot) -> bool {
        snapshot.block_cache.iter().all(|block| {
            !matches!(
                block,
                BlockType::Water | BlockType::DeadBush | BlockType::WoodStairs
            )
        })
    }

    /// Counts the output capacity using the same visibility rules as the GPU.
    /// This tiny CPU prepass only allocates memory; it does not create mesh
    /// descriptors or vertices.
    pub fn visible_face_count(snapshot: &SubchunkMeshSnapshot) -> u32 {
        let mut count = 0;
        for x in 1..17 {
            for y in 1..17 {
                for z in 1..17 {
                    let block = snapshot.block_cache[voxel_index(x, y, z)];
                    if block == BlockType::Air {
                        continue;
                    }
                    for (dx, dy, dz) in [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ] {
                        if block.should_render_face_against(
                            snapshot.block_cache[voxel_index(
                                (x as i32 + dx) as usize,
                                (y as i32 + dy) as usize,
                                (z as i32 + dz) as usize,
                            )],
                        ) {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        manager: &IndirectManager,
        snapshot: &SubchunkMeshSnapshot,
        allocation: GpuQuadAllocation,
    ) {
        let voxels = snapshot.block_cache.map(block_id);
        queue.write_buffer(&self.voxel_buffer, 0, bytemuck::cast_slice(&voxels));
        queue.write_buffer(&self.face_counter, 0, &0u32.to_ne_bytes());
        queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&FaceExtractUniforms {
                output_quad_offset: allocation.quad_offset,
                quad_capacity: allocation.quad_capacity,
                subchunk_slot: allocation.subchunk_slot,
                _padding: 0,
            }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Face Mesher Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: manager.quad_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: manager.subchunk_meta_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.face_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniforms.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Face Extraction Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU Face Extraction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(64, 1, 1);
            pass.set_pipeline(&self.finalize_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn voxel_index(x: usize, y: usize, z: usize) -> usize {
    x * PADDED_SIDE * PADDED_SIDE + y * PADDED_SIDE + z
}

fn block_id(block: BlockType) -> u32 {
    match block {
        BlockType::Air => 0,
        BlockType::Grass => 1,
        BlockType::Dirt => 2,
        BlockType::Stone => 3,
        BlockType::Sand => 4,
        BlockType::Water => 5,
        BlockType::Wood => 6,
        BlockType::Leaves => 7,
        BlockType::Bedrock => 8,
        BlockType::Snow => 9,
        BlockType::Gravel => 10,
        BlockType::Clay => 11,
        BlockType::Ice => 12,
        BlockType::Cactus => 13,
        BlockType::DeadBush => 14,
        BlockType::WoodStairs => 15,
        BlockType::WoodLogX => 16,
        BlockType::WoodLogZ => 17,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(blocks: &[(usize, usize, usize, BlockType)]) -> SubchunkMeshSnapshot {
        let mut block_cache = [BlockType::Air; PADDED_VOXEL_COUNT];
        for &(x, y, z, block) in blocks {
            block_cache[voxel_index(x, y, z)] = block;
        }
        SubchunkMeshSnapshot {
            chunk_x: 0,
            chunk_z: 0,
            subchunk_y: 0,
            mesh_version: 0,
            has_blocks: !blocks.is_empty(),
            block_cache,
            sky_height_cache: [-1; PADDED_SIDE * PADDED_SIDE],
        }
    }

    #[test]
    fn count_matches_cube_face_visibility() {
        let one = snapshot_with(&[(8, 8, 8, BlockType::Stone)]);
        assert_eq!(GpuFaceMesher::visible_face_count(&one), 6);

        let two = snapshot_with(&[(8, 8, 8, BlockType::Stone), (9, 8, 8, BlockType::Stone)]);
        assert_eq!(GpuFaceMesher::visible_face_count(&two), 10);
    }

    #[test]
    fn custom_or_water_blocks_use_cpu_fallback() {
        assert!(!GpuFaceMesher::supports(&snapshot_with(&[(
            8,
            8,
            8,
            BlockType::Water
        )])));
        assert!(!GpuFaceMesher::supports(&snapshot_with(&[(
            8,
            8,
            8,
            BlockType::WoodStairs
        )])));
    }

    #[test]
    fn face_extraction_shader_parses_and_validates() {
        let module =
            wgpu::naga::front::wgsl::parse_str(include_str!("../shaders/face_extract.wgsl"))
                .expect("face extraction WGSL must parse");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("face extraction WGSL must validate");
    }
}
