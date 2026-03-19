use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::core::vertex::Vertex;
use crate::render::frustum::AABB;

use ::std::collections::BTreeMap;

const MAX_SUBCHUNKS: usize = 65536;
const MAX_VERTICES: usize = 20_000_000;
const MAX_INDICES: usize = 60_000_000;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SubchunkGpuMeta {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub draw_data: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CullUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub frustum_planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub subchunk_count: u32,
    pub hiz_size: [f32; 2],
    pub screen_size: [f32; 2],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubchunkKey {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub subchunk_y: i32,
}

#[derive(Copy, Clone, Debug)]
struct SubchunkAlloc {
    vertex_offset: u32,
    vertex_count: u32,
    index_offset: u32,
    index_count: u32,
    slot_index: usize,
}

#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    offset: u32,
    count: u32,
}

pub struct IndirectManager {
    unified_vertex_buffer: wgpu::Buffer,
    unified_index_buffer: wgpu::Buffer,

    #[allow(dead_code)]
    draw_commands_buffer: wgpu::Buffer,
    visible_draw_commands_buffer: wgpu::Buffer,

    subchunk_meta_buffer: wgpu::Buffer,

    visible_count_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    visible_count_staging: wgpu::Buffer,

    allocations: FxHashMap<SubchunkKey, SubchunkAlloc>,
    next_vertex_offset: u32,
    next_index_offset: u32,
    active_subchunk_count: u32,
    max_slot_bound: u32,
    free_slots: Vec<usize>,

    free_index_blocks: BTreeMap<u32, Vec<FreeBlock>>,

    cull_pipeline: wgpu::ComputePipeline,
    cull_bind_group_layout: wgpu::BindGroupLayout,
    cull_bind_group: Option<wgpu::BindGroup>,
    cull_uniforms_buffer: wgpu::Buffer,

    hiz_sampler: wgpu::Sampler,

    shadow_visible_commands: Vec<wgpu::Buffer>,
    shadow_visible_counts: Vec<wgpu::Buffer>,
    shadow_bind_groups: Vec<wgpu::BindGroup>,
    shadow_uniform_buffers: Vec<wgpu::Buffer>,

    free_vertex_blocks: BTreeMap<u32, Vec<FreeBlock>>,
    coalesce_counter: usize,
}

impl IndirectManager {
    pub fn new(device: &wgpu::Device) -> Self {
        let unified_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Unified Vertex Buffer"),
            size: (MAX_VERTICES * size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let unified_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Unified Index Buffer"),
            size: (MAX_INDICES * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<DrawIndexedIndirect>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<DrawIndexedIndirect>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let subchunk_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Subchunk Metadata Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<SubchunkGpuMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let visible_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Count Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cull_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Uniforms Buffer"),
            size: size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cull.wgsl").into()),
        });

        let cull_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cull Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cull Pipeline Layout"),
            bind_group_layouts: &[&cull_bind_group_layout],
            immediate_size: 0,
        });

        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cull Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Hi-Z Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            unified_vertex_buffer,
            unified_index_buffer,
            draw_commands_buffer,
            visible_draw_commands_buffer,
            subchunk_meta_buffer,
            visible_count_buffer,
            visible_count_staging,
            allocations: FxHashMap::default(),
            next_vertex_offset: 0,
            next_index_offset: 0,
            active_subchunk_count: 0,
            max_slot_bound: 0,
            free_slots: {
                let mut v = Vec::with_capacity(MAX_SUBCHUNKS);
                v.extend((0..MAX_SUBCHUNKS).rev());
                v
            },
            free_vertex_blocks: BTreeMap::new(),
            free_index_blocks: BTreeMap::new(),
            cull_pipeline,
            cull_bind_group_layout,
            cull_bind_group: None,
            cull_uniforms_buffer,
            hiz_sampler,
            shadow_visible_commands: Vec::new(),
            shadow_visible_counts: Vec::new(),
            shadow_bind_groups: Vec::new(),
            shadow_uniform_buffers: Vec::new(),
            coalesce_counter: 0,
        }
    }

    pub fn update_bind_group(&mut self, device: &wgpu::Device, hiz_view: &wgpu::TextureView) {
        self.cull_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cull Bind Group"),
            layout: &self.cull_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cull_uniforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.subchunk_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.visible_draw_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
            ],
        }));
    }

    pub fn init_shadow_resources(&mut self, device: &wgpu::Device) {
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shared Shadow Dummy Hi-Z"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        for i in 0..4 {
            let visible_commands = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Shadow Visible Draw Commands Buffer {}", i)),
                size: (MAX_SUBCHUNKS * size_of::<DrawIndexedIndirect>()) as u64,
                usage: wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let visible_count = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Shadow Visible Count Buffer {}", i)),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Shadow Cull Uniforms Buffer {}", i)),
                size: size_of::<CullUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Shadow Cull Bind Group {}", i)),
                layout: &self.cull_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.subchunk_meta_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: visible_commands.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: visible_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                    },
                ],
            });

            self.shadow_visible_commands.push(visible_commands);
            self.shadow_visible_counts.push(visible_count);
            self.shadow_bind_groups.push(bind_group);
            self.shadow_uniform_buffers.push(uniform_buffer);
        }
    }

    pub fn upload_subchunk(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        vertices: &[Vertex],
        indices: &[u32],
        aabb: &AABB,
    ) -> bool {
        if vertices.is_empty() || indices.is_empty() {
            if let Some(old_alloc) = self.allocations.remove(&key) {
                if old_alloc.vertex_count > 0 {
                    Self::add_free_block(
                        &mut self.free_vertex_blocks,
                        FreeBlock {
                            offset: old_alloc.vertex_offset,
                            count: old_alloc.vertex_count,
                        },
                    );
                }
                if old_alloc.index_count > 0 {
                    Self::add_free_block(
                        &mut self.free_index_blocks,
                        FreeBlock {
                            offset: old_alloc.index_offset,
                            count: old_alloc.index_count,
                        },
                    );
                }
                self.free_slots.push(old_alloc.slot_index);
                self.active_subchunk_count = self.active_subchunk_count.saturating_sub(1);
            }
            return true;
        }

        if let Some(old_alloc) = self.allocations.remove(&key) {
            if old_alloc.vertex_count > 0 {
                Self::add_free_block(
                    &mut self.free_vertex_blocks,
                    FreeBlock {
                        offset: old_alloc.vertex_offset,
                        count: old_alloc.vertex_count,
                    },
                )
            }
            if old_alloc.index_count > 0 {
                Self::add_free_block(
                    &mut self.free_index_blocks,
                    FreeBlock {
                        offset: old_alloc.index_offset,
                        count: old_alloc.index_count,
                    },
                );
            }
            self.free_slots.push(old_alloc.slot_index);
        }

        let vertex_count = vertices.len() as u32;
        let index_count = indices.len() as u32;

        let vertex_alloc =
            Self::find_and_remove_free_block(&mut self.free_vertex_blocks, vertex_count);
        let index_alloc =
            Self::find_and_remove_free_block(&mut self.free_index_blocks, index_count);

        let (vertex_offset, reused_vertex) = match vertex_alloc {
            Some(block) => {
                if block.count > vertex_count {
                    Self::add_free_block(
                        &mut self.free_vertex_blocks,
                        FreeBlock {
                            offset: block.offset + vertex_count,
                            count: block.count - vertex_count,
                        },
                    )
                }
                (block.offset, true)
            }
            None => {
                if self.next_vertex_offset + vertex_count > MAX_VERTICES as u32 {
                    tracing::warn!(
                        "Unified vertex buffer full ({}/{} vertices used), clearing indirect draw cache...",
                        self.next_vertex_offset,
                        MAX_VERTICES
                    );
                    self.clear_gpu_data(queue);
                    return false;
                }
                (self.next_vertex_offset, false)
            }
        };

        let (index_offset, reused_index) = match index_alloc {
            Some(block) => {
                if block.count > index_count {
                    Self::add_free_block(
                        &mut self.free_index_blocks,
                        FreeBlock {
                            offset: block.offset + index_count,
                            count: block.count - index_count,
                        },
                    );
                }

                (block.offset, true)
            }
            None => {
                if self.next_index_offset + index_count > MAX_INDICES as u32 {
                    tracing::warn!(
                        "Unified index buffer full ({}/{} indices used), clearing indirect draw cache...",
                        self.next_index_offset,
                        MAX_INDICES
                    );
                    self.clear_gpu_data(queue);
                    return false;
                }
                let offset = self.next_index_offset;
                (offset, false)
            }
        };

        let slot_index = match self.free_slots.pop() {
            Some(idx) => idx,
            None => {
                tracing::warn!("Max subchunks reached!");
                return false;
            }
        };

        let alloc = SubchunkAlloc {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            slot_index,
        };

        let vertex_byte_offset = alloc.vertex_offset as u64 * size_of::<Vertex>() as u64;
        queue.write_buffer(
            &self.unified_vertex_buffer,
            vertex_byte_offset,
            bytemuck::cast_slice(vertices),
        );

        let index_byte_offset = alloc.index_offset as u64 * size_of::<u32>() as u64;
        queue.write_buffer(
            &self.unified_index_buffer,
            index_byte_offset,
            bytemuck::cast_slice(indices),
        );

        let subchunk_meta = SubchunkGpuMeta {
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, slot_index as f32],
            draw_data: [index_count, alloc.index_offset, alloc.vertex_offset, 1],
        };
        let meta_byte_offset = slot_index * size_of::<SubchunkGpuMeta>();
        queue.write_buffer(
            &self.subchunk_meta_buffer,
            meta_byte_offset as u64,
            bytemuck::bytes_of(&subchunk_meta),
        );

        if !reused_vertex {
            self.next_vertex_offset += vertex_count;
        }
        if !reused_index {
            self.next_index_offset += index_count;
        }
        self.allocations.insert(key, alloc);
        self.active_subchunk_count = self.allocations.len() as u32;
        self.max_slot_bound = self.max_slot_bound.max(slot_index as u32 + 1);
        self.maybe_coalesce();
        true
    }

    pub fn get_slot_index(&self, key: &SubchunkKey) -> Option<usize> {
        self.allocations.get(key).map(|a| a.slot_index)
    }

    pub fn remove_subchunk(&mut self, queue: &wgpu::Queue, key: SubchunkKey) {
        if let Some(alloc) = self.allocations.remove(&key) {
            let subchunk_meta = SubchunkGpuMeta {
                aabb_min: [0.0; 4],
                aabb_max: [0.0; 4],
                draw_data: [0, 0, 0, 0],
            };
            let meta_byte_offset = alloc.slot_index * size_of::<SubchunkGpuMeta>();
            queue.write_buffer(
                &self.subchunk_meta_buffer,
                meta_byte_offset as u64,
                bytemuck::bytes_of(&subchunk_meta),
            );
            self.free_slots.push(alloc.slot_index);

            if alloc.vertex_count > 0 {
                Self::add_free_block(
                    &mut self.free_vertex_blocks,
                    FreeBlock {
                        offset: alloc.vertex_offset,
                        count: alloc.vertex_count,
                    },
                )
            }
            if alloc.index_count > 0 {
                Self::add_free_block(
                    &mut self.free_index_blocks,
                    FreeBlock {
                        offset: alloc.index_offset,
                        count: alloc.index_count,
                    },
                );
            }

            self.active_subchunk_count = self.allocations.len() as u32;
            self.maybe_coalesce();
        }
    }

    fn coalesce_vertex_blocks(blocks: &mut BTreeMap<u32, Vec<FreeBlock>>) {
        let mut all_blocks: Vec<FreeBlock> =
            blocks.values().flat_map(|v| v.iter().cloned()).collect();

        if all_blocks.len() < 2 {
            return;
        }

        all_blocks.sort_by_key(|b| b.offset);

        let mut merged = Vec::with_capacity(all_blocks.len());
        let mut current = all_blocks[0];

        for block in all_blocks.into_iter().skip(1) {
            if current.offset + current.count == block.offset {
                current.count += block.count;
            } else {
                merged.push(current);
                current = block;
            }
        }
        merged.push(current);

        blocks.clear();
        for block in merged {
            Self::add_free_block(blocks, block);
        }
    }

    fn maybe_coalesce(&mut self) {
        const COALESCE_THRESHOLD: usize = 50;

        self.coalesce_counter += 1;
        if self.coalesce_counter >= COALESCE_THRESHOLD {
            Self::coalesce_vertex_blocks(&mut self.free_vertex_blocks);
            Self::coalesce_vertex_blocks(&mut self.free_index_blocks);
            self.coalesce_counter = 0;
        }
    }

    fn find_and_remove_free_block(
        blocks: &mut BTreeMap<u32, Vec<FreeBlock>>,
        count: u32,
    ) -> Option<FreeBlock> {
        let size_key = blocks.range(count..).next().map(|(k, _)| *k)?;
        let vec = blocks.get_mut(&size_key)?;
        let block = vec.pop()?;
        if vec.is_empty() {
            blocks.remove(&size_key);
        }
        Some(block)
    }

    fn add_free_block(blocks: &mut BTreeMap<u32, Vec<FreeBlock>>, block: FreeBlock) {
        blocks
            .entry(block.count)
            .or_insert_with(Vec::new)
            .push(block);
    }

    fn clear_gpu_data(&mut self, queue: &wgpu::Queue) {
        for alloc in self.allocations.values() {
            let subchunk_meta = SubchunkGpuMeta {
                aabb_min: [0.0; 4],
                aabb_max: [0.0; 4],
                draw_data: [0, 0, 0, 0],
            };
            let meta_byte_offset = alloc.slot_index * size_of::<SubchunkGpuMeta>();
            queue.write_buffer(
                &self.subchunk_meta_buffer,
                meta_byte_offset as u64,
                bytemuck::bytes_of(&subchunk_meta),
            );
        }

        self.allocations.clear();
        self.next_vertex_offset = 0;
        self.next_index_offset = 0;
        self.active_subchunk_count = 0;
        self.max_slot_bound = 0;
        self.free_vertex_blocks.clear();
        self.free_index_blocks.clear();

        self.free_slots.clear();
        self.free_slots.extend((0..MAX_SUBCHUNKS).rev());
    }

    pub fn dispatch_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view_proj: &cgmath::Matrix4<f32>,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
    ) {
        if self.active_subchunk_count == 0 {
            return;
        }

        queue.write_buffer(&self.visible_count_buffer, 0, &0u32.to_le_bytes());

        let active = self.max_slot_bound;

        let uniforms = CullUniforms {
            view_proj: (*view_proj).into(),
            frustum_planes: *frustum_planes,
            camera_pos,
            subchunk_count: active,
            hiz_size,
            screen_size,
        };
        queue.write_buffer(&self.cull_uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        if let Some(bind_group) = &self.cull_bind_group {
            let bytes_to_clear = (active as u64) * size_of::<DrawIndexedIndirect>() as u64;
            if bytes_to_clear > 0 {
                encoder.clear_buffer(&self.visible_draw_commands_buffer, 0, Some(bytes_to_clear));
            }

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Culling Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            let workgroup_count = (active + 63) / 64;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.unified_vertex_buffer
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.unified_index_buffer
    }

    pub fn draw_commands(&self) -> &wgpu::Buffer {
        &self.visible_draw_commands_buffer
    }

    pub fn shadow_draw_commands(&self, cascade_idx: usize) -> &wgpu::Buffer {
        &self.shadow_visible_commands[cascade_idx]
    }

    pub fn visible_count_buffer(&self) -> &wgpu::Buffer {
        &self.visible_count_buffer
    }

    pub fn shadow_visible_count_buffer(&self, cascade_idx: usize) -> &wgpu::Buffer {
        &self.shadow_visible_counts[cascade_idx]
    }

    pub fn active_count(&self) -> u32 {
        self.active_subchunk_count
    }

    pub fn dispatch_shadow_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        cascade_idx: usize,
        frustum_planes: &[[f32; 4]; 6],
    ) {
        if self.active_subchunk_count == 0 || cascade_idx >= self.shadow_bind_groups.len() {
            return;
        }

        queue.write_buffer(
            &self.shadow_visible_counts[cascade_idx],
            0,
            &0u32.to_le_bytes(),
        );

        let active = self.max_slot_bound;

        let uniforms = CullUniforms {
            view_proj: [[0.0; 4]; 4],
            frustum_planes: *frustum_planes,
            camera_pos: [0.0, 0.0, 0.0],
            subchunk_count: active,
            hiz_size: [0.0, 0.0],
            screen_size: [0.0, 0.0],
        };
        queue.write_buffer(
            &self.shadow_uniform_buffers[cascade_idx],
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let bytes_to_clear = (active as u64) * size_of::<DrawIndexedIndirect>() as u64;
        if bytes_to_clear > 0 {
            encoder.clear_buffer(
                &self.shadow_visible_commands[cascade_idx],
                0,
                Some(bytes_to_clear),
            );
        }

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Shadow Culling Pass {}", cascade_idx)),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.cull_pipeline);
        cpass.set_bind_group(0, &self.shadow_bind_groups[cascade_idx], &[]);

        let workgroup_count = (active + 63) / 64;
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    pub fn has_subchunk(&self, key: &SubchunkKey) -> bool {
        self.allocations.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.allocations.clear();
        self.next_vertex_offset = 0;
        self.next_index_offset = 0;
        self.active_subchunk_count = 0;
        self.max_slot_bound = 0;
        self.free_slots.clear();
        self.free_slots.extend((0..MAX_SUBCHUNKS).rev());
        self.free_vertex_blocks.clear();
        self.free_index_blocks.clear();
    }
}