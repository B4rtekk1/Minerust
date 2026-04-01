use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::core::vertex::Vertex;
use crate::render::frustum::AABB;

use crate::logger::{LogLevel, log};
use ::std::collections::BTreeMap;

/// Maximum number of subchunks that can be tracked simultaneously.
const MAX_SUBCHUNKS: usize = 65536;

/// Maximum number of vertices across all subchunks in the unified vertex buffer.
const MAX_VERTICES: usize = 20_000_000;

/// Maximum number of indices across all subchunks in the unified index buffer.
const MAX_INDICES: usize = 60_000_000;

/// GPU-side arguments for a single `draw_indexed_indirect` call.
///
/// The memory layout matches the `VkDrawIndexedIndirectCommand` / wgpu
/// `DrawIndexedIndirectArgs` spec so the buffer can be consumed directly by
/// the GPU without additional marshaling.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirect {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw (typically 1).
    pub instance_count: u32,
    /// Offset into the index buffer where this draw starts.
    pub first_index: u32,
    /// Value added to each index before fetching a vertex (base vertex offset).
    pub base_vertex: i32,
    /// Instance ID of the first instance.
    pub first_instance: u32,
}

/// Per-subchunk metadata uploaded to the GPU for use during the culling pass.
///
/// Padded to 16-byte alignment (`[f32; 4]`) to satisfy WGSL `struct` layout rules.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SubchunkGpuMeta {
    /// World-space AABB minimum corner (w component unused, set to 0).
    pub aabb_min: [f32; 4],
    /// World-space AABB maximum corner (w component stores the slot index).
    pub aabb_max: [f32; 4],
    /// Packed draw arguments: `[index_count, index_offset, vertex_offset, 1]`.
    pub draw_data: [u32; 4],
}

/// Uniform data consumed by the GPU culling compute shader.
///
/// Contains everything the shader needs to perform frustum + Hi-Z occlusion
/// culling in a single pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CullUniforms {
    /// Combined view-projection matrix (column-major).
    pub view_proj: [[f32; 4]; 4],
    /// Six frustum planes in world space (normal + distance).
    pub frustum_planes: [[f32; 4]; 6],
    /// World-space camera position (used for LOD or distance culling).
    pub camera_pos: [f32; 3],
    /// Total number of subchunk slots to evaluate this frame.
    pub subchunk_count: u32,
    /// Dimensions of the Hi-Z (hierarchical depth) texture in pixels.
    pub hiz_size: [f32; 2],
    /// Dimensions of the render target in pixels.
    pub screen_size: [f32; 2],
}

/// Uniquely identifies a subchunk by its chunk column and vertical slice index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubchunkKey {
    /// X coordinate of the parent chunk column (in chunk units).
    pub chunk_x: i32,
    /// Z coordinate of the parent chunk column (in chunk units).
    pub chunk_z: i32,
    /// Vertical index of this subchunk within its chunk column.
    pub subchunk_y: i32,
}

/// CPU-side record of where a subchunk's data lives inside the unified buffers.
#[derive(Copy, Clone, Debug)]
struct SubchunkAlloc {
    /// First vertex index in the unified vertex buffer.
    vertex_offset: u32,
    /// Number of vertices belonging to this subchunk.
    vertex_count: u32,
    /// First index in the unified index buffer.
    index_offset: u32,
    /// Number of indices belonging to this subchunk.
    index_count: u32,
    /// Slot in the `SubchunkGpuMeta` array assigned to this subchunk.
    slot_index: usize,
}

/// A contiguous run of free elements inside a unified buffer.
#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    /// Start of the free run (in elements, not bytes).
    offset: u32,
    /// Length of the free run (in elements, not bytes).
    count: u32,
}

/// Manages GPU-side geometry and indirect draw commands for all visible subchunks.
///
/// `IndirectManager` owns a pair of large, pre-allocated unified buffers (vertex +
/// index) and assigns subregions of those buffers to individual subchunks via a
/// free-list allocator.  A GPU compute pass then performs frustum and Hi-Z
/// occlusion culling each frame, writing surviving draw commands into a separate
/// indirect command buffer that is consumed by the main render pass.
///
/// Shadow cascades each get their own command + count buffers so culling can be
/// dispatched independently per cascade without CPU readbacks.
pub struct IndirectManager {
    /// Single large vertex buffer shared by all subchunks.
    unified_vertex_buffer: wgpu::Buffer,
    /// Single large index buffer shared by all subchunks.
    unified_index_buffer: wgpu::Buffer,

    /// Staging buffer for all draw commands before culling (written by CPU).
    #[allow(dead_code)]
    draw_commands_buffer: wgpu::Buffer,
    /// Output buffer for draw commands that survive the culling pass.
    visible_draw_commands_buffer: wgpu::Buffer,

    /// Per-slot AABB and draw-argument metadata consumed by the culling shader.
    subchunk_meta_buffer: wgpu::Buffer,

    /// Atomic counter incremented by the culling shader for each visible subchunk.
    visible_count_buffer: wgpu::Buffer,
    /// CPU-readable staging copy of `visible_count_buffer` (for debugging/stats).
    #[allow(dead_code)]
    visible_count_staging: wgpu::Buffer,

    /// Map from subchunk identity to its current buffer allocation.
    allocations: FxHashMap<SubchunkKey, SubchunkAlloc>,
    /// High-water mark for vertex allocations (used when no free block fits).
    next_vertex_offset: u32,
    /// High-water mark for index allocations (used when no free block fits).
    next_index_offset: u32,
    /// Number of subchunks currently allocated.
    active_subchunk_count: u32,
    /// One past the highest slot index ever assigned; bounds the culling dispatch.
    max_slot_bound: u32,
    /// Stack of recycled metadata slot indices ready for reuse.
    free_slots: Vec<usize>,

    /// Free-list for index buffer regions, keyed by block size for O(log n) lookup.
    free_index_blocks: BTreeMap<u32, Vec<FreeBlock>>,

    /// Compute pipeline that performs per-subchunk frustum + Hi-Z culling.
    cull_pipeline: wgpu::ComputePipeline,
    /// Bind group layout shared between the main and shadow culling passes.
    cull_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the main (camera) culling pass; rebuilt when the Hi-Z changes.
    cull_bind_group: Option<wgpu::BindGroup>,
    /// Uniform buffer uploaded each frame with camera matrices and frustum planes.
    cull_uniforms_buffer: wgpu::Buffer,

    /// Nearest-neighbor sampler used to read the Hi-Z mip chain.
    hiz_sampler: wgpu::Sampler,

    /// One indirect command output buffer per shadow cascade.
    shadow_visible_commands: Vec<wgpu::Buffer>,
    /// One visible-count atomic buffer per shadow cascade.
    shadow_visible_counts: Vec<wgpu::Buffer>,
    /// Pre-built bind groups for each shadow cascade culling pass.
    shadow_bind_groups: Vec<wgpu::BindGroup>,
    /// Per-cascade uniform buffers (frustum planes differ per cascade).
    shadow_uniform_buffers: Vec<wgpu::Buffer>,

    /// Free-list for vertex buffer regions, keyed by block size for O(log n) lookup.
    free_vertex_blocks: BTreeMap<u32, Vec<FreeBlock>>,
    /// Counts uploads/removals since the last free-list coalescing pass.
    coalesce_counter: usize,
}

impl IndirectManager {
    /// Creates a new `IndirectManager` and allocates all GPU-side buffers.
    ///
    /// No geometry is uploaded at construction time; call [`upload_subchunk`]
    /// to populate the buffers before rendering.
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

        // Bindings:
        //   0 – CullUniforms (uniform)
        //   1 – SubchunkGpuMeta array (read-only storage)
        //   2 – visible draw commands output (read-write storage)
        //   3 – visible count atomic (read-write storage)
        //   4 – Hi-Z texture (non-filtered float)
        //   5 – Hi-Z sampler (non-filtering)
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

        // Nearest-neighbor clamp sampler; no filtering needed for depth comparisons.
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
            // Pre-populate the free-slot stack in reverse so slot 0 is popped first.
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

    /// Rebuilds the main culling bind group after the Hi-Z texture is recreated.
    ///
    /// Must be called whenever the depth pyramid texture or its view changes
    /// (e.g. on window resize), before the next call to [`dispatch_culling`].
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

    /// Allocates per-cascade GPU buffers and bind groups for shadow culling.
    ///
    /// Creates four sets of resources (one per shadow cascade).  Each cascade
    /// gets its own indirect command buffer, visible-count buffer, uniform
    /// buffer, and bind group.  A shared 1×1 dummy Hi-Z texture is bound for
    /// shadow passes because shadow culling skips the occlusion test.
    ///
    /// Must be called once after [`new`] before [`dispatch_shadow_culling`].
    pub fn init_shadow_resources(&mut self, device: &wgpu::Device) {
        // A 1×1 placeholder texture satisfies the Hi-Z binding slot for shadow
        // passes, which do not perform occlusion culling.
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

    /// Uploads or replaces a subchunk's geometry in the unified buffers.
    ///
    /// If `vertices` or `indices` is empty the existing allocation for `key`
    /// is freed and `true` is returned immediately.
    ///
    /// If a previous allocation exists for `key` it is freed before the new
    /// geometry is written, so callers do not need to call [`remove_subchunk`]
    /// first.
    ///
    /// Returns `true` on success.  Returns `false` if either unified buffer is
    /// full and had to be cleared; the caller should re-submit all subchunks
    /// in that case.
    pub fn upload_subchunk(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        vertices: &[Vertex],
        indices: &[u32],
        aabb: &AABB,
    ) -> bool {
        // Empty geometry means the subchunk should be removed.
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

        // Release the old allocation so its regions can be reused below.
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

        // Try to reuse a free block; fall back to the high-water mark.
        let vertex_alloc =
            Self::find_and_remove_free_block(&mut self.free_vertex_blocks, vertex_count);
        let index_alloc =
            Self::find_and_remove_free_block(&mut self.free_index_blocks, index_count);

        let (vertex_offset, reused_vertex) = match vertex_alloc {
            Some(block) => {
                // Return the leftover tail of the block to the free list.
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
                    log(
                        LogLevel::Warning,
                        &format!(
                            "Unified vertex buffer full ({}/{} vertices used), clearing indirect draw cache...",
                            self.next_vertex_offset, MAX_VERTICES
                        ),
                    );
                    self.clear_gpu_data(queue);
                    return false;
                }
                (self.next_vertex_offset, false)
            }
        };

        let (index_offset, reused_index) = match index_alloc {
            Some(block) => {
                // Return the leftover tail of the block to the free list.
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
                    log(
                        LogLevel::Warning,
                        &format!(
                            "Unified index buffer full ({}/{} indices used), clearing indirect draw cache...",
                            self.next_index_offset, MAX_INDICES
                        ),
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
                log(
                    LogLevel::Warning,
                    "No free metadata slots available, clearing indirect draw cache...",
                );
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

        // Upload vertex data at the allocated offset.
        let vertex_byte_offset = alloc.vertex_offset as u64 * size_of::<Vertex>() as u64;
        queue.write_buffer(
            &self.unified_vertex_buffer,
            vertex_byte_offset,
            bytemuck::cast_slice(vertices),
        );

        // Upload index data at the allocated offset.
        let index_byte_offset = alloc.index_offset as u64 * size_of::<u32>() as u64;
        queue.write_buffer(
            &self.unified_index_buffer,
            index_byte_offset,
            bytemuck::cast_slice(indices),
        );

        // Write the culling metadata for this slot.
        let subchunk_meta = SubchunkGpuMeta {
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
            // Slot index is packed into the w component of aabb_max.
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, slot_index as f32],
            draw_data: [index_count, alloc.index_offset, alloc.vertex_offset, 1],
        };
        let meta_byte_offset = slot_index * size_of::<SubchunkGpuMeta>();
        queue.write_buffer(
            &self.subchunk_meta_buffer,
            meta_byte_offset as u64,
            bytemuck::bytes_of(&subchunk_meta),
        );

        // Advance the high-water marks only when no free block was reused.
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

    /// Returns the metadata slot index assigned to `key`, if it is allocated.
    pub fn get_slot_index(&self, key: &SubchunkKey) -> Option<usize> {
        self.allocations.get(key).map(|a| a.slot_index)
    }

    /// Frees all GPU resources belonging to `key` and zeros its metadata slot.
    ///
    /// After this call the slot is returned to the free pool and may be reused
    /// by a subsequent [`upload_subchunk`].  Does nothing if `key` is not
    /// currently allocated.
    pub fn remove_subchunk(&mut self, queue: &wgpu::Queue, key: SubchunkKey) {
        if let Some(alloc) = self.allocations.remove(&key) {
            // Zero the metadata slot so the culling shader ignores it.
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

    /// Merges adjacent free blocks in `blocks` to reduce fragmentation.
    ///
    /// Sorts all blocks by offset, walks them linearly, and merges any two
    /// blocks whose ranges are directly contiguous.  The map is then rebuilt
    /// from the merged result.
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
                // Blocks are adjacent — extend the current run.
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

    /// Runs free-list coalescing every `COALESCE_THRESHOLD` mutations.
    ///
    /// Coalescing is amortized over many uploads/removals to keep individual
    /// operations O(log n) while still preventing unbounded fragmentation.
    fn maybe_coalesce(&mut self) {
        const COALESCE_THRESHOLD: usize = 50;

        self.coalesce_counter += 1;
        if self.coalesce_counter >= COALESCE_THRESHOLD {
            Self::coalesce_vertex_blocks(&mut self.free_vertex_blocks);
            Self::coalesce_vertex_blocks(&mut self.free_index_blocks);
            self.coalesce_counter = 0;
        }
    }

    /// Finds and removes the smallest free block that can satisfy `count` elements.
    ///
    /// Uses `BTreeMap::range` to find the best-fit block in O(log n).
    /// Returns `None` if no free block is large enough.
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

    /// Inserts a free block into the size-keyed free-list map.
    fn add_free_block(blocks: &mut BTreeMap<u32, Vec<FreeBlock>>, block: FreeBlock) {
        blocks
            .entry(block.count)
            .or_insert_with(Vec::new)
            .push(block);
    }

    /// Zeros all metadata slots and resets every CPU-side allocator to empty.
    ///
    /// Called as a last resort when a unified buffer overflows.  After this
    /// returns the caller must re-upload all subchunks from scratch.
    pub fn clear_gpu_data(&mut self, queue: &wgpu::Queue) {
        // Zero every live metadata slot so stale entries don't survive.
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

    /// Uploads cull uniforms and dispatches the main camera culling compute pass.
    ///
    /// Clears `visible_count_buffer` and the visible command buffer before
    /// dispatching so that only subchunks that pass culling this frame are drawn.
    /// One workgroup of 64 threads is launched per 64 subchunk slots.
    ///
    /// Does nothing if no subchunks are currently allocated or if the bind
    /// group has not yet been created via [`update_bind_group`].
    pub fn dispatch_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view_proj: &glam::Mat4,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
    ) {
        if self.active_subchunk_count == 0 {
            return;
        }

        // Reset the visible counter before the pass writes into it.
        queue.write_buffer(&self.visible_count_buffer, 0, &0u32.to_le_bytes());

        let active = self.max_slot_bound;

        let uniforms = CullUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            frustum_planes: *frustum_planes,
            camera_pos,
            subchunk_count: active,
            hiz_size,
            screen_size,
        };
        queue.write_buffer(&self.cull_uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        if let Some(bind_group) = &self.cull_bind_group {
            // Clear only the portion of the command buffer that will be written.
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

            // Round up to a full workgroup; the shader discards out-of-range threads.
            let workgroup_count = (active + 63) / 64;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Returns a reference to the unified vertex buffer.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.unified_vertex_buffer
    }

    /// Returns a reference to the unified index buffer.
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.unified_index_buffer
    }

    /// Returns a reference to the visible (post-cull) indirect draw command buffer.
    pub fn draw_commands(&self) -> &wgpu::Buffer {
        &self.visible_draw_commands_buffer
    }

    /// Returns the shadow cascade's indirect draw command buffer.
    pub fn shadow_draw_commands(&self, cascade_idx: usize) -> &wgpu::Buffer {
        &self.shadow_visible_commands[cascade_idx]
    }

    /// Returns the main visible-count buffer (used as an indirect dispatch argument).
    pub fn visible_count_buffer(&self) -> &wgpu::Buffer {
        &self.visible_count_buffer
    }

    /// Returns the visible-count buffer for the given shadow cascade.
    pub fn shadow_visible_count_buffer(&self, cascade_idx: usize) -> &wgpu::Buffer {
        &self.shadow_visible_counts[cascade_idx]
    }

    /// Returns the number of subchunks currently allocated.
    pub fn active_count(&self) -> u32 {
        self.active_subchunk_count
    }

    /// Dispatches a frustum culling compute pass for one shadow cascade.
    ///
    /// Shadow culling uses the cascade's own frustum planes but skips Hi-Z
    /// occlusion (the Hi-Z slot is bound to a 1×1 dummy texture).
    ///
    /// Does nothing if no subchunks are allocated or `cascade_idx` is out of range.
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

        // Reset the per-cascade visible counter.
        queue.write_buffer(
            &self.shadow_visible_counts[cascade_idx],
            0,
            &0u32.to_le_bytes(),
        );

        let active = self.max_slot_bound;

        // view_proj, camera_pos, and screen/Hi-Z sizes are unused in shadow mode.
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

    /// Returns `true` if `key` currently has an active GPU allocation.
    pub fn has_subchunk(&self, key: &SubchunkKey) -> bool {
        self.allocations.contains_key(key)
    }

    /// Resets all CPU-side allocator state without touching GPU buffers.
    ///
    /// Use this when the GPU buffers will be discarded or recreated.  If the
    /// buffers are still in use, call [`clear_gpu_data`] instead to also zero
    /// the metadata slots.
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
