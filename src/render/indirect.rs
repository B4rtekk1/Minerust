use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::constants::{CHUNK_UNLOAD_DISTANCE, NUM_SUBCHUNKS};
use crate::render::frustum::AABB;
use crate::render::quad::PackedQuad;

use crate::logger::{LogLevel, log};
use std::collections::BTreeMap;

/// Maximum number of subchunks that can be tracked simultaneously.
///
/// This follows the chunk unload radius instead of reserving for an arbitrary
/// world-sized ceiling. With the current render settings this is
/// `(CHUNK_UNLOAD_DISTANCE * 2 + 1)^2` chunk columns times 16 vertical
/// subchunks per manager.
const MAX_CHUNK_COLUMNS: usize =
    (CHUNK_UNLOAD_DISTANCE as usize * 2 + 1) * (CHUNK_UNLOAD_DISTANCE as usize * 2 + 1);
const MAX_SUBCHUNKS: usize = MAX_CHUNK_COLUMNS * NUM_SUBCHUNKS as usize;

/// Initial arena sizes. Geometry grows on demand, so render distance does not
/// translate directly into a permanent VRAM reservation.
const TERRAIN_INITIAL_ARENA_BYTES: usize = 32 * 1024 * 1024;
const WATER_INITIAL_ARENA_BYTES: usize = 4 * 1024 * 1024;

/// Geometry allocation policy for one [`IndirectManager`].
#[derive(Clone, Copy, Debug)]
pub struct IndirectBufferBudget {
    label: &'static str,
    initial_arena_bytes: usize,
}

impl IndirectBufferBudget {
    /// Budget used for opaque terrain geometry.
    pub const TERRAIN: Self = Self {
        label: "Terrain",
        initial_arena_bytes: TERRAIN_INITIAL_ARENA_BYTES,
    };

    /// Budget used for water geometry.
    pub const WATER: Self = Self {
        label: "Water",
        initial_arena_bytes: WATER_INITIAL_ARENA_BYTES,
    };

    fn initial_quads(&self) -> usize {
        self.initial_arena_bytes / size_of::<PackedQuad>()
    }

    fn label(&self) -> &'static str {
        self.label
    }
}

/// GPU-side arguments for a single non-indexed indirect draw.
///
/// The memory layout matches wgpu `DrawIndirectArgs` so the buffer can be consumed directly by
/// the GPU without additional marshaling.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndirect {
    /// Number of procedurally pulled vertices to draw.
    pub vertex_count: u32,
    /// Number of instances to draw (typically 1).
    pub instance_count: u32,
    /// Offset into the implicit six-vertices-per-quad stream.
    pub first_vertex: u32,
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
    /// World-space AABB maximum corner (w component unused).
    pub aabb_max: [f32; 4],
    /// Terrain draw arguments: `[vertex_count, first_vertex, unused, active]`.
    pub terrain_draw_data: [u32; 4],
    /// Water draw arguments: `[vertex_count, first_vertex, unused, active]`.
    pub water_draw_data: [u32; 4],
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
    quad_offset: u32,
    /// Number of quad descriptors belonging to this subchunk.
    quad_count: u32,
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
/// `IndirectManager` owns a growable unified quad arena and assigns subregions of
/// that buffer to individual subchunks via a
/// free-list allocator.  A GPU compute pass then performs frustum and Hi-Z
/// occlusion culling each frame, writing surviving draw commands into a separate
/// indirect command buffer that is consumed by the main render pass.
///
pub struct IndirectManager {
    /// Single large vertex buffer shared by all subchunks.
    unified_quad_buffer: wgpu::Buffer,

    /// Staging buffer for all draw commands before culling (written by CPU).
    #[allow(dead_code)]
    draw_commands_buffer: wgpu::Buffer,
    /// Output buffer for draw commands that survive the culling pass.
    visible_draw_commands_buffer: wgpu::Buffer,

    /// Per-slot AABB and draw-argument metadata consumed by the culling shader.
    subchunk_meta_buffer: Option<wgpu::Buffer>,

    /// Atomic counter incremented by the culling shader for each visible subchunk.
    visible_count_buffer: wgpu::Buffer,
    /// CPU-readable staging copy of `visible_count_buffer` (for debugging/stats).
    #[allow(dead_code)]
    visible_count_staging: wgpu::Buffer,

    /// Map from subchunk identity to its current buffer allocation.
    allocations: FxHashMap<SubchunkKey, SubchunkAlloc>,
    /// High-water mark for vertex allocations (used when no free block fits).
    next_quad_offset: u32,
    /// Number of subchunks currently allocated.
    active_subchunk_count: u32,
    /// Shared culling slots. Only the terrain manager owns these; its metadata
    /// contains draw ranges for both terrain and water.
    cull_allocations: FxHashMap<SubchunkKey, usize>,
    /// One past the highest culling slot, bounding the single dispatch.
    max_slot_bound: u32,
    free_slots: Vec<usize>,

    /// Free-list for index buffer regions, keyed by block size for O(log n) lookup.

    /// Compute pipeline that performs per-subchunk frustum + Hi-Z culling.
    cull_pipeline: wgpu::ComputePipeline,
    /// Bind group layout used by the main camera culling pass.
    cull_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the main (camera) culling pass; rebuilt when the Hi-Z changes.
    cull_bind_group: Option<wgpu::BindGroup>,
    /// Uniform buffer uploaded each frame with camera matrices and frustum planes.
    cull_uniforms_buffer: wgpu::Buffer,

    /// Nearest-neighbor sampler used to read the Hi-Z mip chain.
    hiz_sampler: wgpu::Sampler,

    /// Free-list for vertex buffer regions, keyed by block size for O(log n) lookup.
    free_quad_blocks: BTreeMap<u32, Vec<FreeBlock>>,
    /// Counts uploads/removals since the last free-list coalescing pass.
    coalesce_counter: usize,
    /// Current capacity of the growable quad arena.
    max_quads: usize,
    /// Set after the arena is replaced; the renderer must rebuild its vertex-pulling bind group.
    quad_buffer_rebind_pending: bool,
    /// Short label used in logs.
    label: &'static str,
}

impl IndirectManager {
    /// Creates a new `IndirectManager` and allocates all GPU-side buffers.
    ///
    /// No geometry is uploaded at construction time; call [`upload_subchunk`]
    /// to populate the buffers before rendering.
    pub fn new(device: &wgpu::Device) -> Self {
        Self::with_budget(device, IndirectBufferBudget::TERRAIN)
    }

    /// Creates a new `IndirectManager` with a custom geometry budget.
    pub fn with_budget(device: &wgpu::Device, budget: IndirectBufferBudget) -> Self {
        let max_quads = budget.initial_quads();
        let label = budget.label();

        let unified_quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Unified Quad Buffer")),
            size: (max_quads * size_of::<PackedQuad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<DrawIndirect>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<DrawIndirect>()) as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Only the terrain manager owns the shared record buffer. Water uses
        // that same buffer for vertex pulling and no longer allocates a copy.
        let subchunk_meta_buffer = (label == IndirectBufferBudget::TERRAIN.label()).then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shared Subchunk Metadata Buffer"),
                size: (MAX_SUBCHUNKS * size_of::<SubchunkGpuMeta>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
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
        //   2/3 – terrain draw commands and count; 4/5 – water equivalents
        //   6 – Hi-Z texture (non-filtered float); 7 – Hi-Z sampler
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cull Pipeline Layout"),
            bind_group_layouts: &[Some(&cull_bind_group_layout)],
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
            unified_quad_buffer,
            draw_commands_buffer,
            visible_draw_commands_buffer,
            subchunk_meta_buffer,
            visible_count_buffer,
            visible_count_staging,
            allocations: FxHashMap::default(),
            next_quad_offset: 0,
            active_subchunk_count: 0,
            cull_allocations: FxHashMap::default(),
            max_slot_bound: 0,
            // Pre-populate the free-slot stack in reverse so slot 0 is popped first.
            free_slots: {
                let mut v = Vec::with_capacity(MAX_SUBCHUNKS);
                v.extend((0..MAX_SUBCHUNKS).rev());
                v
            },
            free_quad_blocks: BTreeMap::new(),
            cull_pipeline,
            cull_bind_group_layout,
            cull_bind_group: None,
            cull_uniforms_buffer,
            hiz_sampler,
            coalesce_counter: 0,
            max_quads,
            quad_buffer_rebind_pending: false,
            label,
        }
    }

    /// Rebuilds the main culling bind group after the Hi-Z texture is recreated.
    ///
    /// Must be called whenever the depth pyramid texture or its view changes
    /// (e.g. on window resize), before the next call to [`dispatch_culling`].
    pub fn update_bind_group(
        &mut self,
        device: &wgpu::Device,
        hiz_view: &wgpu::TextureView,
        water: &IndirectManager,
    ) {
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
                    resource: self.subchunk_meta_buffer().as_entire_binding(),
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
                    resource: water.visible_draw_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: water.visible_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                },
            ],
        }));
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
    /// Returns `true` on success. The arena is compacted or grown in-place when
    /// necessary; live geometry is copied GPU-to-GPU and is never evicted.
    pub fn upload_subchunk(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        quads: &[PackedQuad],
        _aabb: &AABB,
    ) -> bool {
        // Empty geometry means the subchunk should be removed.
        if quads.is_empty() {
            self.remove_subchunk(queue, key);
            return true;
        }

        // Release the old allocation so its regions can be reused below.
        if let Some(old_alloc) = self.allocations.remove(&key) {
            if old_alloc.quad_count > 0 {
                Self::add_free_block(
                    &mut self.free_quad_blocks,
                    FreeBlock {
                        offset: old_alloc.quad_offset,
                        count: old_alloc.quad_count,
                    },
                );
            }
        }

        let quad_count = quads.len() as u32;

        // Try to reuse a free block; fall back to the high-water mark.
        let quad_alloc = Self::find_and_remove_free_block(&mut self.free_quad_blocks, quad_count);
        let (quad_offset, reused_quad) = match quad_alloc {
            Some(block) => {
                // Return the leftover tail of the block to the free list.
                if block.count > quad_count {
                    Self::add_free_block(
                        &mut self.free_quad_blocks,
                        FreeBlock {
                            offset: block.offset + quad_count,
                            count: block.count - quad_count,
                        },
                    );
                }
                (block.offset, true)
            }
            None => {
                if self.next_quad_offset + quad_count > self.max_quads as u32 {
                    // If enough space exists but is fragmented, pack live regions into a
                    // fresh GPU buffer first. This changes no mesh data and needs no CPU readback.
                    let free_quads: u32 = self
                        .free_quad_blocks
                        .values()
                        .flatten()
                        .map(|block| block.count)
                        .sum();
                    if free_quads >= quad_count {
                        self.compact_quad_arena(device, queue);
                    }
                }
                if self.next_quad_offset + quad_count > self.max_quads as u32 {
                    log(
                        LogLevel::Info,
                        &format!(
                            "Growing {} quad arena from {} to at least {} quads",
                            self.label,
                            self.max_quads,
                            self.next_quad_offset as usize + quad_count as usize
                        ),
                    );
                    self.grow_quad_arena(device, queue, quad_count);
                }
                (self.next_quad_offset, false)
            }
        };

        let alloc = SubchunkAlloc {
            quad_offset,
            quad_count,
        };

        let quad_byte_offset = alloc.quad_offset as u64 * size_of::<PackedQuad>() as u64;
        queue.write_buffer(
            &self.unified_quad_buffer,
            quad_byte_offset,
            bytemuck::cast_slice(quads),
        );

        // Advance the high-water marks only when no free block was reused.
        if !reused_quad {
            self.next_quad_offset += quad_count;
        }
        self.allocations.insert(key, alloc);
        self.active_subchunk_count = self.allocations.len() as u32;
        self.maybe_coalesce();
        true
    }

    fn create_quad_buffer(
        device: &wgpu::Device,
        label: &'static str,
        quads: usize,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Unified Quad Buffer")),
            size: (quads * size_of::<PackedQuad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Replaces the arena with a larger one and copies its occupied prefix entirely on the GPU.
    fn grow_quad_arena(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, required: u32) {
        let required_capacity = self.next_quad_offset as usize + required as usize;
        let new_capacity = (self.max_quads.saturating_mul(3) / 2)
            .max(required_capacity)
            .max(1);
        self.replace_quad_arena(device, queue, new_capacity, false);
    }

    /// GPU-side defragmentation. Metadata slots stay stable; only quad offsets are rewritten.
    fn compact_quad_arena(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.replace_quad_arena(device, queue, self.max_quads, true);
    }

    fn replace_quad_arena(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        capacity: usize,
        compact: bool,
    ) {
        let new_buffer = Self::create_quad_buffer(device, self.label, capacity);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Indirect Quad Arena Relocation"),
        });

        if compact {
            let mut live: Vec<_> = self
                .allocations
                .iter()
                .map(|(key, alloc)| (*key, *alloc))
                .collect();
            live.sort_by_key(|(_, alloc)| alloc.quad_offset);
            let mut next_offset = 0u32;
            for (key, old) in live {
                let byte_count = old.quad_count as u64 * size_of::<PackedQuad>() as u64;
                encoder.copy_buffer_to_buffer(
                    &self.unified_quad_buffer,
                    old.quad_offset as u64 * size_of::<PackedQuad>() as u64,
                    &new_buffer,
                    next_offset as u64 * size_of::<PackedQuad>() as u64,
                    byte_count,
                );
                let alloc = self
                    .allocations
                    .get_mut(&key)
                    .expect("live allocation disappeared");
                alloc.quad_offset = next_offset;
                next_offset += old.quad_count;
            }
            self.next_quad_offset = next_offset;
            self.free_quad_blocks.clear();
        } else if self.next_quad_offset > 0 {
            encoder.copy_buffer_to_buffer(
                &self.unified_quad_buffer,
                0,
                &new_buffer,
                0,
                self.next_quad_offset as u64 * size_of::<PackedQuad>() as u64,
            );
        }
        queue.submit(Some(encoder.finish()));
        self.unified_quad_buffer = new_buffer;
        self.max_quads = capacity;
        self.quad_buffer_rebind_pending = true;
    }

    /// Returns whether a replaced arena requires a new vertex-pulling bind group.
    pub fn take_quad_buffer_rebind(&mut self) -> bool {
        std::mem::take(&mut self.quad_buffer_rebind_pending)
    }

    /// Frees geometry belonging to `key`. Shared culling metadata is updated by
    /// `update_cull_subchunk` after both terrain and water allocations change.
    ///
    /// After this call the slot is returned to the free pool and may be reused
    /// by a subsequent [`upload_subchunk`].  Does nothing if `key` is not
    /// currently allocated.
    pub fn remove_subchunk(&mut self, _queue: &wgpu::Queue, key: SubchunkKey) {
        if let Some(alloc) = self.allocations.remove(&key) {
            if alloc.quad_count > 0 {
                Self::add_free_block(
                    &mut self.free_quad_blocks,
                    FreeBlock {
                        offset: alloc.quad_offset,
                        count: alloc.quad_count,
                    },
                );
            }

            self.active_subchunk_count = self.allocations.len() as u32;
            self.maybe_coalesce();
        }
    }

    /// Updates the single shared culling record for `key` after terrain and
    /// water geometry have both been uploaded or removed.
    pub fn update_cull_subchunk(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        aabb: &AABB,
        water: &IndirectManager,
    ) {
        let terrain_draw = self
            .allocations
            .get(&key)
            .map(|alloc| [alloc.quad_count * 6, alloc.quad_offset * 6, 0, 1])
            .unwrap_or([0; 4]);
        let water_draw = water
            .allocations
            .get(&key)
            .map(|alloc| [alloc.quad_count * 6, alloc.quad_offset * 6, 0, 1])
            .unwrap_or([0; 4]);

        let has_geometry = terrain_draw[3] != 0 || water_draw[3] != 0;
        let slot_index = if has_geometry {
            match self.cull_allocations.get(&key) {
                Some(&slot) => slot,
                None => match self.free_slots.pop() {
                    Some(slot) => {
                        self.cull_allocations.insert(key, slot);
                        self.max_slot_bound = self.max_slot_bound.max(slot as u32 + 1);
                        slot
                    }
                    None => {
                        log(
                            LogLevel::Warning,
                            "No free shared culling metadata slots available",
                        );
                        return;
                    }
                },
            }
        } else if let Some(slot) = self.cull_allocations.remove(&key) {
            self.free_slots.push(slot);
            self.max_slot_bound = self
                .cull_allocations
                .values()
                .map(|slot| *slot as u32 + 1)
                .max()
                .unwrap_or(0);
            slot
        } else {
            return;
        };

        let meta = if has_geometry {
            SubchunkGpuMeta {
                aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
                aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
                terrain_draw_data: terrain_draw,
                water_draw_data: water_draw,
            }
        } else {
            SubchunkGpuMeta::zeroed()
        };
        queue.write_buffer(
            self.subchunk_meta_buffer(),
            (slot_index * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&meta),
        );
    }

    /// Removes a shared culling record once both geometry managers have freed it.
    pub fn remove_cull_subchunk(&mut self, queue: &wgpu::Queue, key: SubchunkKey) {
        let Some(slot) = self.cull_allocations.remove(&key) else {
            return;
        };
        self.free_slots.push(slot);
        self.max_slot_bound = self
            .cull_allocations
            .values()
            .map(|slot| *slot as u32 + 1)
            .max()
            .unwrap_or(0);
        queue.write_buffer(
            self.subchunk_meta_buffer(),
            (slot * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&SubchunkGpuMeta::zeroed()),
        );
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
            Self::coalesce_vertex_blocks(&mut self.free_quad_blocks);
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
        for slot in self.cull_allocations.values() {
            let meta_byte_offset = slot * size_of::<SubchunkGpuMeta>();
            queue.write_buffer(
                self.subchunk_meta_buffer(),
                meta_byte_offset as u64,
                bytemuck::bytes_of(&SubchunkGpuMeta::zeroed()),
            );
        }

        self.allocations.clear();
        self.next_quad_offset = 0;
        self.active_subchunk_count = 0;
        self.cull_allocations.clear();
        self.max_slot_bound = 0;
        self.free_quad_blocks.clear();

        self.free_slots.clear();
        self.free_slots.extend((0..MAX_SUBCHUNKS).rev());
    }

    /// Uploads cull uniforms and dispatches the main camera culling compute pass.
    ///
    /// Clears both visible-count and command buffers before dispatching so that
    /// only subchunks that pass culling this frame are drawn.
    /// One workgroup of 64 threads is launched per 64 subchunk slots.
    ///
    /// Does nothing if no subchunks are currently allocated or if the bind
    /// group has not yet been created via [`update_bind_group`].
    pub fn dispatch_culling(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        timestamp_query_set: Option<&wgpu::QuerySet>,
        timestamp_start: u32,
        timestamp_end: u32,
        water: &IndirectManager,
        view_proj: &glam::Mat4,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
    ) {
        queue.write_buffer(&water.visible_count_buffer, 0, &0u32.to_le_bytes());
        if !self.cull_allocations.is_empty() {
            let bytes = self.max_slot_bound as u64 * size_of::<DrawIndirect>() as u64;
            encoder.clear_buffer(&water.visible_draw_commands_buffer, 0, Some(bytes));
        }
        self.dispatch_culling_into(
            encoder,
            queue,
            view_proj,
            frustum_planes,
            camera_pos,
            hiz_size,
            screen_size,
            &self.cull_uniforms_buffer,
            &self.visible_count_buffer,
            &self.visible_draw_commands_buffer,
            self.cull_bind_group.as_ref(),
            "Culling Pass",
            timestamp_query_set,
            timestamp_start,
            timestamp_end,
        );
    }

    fn dispatch_culling_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view_proj: &glam::Mat4,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
        uniforms_buffer: &wgpu::Buffer,
        count_buffer: &wgpu::Buffer,
        commands_buffer: &wgpu::Buffer,
        bind_group: Option<&wgpu::BindGroup>,
        label: &'static str,
        timestamp_query_set: Option<&wgpu::QuerySet>,
        timestamp_start: u32,
        timestamp_end: u32,
    ) {
        queue.write_buffer(count_buffer, 0, &0u32.to_le_bytes());

        if self.cull_allocations.is_empty() {
            // Still write a valid pair every profiled frame. Resolving an
            // unwritten query would otherwise report stale data from a prior
            // frame while the world is still loading.
            if let Some(query_set) = timestamp_query_set {
                let _timestamp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Empty Culling Timestamp Pass"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(timestamp_start),
                        end_of_pass_write_index: Some(timestamp_end),
                    }),
                });
            }
            return;
        }

        let active = self.max_slot_bound;
        let uniforms = CullUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            frustum_planes: *frustum_planes,
            camera_pos,
            subchunk_count: active,
            hiz_size,
            screen_size,
        };
        queue.write_buffer(uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        if let Some(bind_group) = bind_group {
            let bytes_to_clear = (active as u64) * size_of::<DrawIndirect>() as u64;
            if bytes_to_clear > 0 {
                encoder.clear_buffer(commands_buffer, 0, Some(bytes_to_clear));
            }

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: timestamp_query_set.map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(timestamp_start),
                        end_of_pass_write_index: Some(timestamp_end),
                    }
                }),
            });
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, bind_group, &[]);

            let workgroup_count = (active + 127) / 128;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Returns the compact descriptor buffer consumed by vertex-pulling shaders.
    pub fn quad_buffer(&self) -> &wgpu::Buffer {
        &self.unified_quad_buffer
    }

    /// Returns metadata indexed by `first_instance` in the draw command.
    pub fn subchunk_meta_buffer(&self) -> &wgpu::Buffer {
        self.subchunk_meta_buffer
            .as_ref()
            .expect("only the shared terrain culler owns metadata")
    }

    /// Returns a reference to the visible (post-cull) indirect draw command buffer.
    pub fn draw_commands(&self) -> &wgpu::Buffer {
        &self.visible_draw_commands_buffer
    }

    /// Returns the main visible-count buffer (used as an indirect dispatch argument).
    pub fn visible_count_buffer(&self) -> &wgpu::Buffer {
        &self.visible_count_buffer
    }

    /// Returns the number of subchunks currently allocated.
    pub fn active_count(&self) -> u32 {
        self.active_subchunk_count
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
        self.next_quad_offset = 0;
        self.active_subchunk_count = 0;
        self.cull_allocations.clear();
        self.max_slot_bound = 0;
        self.free_slots.clear();
        self.free_slots.extend((0..MAX_SUBCHUNKS).rev());
        self.free_quad_blocks.clear();
    }
}
