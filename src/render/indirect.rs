use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::constants::{
    CHUNK_SIZE, CHUNK_UNLOAD_DISTANCE, NUM_SUBCHUNKS, RENDER_DISTANCE, SUBCHUNK_HEIGHT,
};
use crate::core::quad::PackedQuad;
use crate::core::uniforms::UploadBatch;
use crate::render::frustum::AABB;

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
/// A hierarchy node spans two chunk columns in X/Z and four subchunks in Y.
const CLUSTER_CHUNK_SPAN: i32 = 2;
const CLUSTER_SUBCHUNK_SPAN: i32 = 4;
const CLUSTER_EXTENT: [i32; 3] = [
    CHUNK_SIZE * CLUSTER_CHUNK_SPAN,
    SUBCHUNK_HEIGHT * CLUSTER_SUBCHUNK_SPAN,
    CHUNK_SIZE * CLUSTER_CHUNK_SPAN,
];
const MAX_CLUSTER_COLUMNS_PER_AXIS: usize = (CHUNK_UNLOAD_DISTANCE as usize * 2 + 2) / 2;
const MAX_CLUSTERS: usize = MAX_CLUSTER_COLUMNS_PER_AXIS
    * MAX_CLUSTER_COLUMNS_PER_AXIS
    * ((NUM_SUBCHUNKS as usize + 3) / 4);

/// Default terrain geometry budget for the unified buffers.
///
/// Greedy meshing keeps most subchunks far below these averages. If a pathological
/// area exceeds the budget, upload_subchunk falls back to the existing cache clear
/// and retry path instead of overflowing the GPU buffers.
const TERRAIN_QUADS_PER_SUBCHUNK_BUDGET: usize = 192;

/// Water is meshed separately from terrain but normally needs far fewer faces
/// per subchunk. Keeping a smaller water budget avoids duplicating the large
/// terrain buffers while preserving the same render distance and culling range.
const WATER_QUADS_PER_SUBCHUNK_BUDGET: usize = 32;

/// Geometry allocation policy for one [`IndirectManager`].
#[derive(Clone, Copy, Debug)]
pub struct IndirectBufferBudget {
    label: &'static str,
    quads_per_subchunk: usize,
}

impl IndirectBufferBudget {
    /// Budget used for opaque terrain geometry.
    pub const TERRAIN: Self = Self {
        label: "Terrain",
        quads_per_subchunk: TERRAIN_QUADS_PER_SUBCHUNK_BUDGET,
    };

    /// Budget used for water geometry.
    pub const WATER: Self = Self {
        label: "Water",
        quads_per_subchunk: WATER_QUADS_PER_SUBCHUNK_BUDGET,
    };

    fn max_quads(&self) -> usize {
        MAX_SUBCHUNKS * self.quads_per_subchunk
    }

    fn label(&self) -> &'static str {
        self.label
    }
}

/// GPU-side arguments for a single non-indexed indirect draw.
///
/// The memory layout matches `VkDrawIndirectCommand` / wgpu
/// `DrawIndirectArgs` so the buffer can be consumed directly by
/// the GPU without additional marshaling.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndirect {
    /// Number of procedural vertices to draw (`quad_count * 6`).
    pub vertex_count: u32,
    /// Number of instances to draw (typically 1).
    pub instance_count: u32,
    /// First procedural vertex (`quad_offset * 6`).
    pub first_vertex: u32,
    /// Instance ID of the first instance.
    pub first_instance: u32,
}

/// A pre-reserved descriptor range owned by a GPU face-extraction job.
#[derive(Copy, Clone, Debug)]
pub struct GpuQuadAllocation {
    pub quad_offset: u32,
    pub quad_capacity: u32,
    pub subchunk_slot: u32,
}

/// Per-subchunk metadata uploaded to the GPU for vertex pulling and culling.
///
/// Geometry descriptors are local to this origin.  Keeping it as integers
/// avoids baking lossy world-space `f32` values into every quad and permits a
/// future camera-relative/floating-origin transform without re-uploading mesh
/// data.  The fourth component is explicit WGSL `vec4` alignment padding.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SubchunkGpuMeta {
    /// Integer world-space origin of the owning subchunk.
    pub world_origin: [i32; 4],
    /// Packed draw arguments: `[vertex_count, first_vertex, slot, active]`.
    pub draw_data: [u32; 4],
}

/// GPU metadata for one coarse culling node.  Its extent is fixed at
/// `2 * 16` by `4 * 16` by `2 * 16` blocks, so only its origin is needed.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ClusterGpuMeta {
    world_origin: [i32; 4],
}

/// Uniform data consumed by the GPU culling compute shader.
///
/// Contains everything the shader needs to perform frustum + Hi-Z occlusion
/// culling in a single pass.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CullUniforms {
    /// View-projection matrix of the frame that produced the Hi-Z pyramid.
    pub occlusion_view_proj: [[f32; 4]; 4],
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
    /// Horizontal world-space radius accepted by the coarse distance test.
    pub cull_distance: f32,
    /// Number of active coarse hierarchy nodes.
    pub cluster_count: u32,
    /// Non-zero only when Hi-Z has valid history for `occlusion_view_proj`.
    pub occlusion_enabled: u32,
    /// Explicit padding to preserve the uniform block's 16-byte alignment.
    pub _padding: u32,
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
    quad_count: u32,
    /// Slot in the `SubchunkGpuMeta` array assigned to this subchunk.
    slot_index: usize,
    /// CPU copy used to move this entry when a dense slot is removed.
    gpu_meta: SubchunkGpuMeta,
    /// Unculled draw command, likewise rewritten when its slot changes.
    draw_command: DrawIndirect,
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
/// `IndirectManager` owns one large packed-quad storage buffer and assigns
/// subregions of that buffer to individual subchunks via a
/// free-list allocator.  A GPU compute pass then performs frustum and Hi-Z
/// occlusion culling each frame, writing surviving draw commands into a separate
/// indirect command buffer that is consumed by the main render pass.
///
pub struct IndirectManager {
    /// Single large vertex buffer shared by all subchunks.
    unified_quad_buffer: wgpu::Buffer,

    /// Staging buffer for all draw commands before culling (written by CPU).
    draw_commands_buffer: wgpu::Buffer,
    /// Output buffer for draw commands that survive the culling pass.
    visible_draw_commands_buffer: wgpu::Buffer,

    /// Per-slot AABB and draw-argument metadata consumed by the culling shader.
    subchunk_meta_buffer: wgpu::Buffer,
    /// Coarse AABBs rebuilt when the dense subchunk set changes.
    cluster_meta_buffer: wgpu::Buffer,
    /// Maps each dense subchunk slot to its parent cluster slot.
    subchunk_cluster_buffer: wgpu::Buffer,
    /// Visibility result produced by the cluster pass and consumed by children.
    cluster_visibility_buffer: wgpu::Buffer,

    /// Atomic counter incremented once by each non-empty culling workgroup.
    visible_count_buffer: wgpu::Buffer,
    /// CPU-readable staging copy of `visible_count_buffer` (for debugging/stats).
    #[allow(dead_code)]
    visible_count_staging: wgpu::Buffer,

    /// Map from subchunk identity to its current buffer allocation.
    allocations: FxHashMap<SubchunkKey, SubchunkAlloc>,
    /// High-water mark for quad allocations (used when no free block fits).
    next_quad_offset: u32,
    /// Number of subchunks currently allocated.
    active_subchunk_count: u32,
    /// Slot-to-key map. Its indices are always dense in `0..active_subchunk_count`.
    slot_keys: Vec<SubchunkKey>,

    /// Free-list for index buffer regions, keyed by block size for O(log n) lookup.

    /// Compute pipeline that performs per-subchunk frustum + Hi-Z culling.
    cull_pipeline: wgpu::ComputePipeline,
    /// First hierarchy stage: frustum/Hi-Z tests one AABB per cluster.
    cluster_cull_pipeline: wgpu::ComputePipeline,
    /// Bind group layout used by the main camera culling pass.
    cull_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the main (camera) culling pass; rebuilt when the Hi-Z changes.
    cull_bind_group: Option<wgpu::BindGroup>,
    /// Uniform buffer uploaded each frame with camera matrices and frustum planes.
    cull_uniforms_buffer: wgpu::Buffer,

    /// Nearest-neighbor sampler used to read the Hi-Z mip chain.
    hiz_sampler: wgpu::Sampler,

    /// Free-list for packed-quad buffer regions, keyed by block size for O(log n) lookup.
    free_quad_blocks: BTreeMap<u32, Vec<FreeBlock>>,
    /// Counts uploads/removals since the last free-list coalescing pass.
    coalesce_counter: usize,
    /// Maximum vertex count reserved by this manager.
    max_quads: usize,
    /// Short label used in logs.
    label: &'static str,
    cluster_count: u32,
    hierarchy_dirty: bool,
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
        let max_quads = budget.max_quads();
        let label = budget.label();

        let unified_quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Packed Quad Buffer")),
            size: (max_quads * size_of::<PackedQuad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<DrawIndirect>()) as u64,
            // The sun pass reads the unculled commands so off-camera casters
            // still contribute to shadows. The main pass uses the separate
            // visible command buffer produced by the culling compute shader.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
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

        let subchunk_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Subchunk Metadata Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<SubchunkGpuMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cluster_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Cluster Metadata Buffer"),
            size: (MAX_CLUSTERS * size_of::<ClusterGpuMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let subchunk_cluster_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Subchunk Cluster Index Buffer"),
            size: (MAX_SUBCHUNKS * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cluster_visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cull Cluster Visibility Buffer"),
            size: (MAX_CLUSTERS * size_of::<u32>()) as u64,
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
        //   6 – cluster AABBs, 7 – child-to-cluster indices,
        //   8 – cluster visibility written by the first pass
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
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
        let cluster_cull_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Cluster Cull Pipeline"),
                layout: Some(&cull_pipeline_layout),
                module: &cull_shader,
                entry_point: Some("cull_clusters"),
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
            cluster_meta_buffer,
            subchunk_cluster_buffer,
            cluster_visibility_buffer,
            visible_count_buffer,
            visible_count_staging,
            allocations: FxHashMap::default(),
            next_quad_offset: 0,
            active_subchunk_count: 0,
            slot_keys: Vec::with_capacity(MAX_SUBCHUNKS),
            free_quad_blocks: BTreeMap::new(),
            cull_pipeline,
            cluster_cull_pipeline,
            cull_bind_group_layout,
            cull_bind_group: None,
            cull_uniforms_buffer,
            hiz_sampler,
            coalesce_counter: 0,
            max_quads,
            label,
            cluster_count: 0,
            hierarchy_dirty: true,
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.cluster_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.subchunk_cluster_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cluster_visibility_buffer.as_entire_binding(),
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
    /// Returns `true` on success.  Returns `false` if either unified buffer is
    /// full and had to be cleared; the caller should re-submit all subchunks
    /// in that case.
    pub fn upload_subchunk(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        quads: &[PackedQuad],
        _aabb: &AABB,
    ) -> bool {
        // Empty geometry means the subchunk should be removed.
        if quads.is_empty() {
            if let Some(old_alloc) = self.remove_allocation(queue, key) {
                Self::add_free_block(
                    &mut self.free_quad_blocks,
                    FreeBlock {
                        offset: old_alloc.quad_offset,
                        count: old_alloc.quad_count,
                    },
                );
                self.maybe_coalesce();
            }
            return true;
        }

        // Release the old allocation so its regions can be reused below.
        if let Some(old_alloc) = self.remove_allocation(queue, key) {
            Self::add_free_block(
                &mut self.free_quad_blocks,
                FreeBlock {
                    offset: old_alloc.quad_offset,
                    count: old_alloc.quad_count,
                },
            );
        }

        let quad_count = quads.len() as u32;
        let quad_alloc = Self::find_and_remove_free_block(&mut self.free_quad_blocks, quad_count);
        let (quad_offset, reused_quad) = match quad_alloc {
            Some(block) => {
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
                    log(
                        LogLevel::Warning,
                        &format!(
                            "{} packed quad buffer full ({}/{} quads used), clearing indirect draw cache...",
                            self.label, self.next_quad_offset, self.max_quads
                        ),
                    );
                    self.clear_gpu_data(queue);
                    return false;
                }
                (self.next_quad_offset, false)
            }
        };

        if self.active_subchunk_count as usize >= MAX_SUBCHUNKS {
            log(
                LogLevel::Warning,
                "No free metadata slots available, clearing indirect draw cache...",
            );
            self.clear_gpu_data(queue);
            return false;
        }
        let slot_index = self.active_subchunk_count as usize;

        let gpu_meta = SubchunkGpuMeta {
            world_origin: [
                key.chunk_x * CHUNK_SIZE,
                key.subchunk_y * SUBCHUNK_HEIGHT,
                key.chunk_z * CHUNK_SIZE,
                0,
            ],
            draw_data: [quad_count * 6, quad_offset * 6, slot_index as u32, 1],
        };
        let draw_command = DrawIndirect {
            vertex_count: quad_count * 6,
            instance_count: 1,
            first_vertex: quad_offset * 6,
            first_instance: slot_index as u32,
        };
        let alloc = SubchunkAlloc {
            quad_offset,
            quad_count,
            slot_index,
            gpu_meta,
            draw_command,
        };

        let quad_byte_offset = alloc.quad_offset as u64 * size_of::<PackedQuad>() as u64;
        queue.write_buffer(
            &self.unified_quad_buffer,
            quad_byte_offset,
            bytemuck::cast_slice(quads),
        );

        self.write_slot(queue, &alloc);

        // Advance the high-water marks only when no free block was reused.
        if !reused_quad {
            self.next_quad_offset += quad_count;
        }
        self.allocations.insert(key, alloc);
        self.slot_keys.push(key);
        self.active_subchunk_count += 1;
        self.hierarchy_dirty = true;
        self.maybe_coalesce();
        true
    }

    /// Reserves a descriptor range for a compute meshing dispatch.
    ///
    /// Unlike [`Self::upload_subchunk`], this does not upload descriptors.
    /// The compute shader fills the range and updates `draw_data.x` once it
    /// knows the number of visible faces.  The allocation is deliberately
    /// sized by the CPU face-count prepass, not by a worst-case 24,576-face
    /// reservation per subchunk.
    pub fn prepare_gpu_subchunk(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        quad_capacity: u32,
    ) -> Option<GpuQuadAllocation> {
        if quad_capacity == 0 {
            self.remove_subchunk(queue, key);
            return Some(GpuQuadAllocation {
                quad_offset: 0,
                quad_capacity: 0,
                subchunk_slot: 0,
            });
        }
        if let Some(old) = self.remove_allocation(queue, key) {
            Self::add_free_block(
                &mut self.free_quad_blocks,
                FreeBlock {
                    offset: old.quad_offset,
                    count: old.quad_count,
                },
            );
        }
        let allocation =
            Self::find_and_remove_free_block(&mut self.free_quad_blocks, quad_capacity);
        let (quad_offset, reused) = match allocation {
            Some(block) => {
                if block.count > quad_capacity {
                    Self::add_free_block(
                        &mut self.free_quad_blocks,
                        FreeBlock {
                            offset: block.offset + quad_capacity,
                            count: block.count - quad_capacity,
                        },
                    );
                }
                (block.offset, true)
            }
            None => {
                if self.next_quad_offset + quad_capacity > self.max_quads as u32
                    || self.active_subchunk_count as usize >= MAX_SUBCHUNKS
                {
                    log(
                        LogLevel::Warning,
                        &format!(
                            "{} packed quad buffer full while reserving GPU mesh",
                            self.label
                        ),
                    );
                    self.clear_gpu_data(queue);
                    return None;
                }
                (self.next_quad_offset, false)
            }
        };
        let slot_index = self.active_subchunk_count as usize;
        let alloc = SubchunkAlloc {
            quad_offset,
            // Keep the reserved capacity here so remove/free returns the full range.
            quad_count: quad_capacity,
            slot_index,
            gpu_meta: SubchunkGpuMeta {
                world_origin: [
                    key.chunk_x * CHUNK_SIZE,
                    key.subchunk_y * SUBCHUNK_HEIGHT,
                    key.chunk_z * CHUNK_SIZE,
                    0,
                ],
                // The CPU prepass and the shader use identical visibility
                // rules, so capacity is also the expected final count. Keep
                // it here as well as in GPU metadata: a later swap-remove
                // must not resurrect a stale zero-count draw.
                draw_data: [quad_capacity * 6, quad_offset * 6, slot_index as u32, 1],
            },
            draw_command: DrawIndirect {
                vertex_count: quad_capacity * 6,
                instance_count: 1,
                first_vertex: quad_offset * 6,
                first_instance: slot_index as u32,
            },
        };
        self.write_slot(queue, &alloc);
        if !reused {
            self.next_quad_offset += quad_capacity;
        }
        self.allocations.insert(key, alloc);
        self.slot_keys.push(key);
        self.active_subchunk_count += 1;
        self.hierarchy_dirty = true;
        self.maybe_coalesce();
        Some(GpuQuadAllocation {
            quad_offset,
            quad_capacity,
            subchunk_slot: slot_index as u32,
        })
    }

    /// Batched counterpart to [`Self::upload_subchunk`]. It performs the same
    /// allocator updates, but appends geometry and metadata writes to one
    /// frame-owned upload batch instead of issuing individual queue writes.
    pub fn upload_subchunk_batched(
        &mut self,
        queue: &wgpu::Queue,
        batch: &mut UploadBatch,
        key: SubchunkKey,
        quads: &[PackedQuad],
        _aabb: &AABB,
    ) -> bool {
        if quads.is_empty() {
            if let Some(old_alloc) = self.remove_allocation_batched(batch, key) {
                Self::add_free_block(
                    &mut self.free_quad_blocks,
                    FreeBlock {
                        offset: old_alloc.quad_offset,
                        count: old_alloc.quad_count,
                    },
                );
                self.maybe_coalesce();
            }
            return true;
        }

        if let Some(old_alloc) = self.remove_allocation_batched(batch, key) {
            Self::add_free_block(
                &mut self.free_quad_blocks,
                FreeBlock {
                    offset: old_alloc.quad_offset,
                    count: old_alloc.quad_count,
                },
            );
        }

        let quad_count = quads.len() as u32;
        let quad_alloc = Self::find_and_remove_free_block(&mut self.free_quad_blocks, quad_count);
        let (quad_offset, reused_quad) = match quad_alloc {
            Some(block) => {
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
                    log(
                        LogLevel::Warning,
                        &format!(
                            "{} packed quad buffer full ({}/{} quads used), clearing indirect draw cache...",
                            self.label, self.next_quad_offset, self.max_quads
                        ),
                    );
                    // Discard queued writes: clearing the cache must be the
                    // final operation visible to the GPU this frame.
                    batch.clear();
                    self.clear_gpu_data(queue);
                    return false;
                }
                (self.next_quad_offset, false)
            }
        };

        if self.active_subchunk_count as usize >= MAX_SUBCHUNKS {
            log(
                LogLevel::Warning,
                "No free metadata slots available, clearing indirect draw cache...",
            );
            batch.clear();
            self.clear_gpu_data(queue);
            return false;
        }
        let slot_index = self.active_subchunk_count as usize;
        let alloc = SubchunkAlloc {
            quad_offset,
            quad_count,
            slot_index,
            gpu_meta: SubchunkGpuMeta {
                world_origin: [
                    key.chunk_x * CHUNK_SIZE,
                    key.subchunk_y * SUBCHUNK_HEIGHT,
                    key.chunk_z * CHUNK_SIZE,
                    0,
                ],
                draw_data: [quad_count * 6, quad_offset * 6, slot_index as u32, 1],
            },
            draw_command: DrawIndirect {
                vertex_count: quad_count * 6,
                instance_count: 1,
                first_vertex: quad_offset * 6,
                first_instance: slot_index as u32,
            },
        };

        batch.push(
            &self.unified_quad_buffer,
            alloc.quad_offset as u64 * size_of::<PackedQuad>() as u64,
            bytemuck::cast_slice(quads),
        );
        self.write_slot_batched(batch, &alloc);

        if !reused_quad {
            self.next_quad_offset += quad_count;
        }
        self.allocations.insert(key, alloc);
        self.slot_keys.push(key);
        self.active_subchunk_count += 1;
        self.hierarchy_dirty = true;
        self.maybe_coalesce();
        true
    }

    /// Returns the metadata slot index assigned to `key`, if it is allocated.
    pub fn get_slot_index(&self, key: &SubchunkKey) -> Option<usize> {
        self.allocations.get(key).map(|a| a.slot_index)
    }

    /// Changes a subchunk's integer world origin without touching its packed
    /// quad allocation.  This is the upload primitive needed by a floating
    /// origin or by streaming systems that reposition an already-built mesh.
    ///
    /// Returns `false` when `key` is not currently resident in this manager.
    pub fn set_subchunk_world_origin(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        world_origin: [i32; 3],
    ) -> bool {
        let updated_alloc = match self.allocations.get_mut(&key) {
            Some(alloc) => {
                alloc.gpu_meta.world_origin =
                    [world_origin[0], world_origin[1], world_origin[2], 0];
                *alloc
            }
            None => return false,
        };
        self.write_slot(queue, &updated_alloc);
        self.hierarchy_dirty = true;
        true
    }

    /// Frees all GPU resources belonging to `key` using swap-remove to keep
    /// metadata and unculled draw-command slots dense.
    pub fn remove_subchunk(&mut self, queue: &wgpu::Queue, key: SubchunkKey) {
        if let Some(alloc) = self.remove_allocation(queue, key) {
            Self::add_free_block(
                &mut self.free_quad_blocks,
                FreeBlock {
                    offset: alloc.quad_offset,
                    count: alloc.quad_count,
                },
            );
            self.maybe_coalesce();
        }
    }

    /// Removes an allocation from the dense slot array and returns its geometry
    /// allocation. If it was not last, rewrites the moved entry at the removed
    /// index and updates its CPU-side slot index.
    fn remove_allocation(
        &mut self,
        queue: &wgpu::Queue,
        key: SubchunkKey,
    ) -> Option<SubchunkAlloc> {
        let removed = self.allocations.remove(&key)?;
        let removed_slot = removed.slot_index;
        let last_slot = self.active_subchunk_count as usize - 1;
        debug_assert_eq!(self.slot_keys[removed_slot], key);

        if removed_slot != last_slot {
            let moved_key = self.slot_keys[last_slot];
            let moved_alloc = {
                let alloc = self
                    .allocations
                    .get_mut(&moved_key)
                    .expect("dense slot map must reference an allocation");
                alloc.slot_index = removed_slot;
                *alloc
            };
            self.write_slot(queue, &moved_alloc);
            self.slot_keys[removed_slot] = moved_key;
        }

        self.slot_keys.pop();
        self.active_subchunk_count -= 1;
        self.hierarchy_dirty = true;
        // The last slot is outside the dense range now. Clearing it avoids
        // stale entries if a future caller accidentally uses a wider range.
        self.zero_metadata_slot(queue, last_slot);
        Some(removed)
    }

    fn remove_allocation_batched(
        &mut self,
        batch: &mut UploadBatch,
        key: SubchunkKey,
    ) -> Option<SubchunkAlloc> {
        let removed = self.allocations.remove(&key)?;
        let removed_slot = removed.slot_index;
        let last_slot = self.active_subchunk_count as usize - 1;
        debug_assert_eq!(self.slot_keys[removed_slot], key);

        if removed_slot != last_slot {
            let moved_key = self.slot_keys[last_slot];
            let moved_alloc = {
                let alloc = self
                    .allocations
                    .get_mut(&moved_key)
                    .expect("dense slot map must reference an allocation");
                alloc.slot_index = removed_slot;
                *alloc
            };
            self.write_slot_batched(batch, &moved_alloc);
            self.slot_keys[removed_slot] = moved_key;
        }
        self.slot_keys.pop();
        self.active_subchunk_count -= 1;
        self.hierarchy_dirty = true;
        self.zero_metadata_slot_batched(batch, last_slot);
        Some(removed)
    }

    /// Writes one allocation at its current dense slot, fixing slot-dependent
    /// draw data used by the culler and vertex pulling.
    fn write_slot(&self, queue: &wgpu::Queue, alloc: &SubchunkAlloc) {
        let slot = alloc.slot_index;
        let mut meta = alloc.gpu_meta;
        meta.draw_data[2] = slot as u32;
        queue.write_buffer(
            &self.subchunk_meta_buffer,
            (slot * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&meta),
        );

        let mut draw = alloc.draw_command;
        draw.first_instance = slot as u32;
        queue.write_buffer(
            &self.draw_commands_buffer,
            (slot * size_of::<DrawIndirect>()) as u64,
            bytemuck::bytes_of(&draw),
        );
    }

    fn write_slot_batched(&self, batch: &mut UploadBatch, alloc: &SubchunkAlloc) {
        let slot = alloc.slot_index;
        let mut meta = alloc.gpu_meta;
        meta.draw_data[2] = slot as u32;
        batch.push(
            &self.subchunk_meta_buffer,
            (slot * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&meta),
        );
        let mut draw = alloc.draw_command;
        draw.first_instance = slot as u32;
        batch.push(
            &self.draw_commands_buffer,
            (slot * size_of::<DrawIndirect>()) as u64,
            bytemuck::bytes_of(&draw),
        );
    }

    /// Zeros one metadata and draw slot so neither camera nor light passes can
    /// reuse stale geometry after a subchunk has been unloaded.
    fn zero_metadata_slot(&self, queue: &wgpu::Queue, slot_index: usize) {
        let subchunk_meta = SubchunkGpuMeta {
            world_origin: [0; 4],
            draw_data: [0, 0, 0, 0],
        };
        let meta_byte_offset = slot_index * size_of::<SubchunkGpuMeta>();
        queue.write_buffer(
            &self.subchunk_meta_buffer,
            meta_byte_offset as u64,
            bytemuck::bytes_of(&subchunk_meta),
        );
        let empty_draw = DrawIndirect::zeroed();
        let draw_byte_offset = slot_index * size_of::<DrawIndirect>();
        queue.write_buffer(
            &self.draw_commands_buffer,
            draw_byte_offset as u64,
            bytemuck::bytes_of(&empty_draw),
        );
    }

    fn zero_metadata_slot_batched(&self, batch: &mut UploadBatch, slot_index: usize) {
        let meta = SubchunkGpuMeta {
            world_origin: [0; 4],
            draw_data: [0; 4],
        };
        batch.push(
            &self.subchunk_meta_buffer,
            (slot_index * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&meta),
        );
        batch.push(
            &self.draw_commands_buffer,
            (slot_index * size_of::<DrawIndirect>()) as u64,
            bytemuck::bytes_of(&DrawIndirect::zeroed()),
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
        for alloc in self.allocations.values() {
            let subchunk_meta = SubchunkGpuMeta {
                world_origin: [0; 4],
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
        self.next_quad_offset = 0;
        self.active_subchunk_count = 0;
        self.slot_keys.clear();
        self.free_quad_blocks.clear();
        self.cluster_count = 0;
        self.hierarchy_dirty = true;
    }

    /// Rebuilds the compact cluster table after streaming or a slot swap.
    ///
    /// The child metadata remains dense for drawing, while this table groups
    /// arbitrary dense slots by their fixed world-space 2×2×4 parent.  This
    /// indirection avoids constraining the allocator to a spatial slot order.
    fn upload_hierarchy_if_dirty(&mut self, queue: &wgpu::Queue) {
        if !self.hierarchy_dirty {
            return;
        }

        let active = self.active_subchunk_count as usize;
        let mut clusters = Vec::<ClusterGpuMeta>::new();
        let mut cluster_slots = BTreeMap::<(i32, i32, i32), u32>::new();
        let mut child_clusters = vec![0u32; active];

        for (slot, key) in self.slot_keys.iter().enumerate() {
            let origin = self
                .allocations
                .get(key)
                .expect("dense slot map must reference an allocation")
                .gpu_meta
                .world_origin;
            let cluster_key = (
                origin[0].div_euclid(CLUSTER_EXTENT[0]),
                origin[1].div_euclid(CLUSTER_EXTENT[1]),
                origin[2].div_euclid(CLUSTER_EXTENT[2]),
            );
            let cluster_slot = match cluster_slots.get(&cluster_key) {
                Some(&slot) => slot,
                None => {
                    let slot = clusters.len() as u32;
                    assert!(
                        (slot as usize) < MAX_CLUSTERS,
                        "cluster buffer capacity exceeded"
                    );
                    cluster_slots.insert(cluster_key, slot);
                    clusters.push(ClusterGpuMeta {
                        world_origin: [
                            cluster_key.0 * CLUSTER_EXTENT[0],
                            cluster_key.1 * CLUSTER_EXTENT[1],
                            cluster_key.2 * CLUSTER_EXTENT[2],
                            0,
                        ],
                    });
                    slot
                }
            };
            child_clusters[slot] = cluster_slot;
        }

        if !clusters.is_empty() {
            queue.write_buffer(
                &self.cluster_meta_buffer,
                0,
                bytemuck::cast_slice(&clusters),
            );
            queue.write_buffer(
                &self.subchunk_cluster_buffer,
                0,
                bytemuck::cast_slice(&child_clusters),
            );
        }
        self.cluster_count = clusters.len() as u32;
        self.hierarchy_dirty = false;
    }

    /// Uploads cull uniforms and dispatches the two-stage camera culling pass.
    ///
    /// Clears `visible_count_buffer` before dispatching so that only subchunks
    /// that pass culling this frame are drawn. When indirect-count drawing is
    /// unavailable, also clears the command range consumed by the fixed-count
    /// fallback so stale commands become no-op draws.
    /// One workgroup of 128 threads is launched per 128 active subchunks. Each
    /// workgroup compacts visible entries locally and reserves its output span
    /// with one global atomic operation.
    ///
    /// Does nothing if no subchunks are currently allocated or if the bind
    /// group has not yet been created via [`update_bind_group`].
    pub fn dispatch_culling(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        occlusion_view_proj: &glam::Mat4,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
        occlusion_enabled: bool,
        supports_indirect_count: bool,
    ) {
        self.upload_hierarchy_if_dirty(queue);
        self.dispatch_culling_into(
            encoder,
            queue,
            occlusion_view_proj,
            frustum_planes,
            camera_pos,
            hiz_size,
            screen_size,
            occlusion_enabled,
            supports_indirect_count,
            &self.cull_uniforms_buffer,
            &self.visible_count_buffer,
            &self.visible_draw_commands_buffer,
            self.cull_bind_group.as_ref(),
            "Culling Pass",
        );
    }

    fn dispatch_culling_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        occlusion_view_proj: &glam::Mat4,
        frustum_planes: &[[f32; 4]; 6],
        camera_pos: [f32; 3],
        hiz_size: [f32; 2],
        screen_size: [f32; 2],
        occlusion_enabled: bool,
        supports_indirect_count: bool,
        uniforms_buffer: &wgpu::Buffer,
        count_buffer: &wgpu::Buffer,
        commands_buffer: &wgpu::Buffer,
        bind_group: Option<&wgpu::BindGroup>,
        label: &'static str,
    ) {
        // Keep this reset in the same command buffer as culling and drawing.
        // With indirect-count support, the draw pass consumes only this many
        // commands, so commands left from prior frames need not be cleared.
        encoder.clear_buffer(count_buffer, 0, None);

        if self.active_subchunk_count == 0 {
            return;
        }

        let active = self.active_subchunk_count;
        let uniforms = CullUniforms {
            occlusion_view_proj: occlusion_view_proj.to_cols_array_2d(),
            frustum_planes: *frustum_planes,
            camera_pos,
            subchunk_count: active,
            hiz_size,
            screen_size,
            cull_distance: (RENDER_DISTANCE * CHUNK_SIZE) as f32,
            cluster_count: self.cluster_count,
            occlusion_enabled: u32::from(occlusion_enabled),
            _padding: 0,
        };
        queue.write_buffer(uniforms_buffer, 0, bytemuck::bytes_of(&uniforms));

        if let Some(bind_group) = bind_group {
            if !supports_indirect_count {
                let bytes_to_clear = (active as u64) * size_of::<DrawIndirect>() as u64;
                if bytes_to_clear > 0 {
                    encoder.clear_buffer(commands_buffer, 0, Some(bytes_to_clear));
                }
            }

            let mut cluster_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cluster Culling Pass"),
                timestamp_writes: None,
            });
            cluster_pass.set_pipeline(&self.cluster_cull_pipeline);
            cluster_pass.set_bind_group(0, bind_group, &[]);
            cluster_pass.dispatch_workgroups((self.cluster_count + 127) / 128, 1, 1);
            drop(cluster_pass);

            let mut child_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            child_pass.set_pipeline(&self.cull_pipeline);
            child_pass.set_bind_group(0, bind_group, &[]);

            let workgroup_count = (active + 127) / 128;
            child_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Returns the storage buffer containing packed quad descriptors.
    pub fn quad_buffer(&self) -> &wgpu::Buffer {
        &self.unified_quad_buffer
    }

    /// Returns per-subchunk origins and draw metadata for vertex pulling.
    pub fn subchunk_meta_buffer(&self) -> &wgpu::Buffer {
        &self.subchunk_meta_buffer
    }

    /// Returns a reference to the visible (post-cull) indirect draw command buffer.
    pub fn draw_commands(&self) -> &wgpu::Buffer {
        &self.visible_draw_commands_buffer
    }

    /// Commands for every loaded subchunk, before camera frustum/Hi-Z culling.
    /// Used by light-space passes, whose visibility cannot depend on the camera.
    pub fn all_draw_commands(&self) -> &wgpu::Buffer {
        &self.draw_commands_buffer
    }

    /// Number of dense commands in the unculled stream.
    pub fn all_draw_command_count(&self) -> u32 {
        self.active_subchunk_count
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
        self.slot_keys.clear();
        self.free_quad_blocks.clear();
        self.cluster_count = 0;
        self.hierarchy_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cull_shader_parses_and_validates() {
        let source = include_str!("../shaders/cull.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source).expect("cull WGSL must parse");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("cull WGSL must validate");
    }
}
