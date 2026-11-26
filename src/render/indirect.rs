use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;

use crate::constants::{CHUNK_UNLOAD_DISTANCE, NUM_SUBCHUNKS};
use crate::render::frustum::AABB;
use crate::render::quad::{PackedQuad, TERRAIN_GROUPS, group_terrain_quads};

use crate::logger::{LogLevel, log};
use std::collections::{BTreeMap, BTreeSet};

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
const TERRAIN_INITIAL_ARENA_BYTES: usize = 64 * 1024 * 1024;
const WATER_INITIAL_ARENA_BYTES: usize = 6 * 1024 * 1024;

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

/// GPU-side arguments for one indexed indirect draw (terrain): layout of
/// wgpu `DrawIndexedIndirectArgs`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Largest number of quads a single terrain draw can address through the
/// shared index pattern. One subchunk face direction holds at most
/// 16×16×16 unit faces, and the cutout group at most six times that.
pub const MAX_QUADS_PER_DRAW: usize = 65_536;
/// Distance buckets used to order terrain draws front to back.
pub const SORT_BUCKETS: usize = 8;

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
    /// Greedily merged far-terrain draw arguments, in the same format.
    pub terrain_far_draw_data: [u32; 4],
    /// 2×2×2 terrain LOD draw arguments, in the same format.
    pub terrain_lod_draw_data: [u32; 4],
    /// Terrain quad counts of draw groups 0..4 (face directions −X, +X, −Y, +Y).
    pub terrain_groups_a: [u32; 4],
    /// Terrain quad counts of groups 4..8 (−Z, +Z, cutout, unused).
    pub terrain_groups_b: [u32; 4],
    /// Far-terrain quad counts, split into the same seven draw groups.
    pub terrain_far_groups_a: [u32; 4],
    pub terrain_far_groups_b: [u32; 4],
    /// LOD quad counts, split into the same seven draw groups.
    pub terrain_lod_groups_a: [u32; 4],
    pub terrain_lod_groups_b: [u32; 4],
}

/// Terrain draws emitted per visible subchunk at most: one per face direction.
pub const MAX_TERRAIN_DRAWS_PER_SUBCHUNK: usize = 6;

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
    /// World-space **eye** position, used for per-direction back-face
    /// culling (must not be the player's feet).
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
    /// Quads per draw group (terrain only; see `TERRAIN_GROUPS`).
    groups: [u32; TERRAIN_GROUPS],
    /// Far and LOD ranges are relative to `quad_offset`, so compaction only
    /// needs to relocate the allocation's base offset.
    far_range: Option<TerrainDrawRange>,
    lod_range: Option<TerrainDrawRange>,
    /// Leading quads drawn near the camera; the rest (`quad_count -
    /// detail_count`) are a merged stand-in drawn far away (water only).
    detail_count: u32,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct TerrainDrawRange {
    quad_offset: u32,
    quad_count: u32,
    groups: [u32; TERRAIN_GROUPS],
}

impl TerrainDrawRange {
    fn with_base(mut self, base: u32) -> Self {
        self.quad_offset += base;
        self
    }

    fn draw_data(self) -> [u32; 4] {
        [
            self.quad_count * 6,
            self.quad_offset * 6,
            0,
            u32::from(self.quad_count > 0),
        ]
    }

    fn group_fields(self) -> ([u32; 4], [u32; 4]) {
        (
            [
                self.groups[0],
                self.groups[1],
                self.groups[2],
                self.groups[3],
            ],
            [self.groups[4], self.groups[5], self.groups[6], 0],
        )
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct TerrainVariantRanges {
    near: TerrainDrawRange,
    far: TerrainDrawRange,
    lod: TerrainDrawRange,
}

impl SubchunkAlloc {
    fn terrain_ranges(self) -> TerrainVariantRanges {
        let near = TerrainDrawRange {
            quad_offset: self.quad_offset,
            quad_count: self.quad_count,
            groups: self.groups,
        };
        let far = self
            .far_range
            .map(|range| range.with_base(self.quad_offset))
            .unwrap_or(near);
        let lod = self
            .lod_range
            .map(|range| range.with_base(self.quad_offset))
            .unwrap_or(far);
        TerrainVariantRanges { near, far, lod }
    }
}

fn group_terrain_variants(
    near: &[PackedQuad],
    far: &[PackedQuad],
    lod: &[PackedQuad],
) -> (Vec<PackedQuad>, TerrainVariantRanges) {
    let (near_quads, near_groups) = group_terrain_quads(near);
    let near_range = TerrainDrawRange {
        quad_offset: 0,
        quad_count: near_quads.len() as u32,
        groups: near_groups,
    };
    let mut grouped = near_quads;

    let far_range = if far.is_empty() {
        near_range
    } else {
        let (far_quads, groups) = group_terrain_quads(far);
        let range = TerrainDrawRange {
            quad_offset: grouped.len() as u32,
            quad_count: far_quads.len() as u32,
            groups,
        };
        grouped.extend(far_quads);
        range
    };
    let lod_range = if lod.is_empty() {
        far_range
    } else {
        let (lod_quads, groups) = group_terrain_quads(lod);
        let range = TerrainDrawRange {
            quad_offset: grouped.len() as u32,
            quad_count: lod_quads.len() as u32,
            groups,
        };
        grouped.extend(lod_quads);
        range
    };

    (
        grouped,
        TerrainVariantRanges {
            near: near_range,
            far: far_range,
            lod: lod_range,
        },
    )
}

/// A contiguous run of free elements inside a unified buffer.
#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    /// Start of the free run (in elements, not bytes).
    offset: u32,
    /// Length of the free run (in elements, not bytes).
    count: u32,
}

/// Free runs of a quad arena. Neighbouring runs are merged the moment one
/// is freed, so the list never needs a periodic sort-and-merge pass, and
/// best-fit lookups, inserts and merges are all O(log n).
#[derive(Default)]
struct FreeList {
    /// Run length by start offset (finds neighbours to merge with).
    by_offset: BTreeMap<u32, u32>,
    /// `(length, offset)` of every run (best-fit search).
    by_size: BTreeSet<(u32, u32)>,
    /// Total free elements.
    total: u64,
}

impl FreeList {
    fn remove_run(&mut self, offset: u32, count: u32) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(count, offset));
        self.total -= count as u64;
    }

    /// Adds a free run, merging it with directly adjacent runs.
    fn insert(&mut self, FreeBlock { mut offset, mut count }: FreeBlock) {
        if count == 0 {
            return;
        }
        if let Some((&prev_offset, &prev_count)) = self.by_offset.range(..offset).next_back() {
            if prev_offset + prev_count == offset {
                self.remove_run(prev_offset, prev_count);
                offset = prev_offset;
                count += prev_count;
            }
        }
        if let Some(&next_count) = self.by_offset.get(&(offset + count)) {
            self.remove_run(offset + count, next_count);
            count += next_count;
        }
        self.by_offset.insert(offset, count);
        self.by_size.insert((count, offset));
        self.total += count as u64;
    }

    /// Removes and returns the smallest run holding at least `count`.
    fn take_best_fit(&mut self, count: u32) -> Option<FreeBlock> {
        let &(run, offset) = self.by_size.range((count, 0)..).next()?;
        self.remove_run(offset, run);
        Some(FreeBlock { offset, count: run })
    }

    /// Removes the run ending exactly at `end`, if any (lets the arena's
    /// high-water mark shrink back over freed tail space).
    fn take_ending_at(&mut self, end: u32) -> Option<FreeBlock> {
        let (&offset, &count) = self.by_offset.range(..end).next_back()?;
        if offset + count != end {
            return None;
        }
        self.remove_run(offset, count);
        Some(FreeBlock { offset, count })
    }

    fn clear(&mut self) {
        self.by_offset.clear();
        self.by_size.clear();
        self.total = 0;
    }
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
    /// Visible alpha-tested (cutout) terrain draws, drawn with their own
    /// pipeline so the opaque pipeline can skip `discard`.
    cutout_draw_commands_buffer: wgpu::Buffer,
    /// Count of `cutout_draw_commands_buffer` entries written this frame.
    cutout_count_buffer: wgpu::Buffer,
    /// Visible terrain draws before front-to-back sorting.
    unsorted_draw_commands_buffer: wgpu::Buffer,
    /// Distance bucket of each unsorted terrain draw.
    draw_buckets_buffer: wgpu::Buffer,
    /// Per-bucket draw counts and scatter cursors (2 × `SORT_BUCKETS`).
    sort_counters_buffer: wgpu::Buffer,
    /// Static `(0,1,2)(0,2,3)` index pattern shared by every terrain draw.
    quad_index_buffer: wgpu::Buffer,
    /// Scatters culled terrain draws into bucket order.
    sort_pipeline: wgpu::ComputePipeline,
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
    /// CPU copy of the AABB for every shared metadata slot. This lets a quad
    /// arena relocation rebuild draw ranges for all live subchunks.
    cull_aabbs: FxHashMap<SubchunkKey, AABB>,
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
    free_quad_blocks: FreeList,
    /// Metadata slots freed this frame whose zeroing is still to be
    /// uploaded; written together (in contiguous runs) once per frame
    /// instead of one `write_buffer` per subchunk.
    pending_slot_clears: rustc_hash::FxHashSet<usize>,
    /// Current capacity of the growable quad arena.
    max_quads: usize,
    /// Set after the arena is replaced; the renderer must rebuild its vertex-pulling bind group.
    quad_buffer_rebind_pending: bool,
    /// Set when compaction changes existing allocation offsets.
    cull_metadata_refresh_pending: bool,
    /// Short label used in logs.
    label: &'static str,
    /// Quads in live allocations (excludes free blocks).
    live_quads: u64,
    /// A larger arena being allocated on a background thread. Allocating a
    /// buffer of hundreds of MiB takes the driver tens of milliseconds, so
    /// growth is started early, off the main thread, and only swapped in
    /// (a GPU-side copy) once ready.
    pending_arena: Option<std::sync::mpsc::Receiver<(wgpu::Buffer, usize)>>,
}

/// Start allocating a larger arena in the background once live geometry
/// fills this fraction of the current one.
const ARENA_PREGROW_FILL: f64 = 0.7;

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

        // Terrain emits up to one draw per visible face direction.
        let is_terrain = label == IndirectBufferBudget::TERRAIN.label();
        let draws_per_subchunk = if is_terrain {
            MAX_TERRAIN_DRAWS_PER_SUBCHUNK
        } else {
            1
        };
        // Terrain draws are indexed (20-byte args); water draws are not.
        let draw_args_size = if is_terrain {
            size_of::<DrawIndexedIndirect>()
        } else {
            size_of::<DrawIndirect>()
        };
        let visible_draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visible Draw Commands Buffer"),
            size: (MAX_SUBCHUNKS * draws_per_subchunk * draw_args_size) as u64,
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

        let cutout_draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cutout Draw Commands Buffer"),
            size: (if is_terrain { MAX_SUBCHUNKS } else { 1 } * draw_args_size) as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cutout_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cutout Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sort_capacity = if is_terrain {
            MAX_SUBCHUNKS * MAX_TERRAIN_DRAWS_PER_SUBCHUNK
        } else {
            1
        };
        let unsorted_draw_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Unsorted Terrain Draw Commands Buffer"),
            size: (sort_capacity * draw_args_size) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let draw_buckets_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Draw Buckets Buffer"),
            size: (sort_capacity * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sort_counters_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Sort Counters Buffer"),
            size: (SORT_BUCKETS * 2 * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pattern_quads = if is_terrain { MAX_QUADS_PER_DRAW } else { 1 };
        let pattern: Vec<u32> = (0..pattern_quads as u32)
            .flat_map(|quad| {
                let v = quad * 4;
                [v, v + 1, v + 2, v, v + 2, v + 3]
            })
            .collect();
        let quad_index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Quad Index Pattern Buffer"),
                contents: bytemuck::cast_slice(&pattern),
                usage: wgpu::BufferUsages::INDEX,
            },
        );

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
        //   8/9 – cutout terrain draw commands and count
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
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

        let sort_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Draw Sort Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("sort_draws"),
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
            cutout_draw_commands_buffer,
            cutout_count_buffer,
            unsorted_draw_commands_buffer,
            draw_buckets_buffer,
            sort_counters_buffer,
            quad_index_buffer,
            sort_pipeline,
            visible_count_staging,
            allocations: FxHashMap::default(),
            next_quad_offset: 0,
            active_subchunk_count: 0,
            cull_allocations: FxHashMap::default(),
            cull_aabbs: FxHashMap::default(),
            max_slot_bound: 0,
            // Pre-populate the free-slot stack in reverse so slot 0 is popped first.
            free_slots: {
                let mut v = Vec::with_capacity(MAX_SUBCHUNKS);
                v.extend((0..MAX_SUBCHUNKS).rev());
                v
            },
            free_quad_blocks: FreeList::default(),
            pending_slot_clears: Default::default(),
            cull_pipeline,
            cull_bind_group_layout,
            cull_bind_group: None,
            cull_uniforms_buffer,
            hiz_sampler,
            max_quads,
            quad_buffer_rebind_pending: false,
            cull_metadata_refresh_pending: false,
            label,
            live_quads: 0,
            pending_arena: None,
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
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.cutout_draw_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.cutout_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.unsorted_draw_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.draw_buckets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.sort_counters_buffer.as_entire_binding(),
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
        self.upload_subchunk_with_detail(device, queue, key, quads, quads.len() as u32)
    }

    /// Like [`Self::upload_subchunk`], but only the first `detail_count`
    /// quads are drawn near the camera; the remaining quads replace them
    /// beyond the distant-water range (see `cull.wgsl`).
    pub fn upload_subchunk_with_detail(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        quads: &[PackedQuad],
        detail_count: u32,
    ) -> bool {
        self.upload_subchunk_internal(device, queue, key, quads, detail_count, None)
    }

    /// Uploads the near, merged-far and 2×2×2 LOD terrain ranges together.
    /// Ranges share one allocation so arena relocation preserves their offsets.
    pub fn upload_terrain_variants(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        near: &[PackedQuad],
        far: &[PackedQuad],
        lod: &[PackedQuad],
        _aabb: &AABB,
    ) -> bool {
        if self.label != IndirectBufferBudget::TERRAIN.label() {
            return false;
        }
        let (grouped, ranges) = group_terrain_variants(near, far, lod);
        self.upload_subchunk_internal(
            device,
            queue,
            key,
            &grouped,
            grouped.len() as u32,
            Some(ranges),
        )
    }

    fn upload_subchunk_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: SubchunkKey,
        quads: &[PackedQuad],
        detail_count: u32,
        terrain_variants: Option<TerrainVariantRanges>,
    ) -> bool {
        // Empty geometry means the subchunk should be removed.
        if quads.is_empty() {
            self.remove_subchunk(queue, key);
            return true;
        }

        // A background-allocated larger arena is swapped in as soon as it
        // is ready.
        self.adopt_prepared_arena(device, queue, false);

        // Release the old allocation so its regions can be reused below.
        if let Some(old_alloc) = self.allocations.remove(&key) {
            self.live_quads -= old_alloc.quad_count as u64;
            self.release_quads(old_alloc.quad_offset, old_alloc.quad_count);
        }

        // Terrain geometry is stored grouped by face direction (plus a cutout
        // group) so the culling pass can skip back-facing directions.
        let (grouped, groups, terrain_ranges) = if let Some(ranges) = terrain_variants {
            (None, ranges.near.groups, Some(ranges))
        } else if self.label == IndirectBufferBudget::TERRAIN.label() {
            let (grouped, groups) = group_terrain_quads(quads);
            (Some(grouped), groups, None)
        } else {
            (None, [0; TERRAIN_GROUPS], None)
        };
        let quads = grouped.as_deref().unwrap_or(quads);
        let quad_count = quads.len() as u32;

        // Try to reuse a free block; fall back to the high-water mark.
        let quad_alloc = self.free_quad_blocks.take_best_fit(quad_count);
        let (quad_offset, reused_quad) = match quad_alloc {
            Some(block) => {
                // Return the leftover tail of the block to the free list.
                if block.count > quad_count {
                    self.free_quad_blocks.insert(FreeBlock {
                        offset: block.offset + quad_count,
                        count: block.count - quad_count,
                    });
                }
                (block.offset, true)
            }
            None => {
                if self.next_quad_offset + quad_count > self.max_quads as u32 {
                    // If enough space exists but is fragmented, pack live regions into a
                    // fresh GPU buffer first. This changes no mesh data and needs no CPU readback.
                    if self.free_quad_blocks.total >= quad_count as u64 {
                        self.compact_quad_arena(device, queue);
                    }
                }
                if self.next_quad_offset + quad_count > self.max_quads as u32
                    && self.pending_arena.is_some()
                {
                    // Growth is already under way: wait for it rather than
                    // allocating a second buffer here.
                    self.adopt_prepared_arena(device, queue, true);
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
            groups,
            far_range: terrain_ranges.map(|ranges| ranges.far),
            lod_range: terrain_ranges.map(|ranges| ranges.lod),
            detail_count: detail_count.min(quad_count),
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
        self.live_quads += quad_count as u64;
        self.active_subchunk_count = self.allocations.len() as u32;
        self.maybe_pregrow(device);
        true
    }

    /// Starts allocating a 2× larger arena on a background thread once
    /// live geometry fills [`ARENA_PREGROW_FILL`] of the current one.
    fn maybe_pregrow(&mut self, device: &wgpu::Device) {
        if self.pending_arena.is_some()
            || (self.live_quads as f64) < self.max_quads as f64 * ARENA_PREGROW_FILL
        {
            return;
        }
        // Doubling keeps the number of swaps (each a GPU copy of the whole
        // arena) small while the world streams in.
        let capacity = self.max_quads.saturating_mul(2);
        let (sender, receiver) = std::sync::mpsc::channel();
        let device = device.clone();
        let label = self.label;
        let spawned = std::thread::Builder::new()
            .name(format!("{label}-arena-alloc"))
            .spawn(move || {
                let buffer = Self::create_quad_buffer(&device, label, capacity);
                let _ = sender.send((buffer, capacity));
            });
        if spawned.is_ok() {
            log(
                LogLevel::Info,
                &format!(
                    "Pre-allocating {} quad arena of {} quads in the background",
                    self.label, capacity
                ),
            );
            self.pending_arena = Some(receiver);
        }
    }

    /// Swaps in the background-allocated arena if it is ready (or, with
    /// `wait`, once it is). The live prefix is copied on the GPU.
    fn adopt_prepared_arena(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, wait: bool) {
        let Some(receiver) = &self.pending_arena else {
            return;
        };
        let result = if wait {
            receiver.recv().map_err(|_| std::sync::mpsc::TryRecvError::Disconnected)
        } else {
            receiver.try_recv()
        };
        match result {
            Ok((buffer, capacity)) => {
                self.pending_arena = None;
                if capacity > self.max_quads {
                    self.replace_quad_arena_with(device, queue, buffer, capacity, false);
                }
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending_arena = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
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
        self.replace_quad_arena_with(device, queue, new_buffer, capacity, compact);
    }

    /// Moves the arena into `new_buffer` (GPU copies only).
    fn replace_quad_arena_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_buffer: wgpu::Buffer,
        capacity: usize,
        compact: bool,
    ) {
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
            let mut relocated = false;
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
                relocated |= old.quad_offset != next_offset;
                alloc.quad_offset = next_offset;
                next_offset += old.quad_count;
            }
            self.next_quad_offset = next_offset;
            self.free_quad_blocks.clear();
            self.cull_metadata_refresh_pending |= relocated;
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

    /// Returns whether compaction relocated live quad ranges since the last
    /// call. The terrain manager owns the shared metadata and uses this after
    /// both terrain and water uploads have completed.
    pub fn take_cull_metadata_refresh(&mut self) -> bool {
        std::mem::take(&mut self.cull_metadata_refresh_pending)
    }

    /// Frees geometry belonging to `key`. Shared culling metadata is updated by
    /// `update_cull_subchunk` after both terrain and water allocations change.
    ///
    /// After this call the slot is returned to the free pool and may be reused
    /// by a subsequent [`upload_subchunk`].  Does nothing if `key` is not
    /// currently allocated.
    pub fn remove_subchunk(&mut self, _queue: &wgpu::Queue, key: SubchunkKey) {
        if let Some(alloc) = self.allocations.remove(&key) {
            self.live_quads -= alloc.quad_count as u64;
            self.release_quads(alloc.quad_offset, alloc.quad_count);
            self.active_subchunk_count = self.allocations.len() as u32;
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
        let terrain_ranges = self
            .allocations
            .get(&key)
            .map(|alloc| alloc.terrain_ranges())
            .unwrap_or_default();
        let near_range = terrain_ranges.near;
        let far_range = terrain_ranges.far;
        let lod_range = terrain_ranges.lod;
        let terrain_draw = near_range.draw_data();
        let terrain_far_draw = far_range.draw_data();
        let terrain_lod_draw = lod_range.draw_data();
        let water_draw = water
            .allocations
            .get(&key)
            .map(|alloc| [alloc.detail_count * 6, alloc.quad_offset * 6, (alloc.quad_count - alloc.detail_count) * 6, 1])
            .unwrap_or([0; 4]);

        let has_geometry = terrain_draw[3] != 0
            || terrain_far_draw[3] != 0
            || terrain_lod_draw[3] != 0
            || water_draw[3] != 0;
        let slot_index = if has_geometry {
            match self.cull_allocations.get(&key) {
                Some(&slot) => slot,
                None => match self.free_slots.pop() {
                    Some(slot) => {
                        // Written below; a pending zeroing would erase it.
                        self.pending_slot_clears.remove(&slot);
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
            // Freed slots hold zeroed metadata that the cull shader skips, so
            // the bound stays at its high-water mark; rescanning every
            // allocation on each removal was quadratic during unloads.
            if self.cull_allocations.is_empty() {
                self.max_slot_bound = 0;
            }
            slot
        } else {
            return;
        };

        let meta = if has_geometry {
            self.cull_aabbs.insert(key, *aabb);
            SubchunkGpuMeta {
                aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
                aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
                terrain_draw_data: terrain_draw,
                water_draw_data: water_draw,
                terrain_far_draw_data: terrain_far_draw,
                terrain_lod_draw_data: terrain_lod_draw,
                terrain_groups_a: near_range.group_fields().0,
                terrain_groups_b: near_range.group_fields().1,
                terrain_far_groups_a: far_range.group_fields().0,
                terrain_far_groups_b: far_range.group_fields().1,
                terrain_lod_groups_a: lod_range.group_fields().0,
                terrain_lod_groups_b: lod_range.group_fields().1,
            }
        } else {
            self.cull_aabbs.remove(&key);
            SubchunkGpuMeta::zeroed()
        };
        queue.write_buffer(
            self.subchunk_meta_buffer(),
            (slot_index * size_of::<SubchunkGpuMeta>()) as u64,
            bytemuck::bytes_of(&meta),
        );
    }

    /// Rebuilds every shared culling record after either quad arena was
    /// compacted.  Both draw ranges are reconstructed together because the
    /// records are shared by the terrain and water managers.
    pub fn refresh_cull_metadata(&self, queue: &wgpu::Queue, water: &IndirectManager) {
        for (key, &slot) in &self.cull_allocations {
            let aabb = self
                .cull_aabbs
                .get(key)
                .expect("active culling slot is missing its AABB");
            let terrain_ranges = self
                .allocations
                .get(key)
                .map(|alloc| alloc.terrain_ranges())
                .unwrap_or_default();
            let near_range = terrain_ranges.near;
            let far_range = terrain_ranges.far;
            let lod_range = terrain_ranges.lod;
            let terrain_draw = near_range.draw_data();
            let terrain_far_draw = far_range.draw_data();
            let terrain_lod_draw = lod_range.draw_data();
            let water_draw = water
                .allocations
                .get(key)
                .map(|alloc| [alloc.detail_count * 6, alloc.quad_offset * 6, (alloc.quad_count - alloc.detail_count) * 6, 1])
                .unwrap_or([0; 4]);
            let meta = SubchunkGpuMeta {
                aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
                aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
                terrain_draw_data: terrain_draw,
                water_draw_data: water_draw,
                terrain_far_draw_data: terrain_far_draw,
                terrain_lod_draw_data: terrain_lod_draw,
                terrain_groups_a: near_range.group_fields().0,
                terrain_groups_b: near_range.group_fields().1,
                terrain_far_groups_a: far_range.group_fields().0,
                terrain_far_groups_b: far_range.group_fields().1,
                terrain_lod_groups_a: lod_range.group_fields().0,
                terrain_lod_groups_b: lod_range.group_fields().1,
            };
            queue.write_buffer(
                self.subchunk_meta_buffer(),
                (slot * size_of::<SubchunkGpuMeta>()) as u64,
                bytemuck::bytes_of(&meta),
            );
        }
    }

    /// Removes a shared culling record once both geometry managers have freed it.
    pub fn remove_cull_subchunk(&mut self, queue: &wgpu::Queue, key: SubchunkKey) {
        let Some(slot) = self.cull_allocations.remove(&key) else {
            return;
        };
        self.free_slots.push(slot);
        self.cull_aabbs.remove(&key);
        // See `update_cull_subchunk`: zeroed slots are skipped by the shader.
        if self.cull_allocations.is_empty() {
            self.max_slot_bound = 0;
        }
        let _ = queue;
        self.pending_slot_clears.insert(slot);
    }

    /// Returns a quad range to the free list. A range that ends at the
    /// high-water mark (with any free run before it) lowers the mark
    /// instead, so the arena's used prefix shrinks as geometry is unloaded.
    fn release_quads(&mut self, offset: u32, count: u32) {
        if count == 0 {
            return;
        }
        self.free_quad_blocks.insert(FreeBlock { offset, count });
        while let Some(tail) = self.free_quad_blocks.take_ending_at(self.next_quad_offset) {
            self.next_quad_offset = tail.offset;
        }
    }

    /// Uploads the zeroing of metadata slots freed since the last call, one
    /// `write_buffer` per contiguous run. Call once per frame before culling.
    pub fn flush_slot_clears(&mut self, queue: &wgpu::Queue) {
        if self.pending_slot_clears.is_empty() {
            return;
        }
        let mut slots: Vec<usize> = self.pending_slot_clears.drain().collect();
        slots.sort_unstable();
        let stride = size_of::<SubchunkGpuMeta>();
        let mut zeros: Vec<u8> = Vec::new();
        let mut start = 0;
        while start < slots.len() {
            let mut end = start + 1;
            while end < slots.len() && slots[end] == slots[end - 1] + 1 {
                end += 1;
            }
            zeros.resize((end - start) * stride, 0);
            queue.write_buffer(
                self.subchunk_meta_buffer(),
                (slots[start] * stride) as u64,
                &zeros,
            );
            start = end;
        }
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
        self.cull_aabbs.clear();
        self.max_slot_bound = 0;
        self.cull_metadata_refresh_pending = false;
        self.free_quad_blocks.clear();
        self.live_quads = 0;
        // Freed slots awaiting their zeroing upload must be cleared too.
        self.flush_slot_clears(queue);

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
        // Counters are reset on the GPU timeline: no per-frame staging
        // uploads (each `queue.write_buffer` allocates a staging copy).
        encoder.clear_buffer(&water.visible_count_buffer, 0, None);
        encoder.clear_buffer(&self.cutout_count_buffer, 0, None);
        encoder.clear_buffer(&self.sort_counters_buffer, 0, None);
        if !self.cull_allocations.is_empty() {
            let slots = self.max_slot_bound as u64;
            encoder.clear_buffer(
                &water.visible_draw_commands_buffer,
                0,
                Some(slots * size_of::<DrawIndirect>() as u64),
            );
            encoder.clear_buffer(
                &self.cutout_draw_commands_buffer,
                0,
                Some(slots * size_of::<DrawIndexedIndirect>() as u64),
            );
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
        encoder.clear_buffer(count_buffer, 0, None);

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
            let bytes_to_clear = (active as u64)
                * (MAX_TERRAIN_DRAWS_PER_SUBCHUNK * size_of::<DrawIndexedIndirect>()) as u64;
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

            // Order the visible terrain draws front to back (wgpu inserts the
            // storage barrier between the two dispatches).
            cpass.set_pipeline(&self.sort_pipeline);
            let max_draws = active * MAX_TERRAIN_DRAWS_PER_SUBCHUNK as u32;
            cpass.dispatch_workgroups(max_draws.div_ceil(128), 1, 1);
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
    /// Static index pattern for indexed terrain draws.
    pub fn quad_index_buffer(&self) -> &wgpu::Buffer {
        &self.quad_index_buffer
    }

    /// Visible alpha-tested terrain draws (terrain manager only).
    pub fn cutout_draw_commands(&self) -> &wgpu::Buffer {
        &self.cutout_draw_commands_buffer
    }

    /// Number of entries in [`Self::cutout_draw_commands`] this frame.
    pub fn cutout_count_buffer(&self) -> &wgpu::Buffer {
        &self.cutout_count_buffer
    }

    /// Upper bound on terrain draws the culling pass can emit this frame.
    pub fn max_terrain_draws(&self) -> u32 {
        self.max_slot_bound * MAX_TERRAIN_DRAWS_PER_SUBCHUNK as u32
    }

    /// Upper bound on cutout draws the culling pass can emit this frame.
    pub fn max_cutout_draws(&self) -> u32 {
        self.max_slot_bound
    }

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
        self.live_quads = 0;
        self.next_quad_offset = 0;
        self.active_subchunk_count = 0;
        self.cull_allocations.clear();
        self.cull_aabbs.clear();
        self.max_slot_bound = 0;
        self.cull_metadata_refresh_pending = false;
        self.free_slots.clear();
        self.free_slots.extend((0..MAX_SUBCHUNKS).rev());
        self.free_quad_blocks.clear();
    }
}

/// CPU mirror of `direction_visible` in `cull.wgsl` (kept in sync by the
/// test below): can any face of direction `dir` in the AABB face the eye?
#[cfg(test)]
fn direction_visible(dir: usize, aabb_min: [f32; 3], aabb_max: [f32; 3], eye: [f32; 3]) -> bool {
    const MARGIN: f32 = 0.05;
    match dir {
        0 => eye[0] < aabb_max[0] + MARGIN,
        1 => eye[0] > aabb_min[0] - MARGIN,
        2 => eye[1] < aabb_max[1] + MARGIN,
        3 => eye[1] > aabb_min[1] - MARGIN,
        4 => eye[2] < aabb_max[2] + MARGIN,
        _ => eye[2] > aabb_min[2] - MARGIN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terrain_quad(face: u8) -> PackedQuad {
        PackedQuad::terrain([0; 3], face, 2, 2, 3, [7; 3], [3; 4], [15; 4], 0)
    }

    #[test]
    fn terrain_variant_ranges_keep_grouped_offsets_and_fallbacks() {
        assert_eq!(size_of::<SubchunkGpuMeta>(), 192);

        let near = [terrain_quad(0), terrain_quad(5)];
        let far = [terrain_quad(1)];
        let lod = [terrain_quad(4), terrain_quad(4)];
        let (grouped, ranges) = group_terrain_variants(&near, &far, &lod);

        assert_eq!(grouped.len(), 5);
        assert_eq!((ranges.near.quad_offset, ranges.near.quad_count), (0, 2));
        assert_eq!((ranges.far.quad_offset, ranges.far.quad_count), (2, 1));
        assert_eq!((ranges.lod.quad_offset, ranges.lod.quad_count), (3, 2));
        assert_eq!(ranges.near.groups[0], 1);
        assert_eq!(ranges.near.groups[5], 1);
        assert_eq!(ranges.far.groups[1], 1);
        assert_eq!(ranges.lod.groups[4], 2);

        let allocation_base = 37;
        let far_absolute = ranges.far.with_base(allocation_base);
        let lod_absolute = ranges.lod.with_base(allocation_base);
        assert_eq!(far_absolute.quad_offset, 39);
        assert_eq!(lod_absolute.quad_offset, 40);
        assert_eq!(far_absolute.draw_data(), [6, 234, 0, 1]);

        let (_, fallback) = group_terrain_variants(&near, &[], &[]);
        assert_eq!(fallback.far, fallback.near);
        assert_eq!(fallback.lod, fallback.near);
    }

    #[test]
    fn free_list_merges_neighbours_and_finds_best_fit() {
        let mut list = FreeList::default();
        list.insert(FreeBlock { offset: 0, count: 4 });
        list.insert(FreeBlock { offset: 10, count: 2 });
        list.insert(FreeBlock { offset: 4, count: 6 }); // bridges both runs
        assert_eq!(list.by_offset.len(), 1);
        assert_eq!(list.total, 12);
        list.insert(FreeBlock { offset: 20, count: 3 });
        // Best fit picks the smallest run that is large enough.
        let fit = list.take_best_fit(3).unwrap();
        assert_eq!((fit.offset, fit.count), (20, 3));
        assert!(list.take_best_fit(13).is_none());
        let tail = list.take_ending_at(12).unwrap();
        assert_eq!((tail.offset, tail.count), (0, 12));
        assert_eq!(list.total, 0);
    }

    /// Every face that is front-facing from the eye must survive direction
    /// culling, for eyes inside, on the boundary of and around a subchunk.
    #[test]
    fn direction_culling_never_drops_front_facing_faces() {
        let (min, max) = ([0.0f32, 64.0, 0.0], [16.0f32, 80.0, 16.0]);
        let normals = [
            [-1.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ];
        let coords = [-3.0f32, 0.0, 0.5, 8.0, 15.99, 16.0, 19.0];
        let heights = [60.0f32, 64.0, 64.0001, 65.62, 79.9, 80.0, 81.6];
        for &ex in &coords {
            for &ey in &heights {
                for &ez in &coords {
                    let eye = [ex, ey, ez];
                    for (dir, n) in normals.iter().enumerate() {
                        // Faces of this direction lie on integer planes in the AABB.
                        for plane in 0..=16 {
                            let p = [
                                min[0] + plane as f32,
                                min[1] + plane as f32,
                                min[2] + plane as f32,
                            ];
                            let axis = dir / 2;
                            let to_eye = eye[axis] - p[axis];
                            let front_facing = to_eye * n[axis] > 0.0;
                            if front_facing {
                                assert!(
                                    direction_visible(dir, min, max, eye),
                                    "dir {dir} plane {plane} eye {eye:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cull_shader_mirrors_cpu_direction_test() {
        let shader = include_str!("../shaders/cull.wgsl");
        for line in [
            "case 0u: { return cam.x < aabb_max.x + m; }",
            "case 3u: { return cam.y > aabb_min.y - m; }",
            "const DIRECTION_MARGIN: f32 = 0.05;",
        ] {
            assert!(shader.contains(line), "cull.wgsl changed: {line}");
        }
    }
}
