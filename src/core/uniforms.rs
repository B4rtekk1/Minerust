use bytemuck::{Pod, Zeroable};

/// Mutable GPU bookkeeping for one voxel subchunk.
///
/// The face-extraction compute pass uses this layout when writing persistent
/// [`crate::core::quad::PackedQuad`] descriptors.  `quad_capacity` is supplied
/// by the CPU allocator; the shader never writes past that range.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuSubchunk {
    pub voxel_offset: u32,
    pub quad_offset: u32,
    pub quad_count: u32,
    pub quad_capacity: u32,
    pub flags: u32,
}

/// One deferred copy from a frame upload staging buffer to a GPU buffer.
///
/// The source offset points into [`UploadBatch::data`]. All offsets and sizes
/// are kept four-byte aligned, as required by `copy_buffer_to_buffer`.
pub struct PendingCopy {
    pub source_offset: u64,
    pub destination: wgpu::Buffer,
    pub destination_offset: u64,
    pub size: u64,
}

/// CPU-side upload accumulator for one frame.
///
/// Mesh descriptors and their metadata are appended to one contiguous byte
/// buffer. At the end of the frame update it is written once to a staging-ring
/// region, then copied into the individual GPU buffers by a command encoder.
#[derive(Default)]
pub struct UploadBatch {
    pub data: Vec<u8>,
    pub copies: Vec<PendingCopy>,
}

impl UploadBatch {
    /// Appends `data` and schedules its copy into `destination`.
    pub fn push(&mut self, destination: &wgpu::Buffer, destination_offset: u64, data: &[u8]) {
        debug_assert_eq!(destination_offset % wgpu::COPY_BUFFER_ALIGNMENT, 0);
        debug_assert_eq!(data.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT, 0);

        let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
        let source_offset = (self.data.len() + alignment - 1) & !(alignment - 1);
        self.data.resize(source_offset, 0);
        self.data.extend_from_slice(data);
        self.copies.push(PendingCopy {
            source_offset: source_offset as u64,
            destination: destination.clone(),
            destination_offset,
            size: data.len() as u64,
        });
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.copies.clear();
    }
}

/// Per-frame uniform data uploaded to the GPU at the start of each render pass.
///
/// All matrices are stored in column-major order to match WGSL/GLSL conventions.
/// The struct is `#[repr(C)]` and implements [`Pod`] / [`Zeroable`] for safe
/// byte-slice casting into a uniform buffer.
///
/// # Alignment
/// Fields are ordered to satisfy `std140`/`std430` alignment rules without
/// implicit padding. Explicit padding fields (prefixed `_pad`) are included
/// where necessary to maintain 16-byte alignment boundaries.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    /// Combined view-projection matrix for the main camera.
    pub view_proj: [[f32; 4]; 4],

    /// Inverse of [`Self::view_proj`], used for reconstructing world-space
    /// positions from NDC (e.g. in deferred or post-process passes).
    pub inv_view_proj: [[f32; 4]; 4],

    /// World-space camera position `[x, y, z]`.
    ///
    /// Packed with [`Self::time`] to fill a `vec4` alignment slot.
    pub camera_pos: [f32; 3],

    /// Elapsed time in seconds since application start.
    ///
    /// Used for animating effects such as water waves, wind, or sky scattering.
    pub time: f32,

    /// Normalized direction vector toward the sun `[x, y, z]` in world space.
    ///
    /// Packed with [`Self::is_underwater`] to fill a `vec4` alignment slot.
    pub sun_position: [f32; 3],

    /// Non-zero when the camera is below the water surface, zero otherwise.
    ///
    /// Treated as a boolean in shaders (`> 0.0` = underwater). Stored as `f32`
    /// to avoid padding issues.
    pub is_underwater: f32,

    /// Render target dimensions in pixels `[width, height]`.
    ///
    /// Used for UV reconstruction, TAA jitter, and screen-space effects.
    pub screen_size: [f32; 2],

    /// World-space Y coordinate of the water plane.
    ///
    /// Used by water shaders and above/below-surface transitions.
    pub water_level: f32,

    /// Selects the active water-reflection technique.
    ///
    /// Kept in this slot to match the WGSL `Uniforms` layout used by the
    /// current terrain and water renderers.
    pub reflection_mode: f32,

    /// Normalized direction vector toward the moon `[x, y, z]` in world space.
    ///
    /// Packed with [`Self::_pad1_moon`] to fill a `vec4` alignment slot.
    pub moon_position: [f32; 3],

    /// Explicit padding to align `moon_position` to a 16-byte boundary.
    ///
    /// Not intended for use in shaders.
    pub _pad1_moon: f32,

    /// Current moon light intensity in the range `[0.0, 1.0]`.
    pub moon_intensity: f32,
    /// Normalized wind direction in XZ space `[x, z]`.
    pub wind_dir: [f32; 2],
    /// Multiplier applied to the water wave phase speed.
    pub wind_speed: f32,

    /// Rain intensity in the range `[0.0, 1.0]`.
    ///
    /// Used by the sky shader to desaturate the atmosphere and dim the sun
    /// / cloud response under overcast conditions.
    pub rain_factor: f32,

    /// Approximate fraction of open sky visible above the camera.
    ///
    /// `1.0` means the camera is outdoors, `0.0` means a solid ceiling blocks
    /// the column above it. Terrain GI uses this to darken caves and tunnels.
    pub sky_visibility: f32,

    /// Non-zero while the main menu is open, enabling menu-only post effects.
    pub menu_blur: f32,

    /// Explicit padding so the uniform block remains 16-byte aligned.
    pub _pad_uniforms: f32,
}
