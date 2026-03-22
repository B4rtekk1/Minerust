use bytemuck::{Pod, Zeroable};

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

    /// Cascaded Shadow Map view-projection matrices.
    ///
    /// Layout: one `mat4x4` per cascade, 4 cascades total. Each matrix
    /// transforms world-space into the light-space clip space of that cascade.
    pub csm_view_proj: [[[f32; 4]; 4]; 4],

    /// World-space depth at which each CSM cascade ends.
    ///
    /// `csm_split_distances[i]` is the far-plane distance of cascade `i`.
    /// Compare against the camera-space fragment depth to select the correct cascade.
    pub csm_split_distances: [f32; 4],

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

    /// Selects the active water reflection technique.
    ///
    /// Interpreted as an integer enum in shaders:
    /// - `0.0` — no reflection
    /// - `1.0` — planar reflection
    /// - `2.0` — screen-space reflection (SSR)
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
    /// Explicit padding to keep the struct 16-byte aligned.
    pub _pad: f32,

    /// Rain intensity in the range `[0.0, 1.0]`.
    ///
    /// Used by the sky shader to desaturate the atmosphere and dim the sun
    /// / cloud response under overcast conditions.
    pub rain_factor: f32,
}

/// Small shadow-specific configuration uploaded separately from the main
/// per-frame uniform block.
///
/// This keeps shadow quality knobs isolated from the large `Uniforms` struct
/// and matches the `ShadowConfig` block used by `terrain.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ShadowConfig {
    /// Shadow map resolution in texels for one cascade face.
    pub shadow_map_size: f32,
    /// Number of PCF taps used when filtering the shadow map.
    pub pcf_samples: u32,
    /// Explicit padding so the buffer remains 16 bytes wide.
    pub _pad: [u32; 2],
}
