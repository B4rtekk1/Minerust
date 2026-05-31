use glam::{Mat4, Vec3, Vec4};

use crate::constants::{
    CSM_ACTIVE_CASCADE_COUNT, CSM_CASCADE_COUNT, CSM_CASCADE_SPLITS, CSM_SHADOW_MAP_SIZES,
};

// Voxel shadows shimmer if the shadow-map grid rotates every frame with the
// day cycle. Update the light direction in small angular steps so a stationary
// camera keeps a stable shadow map between steps.
const CSM_SUN_DIRECTION_UPDATE_RADIANS: f32 = 0.0015;

#[derive(Debug, Clone, Copy)]
pub struct CascadeData {
    pub view_proj: Mat4,
    pub split_distance: f32,
}

impl Default for CascadeData {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            split_distance: 0.0,
        }
    }
}

pub struct CsmManager {
    pub cascades: [CascadeData; CSM_CASCADE_COUNT],
    snapped_centers: [Vec3; CSM_CASCADE_COUNT],
    raw_centers: [Vec3; CSM_CASCADE_COUNT],
    stable_sun_dir: Vec3,
    sun_direction_initialized: bool,
    centers_initialized: bool,
}

impl CsmManager {
    pub fn new() -> Self {
        Self {
            cascades: [CascadeData::default(); CSM_CASCADE_COUNT],
            snapped_centers: [Vec3::ZERO; CSM_CASCADE_COUNT],
            raw_centers: [Vec3::ZERO; CSM_CASCADE_COUNT],
            stable_sun_dir: Vec3::Y,
            sun_direction_initialized: false,
            centers_initialized: false,
        }
    }
    pub fn update(
        &mut self,
        camera_view: &Mat4,
        sun_dir: Vec3,
        near: f32,
        far: f32,
        aspect: f32,
        fov_y: f32,
    ) {
        let inv_view = camera_view.inverse();
        let (light_z, light_direction_changed) = self.stable_light_direction(sun_dir);
        let (light_x, light_y) = stable_light_basis(light_z);

        let mut split_distances = [0.0_f32; CSM_CASCADE_COUNT + 1];
        split_distances[0] = near;

        for i in 0..CSM_CASCADE_COUNT {
            split_distances[i + 1] = CSM_CASCADE_SPLITS[i].min(far);
        }

        let active_cascade_count = CSM_ACTIVE_CASCADE_COUNT.min(CSM_CASCADE_COUNT);
        for cascade_idx in 0..active_cascade_count {
            let cascade_near = split_distances[cascade_idx];
            let cascade_far = split_distances[cascade_idx + 1];

            let slice_depth = (cascade_far - cascade_near).max(1.0);

            // Use a bounding sphere instead of a tight AABB. The square
            // projection is slightly larger, but its size does not breathe as
            // the camera moves, which prevents front/back cascade jumps.
            let shadow_size = CSM_SHADOW_MAP_SIZES[cascade_idx].max(1) as f32;
            let center_depth = stable_cascade_center_depth(cascade_idx, cascade_near, cascade_far);
            let radius = calculate_frustum_bounding_radius(
                cascade_near,
                cascade_far,
                center_depth,
                fov_y,
                aspect,
                shadow_size,
            );
            let extent = radius * 2.0;
            let texel_size = extent / shadow_size;
            let center = calculate_frustum_center(center_depth, &inv_view);
            let snapped_center = self.stable_snapped_center(
                cascade_idx,
                center,
                light_x,
                light_y,
                texel_size,
                light_direction_changed,
            );
            let depth_pad = (slice_depth * 1.5).clamp(64.0, 256.0);
            let light_distance = radius + depth_pad;
            let light_pos = snapped_center + light_z * light_distance;
            let light_view = Mat4::look_at_rh(light_pos, snapped_center, light_y);

            let left = -radius;
            let right = radius;
            let bottom = -radius;
            let top = radius;

            // Keep the depth range independent of sub-texel camera motion.
            // Tight near/far fitting changes the normalized shadow depth every
            // frame, which makes filtered shadow edges crawl while moving.
            let near_plane = 0.1;
            let far_plane = light_distance + radius + depth_pad;

            let light_proj = Mat4::orthographic_rh(left, right, bottom, top, near_plane, far_plane);
            let shadow_matrix = light_proj * light_view;

            self.cascades[cascade_idx] = CascadeData {
                view_proj: shadow_matrix,
                split_distance: cascade_far,
            };
        }

        if active_cascade_count > 0 {
            for cascade_idx in active_cascade_count..CSM_CASCADE_COUNT {
                self.cascades[cascade_idx] = self.cascades[0];
            }
        }

        self.centers_initialized = true;
    }

    fn stable_light_direction(&mut self, sun_dir: Vec3) -> (Vec3, bool) {
        let light_z = if sun_dir.length_squared() > 1e-6 {
            sun_dir.normalize()
        } else {
            Vec3::Y
        };

        if !self.sun_direction_initialized {
            self.stable_sun_dir = light_z;
            self.sun_direction_initialized = true;
            return (self.stable_sun_dir, true);
        }

        let dot = self.stable_sun_dir.dot(light_z).clamp(-1.0, 1.0);
        let angle_delta = dot.acos();
        if angle_delta < CSM_SUN_DIRECTION_UPDATE_RADIANS {
            return (self.stable_sun_dir, false);
        }

        self.stable_sun_dir = light_z;
        (self.stable_sun_dir, true)
    }

    fn stable_snapped_center(
        &mut self,
        cascade_idx: usize,
        center: Vec3,
        light_x: Vec3,
        light_y: Vec3,
        texel_size: f32,
        force_resnap: bool,
    ) -> Vec3 {
        let center_movement = (center - self.raw_centers[cascade_idx]).length();
        if self.centers_initialized && !force_resnap && center_movement < texel_size {
            return self.snapped_centers[cascade_idx];
        }

        let snapped_center =
            snap_cascade_center_to_texel_grid(center, light_x, light_y, texel_size);
        self.raw_centers[cascade_idx] = center;
        self.snapped_centers[cascade_idx] = snapped_center;
        snapped_center
    }
}
impl Default for CsmManager {
    fn default() -> Self {
        Self::new()
    }
}

fn stable_cascade_center_depth(cascade_idx: usize, near: f32, far: f32) -> f32 {
    if cascade_idx == 0 {
        (near + far) * 0.5
    } else {
        0.0
    }
}

fn calculate_frustum_center(center_depth: f32, inv_view: &Mat4) -> Vec3 {
    let center_depth = -center_depth;
    let center = *inv_view * Vec4::new(0.0, 0.0, center_depth, 1.0);
    center.truncate() / center.w
}

fn calculate_frustum_bounding_radius(
    near: f32,
    far: f32,
    center_depth: f32,
    fov_y: f32,
    aspect: f32,
    shadow_size: f32,
) -> f32 {
    let tan_half_fov = (fov_y / 2.0).tan();

    let near_height = near * tan_half_fov;
    let near_width = near_height * aspect;
    let far_height = far * tan_half_fov;
    let far_width = far_height * aspect;

    let near_radius = Vec3::new(near_width, near_height, near - center_depth).length();
    let far_radius = Vec3::new(far_width, far_height, far - center_depth).length();
    snap_up(
        near_radius.max(far_radius).max(1.0),
        1.0 / shadow_size.max(1.0),
    )
}

fn stable_light_basis(light_z: Vec3) -> (Vec3, Vec3) {
    // Project a fixed world axis onto the light plane so the shadow texel grid
    // does not roll when the animated sun moves near vertical.
    let reference = if light_z.x.abs() < 0.95 {
        Vec3::X
    } else {
        Vec3::Z
    };
    let light_x = (reference - light_z * reference.dot(light_z)).normalize();
    let light_y = light_z.cross(light_x).normalize();
    (light_x, light_y)
}

fn snap_cascade_center_to_texel_grid(
    center: Vec3,
    light_x: Vec3,
    light_y: Vec3,
    texel_size: f32,
) -> Vec3 {
    let center_light_x = center.dot(light_x);
    let center_light_y = center.dot(light_y);
    let snapped_center_x = snap_nearest(center_light_x, texel_size);
    let snapped_center_y = snap_nearest(center_light_y, texel_size);

    center
        + light_x * (snapped_center_x - center_light_x)
        + light_y * (snapped_center_y - center_light_y)
}

fn snap_up(value: f32, step: f32) -> f32 {
    if step <= f32::EPSILON {
        value
    } else {
        (value / step).ceil() * step
    }
}

fn snap_nearest(value: f32, step: f32) -> f32 {
    if step <= f32::EPSILON {
        value
    } else {
        (value / step).round() * step
    }
}
