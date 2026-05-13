use glam::{Mat4, Vec3, Vec4};

use crate::constants::{CSM_CASCADE_COUNT, CSM_CASCADE_SPLITS, CSM_SHADOW_MAP_SIZE};
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
}

impl CsmManager {
    pub fn new() -> Self {
        Self {
            cascades: [CascadeData::default(); CSM_CASCADE_COUNT],
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

        let mut split_distances = [0.0_f32; CSM_CASCADE_COUNT + 1];
        split_distances[0] = near;

        for i in 0..CSM_CASCADE_COUNT {
            split_distances[i + 1] = CSM_CASCADE_SPLITS[i].min(far);
        }

        for cascade_idx in 0..CSM_CASCADE_COUNT {
            let cascade_near = split_distances[cascade_idx];
            let cascade_far = split_distances[cascade_idx + 1];

            let frustum_corners =
                calculate_frustum_corners(cascade_near, cascade_far, fov_y, aspect, &inv_view);

            let mut center = Vec3::ZERO;
            for corner in &frustum_corners {
                center += *corner;
            }
            center /= 8.0;

            let light_up = if sun_dir.y.abs() > 0.99 {
                Vec3::Z
            } else {
                Vec3::Y
            };

            let slice_depth = (cascade_far - cascade_near).max(1.0);
            let light_pos = center + sun_dir * (slice_depth + 256.0);
            let light_view = Mat4::look_at_rh(light_pos, center, light_up);

            let mut light_min = Vec3::splat(f32::INFINITY);
            let mut light_max = Vec3::splat(f32::NEG_INFINITY);
            for corner in &frustum_corners {
                let p = light_view * corner.extend(1.0);
                let p = p.truncate() / p.w;
                light_min = light_min.min(p);
                light_max = light_max.max(p);
            }

            // Fit the orthographic projection to the cascade's light-space AABB.
            // This preserves considerably more texels than the old bounding-sphere
            // fit while the snapping below keeps camera movement stable.
            let shadow_size = CSM_SHADOW_MAP_SIZE.max(1) as f32;
            let mut extent = light_max - light_min;
            extent.x = extent.x.max(1.0);
            extent.y = extent.y.max(1.0);

            let texel_x = extent.x / shadow_size;
            let texel_y = extent.y / shadow_size;
            let pad_x = texel_x * 4.0;
            let pad_y = texel_y * 4.0;

            let left = snap_down(light_min.x - pad_x, texel_x);
            let right = snap_up(light_max.x + pad_x, texel_x);
            let bottom = snap_down(light_min.y - pad_y, texel_y);
            let top = snap_up(light_max.y + pad_y, texel_y);

            // Add caster room in front of the camera slice along the light ray
            // so off-screen terrain can still cast into the visible cascade.
            let depth_pad = (slice_depth * 1.5).clamp(64.0, 256.0);
            let near_plane = (-light_max.z - depth_pad).max(0.1);
            let far_plane = (-light_min.z + depth_pad).max(near_plane + 1.0);

            let light_proj = Mat4::orthographic_rh(left, right, bottom, top, near_plane, far_plane);

            let shadow_matrix = light_proj * light_view;

            let opengl_to_wgpu = Mat4::from_cols_array(&[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
            ]);

            self.cascades[cascade_idx] = CascadeData {
                view_proj: opengl_to_wgpu * shadow_matrix,
                split_distance: cascade_far,
            };
        }
    }
}
impl Default for CsmManager {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_frustum_corners(
    near: f32,
    far: f32,
    fov_y: f32,
    aspect: f32,
    inv_view: &Mat4,
) -> [Vec3; 8] {
    let tan_half_fov = (fov_y / 2.0).tan();

    let near_height = near * tan_half_fov;
    let near_width = near_height * aspect;
    let far_height = far * tan_half_fov;
    let far_width = far_height * aspect;

    let corners_view = [
        // Near plane
        Vec3::new(-near_width, -near_height, -near),
        Vec3::new(near_width, -near_height, -near),
        Vec3::new(near_width, near_height, -near),
        Vec3::new(-near_width, near_height, -near),
        // Far plane
        Vec3::new(-far_width, -far_height, -far),
        Vec3::new(far_width, -far_height, -far),
        Vec3::new(far_width, far_height, -far),
        Vec3::new(-far_width, far_height, -far),
    ];

    // Transform to world space
    let mut corners_world = [Vec3::ZERO; 8];
    for (i, corner) in corners_view.iter().enumerate() {
        let world = *inv_view * Vec4::new(corner.x, corner.y, corner.z, 1.0);
        corners_world[i] = Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w);
    }

    corners_world
}

fn snap_down(value: f32, step: f32) -> f32 {
    if step <= f32::EPSILON {
        value
    } else {
        (value / step).floor() * step
    }
}

fn snap_up(value: f32, step: f32) -> f32 {
    if step <= f32::EPSILON {
        value
    } else {
        (value / step).ceil() * step
    }
}
