//! Cascaded Shadow Maps (CSM) implementation
//!
//! CSM divides the view frustum into multiple cascades, each with its own
//! shadow map. Near cascades get higher effective resolution while far
//! cascades cover larger areas at lower resolution.

use cgmath::{Matrix4, Point3, SquareMatrix, Vector3};

use crate::constants::{CSM_CASCADE_COUNT, CSM_CASCADE_SPLITS};
/// Data for a single cascade of the shadow map
#[derive(Debug, Clone, Copy)]
pub struct CascadeData {
    /// View-projection matrix for this cascade (from sun's perspective)
    pub view_proj: Matrix4<f32>,
    /// Split distance (far plane) for this cascade
    pub split_distance: f32,
}

impl Default for CascadeData {
    fn default() -> Self {
        Self {
            view_proj: Matrix4::identity(),
            split_distance: 0.0,
        }
    }
}

/// CSM manager that handles cascade calculations
pub struct CsmManager {
    pub cascades: [CascadeData; CSM_CASCADE_COUNT],
}

impl CsmManager {
    pub fn new() -> Self {
        Self {
            cascades: [CascadeData::default(); CSM_CASCADE_COUNT],
        }
    }

    /// Calculate view-projection matrices for all cascades
    ///
    /// # Arguments
    /// * `camera_pos` - Camera world position
    /// * `camera_view` - Camera view matrix
    /// * `camera_proj` - Camera projection matrix
    /// * `sun_dir` - Normalized direction towards the sun
    /// * `near` - Camera near plane
    /// * `far` - Camera far plane (typically render distance)
    /// * `aspect` - Camera aspect ratio
    /// * `fov_y` - Camera vertical field of view in radians
    pub fn update(
        &mut self,
        camera_view: &Matrix4<f32>,
        sun_dir: Vector3<f32>,
        near: f32,
        far: f32,
        aspect: f32,
        fov_y: f32,
    ) {
        let inv_view = camera_view.invert().unwrap_or(Matrix4::identity());

        // Calculate cascade split distances using practical split scheme
        let mut split_distances = [0.0_f32; CSM_CASCADE_COUNT + 1];
        split_distances[0] = near;

        for i in 0..CSM_CASCADE_COUNT {
            // Use predefined splits, clamped to actual far plane
            split_distances[i + 1] = CSM_CASCADE_SPLITS[i].min(far);
        }

        for cascade_idx in 0..CSM_CASCADE_COUNT {
            let cascade_near = split_distances[cascade_idx];
            let cascade_far = split_distances[cascade_idx + 1];

            // Calculate frustum corners for this cascade slice
            let frustum_corners =
                calculate_frustum_corners(cascade_near, cascade_far, fov_y, aspect, &inv_view);

            // Calculate the center of the frustum slice
            let mut center = Point3::new(0.0, 0.0, 0.0);
            for corner in &frustum_corners {
                center.x += corner.x;
                center.y += corner.y;
                center.z += corner.z;
            }
            center.x /= 8.0;
            center.y /= 8.0;
            center.z /= 8.0;

            // Calculate the radius of the bounding sphere
            let mut radius = 0.0_f32;
            for corner in &frustum_corners {
                let dist = ((corner.x - center.x).powi(2)
                    + (corner.y - center.y).powi(2)
                    + (corner.z - center.z).powi(2))
                .sqrt();
                radius = radius.max(dist);
            }

            // Round up radius to reduce shadow edge flickering
            radius = (radius * 16.0).ceil() / 16.0;

            // Calculate light view matrix looking at frustum center from sun direction
            let sun_distance = radius * 2.0;
            let light_pos = Point3::new(
                center.x + sun_dir.x * sun_distance,
                center.y + sun_dir.y * sun_distance,
                center.z + sun_dir.z * sun_distance,
            );

            // Use stable up vector
            let light_up = if sun_dir.y.abs() > 0.99 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };

            let light_view = Matrix4::look_at_rh(light_pos, center, light_up);

            // Orthographic projection that encompasses the frustum slice
            let light_proj = cgmath::ortho(
                -radius,
                radius,
                -radius,
                radius,
                0.1,
                sun_distance * 2.0 + radius,
            );

            // Snap to texel grid to reduce shadow edge shimmer
            let shadow_matrix = light_proj * light_view;
            let shadow_matrix =
                snap_to_texel_grid(shadow_matrix, crate::constants::CSM_SHADOW_MAP_SIZE as f32);

            // Apply OpenGL to WGPU matrix correction
            let opengl_to_wgpu = Matrix4::new(
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
            );

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

/// Calculate the 8 corners of a frustum slice in world space
fn calculate_frustum_corners(
    near: f32,
    far: f32,
    fov_y: f32,
    aspect: f32,
    inv_view: &Matrix4<f32>,
) -> [Point3<f32>; 8] {
    let tan_half_fov = (fov_y / 2.0).tan();

    let near_height = near * tan_half_fov;
    let near_width = near_height * aspect;
    let far_height = far * tan_half_fov;
    let far_width = far_height * aspect;

    // Corners in view space (camera looking down -Z)
    let corners_view = [
        // Near plane
        Point3::new(-near_width, -near_height, -near),
        Point3::new(near_width, -near_height, -near),
        Point3::new(near_width, near_height, -near),
        Point3::new(-near_width, near_height, -near),
        // Far plane
        Point3::new(-far_width, -far_height, -far),
        Point3::new(far_width, -far_height, -far),
        Point3::new(far_width, far_height, -far),
        Point3::new(-far_width, far_height, -far),
    ];

    // Transform to world space
    let mut corners_world = [Point3::new(0.0, 0.0, 0.0); 8];
    for (i, corner) in corners_view.iter().enumerate() {
        let world = inv_view * cgmath::Vector4::new(corner.x, corner.y, corner.z, 1.0);
        corners_world[i] = Point3::new(world.x / world.w, world.y / world.w, world.z / world.w);
    }

    corners_world
}

/// Snap shadow matrix to texel grid to reduce shadow edge shimmer during camera movement
/// Uses round() for symmetric snapping — floor() is asymmetric around zero and causes a
/// 1-texel discontinuity when the translation crosses zero, which manifests as a visible
/// one-axis shadow jump as the camera moves.
fn snap_to_texel_grid(matrix: Matrix4<f32>, shadow_map_size: f32) -> Matrix4<f32> {
    // Texel size in NDC space (the shadow map covers -1 to 1 range, so 2.0 total)
    let texel_size = 2.0 / shadow_map_size;

    // Project the world origin through the matrix to get the current translation offset
    // in clip space, then snap both X and Y to the nearest texel boundary.
    // round() is symmetric around every integer, so no discontinuity when w crosses zero.
    let mut result = matrix;

    result.w.x = (result.w.x / texel_size).round() * texel_size;
    result.w.y = (result.w.y / texel_size).round() * texel_size;

    result
}
