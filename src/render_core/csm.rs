use cgmath::{Matrix4, Point3, SquareMatrix, Vector3};

use crate::constants::{CSM_CASCADE_COUNT, CSM_CASCADE_SPLITS};
#[derive(Debug, Clone, Copy)]
pub struct CascadeData {
    pub view_proj: Matrix4<f32>,
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
        camera_view: &Matrix4<f32>,
        sun_dir: Vector3<f32>,
        near: f32,
        far: f32,
        aspect: f32,
        fov_y: f32,
    ) {
        let inv_view = camera_view.invert().unwrap_or(Matrix4::identity());

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

            let mut center = Point3::new(0.0, 0.0, 0.0);
            for corner in &frustum_corners {
                center.x += corner.x;
                center.y += corner.y;
                center.z += corner.z;
            }
            center.x /= 8.0;
            center.y /= 8.0;
            center.z /= 8.0;

            let mut radius = 0.0_f32;
            for corner in &frustum_corners {
                let dist = ((corner.x - center.x).powi(2)
                    + (corner.y - center.y).powi(2)
                    + (corner.z - center.z).powi(2))
                    .sqrt();
                radius = radius.max(dist);
            }

            let sun_distance = radius * 2.0;
            let light_pos = Point3::new(
                center.x + sun_dir.x * sun_distance,
                center.y + sun_dir.y * sun_distance,
                center.z + sun_dir.z * sun_distance,
            );

            let light_up = if sun_dir.y.abs() > 0.99 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };

            let light_view = Matrix4::look_at_rh(light_pos, center, light_up);

            let light_proj = cgmath::ortho(
                -radius,
                radius,
                -radius,
                radius,
                0.1,
                sun_distance * 2.0 + radius,
            );

            let shadow_matrix = light_proj * light_view;
            let shadow_matrix = snap_to_texel_grid(
                shadow_matrix,
                center,
                crate::constants::CSM_SHADOW_MAP_SIZE as f32,
            );

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

/// Snaps the shadow matrix so that the projection of `world_center` lands on
/// an exact texel boundary.  This eliminates sub-texel crawling of the shadow
/// map when the camera moves, because the frustum is always anchored to the
/// same texel grid (GPU Gems 3 – "Stable Cascaded Shadow Maps").
fn snap_to_texel_grid(
    matrix: Matrix4<f32>,
    world_center: Point3<f32>,
    shadow_map_size: f32,
) -> Matrix4<f32> {
    if shadow_map_size <= 1.0 {
        return matrix;
    }

    // Project the cascade center into shadow clip space, then shift the
    // matrix so the center lands on the nearest texel boundary. This keeps
    // the shadow map stable as the camera moves.
    let center_clip =
        matrix * cgmath::Vector4::new(world_center.x, world_center.y, world_center.z, 1.0);
    let inv_w = if center_clip.w.abs() > f32::EPSILON {
        1.0 / center_clip.w
    } else {
        1.0
    };
    let center_ndc_x = center_clip.x * inv_w;
    let center_ndc_y = center_clip.y * inv_w;

    let texel_ndc = 2.0 / shadow_map_size;
    let snapped_ndc_x = (center_ndc_x / texel_ndc).round() * texel_ndc;
    let snapped_ndc_y = (center_ndc_y / texel_ndc).round() * texel_ndc;

    let delta_x = snapped_ndc_x - center_ndc_x;
    let delta_y = snapped_ndc_y - center_ndc_y;

    Matrix4::from_translation(Vector3::new(delta_x, delta_y, 0.0)) * matrix
}
