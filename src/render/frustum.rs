use glam::{Mat4, Vec3, Vec4};

/// An axis-aligned bounding box defined by its minimum and maximum corners.
///
/// Used for frustum culling to quickly reject geometry that lies entirely
/// outside the view frustum without inspecting individual vertices.
#[derive(Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Creates a new `AABB` from explicit minimum and maximum corners.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        AABB { min, max }
    }

    /// Tests whether this AABB intersects or lies inside the given view frustum.
    ///
    /// The AABB is expanded by a small `margin` on all sides before testing to
    /// avoid popping artifacts at frustum edges caused by floating-point
    /// imprecision or geometry that slightly overhangs its bounding box.
    ///
    /// The test uses the *positive-vertex* method: for each frustum plane the
    /// corner of the (expanded) box that is furthest along the plane normal is
    /// chosen as the representative point.  If that point lies on the negative
    /// side of any plane the entire box is outside the frustum.
    ///
    /// # Arguments
    /// * `frustum_planes` – Six normalized frustum planes in world space
    ///   (left, right, bottom, top, near, far), each stored as `(nx, ny, nz, d)`
    ///   where the plane equation is `n·p + d ≥ 0` for points inside.
    ///
    /// # Returns
    /// `true` if the AABB is potentially visible; `false` if it is definitely
    /// outside the frustum and can be safely culled.
    pub fn is_visible(&self, frustum_planes: &[Vec4; 6]) -> bool {
        let margin = 2.0;
        let expanded_min = Vec3::new(
            self.min.x - margin,
            self.min.y - margin,
            self.min.z - margin,
        );
        let expanded_max = Vec3::new(
            self.max.x + margin,
            self.max.y + margin,
            self.max.z + margin,
        );

        for plane in frustum_planes {
            let p = Vec3::new(
                if plane.x > 0.0 { expanded_max.x } else { expanded_min.x },
                if plane.y > 0.0 { expanded_max.y } else { expanded_min.y },
                if plane.z > 0.0 { expanded_max.z } else { expanded_min.z },
            );
            if plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

/// Extracts and normalizes the six frustum planes from a combined view-projection matrix.
///
/// Planes are derived using Grib & Hartmann's method of combining rows of the
/// clip-space matrix.  After extraction each plane is divided by the magnitude
/// of its normal so that the `w` component represents the true signed distance
/// from the origin to the plane, enabling accurate distance comparisons.
///
/// The resulting planes follow the convention `n·p + d ≥ 0` for points on the
/// *inside* of the frustum, where `(nx, ny, nz)` is the inward-facing normal
/// and `d` is stored in the `w` component.
///
/// # Arguments
/// * `view_proj` – The combined view-projection matrix (column-major, as glam
///   stores it).  The matrix must use a left-handed clip space (z ∈ [0, 1]),
///   which matches wgpu / Vulkan conventions.
///
/// # Returns
/// An array of six normalized planes in the order:
/// `[left, right, bottom, top, near, far]`.
pub fn extract_frustum_planes(view_proj: &Mat4) -> [Vec4; 6] {
    let m = view_proj;
    let row0 = m.row(0);
    let row1 = m.row(1);
    let row2 = m.row(2);
    let row3 = m.row(3);

    let mut planes = [
        row3 + row0, // Left
        row3 - row0, // Right
        row3 + row1, // Bottom
        row3 - row1, // Top
        row2,        // Near (Z in [0, 1])
        row3 - row2, // Far
    ];

    for plane in &mut planes {
        let length = plane.truncate().length();
        *plane /= length;
    }

    planes
}