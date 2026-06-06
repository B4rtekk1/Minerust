use glam::{Mat4, Vec3};

use crate::{
    DEFAULT_FOV, SHADOW_CASCADE_COUNT, SHADOW_DISTANCE, SHADOW_MAP_SIZE, SHADOW_PCF_RADIUS_TEXELS,
    ShadowCascadeUniform, ShadowUniforms,
};

const CASCADE_SPLIT_LAMBDA: f32 = 0.62;
const SHADOW_CASTER_MARGIN: f32 = 96.0;

pub struct ShadowFrameData {
    pub uniforms: ShadowUniforms,
    pub cascade_uniforms: [ShadowCascadeUniform; SHADOW_CASCADE_COUNT],
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn cascade_splits(near_plane: f32, far_plane: f32) -> [f32; SHADOW_CASCADE_COUNT] {
    std::array::from_fn(|i| {
        let p = (i + 1) as f32 / SHADOW_CASCADE_COUNT as f32;
        let log = near_plane * (far_plane / near_plane).powf(p);
        let uniform = near_plane + (far_plane - near_plane) * p;
        if i + 1 == SHADOW_CASCADE_COUNT {
            far_plane
        } else {
            log * CASCADE_SPLIT_LAMBDA + uniform * (1.0 - CASCADE_SPLIT_LAMBDA)
        }
    })
}

fn frustum_corners_world(
    inv_view: Mat4,
    fov_y: f32,
    aspect: f32,
    near_plane: f32,
    far_plane: f32,
) -> [Vec3; 8] {
    let tan_half_fov = (fov_y * 0.5).tan();
    let near_h = tan_half_fov * near_plane;
    let near_w = near_h * aspect;
    let far_h = tan_half_fov * far_plane;
    let far_w = far_h * aspect;

    let corners = [
        Vec3::new(-near_w, -near_h, -near_plane),
        Vec3::new(near_w, -near_h, -near_plane),
        Vec3::new(-near_w, near_h, -near_plane),
        Vec3::new(near_w, near_h, -near_plane),
        Vec3::new(-far_w, -far_h, -far_plane),
        Vec3::new(far_w, -far_h, -far_plane),
        Vec3::new(-far_w, far_h, -far_plane),
        Vec3::new(far_w, far_h, -far_plane),
    ];

    corners.map(|corner| inv_view.transform_point3(corner))
}

fn light_basis(sun_dir: Vec3) -> (Vec3, Vec3, Vec3, Vec3) {
    let z_axis = sun_dir.normalize();
    let up = if z_axis.dot(Vec3::Y).abs() > 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let x_axis = up.cross(z_axis).normalize();
    let y_axis = z_axis.cross(x_axis).normalize();
    (x_axis, y_axis, z_axis, up)
}

#[inline]
fn to_light_space(point: Vec3, x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Vec3 {
    Vec3::new(point.dot(x_axis), point.dot(y_axis), point.dot(z_axis))
}

fn cascade_matrix(
    inv_view: Mat4,
    aspect: f32,
    cascade_near: f32,
    cascade_far: f32,
    sun_dir: Vec3,
    shadow_map_size: u32,
) -> Mat4 {
    let corners = frustum_corners_world(inv_view, DEFAULT_FOV, aspect, cascade_near, cascade_far);
    let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
    let radius = corners
        .iter()
        .map(|corner| corner.distance(center))
        .fold(0.0f32, f32::max);
    let radius = (radius * 16.0).ceil() / 16.0;

    let (x_axis, y_axis, z_axis, up) = light_basis(sun_dir);

    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for corner in corners {
        let light_space = to_light_space(corner, x_axis, y_axis, z_axis);
        min_z = min_z.min(light_space.z);
        max_z = max_z.max(light_space.z);
    }

    let center_ls = to_light_space(center, x_axis, y_axis, z_axis);
    let world_units_per_texel = (radius * 2.0) / shadow_map_size as f32;
    let snapped_x = (center_ls.x / world_units_per_texel).floor() * world_units_per_texel;
    let snapped_y = (center_ls.y / world_units_per_texel).floor() * world_units_per_texel;

    let z_margin = SHADOW_CASTER_MARGIN + radius * 0.25;
    let eye_ls_z = max_z + z_margin;
    let eye = x_axis * snapped_x + y_axis * snapped_y + z_axis * eye_ls_z;
    let light_view = Mat4::look_at_rh(eye, eye - sun_dir, up);
    let light_proj = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        0.1,
        (eye_ls_z - (min_z - z_margin)).max(1.0),
    );

    light_proj * light_view
}

pub fn build_shadow_frame_data(
    view: Mat4,
    aspect: f32,
    camera_near: f32,
    camera_far: f32,
    camera_forward: Vec3,
    sun_dir: Vec3,
) -> ShadowFrameData {
    let inv_view = view.inverse();
    let shadow_distance = SHADOW_DISTANCE.min(camera_far).max(32.0);
    let splits = cascade_splits(camera_near.max(0.05), shadow_distance);

    let mut last_split = camera_near.max(0.05);
    let matrices = std::array::from_fn(|i| {
        let split = splits[i];
        let matrix = cascade_matrix(
            inv_view,
            aspect,
            last_split,
            split,
            sun_dir,
            SHADOW_MAP_SIZE,
        );
        last_split = split;
        matrix
    });

    let shadow_strength = smoothstep(0.03, 0.22, sun_dir.y);
    let light_view_proj = matrices.map(|matrix| matrix.to_cols_array_2d());
    let cascade_uniforms =
        light_view_proj.map(|light_view_proj| ShadowCascadeUniform { light_view_proj });

    ShadowFrameData {
        uniforms: ShadowUniforms {
            light_view_proj,
            cascade_splits: splits,
            camera_forward: camera_forward.normalize_or_zero().to_array(),
            shadow_strength,
            params: [
                SHADOW_MAP_SIZE as f32,
                SHADOW_CASCADE_COUNT as f32,
                SHADOW_PCF_RADIUS_TEXELS,
                shadow_distance,
            ],
        },
        cascade_uniforms,
    }
}

pub fn create_shadow_texture(
    device: &wgpu::Device,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    Vec<wgpu::TextureView>,
    wgpu::Sampler,
) {
    let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("CSM Shadow Texture"),
        size: wgpu::Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: SHADOW_CASCADE_COUNT as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("CSM Shadow Texture Array View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });

    let cascade_views = (0..SHADOW_CASCADE_COUNT)
        .map(|cascade| {
            shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("CSM Shadow Cascade View {}", cascade)),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: cascade as u32,
                array_layer_count: Some(1),
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            })
        })
        .collect();

    let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("CSM Shadow Compare Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });

    (shadow_texture, shadow_view, cascade_views, shadow_sampler)
}

pub fn shadow_mask_size(width: u32, height: u32) -> [u32; 2] {
    [width.div_ceil(2).max(1), height.div_ceil(2).max(1)]
}

pub fn create_shadow_mask_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, [u32; 2]) {
    let size = shadow_mask_size(width, height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Half Resolution Shadow Mask"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    (texture, view, size)
}
