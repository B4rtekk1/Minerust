/// Hi-Z starts at the former mip 1. Floor division matches texture mip sizing;
/// skinny surfaces still have at least one texel on each axis.
pub fn hiz_base_size(surface_size: [u32; 2]) -> [u32; 2] {
    surface_size.map(|size| (size / 2).max(1))
}

#[cfg(test)]
mod tests {
    #[test]
    fn depth_shaders_parse_and_validate() {
        for source in [
            include_str!("../shaders/depth_resolve.wgsl"),
            include_str!("../shaders/hiz.wgsl"),
        ] {
            let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [16, 16, 1]);
        }
    }
}
