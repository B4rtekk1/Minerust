    /// GPU Frustum + Hi-Z Occlusion Culling Compute Shader (v2)
    ///
    /// Performs frustum culling and Hi-Z occlusion culling on the GPU
    /// for all subchunks in parallel.

    struct SubchunkMeta {
        /// AABB min (xyz), padding in w
        aabb_min: vec4<f32>,
        /// AABB max (xyz), slot_index in w
        aabb_max: vec4<f32>,
        /// draw_data: index_count, first_index, base_vertex, enabled
        draw_data: vec4<u32>,
    }

    struct DrawIndexedIndirect {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    }

    struct CullUniforms {
        /// View-projection matrix for AABB projection
        view_proj: mat4x4<f32>,
        /// 6 frustum planes (each is vec4: xyz=normal, w=distance)
        frustum_planes: array<vec4<f32>, 6>,
        /// Camera position
        camera_pos: vec3<f32>,
        /// Number of active subchunks
        subchunk_count: u32,
        /// Hi-Z texture size (mip 0 dimensions)
        hiz_size: vec2<f32>,
        /// Screen size for UV scaling
        screen_size: vec2<f32>,
    }

    @group(0) @binding(0)
    var<uniform> cull_uniforms: CullUniforms;

    @group(0) @binding(1)
    var<storage, read> subchunks: array<SubchunkMeta>;

    @group(0) @binding(2)
    var<storage, read_write> draw_commands: array<DrawIndexedIndirect>;

    @group(0) @binding(3)
    var<storage, read_write> visible_count: atomic<u32>;

    @group(0) @binding(4)
    var hiz_texture: texture_2d<f32>;

    @group(0) @binding(5)
    var hiz_sampler: sampler;

    // ---------------------------------------------------------------------------
    // Frustum culling
    // ---------------------------------------------------------------------------

    fn aabb_vs_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: vec4<f32>) -> bool {
        let p = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.x > 0.0),
            select(aabb_min.y, aabb_max.y, plane.y > 0.0),
            select(aabb_min.z, aabb_max.z, plane.z > 0.0),
        );
        return dot(plane.xyz, p) + plane.w >= 0.0;
    }

    fn is_frustum_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
        for (var i = 0u; i < 6u; i++) {
            if !aabb_vs_plane(aabb_min, aabb_max, cull_uniforms.frustum_planes[i]) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------------
    // Hi-Z occlusion culling (v2)
    //
    // Key corrections vs v1:
    //  1. Mip selection uses CEIL(log2(max_pixel_dim)) so the chosen mip level
    //     covers the whole projected rectangle in ≤ 2×2 texels — the classic
    //     "two-level" approach.  v1 used floor(log2) which could under-select
    //     and miss occluders, causing false rejections.
    //  2. We sample the 4 texels that straddle the projected rect's corners on
    //     the chosen mip via textureLoad (integer coords), not textureSampleLevel
    //     which adds interpolation error on non-filtered textures.
    //  3. Shadow passes pass hiz_size=(0,0) → early out.
    //  4. Removed erroneous UV scaling by (screen_size / hiz_size) — UV [0,1]
    //     maps directly to the Hi-Z texture regardless of screen aspect ratio
    //     because the Hi-Z mip 0 was already written from the depth pass which
    //     used full-screen NDC coords.
    // ---------------------------------------------------------------------------

    fn is_occlusion_visible(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
        // Shadow passes skip Hi-Z.
        if cull_uniforms.hiz_size.x < 1.0 {
            return true;
        }

        var min_uv  = vec2<f32>(1.0, 1.0);
        var max_uv  = vec2<f32>(0.0, 0.0);
        var min_z   = 1.0f;
        var any_behind = false;

        // Project all 8 AABB corners.
        let corners = array<vec3<f32>, 8>(
            vec3<f32>(aabb_min.x, aabb_min.y, aabb_min.z),
            vec3<f32>(aabb_max.x, aabb_min.y, aabb_min.z),
            vec3<f32>(aabb_min.x, aabb_max.y, aabb_min.z),
            vec3<f32>(aabb_max.x, aabb_max.y, aabb_min.z),
            vec3<f32>(aabb_min.x, aabb_min.y, aabb_max.z),
            vec3<f32>(aabb_max.x, aabb_min.y, aabb_max.z),
            vec3<f32>(aabb_min.x, aabb_max.y, aabb_max.z),
            vec3<f32>(aabb_max.x, aabb_max.y, aabb_max.z),
        );

        for (var c = 0u; c < 8u; c++) {
            let clip = cull_uniforms.view_proj * vec4<f32>(corners[c], 1.0);
            if clip.w <= 0.0 {
                any_behind = true;
            } else {
                let ndc = clip.xyz / clip.w;
                // NDC [-1,1] → UV [0,1], flip Y (NDC +Y = up, UV +Y = down).
                let uv  = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
                min_uv  = min(min_uv, uv);
                max_uv  = max(max_uv, uv);
                // min_z = nearest depth (smallest NDC z = closest to camera).
                min_z   = min(min_z, ndc.z);
            }
        }

        // Any corner behind near plane → straddles clip boundary → conservative visible.
        if any_behind { return true; }

        // Fully off-screen → frustum culling should have handled this already.
        if max_uv.x <= 0.0 || min_uv.x >= 1.0 || max_uv.y <= 0.0 || min_uv.y >= 1.0 {
            return false;
        }

        // Clamp to [0,1].
        let uv_lo = clamp(min_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let uv_hi = clamp(max_uv, vec2<f32>(0.0), vec2<f32>(1.0));

        // --- Mip selection (v2: ceil log2) ---
        //
        // We want the coarsest mip where the projected rectangle fits in ≤ 2×2
        // texels.  That is: mip = ceil( log2( max_pixel_dim ) ).
        // This guarantees a single 2×2 textureLoad covers the whole AABB
        // projection → conservative & fast.
        let max_mip_f   = f32(textureNumLevels(hiz_texture) - 1u);
        let pixel_dim   = (uv_hi - uv_lo) * cull_uniforms.hiz_size;
        let max_dim     = max(pixel_dim.x, pixel_dim.y);

        // ceil(log2(x)) for x >= 1: log2(x-epsilon) rounded up.
        // For very small objects (< 1 px) clamp to mip 0 to avoid log2(0).
        let mip_f = select(ceil(log2(max(max_dim, 1.0))), 0.0, max_dim < 1.0);
        let mip   = u32(clamp(mip_f, 0.0, max_mip_f));

        // --- Sample 4 texels straddling the rectangle corners at chosen mip ---
        //
        // Convert UV to integer texel coords at the selected mip level.
        let mip_size = vec2<f32>(textureDimensions(hiz_texture, mip));
        let lo_px    = vec2<i32>(uv_lo * mip_size);
        let hi_px    = vec2<i32>(uv_hi * mip_size);
        let mip_max  = vec2<i32>(mip_size) - vec2<i32>(1);

        let t00 = textureLoad(hiz_texture, clamp(lo_px,                         vec2<i32>(0), mip_max), i32(mip)).r;
        let t10 = textureLoad(hiz_texture, clamp(vec2<i32>(hi_px.x, lo_px.y),  vec2<i32>(0), mip_max), i32(mip)).r;
        let t01 = textureLoad(hiz_texture, clamp(vec2<i32>(lo_px.x, hi_px.y),  vec2<i32>(0), mip_max), i32(mip)).r;
        let t11 = textureLoad(hiz_texture, clamp(hi_px,                         vec2<i32>(0), mip_max), i32(mip)).r;

        // Hi-Z stores furthest (max) depth → occluder depth is the max of all 4.
        let occluder_z = max(max(t00, t10), max(t01, t11));

        // Skip culling if Hi-Z buffer is uninitialised / all-zero.
        if occluder_z <= 0.00001 {
            return true;
        }

        // AABB is occluded if its nearest point is BEHIND all occluders.
        // Small epsilon guards against self-occlusion at mip seams.
        let nearest_z = clamp(min_z, 0.0, 1.0);
        return nearest_z <= occluder_z + 0.0001;
    }

    // ---------------------------------------------------------------------------
    // Main dispatch
    // ---------------------------------------------------------------------------

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;

        if idx >= cull_uniforms.subchunk_count {
            return;
        }

        let subchunk = subchunks[idx];

        if subchunk.draw_data.w == 0u {
            return;
        }

        let aabb_min = subchunk.aabb_min.xyz;
        let aabb_max = subchunk.aabb_max.xyz;

        // 1. Frustum test (cheap).
        if !is_frustum_visible(aabb_min, aabb_max) {
            return;
        }

        // 2. Hi-Z occlusion test.
        if !is_occlusion_visible(aabb_min, aabb_max) {
            return;
        }

        // Write draw command.
        let slot = atomicAdd(&visible_count, 1u);
        draw_commands[slot].index_count    = subchunk.draw_data.x;
        draw_commands[slot].instance_count = 1u;
        draw_commands[slot].first_index    = subchunk.draw_data.y;
        draw_commands[slot].base_vertex    = i32(subchunk.draw_data.z);
        draw_commands[slot].first_instance = 0u;
    }
