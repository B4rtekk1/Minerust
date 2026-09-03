// Cascaded voxel DDGI.  The world is a camera-centred R8Uint 3-D clipmap;
// block traversal uses Amanatides--Woo DDA, never fixed-size ray marching.
const VOXEL_SIDE: i32 = 256;
const WORLD_HEIGHT: i32 = 256;
const PROBES_X: i32 = 24;
const PROBES_Y: i32 = 16;
const PROBES_Z: i32 = 24;
const PROBES_PER_CASCADE: u32 = 9216u;
const RAYS_PER_UPDATE: u32 = 8u;

struct DdgiConfig {
    // xyz: world position of probe [0, 0, 0], w: spacing in blocks.
    near_origin_spacing: vec4<f32>,
    far_origin_spacing: vec4<f32>,
    // xyz: voxel clipmap world-space minimum, w: monotonically increasing frame.
    voxel_origin_frame: vec4<f32>,
    // xyz: direction towards the sun.
    sun_dir: vec4<f32>,
};

// Six irradiance lobes plus distance mean/variance. vec4 keeps storage layout
// portable across all wgpu backends; alpha channels remain reserved.
struct Probe {
    px: vec4<f32>, nx: vec4<f32>, py: vec4<f32>, ny: vec4<f32>, pz: vec4<f32>, nz: vec4<f32>,
    distance: vec4<f32>,
};

@group(0) @binding(0) var<uniform> config: DdgiConfig;
@group(1) @binding(0) var voxels: texture_3d<u32>;
@group(2) @binding(0) var<storage, read> history: array<Probe>;
@group(3) @binding(0) var<storage, read_write> output: array<Probe>;

fn block_albedo(id: u32) -> vec3<f32> {
    switch id {
        case 1u { return vec3(0.36, 0.70, 0.28); } // grass
        case 2u { return vec3(0.52, 0.37, 0.26); }
        case 3u { return vec3(0.55); }
        case 4u { return vec3(0.89, 0.83, 0.61); }
        case 6u, 16u, 17u { return vec3(0.60, 0.40, 0.20); }
        case 7u { return vec3(0.30, 0.60, 0.20); }
        case 8u { return vec3(0.20); }
        case 9u { return vec3(0.95, 0.95, 0.98); }
        case 10u { return vec3(0.50, 0.50, 0.52); }
        case 11u { return vec3(0.65, 0.65, 0.72); }
        case 12u { return vec3(0.70, 0.85, 0.95); }
        case 13u { return vec3(0.20, 0.55, 0.20); }
        default { return vec3(0.55); }
    }
}

fn solid_for_gi(id: u32) -> bool {
    // Water, foliage and dead bushes deliberately transmit diffuse light.
    return id != 0u && id != 5u && id != 7u && id != 12u && id != 14u;
}

fn hash01(v: u32) -> f32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) * (1.0 / 4294967295.0);
}

fn ray_direction(ray: u32, seed: u32) -> vec3<f32> {
    let u = (f32(ray) + hash01(seed)) / f32(RAYS_PER_UPDATE);
    let z = 1.0 - 2.0 * u;
    let a = 6.28318530718 * fract(hash01(seed + ray * 17u) + f32(ray) * 0.61803398875);
    return vec3(cos(a) * sqrt(max(0.0, 1.0 - z * z)), z, sin(a) * sqrt(max(0.0, 1.0 - z * z)));
}

fn sky_radiance(dir: vec3<f32>) -> vec3<f32> {
    let day = clamp(config.sun_dir.y, 0.0, 1.0);
    let horizon = 0.35 + 0.65 * max(dir.y, 0.0);
    return mix(vec3(0.012, 0.014, 0.022), vec3(0.19, 0.29, 0.48) * horizon, day);
}

// Returns radiance.xyz and travelled distance.w.  The hit face is inferred
// from the last DDA axis for a cheap direct-sun approximation.
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> vec4<f32> {
    var cell = vec3<i32>(floor(origin));
    let step = vec3<i32>(select(-1, 1, direction.x >= 0.0), select(-1, 1, direction.y >= 0.0), select(-1, 1, direction.z >= 0.0));
    let inv_dir = 1.0 / max(abs(direction), vec3(1e-6));
    var t_delta = inv_dir;
    var boundary = vec3<f32>(f32(cell.x + select(0, 1, direction.x >= 0.0)), f32(cell.y + select(0, 1, direction.y >= 0.0)), f32(cell.z + select(0, 1, direction.z >= 0.0)));
    var t_max = abs((boundary - origin) / direction);
    var last_axis = 1u;
    var t = 0.0;

    for (var i = 0u; i < 192u; i++) {
        if t > max_distance { break; }
        let local = cell - vec3<i32>(config.voxel_origin_frame.xyz);
        if all(local >= vec3<i32>(0)) && local.x < VOXEL_SIDE && local.y < WORLD_HEIGHT && local.z < VOXEL_SIDE {
            let id = textureLoad(voxels, local, 0).x;
            if solid_for_gi(id) {
                var normal = vec3(0.0);
                if last_axis == 0u { normal.x = -f32(step.x); }
                if last_axis == 1u { normal.y = -f32(step.y); }
                if last_axis == 2u { normal.z = -f32(step.z); }
                let direct = max(dot(normal, config.sun_dir.xyz), 0.0) * clamp(config.sun_dir.y * 1.4, 0.0, 1.0);
                let sky = 0.12 + 0.18 * max(normal.y, 0.0);
                return vec4(block_albedo(id) * (vec3(0.10, 0.14, 0.20) * sky + vec3(1.0, 0.92, 0.76) * direct), t);
            }
        }
        if t_max.x < t_max.y && t_max.x < t_max.z {
            t = t_max.x; t_max.x += t_delta.x; cell.x += step.x; last_axis = 0u;
        } else if t_max.y < t_max.z {
            t = t_max.y; t_max.y += t_delta.y; cell.y += step.y; last_axis = 1u;
        } else {
            t = t_max.z; t_max.z += t_delta.z; cell.z += step.z; last_axis = 2u;
        }
    }
    return vec4(sky_radiance(direction), max_distance);
}

fn blend(old: vec4<f32>, fresh: vec3<f32>, h: f32) -> vec4<f32> {
    return vec4(mix(fresh, old.xyz, h), 0.0);
}

@compute @workgroup_size(64)
fn cs_update(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= PROBES_PER_CASCADE * 2u { return; }
    let cascade = index / PROBES_PER_CASCADE;
    let local_index = index % PROBES_PER_CASCADE;
    let x = i32(local_index % u32(PROBES_X));
    let y = i32((local_index / u32(PROBES_X)) % u32(PROBES_Y));
    let z = i32(local_index / u32(PROBES_X * PROBES_Y));
    let base = select(config.near_origin_spacing, config.far_origin_spacing, cascade == 1u);
    let probe_pos = base.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * base.w;
    let frame = u32(config.voxel_origin_frame.w);
    let old = history[index];
    if ((local_index + frame) & 7u) != 0u {
        output[index] = old;
        return;
    }

    var px = vec3(0.0); var nx = vec3(0.0); var py = vec3(0.0); var ny = vec3(0.0); var pz = vec3(0.0); var nz = vec3(0.0);
    var distance_sum = 0.0; var distance2_sum = 0.0;
    let seed = index * 31u + frame * 131u;
    let max_distance = select(36.0, 112.0, cascade == 1u);
    for (var r = 0u; r < RAYS_PER_UPDATE; r++) {
        let dir = ray_direction(r, seed);
        let result = trace_ray(probe_pos + dir * 0.05, dir, max_distance);
        px += result.xyz * max(dir.x, 0.0); nx += result.xyz * max(-dir.x, 0.0);
        py += result.xyz * max(dir.y, 0.0); ny += result.xyz * max(-dir.y, 0.0);
        pz += result.xyz * max(dir.z, 0.0); nz += result.xyz * max(-dir.z, 0.0);
        distance_sum += result.w; distance2_sum += result.w * result.w;
    }
    let normalization = 2.0 / f32(RAYS_PER_UPDATE);
    // Lower history weight after an origin reset is handled by clearing probe buffers on CPU.
    let h = 0.92;
    var next: Probe;
    next.px = blend(old.px, px * normalization, h); next.nx = blend(old.nx, nx * normalization, h);
    next.py = blend(old.py, py * normalization, h); next.ny = blend(old.ny, ny * normalization, h);
    next.pz = blend(old.pz, pz * normalization, h); next.nz = blend(old.nz, nz * normalization, h);
    let mean = distance_sum / f32(RAYS_PER_UPDATE);
    next.distance = vec4(mix(mean, old.distance.x, h), mix(distance2_sum / f32(RAYS_PER_UPDATE), old.distance.y, h), 0.0, 0.0);
    output[index] = next;
}
