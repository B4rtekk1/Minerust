@group(0) @binding(0) var<storage, read> ray_count: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1) fn main() { args[0]=(atomicLoad(&ray_count)+63u)/64u; args[1]=1u; args[2]=1u; }
