//! Measures subchunk snapshot + mesh time on generated terrain.
//! `cargo run --release --example mesh_bench -- [seed]`
fn main() {
    let seed = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12345);
    let mut world = minerust::World::new_empty_with_seed(seed);
    let (x, z) = minerust::ChunkGenerator::new(seed).find_spawn_column();
    let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
    world.generate_chunks_in_radius(cx, cz, 5);
    let start = std::time::Instant::now();
    let (mut meshed, mut near, mut far, mut lod, mut water) = (0, 0, 0, 0, 0);
    let (mut far_stored, mut lod_stored) = (0, 0);
    for x in cx - 4..=cx + 4 {
        for z in cz - 4..=cz + 4 {
            for sy in 0..minerust::NUM_SUBCHUNKS {
                if let Some(snapshot) = world.snapshot_subchunk_mesh(x, z, sy) {
                    if snapshot.has_blocks {
                        let meshes = minerust::World::build_subchunk_mesh_from_snapshot(&snapshot);
                        near += meshes.terrain.len();
                        far_stored += meshes.terrain_far.len();
                        lod_stored += meshes.terrain_lod.len();
                        far += if meshes.terrain_far.is_empty() {
                            meshes.terrain.len()
                        } else {
                            meshes.terrain_far.len()
                        };
                        lod += if meshes.terrain_lod.is_empty() {
                            if meshes.terrain_far.is_empty() {
                                meshes.terrain.len()
                            } else {
                                meshes.terrain_far.len()
                            }
                        } else {
                            meshes.terrain_lod.len()
                        };
                        water += meshes.water.len();
                        meshed += 1;
                    }
                }
            }
        }
    }
    let elapsed = start.elapsed();
    let additional_storage = far_stored + lod_stored;
    println!(
        "{meshed} non-empty subchunks: terrain near {near}, far {far} ({:.1}% fewer), 2×2×2 {lod} ({:.1}% fewer), water {water}; stored far+LOD {additional_storage} additional quads (+{:.1}% geometry); {:.3} ms/subchunk",
        100.0 * (1.0 - far as f64 / near.max(1) as f64),
        100.0 * (1.0 - lod as f64 / near.max(1) as f64),
        100.0 * additional_storage as f64 / near.max(1) as f64,
        elapsed.as_secs_f64() * 1000.0 / meshed as f64
    );
}
