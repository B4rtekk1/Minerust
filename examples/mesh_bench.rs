//! Measures subchunk snapshot + mesh time on generated terrain.
//! `cargo run --release --example mesh_bench -- [seed]`
fn main() {
    let seed = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12345);
    let mut world = minerust::World::new_empty_with_seed(seed);
    let (x, z) = minerust::ChunkGenerator::new(seed).find_spawn_column();
    let (cx, cz) = (x.div_euclid(16), z.div_euclid(16));
    world.generate_chunks_in_radius(cx, cz, 5);
    let start = std::time::Instant::now();
    let (mut meshed, mut quads) = (0, 0);
    for x in cx - 4..=cx + 4 {
        for z in cz - 4..=cz + 4 {
            for sy in 0..minerust::NUM_SUBCHUNKS {
                if let Some(snapshot) = world.snapshot_subchunk_mesh(x, z, sy) {
                    if snapshot.has_blocks {
                        let (terrain, water) = minerust::World::build_subchunk_mesh_from_snapshot(&snapshot);
                        quads += terrain.len() + water.len();
                        meshed += 1;
                    }
                }
            }
        }
    }
    let elapsed = start.elapsed();
    println!(
        "{meshed} non-empty subchunks, {quads} quads, {:.3} ms/subchunk",
        elapsed.as_secs_f64() * 1000.0 / meshed as f64
    );
}
