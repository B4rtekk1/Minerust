# Minerust Project - Additional Documentation

## Project-Level Documentation

This document provides supplementary information about key systems and patterns used throughout the Minerust project.

---

## Build & Development

### Building the Project

**Debug Build (Development):**
```bash
cargo build
cargo run
```

**Release Build (Optimized):**
```bash
cargo build --release
cargo run --release
```

The dedicated server is selected at runtime, not by a Cargo feature:

```bash
cargo run --release -- --server --port 25565
```

**Clean Build:**
```bash
cargo clean
cargo build
```

### Dependencies (Cargo.toml)

The current manifest uses `wgpu 30.0.1`, `winit 0.30.13`, `glam 0.33.6`,
`glyphon 0.12.0`, `tokio 1.50.0`, and `postcard 1.1`. `Cargo.toml` is the
authoritative source for the full dependency list and version constraints; networking
currently uses TCP (there is no QUIC dependency or transport implementation).

**Compiler:**
- `edition = "2024"` - Modern Rust syntax

### Development Tools

**Debugging:**

The application writes timestamped logs under the platform-specific application data
directory. Consult the startup log when diagnosing window, device, shader, or asset
initialization issues.

**Performance Profiling:**
```bash
# Build a release binary with debug information
cargo build --release --debug

# Run with profiler (example: perf on Linux)
perf record --call-graph=dwarf cargo run --release
perf report
```

**Memory Analysis:**
```bash
# With valgrind (Linux)
valgrind --leak-check=full cargo run

# Check GPU memory
nvidia-smi  # NVIDIA GPUs
```

---

## Code Organization Principles

### Module Hierarchy

```
minerust (binary root)
├── app/          ← Entry point for game loop
├── core/         ← Fundamental data structures
├── render/       ← GPU rendering pipeline
├── world/        ← Procedural generation
├── player/       ← Camera and movement
├── multiplayer/  ← Networking
├── ui/           ← User interface
├── commands/     ← Command definitions
├── logger/       ← Timestamped file logging
├── minerust_data/← Platform-specific app data paths
└── shaders/      ← WGSL shader sources
```

### Dependency Graph

```
app ──────→ render ─────────→ core
  ├──────→ player              ↑
  ├──────→ world ──────────────┤
  ├──────→ ui                  ↑
  ├──────→ multiplayer ────────┤
  └──────→ commands / logger

Principles:
✓ Lower layers don't depend on higher layers
✓ Clear data flow direction
✓ Circular dependencies avoided
✓ Loosely coupled modules
```

### File Naming

```
mod.rs              - Module declaration and public API
type_name.rs        - Primary type definition
system_name.rs      - System implementation
utilities.rs        - Helper functions
constants.rs        - Global constants
```

---

## Core Systems

### 1. Rendering Pipeline

**High-level frame flow:**
```
Input and app update
  ↓
Compute culling (frustum/Hi-Z) and indirect draw-list generation
  ↓
Terrain and water rendering, then remaining scene/UI work
  ↓
Surface presentation
```
Consult `src/app/render.rs` for the exact pass ordering; optional and platform-dependent
paths may differ.

**Terrain geometry storage:**
```
Packed terrain/water quad arenas
  └─ Per-subchunk relative ranges and draw metadata
       ↓
GPU culling chooses visible ranges and writes indirect commands
```
Near, merged far, and 2×2×2 terrain geometry are uploaded together. Culling selects near
or far; the 2×2×2 mesh is retained but currently not drawn. Arena capacities grow as
needed, so there is no fixed per-project vertex/index memory budget.

### 2. World Generation

**Generation Algorithm:**
```
World seed → procedural chunk generation → chunk data
                                  ↓
                         terrain/water meshing
                                  ↓
                      worker result → GPU upload
```

**Chunk Loading:**
```
Player position
  ↓
World loader schedules chunk generation and mesh work on background workers
  ↓
Completed, current-version mesh result
  ↓
Upload packed near/far/LOD and water geometry to GPU arenas
```

### 3. Networking Protocol

**Message format:** packets use length-prefixed binary data serialized with `postcard`.
See `src/multiplayer/protocol.rs` and `tcp.rs` for the wire framing and message schema;
do not infer packet sizes from this guide.

**Connection Handshake:**
```
Client → connection hello
       ← assigned player ID and world seed
Client ↔ position/rotation, block-change, chat, ping/pong, and disconnect messages
```
The current transport is TCP only. The server is an experimental connection/player-state
foundation, not an authoritative world simulation.

---

## Performance Optimization

### GPU Optimizations

| Technique | Impact | Trade-off |
|-----------|--------|-----------|
| Frustum Culling | 80% fewer draws | CPU cost |
| Indirect Drawing | Single draw call | GPU culling shader |
| Mesh Caching | No regeneration | Memory usage |
| Texture Atlasing | Single bind | Atlas management |
| Early Z | Less pixel work | Extra geometry pass |
| LOD System | Distance quality | Visual quality |

### CPU Optimizations

| Technique | Impact | Trade-off |
|-----------|--------|-----------|
| Async Generation | No frame drops | Threading complexity |
| Frame Budgeting | Smooth FPS | Slower mesh generation |
| Chunk Pooling | Allocation free | Memory overhead |
| Binary Serialization | Fast I/O | Less human readable |
| Compression | Smaller chunks | CPU cost |

### Memory Optimizations

```
Block Data:
  ✓ Store as enum (1 byte per block)
  ✗ Store as full struct (16+ bytes)
  
Vertices:
  ✓ Pack UV, AO into small fields
  ✗ Use separate buffers for each
  
Textures:
  ✓ Use texture atlas (single bind)
  ✗ Separate texture per block type
```

---

## Testing

### Running Tests

```bash
cargo test
cargo test --release
cargo test -- --nocapture  # Show output
```

### Benchmarking

```bash
cargo bench
```

**In-game CPU frame-time benchmark.** Enters a fixed world, flies the camera
along a curving path at 20 blocks/s (forcing chunk streaming, meshing and
unloading) and, after the given number of seconds, prints frame-time
percentiles, a per-section breakdown and the worst frames, then exits:

```bash
MINERUST_AUTOSTART=12345 MINERUST_BENCH=30 cargo run --release
```

`MINERUST_BENCH_SPEED` changes the flight speed; `MINERUST_GPU_LOG=1` adds
GPU pass timings to the log.

### Debug Assertions

```rust
debug_assert!(condition, "message");  // Only in debug
assert!(condition, "message");         // Always
```

---

## Debugging Tips

### Logging

```rust
use tracing::{info, warn, error, debug, trace};

info!("Starting game");
debug!("Loaded chunk at {}", position);
error!("Failed to load texture: {}", path);
```

### Printf Debugging

```rust
println!("value = {:?}", value);
println!("position = {:.2?}", position);  // 2 decimal places
```

### Visual Debugging

**Wireframe Mode:**
```rust
// In render pass
rpass.set_pipeline(&wireframe_pipeline);
```

**Show Chunks:**
```rust
// Render chunk boundaries as lines
```

**Show Frustum:**
```rust
// Render camera frustum visualization
```

**Show Physics:**
```rust
// Draw collision boxes around entities
```

---

## Common Patterns

### Error Handling

```rust
// Using Result
fn load_config() -> Result<Config, LoadError> {
    let data = std::fs::read_to_string("config.yaml")?;
    Ok(parse_config(&data)?)
}

// Using Option
fn find_chunk(x: i32, z: i32) -> Option<&Chunk> {
    chunks.get(&(x, z))
}
```

### Closures & Callbacks

```rust
// Simple closure
let add_one = |x| x + 1;

// With capture
let multiplier = 3;
let multiply = |x| x * multiplier;

// Button callback
button.on_click(|| {
    println!("Button pressed!");
});
```

### Trait Objects

```rust
// Dynamic dispatch
trait Renderable {
    fn render(&self, renderer: &Renderer);
}

let objects: Vec<Box<dyn Renderable>> = vec![
    Box::new(cube),
    Box::new(sphere),
];
```

---

## Common Issues & Solutions

### Issue: Frame Rate Drops

**Causes:**
- Too many chunks loaded
- Mesh generation stalling
- GPU memory pressure

**Solutions:**
- Reduce render distance
- Check GPU memory usage
- Monitor mesh generation queue

### Issue: Memory Leak

**Debugging:**
```bash
valgrind --leak-check=full ./minerust
```

**Common Causes:**
- GPU resources not freed
- Circular references
- Unbounded collections

### Issue: Rendering Artifacts

**Causes:**
- Z-fighting (depth precision)
- Texture coordinates wrong
- Normals incorrect

**Debug:**
- Show normals as colors
- Show UV coordinates
- Verify frustum calculations

### Issue: Network Lag

**Symptoms:**
- Player position jumps
- Block placement delayed
- Chunk loading slow

**Solutions:**
- Check bandwidth usage
- Monitor message latency
- Enable chunk compression

---

## Extending Minerust

### Adding a New Block Type

1. Add the block and its properties in `src/core/block.rs`.
2. Assign its texture layer in `src/constants.rs` and update `assets/textures.png`
   using `tools/generate_block_textures.py` as appropriate.
3. Update `assets/atlas_map_structure.json` if the descriptive mapping needs the new block.
4. Add world-generation or meshing behavior only if required by the block.
5. Add focused tests and check the result in the client.

### Adding a New Biome

1. Add variant to `BiomeType` enum (core/biome.rs)
2. Define biome properties (temperature, humidity, etc.)
3. Implement generation algorithm
4. Add structure spawning (trees, villages, etc.)
5. Test world generation

### Adding a New Render Pass

1. Add the pipeline or pass in the owning app/render component.
2. Add or update shader code in `src/shaders/` when needed.
3. Integrate its resources and dispatch in `src/app/render.rs` or initialization.
4. Validate shaders and run the relevant client path.

### Adding Multiplayer Feature

1. Define message type (multiplayer/protocol.rs)
2. Implement send logic (client.rs or server.rs)
3. Handle received messages
4. Test with multiple clients

---

## Release Checklist

Before publishing a release:

- [ ] All tests pass: `cargo test --release`
- [ ] No compiler warnings: `cargo clippy`
- [ ] Documentation updated: `cargo doc`
- [ ] Performance benchmarks run
- [ ] Memory leaks checked
- [ ] Version bumped in Cargo.toml
- [ ] CHANGELOG.md updated
- [ ] README.md current
- [ ] Build works on all platforms
- [ ] Test on target hardware

---

## License & Attribution

**Minerust License:**
[See LICENSE file in project root]

**Third-Party Libraries:**
All dependencies listed in Cargo.toml with their licenses

**Assets License:**
See `LICENSE` for project licensing and `assets/fonts/OFL.txt` for the bundled font license.

---

## Resources & References

### Graphics Programming
- [wgpu Guide](https://wgpu.rs/)
- [Learn WGSL](https://learn.wgpu.rs/)
- [Game Graphics Programming](https://learnopengl.com/)

### Game Development
- [Minecraft Wiki](https://minecraft.fandom.com/)
- [Voxel Engine Architecture](https://www.youtube.com/@LearnVoxelEng)

### Rust
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### Procedural Generation
- [Noise & Randomness](https://www.redblobgames.com/maps/terrain-from-noise/)
- [Perlin Noise](https://en.wikipedia.org/wiki/Perlin_noise)

### Networking
- Current transport: TCP with length-prefixed `postcard` packets; see `src/multiplayer/`.
- The dedicated server is experimental and does not provide authoritative world simulation.

---

**Last Updated:** 2026-11-26
**Project:** Minerust - Voxel Rendering Engine  
**Status:** Active Development

