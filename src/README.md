# src/ - Source Code Directory

## Overview

The `src/` directory contains all Rust source code for the Minerust project, organized into logical modules by functionality.

## Root Files

### `main.rs`
Entry point of the application. Initializes logging and delegates to the `app` module to start the game loop.

```rust
// Initializes the application logger
// Calls app::run_game() to start the main loop
```

### `lib.rs`
Library root file that declares all modules as a library. Allows code to be reused across different binaries if needed.

### `constants.rs`
Global constants used throughout the project:
- **World constants**: Chunk size, world height, render distance
- **Texture constants**: Texture indices for different block types
- **Optimization constants**: Max chunks per frame, worker count
- **Player constants**: Height, movement speed, jump power
- **Camera constants**: Field of view, near/far planes

### `save.rs`
World save/load functionality. Handles serialization and deserialization of world data to disk.

## Module Structure

```
src/
├── app/           ← Application logic & main loop
├── core/          ← World data structures (blocks, chunks)
├── player/        ← Player character & camera
├── render/        ← Rendering pipeline & mesh management
├── multiplayer/   ← Networking & multiplayer
├── ui/            ← User interface & menus
├── world/         ← World generation & terrain
├── commands/      ← Command definitions
├── logger/        ← Timestamped file logging
├── minerust_data/ ← Platform-specific app data paths
└── shaders/       ← WGSL sources (cull, terrain, water)
```

## Key Patterns

### Module Declaration
Each folder has a `mod.rs` file that:
1. Declares all submodules
2. Defines the public API
3. Re-exports important types

### Type Organization
- Types are defined in focused files (e.g., `block.rs`, `chunk.rs`)
- Related functionality is grouped together
- Public API is intentionally small and focused

### Async/Threading
- World generation happens on background threads
- GPU work is submitted through `wgpu`; it is not run on a project-wide async executor
- Chunk generation and mesh building use worker threads; rendering is coordinated by the app loop

## Building & Compilation

### Debug Build
```bash
cargo build
cargo run
```

### Release Build
```bash
cargo build --release
cargo run --release
```

### Run the mesh benchmark
```bash
cargo run --release --example mesh_bench -- 12345
```

This compares CPU mesh output and quad counts; it does not measure GPU frame time.

## Code Organization Best Practices

1. **Keep modules focused** - Each module has a single responsibility
2. **Use clear naming** - File names match the primary type/system they contain
3. **Document public API** - Use doc comments for public items
4. **Minimize dependencies** - Import only what's needed
5. **Keep hot paths optimized** - Critical paths in rendering loop are optimized

## Common Imports Pattern

Most modules follow this pattern:
```rust
use crate::constants::*;  // Global constants
use crate::core::*;       // Core types
```

## Testing

Run the library unit tests and shader validation with:
```bash
cargo test --lib
```

## Documentation

Generate HTML documentation:
```bash
cargo doc --open
```

This will build and open documentation for all public APIs.

