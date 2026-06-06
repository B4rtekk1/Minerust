# src/ - Source Code Directory

## Overview

The `src/` directory contains all Rust source code for the Minerust project, organized into logical modules by functionality.

## Root Files

### `main.rs`
Entry point of the application. Initializes logging and delegates to the `app` module to start the game loop.

```rust
// Initializes tracing for logging
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
├── render_core/   ← Advanced rendering utilities
├── multiplayer/   ← Networking & multiplayer
├── ui/            ← User interface & menus
├── world/         ← World generation & terrain
├── utils/         ← Configuration & utilities
├── settings/      ← Settings files
└── shaders/       ← Shader module
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
- GPU operations happen on async task executor
- No blocking calls in main game loop

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

### Build with Specific Profile
```bash
cargo build --profile release
```

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

Tests can be added using Rust's built-in testing framework:
```bash
cargo test
```

## Documentation

Generate HTML documentation:
```bash
cargo doc --open
```

This will build and open documentation for all public APIs.

