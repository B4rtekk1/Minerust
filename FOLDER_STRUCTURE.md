# Minerust Project - Folder Structure Documentation

## Overview

This document provides a comprehensive guide to the folder structure and organization of the **Minerust** project. Minerust is a high-performance voxel rendering engine written in Rust, featuring GPU-driven rendering, procedurally generated worlds, and multiplayer support.

---

## Project Root Directory

```
minerust/
├── assets/                       # Texture atlas, menu image, fonts
├── assets_docs/                  # Asset documentation resources
├── examples/                     # Runnable examples, including mesh_bench
├── src/                          # Rust sources and WGSL shaders
├── tests/                        # Integration tests
├── tools/                        # Asset-generation scripts
├── target/                       # Cargo build artifacts (generated)
├── Cargo.toml / Cargo.lock       # Package manifest and lockfile
├── README.md                     # Project overview and current status
├── DEVELOPMENT.md                # Build and developer reference
└── world.minerust                # Local world save (if present)
```

---

## 📁 Core Directories

### 1. **src/** - Source Code

The main Rust source code directory containing all project modules.

#### Structure:
```
src/
├── main.rs / lib.rs              # Binary and library roots
├── constants.rs / save.rs        # Shared constants and world persistence
├── app/                          # Client loop, initialization, input and rendering
├── commands/                     # Command definitions
├── core/                         # Blocks, chunks, items, quads and mobs
├── logger/ / minerust_data/      # Logging and app-data paths
├── multiplayer/                  # TCP client/server and packet protocol
├── player/                       # Camera, input and inventories
├── render/                       # Packed geometry, arenas and GPU helpers
├── shaders/                      # WGSL shader sources
├── ui/                           # Menus, inventory UI, fonts and toolbar
└── world/                        # Generation, streaming, meshing and lighting
```

---

### 2. **src/app/** - Application Logic

**Purpose:** Manages the main game loop, window initialization, input handling, and overall application state.

**Files:**
- **`mod.rs`** - Module root and public API
- **`bench.rs`** - Optional automated rendering benchmark path
- **`graphics.rs`** - Graphics options/state
- **`init.rs`** - Application initialization (window setup, GPU device initialization)
- **`game.rs`** - Event loop entry, argument parsing, and client/server startup
- **`performance.rs`** - Performance measurements
- **`state.rs`** - Application state management
- **`input.rs`** - Keyboard, mouse, and controller input handling
- **`update.rs`** - Game state updates (player movement, chunk loading, etc.)
- **`render.rs`** - Main rendering dispatch and frame composition
- **`resize.rs`** - Window resize event handling and viewport management
- **`server.rs`** - Integrated server functionality for multiplayer hosting
- **`texture_cache.rs`** - Texture loading and caching system

**Key Responsibilities:**
- Window creation and event handling (via `winit`)
- GPU device and queue management (via `wgpu`)
- Rendering pipeline orchestration
- Input event processing and player control
- Server/client initialization for multiplayer mode

---

### 3. **src/core/** - Core World Systems

**Purpose:** Fundamental data structures for world representation, block types, chunks, and voxel storage.

**Files:**
- **`mod.rs`** - Module root
- **`block.rs`** - Block type definitions and properties (solid, opaque, liquid, etc.)
- **`chunk.rs`** - Chunk and SubChunk data structures (16x16x16 voxel storage)
- **`biome.rs`** - Biome definitions and characteristics
- **`item.rs`** - Items, tools, food and loot definitions
- **`quad.rs`** - Packed quad data shared with the renderer
- **`mobs/`** - Early mob types and AI scaffolding
- **`vertex.rs`** - Vertex data structures for rendering
- **`uniforms.rs`** - GPU uniform buffer structures (matrices, lighting, etc.)

**Key Responsibilities:**
- Define block types and their properties (color, solid, opaque, translucent)
- Store and manage voxel data in chunks (SubChunk: 16x16x16)
- Track chunk metadata (empty state, mesh status, AABBs)
- Define biome properties and characteristics
- GPU data structures for vertex and uniform management

**Important Constants:**
- `CHUNK_SIZE = 16` - Horizontal chunk size
- `SUBCHUNK_HEIGHT = 16` - Vertical chunk size
- `WORLD_HEIGHT = 256` - Maximum world height
- `NUM_SUBCHUNKS = 16` - Number of subchunks per column

---

### 4. **src/player/** - Player Mechanics

**Purpose:** Handles player character, camera, and player-specific input.

**Files:**
- **`mod.rs`** - Module root
- **`camera.rs`** - Camera and movement state
- **`input.rs`** - Input and digging state
- **`inventory/`** - Player inventory, transactions, and furnace slot rules
- **`player_stats.rs`** - Player statistics

**Key Responsibilities:**
- First-person camera control and math
- Player movement physics (walking, sprinting, jumping)
- View frustum calculation for culling
- Player position and velocity tracking

---

### 5. **src/render/** - Rendering Pipeline

**Purpose:** Core rendering system using wgpu, including mesh management, texturing, and draw dispatch.

**Files:**
- **`mod.rs`** - Module root
- **`mesh.rs`** - Mesh data structures and vertex/index buffer management
- **`mesh_loader.rs`** - Background mesh workers and result delivery
- **`atlas_map.rs`** - Atlas helper; runtime textures are sampled as array layers
- **`texture.rs`** - Texture loading and GPU texture binding
- **`frustum.rs`** - View frustum and AABB definitions for culling
- **`indirect.rs`** - Packed-quad GPU arenas, metadata and indirect draws
- **`quad.rs`** - Packed quad representation and grouping
- **`depth.rs`, `shadow_grid.rs`, `water_waves.rs`** - Rendering helpers

**Key Responsibilities:**
- Growable packed-quad buffers and indirect draw management
- Texture-array image/layer coordination
- Culling and draw dispatch are coordinated by `app/render.rs` and WGSL shaders
- Frustum culling and visibility determination
- Mesh building from voxel data

**Buffer Architecture:**
- Terrain and water use growable packed-quad GPU arenas with relative allocation ranges.
- Culling emits indirect draw commands and handles buffer relocation through metadata.

---

### 6. **src/shaders/** - WGSL Shaders

**Purpose:** Shader programs compiled and used by the GPU renderer.

**Files:**
- **`cull.wgsl`** - Subchunk visibility and indirect draw generation
- **`terrain.wgsl`** - Terrain vertex pulling, lighting, and shading
- **`water.wgsl`** - Water rendering

---

### 7. **src/multiplayer/** - Networking & Multiplayer

**Purpose:** Client-server architecture with network protocol and player synchronization.

**Files:**
- **`mod.rs`** - Module root
- **`client.rs`** - Client-side networking logic
- **`server.rs`** - Server-side networking and player management
- **`network.rs`** - Shared network utilities
- **`protocol.rs`** - Message protocol definitions
- **`tcp.rs`** - TCP transport implementation (only supported transport)
- **`transport.rs`** - Transport layer abstraction
- **`player.rs`** - Remote player state and synchronization

**Key Responsibilities:**
- TCP client/server communication with length-prefixed binary packets
- Connection acknowledgement, player IDs and world seed exchange
- Position, rotation, block-change and chat packet handling
- Remote player display on clients

This is an early-stage foundation: the server relays/handles network state and is not a
complete authoritative world simulation. Do not treat it as an anti-cheat boundary.

**Architecture:**
- One TCP transport is currently implemented; `Transport` is an abstraction, not a
  promise of multiple available protocols.

---

### 8. **src/ui/** - User Interface

**Purpose:** In-game UI rendering and menu systems.

**Files:**
- **`mod.rs`** - Module root
- **`font.rs`** - Font loading and UI text resources
- **`inventory.rs`** - Inventory interface
- **`ui.rs`** - Core UI rendering and layout
- **`menu.rs`** - Main menu, pause menu, and dialog systems
- **`toolbar.rs`** - Hotbar rendering and interaction

**Key Responsibilities:**
- 2D UI rendering (text, buttons, HUD)
- Font management and text rendering (via `glyphon`)
- Menu navigation and state
- In-game HUD elements (health, inventory, hotbar)

---

### 9. **src/world/** - World Generation & Terrain

**Purpose:** Procedural world generation, terrain simulation, and world loading/saving.

**Files:**
- **`mod.rs`** - Module root
- **`generator.rs`** - Noise-based world generation (FBM, Perlin noise)
- **`terrain.rs`** - World state, chunk access, terrain meshing and water geometry
- **`loader.rs`** - Asynchronous chunk generation/loading
- **`skylight.rs`** - Sky-light propagation
- **`item_entity.rs`** - Dropped item entities
- **`structures/`** - Structure and house generation

**Key Responsibilities:**
- Seed-based procedural terrain and biome generation
- Chunk generation and streaming around the player
- Near, merged far, and stored 2×2×2 terrain mesh variants
- Sky-light sampling, structures, and dropped item entities

The far mesh is drawn from 160 blocks; the stored 2×2×2 LOD is currently not selected by
culling because transitions between independently meshed subchunks are not stitched.

---

### 10. **src/utils/** - Utility prototypes

`mod.rs` is currently empty. `settings.rs` and `gpu_memory_info.rs` contain source
prototypes, and `settings_deafult.yaml` is a draft; these are not declared by the module
root or wired into the runtime. No shared settings API is currently exposed.


---

### 11. **assets/** - Game Assets

**Purpose:** All non-code game resources (textures, fonts, configurations).

**Structure:**
```
assets/
├── textures.png                  # Runtime texture array source
├── atlas_map_structure.json      # Texture mapping metadata
├── menu.png                      # Main-menu background
└── fonts/                        # UI font and license/readme
```

**Key Components:**

#### **atlas_map_structure.json** - Texture Atlas Metadata
- JSON mapping of texture names to atlas coordinates
- UV offset calculations for texture placement
- Supports texture atlasing (combining multiple textures into one)

#### **fonts/** - Typography Resources
- Google Sans 17pt regular font
- OFL license compliance
- Used for in-game text rendering

---

### 12. **target/** - Build Artifacts (Auto-Generated)

**Purpose:** Compiled binaries and intermediate build files.

**Contents:**
```
target/
├── debug/                        # Debug build
│   ├── minerust.exe             # Debug executable
│   ├── minerust.pdb             # Debug symbols
│   ├── libminerust.rlib         # Debug library
│   └── deps/                    # Dependencies
├── release/                      # Release build
│   ├── minerust.exe             # Optimized executable
│   ├── minerust.pdb
│   ├── libminerust.rlib
│   └── deps/
└── flycheck0/                    # IDE temporary files
```

**Note:** This directory is auto-generated by Cargo and should not be committed to version control.

---

## 📊 Architecture Overview

### Rendering Pipeline
```
Input/update → background chunk and mesh workers → GPU upload
  → compute culling/indirect commands → terrain and water rendering → UI → present
```
For exact pass ordering, see `src/app/render.rs` and the files under `src/shaders/`.

### World Structure
```
World
  ├── Chunk (16x16 columns)
      ├── SubChunk 0 (16x16x16 voxels)
      ├── SubChunk 1 (16x16x16 voxels)
      ├── ... 
      └── SubChunk 15 (16x16x16 voxels)
```

### GPU Memory Layout
```
Growable packed-quad arenas
  ├── Terrain near/far/2×2×2 variant ranges
  └── Water ranges
Compute culling → visible draw lists and indirect commands
```
Arena capacity grows with uploaded geometry; there is no fixed 560 MB vertex or 256 MB
index allocation. The 2×2×2 mesh is stored but currently not selected by culling.

---

## 🔧 Key Dependencies

`Cargo.toml` is the authoritative dependency list and version source. The current
renderer uses `wgpu`; the windowed client uses `winit`; multiplayer uses TCP and has no
QUIC dependency. Avoid copying version constraints into this structural guide.

---

## 📋 Important Files at Root

- **`Cargo.toml`** - Project manifest with dependencies and build profile
- **`Cargo.lock`** - Dependency lock file (ensures reproducible builds)
- **`README.md`** - Main project documentation
- **`world.minerust`** - Saved world file (binary format)
- **`FOLDER_STRUCTURE.md`** - This documentation file

---

## 🚀 Running the Project

### Development Build
```bash
cargo run
```

### Release Build
```bash
cargo build --release
cargo run --release
```

### Run Server
```bash
cargo run -- --server --port 25565
```
The server is an experimental TCP connection/player-state foundation, not a fully
authoritative simulation.

---

## 📝 Documentation Standards

Each module should follow these patterns:

1. **Module Header** - Describe the module's purpose
2. **Public API** - Document main types and functions
3. **Examples** - Show common usage patterns
4. **Implementation Details** - Explain non-obvious algorithms

---

## 🔗 Related Documentation

- **README.md** - High-level project overview
- **Performance Optimization** - GPU-driven rendering techniques
- **World Generation** - Procedural generation algorithm details
- **Networking Protocol** - Multiplayer communication format

---

**Last Updated:** 2026-11-26
**Project:** Minerust - Voxel Rendering Engine
**Language:** Rust
**License:** See LICENSE file

