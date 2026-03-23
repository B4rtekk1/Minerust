# Minerust Project - Folder Structure Documentation

## Overview

This document provides a comprehensive guide to the folder structure and organization of the **Minerust** project. Minerust is a high-performance voxel rendering engine written in Rust, featuring GPU-driven rendering, procedurally generated worlds, and multiplayer support.

---

## Project Root Directory

```
minerust/
├── src/                          # Source code (Rust modules)
├── assets/                       # Game assets (textures, fonts, config)
├── shaders/                      # WGSL shader files (currently empty)
├── target/                       # Build artifacts (auto-generated)
├── Cargo.toml                    # Rust project manifest
├── Cargo.lock                    # Dependency lock file
├── README.md                     # Main project documentation
├── FOLDER_STRUCTURE.md           # This file
└── world.minerust                # World save file
```

---

## 📁 Core Directories

### 1. **src/** - Source Code

The main Rust source code directory containing all project modules.

#### Structure:
```
src/
├── main.rs                       # Entry point (delegates to app module)
├── lib.rs                        # Library root
├── constants.rs                  # Global constants and configuration
├── save.rs                       # World save/load functionality
├── app/                          # Application logic
├── core/                         # Core world systems
├── player/                       # Player mechanics
├── render/                       # Rendering pipeline
├── render_core/                  # Rendering utilities (CSM, etc.)
├── multiplayer/                  # Networking and multiplayer
├── ui/                           # User interface
├── utils/                        # Utility functions and helpers
├── world/                        # World generation and terrain
├── settings/                     # Settings configuration
└── shaders/                      # Shader definitions and management
```

---

### 2. **src/app/** - Application Logic

**Purpose:** Manages the main game loop, window initialization, input handling, and overall application state.

**Files:**
- **`mod.rs`** - Module root and public API
- **`init.rs`** - Application initialization (window setup, GPU device initialization)
- **`game.rs`** - Main game logic and core update loop
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
- **`camera.rs`** - Camera system (projection matrices, frustum culling)
- **`input.rs`** - Player input handling (movement, jumping, sprinting)

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
- **`mesh_loader.rs`** - Loads voxel chunk data into GPU buffers
- **`atlas_map.rs`** - Texture atlas mapping and UV coordinate management
- **`texture.rs`** - Texture loading and GPU texture binding
- **`frustum.rs`** - View frustum and AABB definitions for culling
- **`indirect.rs`** - Indirect GPU drawing (indirect buffers for instancing)
- **`passes/`** - Individual render passes (shadows, terrain, water, UI, etc.)

**Key Responsibilities:**
- GPU buffer management (vertex, index, indirect buffers)
- Texture atlas coordination
- Render passes orchestration (shadow pass, terrain pass, composite pass)
- Frustum culling and visibility determination
- Mesh building from voxel data

**Buffer Architecture:**
- Unified large vertex buffer (~560MB capacity)
- Unified large index buffer (~256MB capacity)
- Chunks are sub-allocated regions within these buffers
- Minimizes CPU draw call overhead through indirect drawing

---

### 6. **src/render_core/** - Rendering Utilities

**Purpose:** Advanced rendering techniques and utilities.

**Files:**
- **`mod.rs`** - Module root
- **`csm.rs`** - Cascaded Shadow Maps (4-cascade shadow system with PCF)

**Key Responsibilities:**
- Shadow map management (4 cascades, up to 2048x2048 per cascade)
- Shadow matrix calculations
- Percentage-Closer Filtering (PCF) implementation

---

### 7. **src/multiplayer/** - Networking & Multiplayer

**Purpose:** Client-server architecture with network protocol and player synchronization.

**Files:**
- **`mod.rs`** - Module root
- **`client.rs`** - Client-side networking logic
- **`server.rs`** - Server-side networking and player management
- **`network.rs`** - Shared network utilities
- **`protocol.rs`** - Message protocol definitions
- **`quic.rs`** - QUIC protocol implementation (reliable)
- **`tcp.rs`** - TCP fallback implementation
- **`transport.rs`** - Transport layer abstraction
- **`player.rs`** - Remote player state and synchronization

**Key Responsibilities:**
- Client-server communication (authoritative server)
- QUIC and TCP protocol handling
- Delta compression for chunk data
- Entity state synchronization
- Player movement prediction on client

**Architecture:**
- Authoritative server validates all gameplay
- QUIC for reliable data transfers
- UDP for high-frequency position updates
- Efficient delta compression for chunk data

---

### 8. **src/ui/** - User Interface

**Purpose:** In-game UI rendering and menu systems.

**Files:**
- **`mod.rs`** - Module root
- **`ui.rs`** - Core UI rendering and layout
- **`menu.rs`** - Main menu, pause menu, and dialog systems

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
- **`terrain.rs`** - Terrain features (caves, mountains, water)
- **`loader.rs`** - Chunk loading/unloading and streaming
- **`spline.rs`** - Spline interpolation for smooth terrain transitions
- **`structures/`** - Procedural structure generation (trees, buildings, etc.)

**Key Responsibilities:**
- Multi-threaded procedural generation using FastNoise
- 11 distinct biomes with unique characteristics
- Cave systems ("Cheese" and "Spaghetti" patterns)
- Ore generation and placement
- Chunk loading/unloading based on render distance
- Structure generation (trees, villages, etc.)

**Generation Features:**
- Fractional Brownian Motion (FBM) for terrain
- Deterministic generation (same seed = same world)
- Async generation to avoid frame drops
- Cave systems with procedural placement

---

### 10. **src/utils/** - Utilities & Helpers

**Purpose:** Common utility functions and configuration systems.

**Files:**
- **`mod.rs`** - Module root
- **`settings.rs`** - Runtime settings management
- **`gpu_memory_info.rs`** - GPU memory tracking and diagnostics
- **`server_properties.rs`** - Server configuration properties
- **`settings/`** - Settings definitions and schema

**Key Responsibilities:**
- Load/save game settings (YAML-based)
- GPU memory diagnostics and reporting
- Server configuration management
- Game parameter management

---

### 11. **src/settings/** - Settings Configuration

**Purpose:** Configuration files for game settings.

**Files:**
- **`settings.yaml`** - Runtime configuration (graphics, gameplay, networking)

**Contains:**
- Graphics settings (render distance, shadow quality, etc.)
- Gameplay settings (difficulty, player speed, etc.)
- Networking settings (server address, port, etc.)

---

### 12. **src/shaders/** - Shader Definitions

**Purpose:** Manages shader organization and loading.

**Note:** Actual shader files (.wgsl) are stored in `assets/shaders/`

---

### 13. **assets/** - Game Assets

**Purpose:** All non-code game resources (textures, fonts, configurations).

**Structure:**
```
assets/
├── textures.png                  # Texture atlas (combined textures)
├── atlas_map_structure.json      # Atlas metadata (UV mappings)
├── fonts/                        # Font files
│   ├── GoogleSans_17pt-Regular.ttf
│   ├── OFL.txt                   # Font license
│   └── README.txt
└── textures/                     # Minecraft-compatible texture pack
    ├── pack.png                  # Pack thumbnail
    ├── pack.mcmeta               # Minecraft pack format metadata
    ├── LICENSE.txt               # Texture license
    └── block/                    # Individual block textures
        ├── acacia_door_*.png
        ├── acacia_log*.png
        ├── acacia_leaves.png
        ├── acacia_planks.png
        ├── amethyst_*.png
        ├── ancient_debris_*.png
        ├── andesite.png
        ├── anvil_*.png
        └── ... (100+ block types)
```

**Key Components:**

#### **textures/block/** - Block Textures
- Individual 16x16 PNG files for each block type
- Compatible with Minecraft texture format
- Supports texture variations (top, bottom, side, animated)
- Includes all major block types: ores, plants, wood, stone variants, etc.

#### **atlas_map_structure.json** - Texture Atlas Metadata
- JSON mapping of texture names to atlas coordinates
- UV offset calculations for texture placement
- Supports texture atlasing (combining multiple textures into one)

#### **fonts/** - Typography Resources
- Google Sans 17pt regular font
- OFL license compliance
- Used for in-game text rendering

---

### 14. **target/** - Build Artifacts (Auto-Generated)

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

### 15. **shaders/** - WGSL Shader Files

**Purpose:** GPU shader programs for rendering.

**Status:** Currently empty; shaders may be embedded in code or loaded from assets.

---

## 📊 Architecture Overview

### Rendering Pipeline
```
Input → Update → Generate Chunks → Build Meshes → 
  Upload to GPU → Culling Pass → Shadow Pass → 
  Terrain Pass → Water Pass → Composite Pass → UI Pass → Present
```

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
Unified Vertex Buffer (560MB)
  ├── Chunk 0 (sub-allocated)
  ├── Chunk 1 (sub-allocated)
  └── ...

Unified Index Buffer (256MB)
  ├── Chunk 0 indices
  ├── Chunk 1 indices
  └── ...

Indirect Buffer
  ├── Draw commands
  └── Culling results
```

---

## 🔧 Key Dependencies

From `Cargo.toml`:

```toml
wgpu = "28.0.0"              # Graphics API abstraction
winit = "0.29"               # Window and event handling
glam = { version = "0.29", features = ["bytemuck", "serde"] } # Linear algebra
serde = "1.0"                # Serialization
serde_yaml = "0.9"           # YAML configuration
bincode = "1.3"              # Binary serialization
tokio = "1.40"               # Async runtime
quinn = "0.11"               # QUIC protocol
rustls = "0.23"              # TLS/cryptography
image = "0.25"               # Image loading
glyphon = "0.10"             # Text rendering
fastnoise-lite = "1.1"       # Procedural noise
```

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
cargo run --features "server"
```

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

**Last Updated:** 2026-03-05
**Project:** Minerust - Voxel Rendering Engine
**Language:** Rust
**License:** See LICENSE file

