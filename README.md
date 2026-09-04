# 🎮 Minerust

<div align="center">

![Rust](https://img.shields.io/badge/Rust-2024-CE422B?style=for-the-badge&logo=rust&logoColor=white)
![wgpu](https://img.shields.io/badge/wgpu-30.0.1-4C8CBF?style=for-the-badge)
![Status](https://img.shields.io/badge/status-experimental-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**An experimental high-performance voxel sandbox and rendering engine written in Rust.**

GPU-driven terrain submission, asynchronous world streaming, procedural generation,
inventory/gameplay systems, save files, and early-stage TCP multiplayer.

[Features](#-features) •
[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Controls](#-controls) •
[Project Structure](#-project-structure) •
[Roadmap](#-roadmap)

</div>

---

## 📖 Overview

**Minerust** is a voxel game/engine project focused on modern rendering architecture,
large procedural worlds, and low CPU submission overhead.

The renderer is built on **wgpu** and uses packed quad descriptors, growable shared GPU
arenas, compute-shader visibility culling, and indirect rendering. World generation and
mesh building run asynchronously so the main render loop does not have to generate
terrain synchronously while the player explores.

The project is also evolving into a playable sandbox: block breaking/placement,
inventory and hotbar interaction, item drops, tools and durability, world saving,
and multiplayer foundations are already present.

> **Project status:** active development / experimental. APIs, save data, rendering
> internals, and gameplay systems may change frequently.

### Demo

<https://github.com/user-attachments/assets/3f86d46e-7a33-4144-ae3d-f78887f2b1a7>

---

## ✨ Features

### 🎨 Rendering

- **wgpu 30.0.1** renderer with Vulkan / Direct3D 12 / Metal backend support through wgpu.
- **GPU-driven terrain submission** using compute-generated visibility lists.
- **`multi_draw_indirect_count`** when supported by the active GPU, with a fallback path.
- **GPU frustum culling** for subchunks.
- **Hi-Z occlusion culling** using a hierarchical depth pyramid.
- **Packed quad vertex pulling** instead of a conventional per-subchunk vertex buffer layout.
- **Growable shared quad arenas** for terrain and water geometry.
- **Free-list allocation and arena compaction** for long-running chunk streaming.
- **Greedy meshing** on the CPU mesh path to merge compatible voxel faces.
- **Asynchronous mesh workers** with versioned mesh results so stale work can be rejected safely.
- Separate **opaque terrain** and **transparent water** rendering paths.
- Water shading with **screen-space reflection data, refraction, Fresnel blending, and foam/edge logic**.
- **4× MSAA** in the current renderer.
- Procedural sky/sun rendering.
- Full-screen composite stage with effects such as **underwater fog/color grading and vignette**.
- GPU timestamp profiling and an in-game performance/debug overlay.

### 🌍 World generation and streaming

- Deterministic seed-based procedural generation.
- Chunk columns are **16×16 blocks** horizontally.
- World height is **256 blocks**.
- Each chunk column contains **16 subchunks**, each **16×16×16**.
- Multi-threaded chunk generation through a priority queue.
- Distance-prioritized generation requests.
- Separate generation, render, simulation, and unload distances.
- Caves, terrain variation, vegetation, water bodies, and procedural features.
- Current biome set:
    - Plains
    - Forest
    - Desert
    - Tundra
    - Mountains
    - Swamp
    - Ocean
    - Beach
    - River
    - Lake
    - Island
- Biome-dependent grass/leaf tinting and vegetation density.
- Automatic chunk unloading outside the configured streaming radius.
- Dirty-subchunk tracking and incremental remeshing after world changes.

### 🧱 Blocks and world interaction

The current block model includes ordinary cubes as well as several special cases.

Examples include:

`Grass`, `Dirt`, `Stone`, `Sand`, `Water`, `Wood`, `Leaves`, `Bedrock`,
`Snow`, `Gravel`, `Clay`, `Ice`, `Cactus`, `DeadBush`, `WoodStairs`,
`WoodLogX`, and `WoodLogZ`.

Implemented interaction systems include:

- block breaking with per-block break times,
- block placement from the selected hotbar item,
- collision-aware placement,
- repeated straight-line placement while RMB is held,
- block loot/drop handling,
- dropped item entities with simple world physics and pickup behavior,
- block face visibility rules for transparent and partial blocks.

### 🎒 Inventory and items

Minerust now has a real gameplay inventory rather than only a hotbar mock-up.

- **27-slot main inventory**
- **9-slot hotbar**
- stack merging and maximum stack sizes,
- selected hotbar slot,
- left-click / right-click inventory interaction,
- shift-click quick move between main inventory and hotbar,
- dropping one item or a full stack,
- cursor-stack restoration when the inventory closes,
- hotbar-first insertion for freshly mined drops,
- stable item resource keys for save data,
- item registry with multiple item kinds:
    - blocks,
    - tools,
    - food,
    - materials/generic items,
- tool durability support,
- loot tables for block drops.

The codebase also contains a **furnace inventory/container scaffold** with typed slot
rules. Furnace simulation itself is still a work in progress.

### 💾 Saving

World saves use a compact binary format through `postcard`.

The save system stores:

- world seed,
- player position,
- player rotation,
- inventory and selected hotbar slot,
- item durability,
- player-modified chunks/subchunks.

Procedural terrain that was not modified by the player can be regenerated from the seed,
which keeps save files smaller than serializing the entire loaded world.

Default save file:

```text
world.minerust
```

### 🌐 Multiplayer

Multiplayer is currently an **early-stage TCP implementation**.

Implemented foundations include:

- headless dedicated server mode,
- TCP client/server transport,
- length-prefixed binary packet protocol,
- server-assigned player IDs,
- connection acknowledgement with the server world seed,
- position synchronization,
- rotation synchronization,
- block-change packets,
- chat packets,
- ping/pong packets,
- disconnect propagation,
- remote player rendering/name labels.

The current dedicated server primarily validates/stamps player identity and relays
packets between clients. It is **not yet a complete authoritative world-simulation server**.

---

## 🚀 Quick Start

### Requirements

- **Rust stable with Rust 2024 edition support** — Rust 1.85+ is recommended.
- A GPU and driver supported by **wgpu**.
- Windows is the primary development/release target at the moment.
- Linux/macOS may work through wgpu/winit, but are not guaranteed to be tested on every revision.

### Clone and run

```bash
git clone https://github.com/B4rtekk1/Minerust.git
cd Minerust

cargo run --release
```

For development:

```bash
cargo run
```

Release mode is strongly recommended when evaluating rendering or chunk-generation
performance.

### Build only

```bash
cargo build --release
```

### Run tests

```bash
cargo test
```

### Formatting and linting

```bash
cargo fmt
cargo clippy --all-targets
```

---

## 🖥️ Dedicated Server

Start a headless TCP server on the default port (`25565`):

```bash
cargo run --release -- --server
```

Use a custom port:

```bash
cargo run --release -- --server --port 12345
```

The dedicated server does not create a game window.

---

## 🎮 Controls

| Input | Action |
|---|---|
| `W A S D` | Move |
| `Space` | Jump |
| `Left Shift` | Sprint |
| `Left Ctrl` | Sneak |
| Mouse | Look around |
| `LMB` | Mine / break targeted block |
| `RMB` | Place selected block |
| `E` | Open / close inventory |
| `1`–`9` | Select hotbar slot |
| Mouse wheel | Cycle hotbar |
| `Q` | Drop one selected item |
| `Ctrl + Q` | Drop selected stack |
| `Esc` | Close inventory / return to menu |
| `F1` | Toggle crosshair/HUD element |
| `C` | Toggle debug/performance overlay |
| `F4` | Toggle Hi-Z occlusion culling |
| `F5` | Save world |
| `F9` | Load world |
| `F11` | Toggle borderless fullscreen |

Inventory UI also supports left/right click behavior and Shift + left-click quick moves.

---

## ⚙️ Current Engine Constants

The active world-streaming values are currently compile-time constants in
`src/constants.rs`.

| Constant | Current value | Meaning |
|---|---:|---|
| `WORLD_HEIGHT` | `256` | World height in blocks |
| `CHUNK_SIZE` | `16` | Horizontal chunk size |
| `SUBCHUNK_HEIGHT` | `16` | Vertical subchunk size |
| `NUM_SUBCHUNKS` | `16` | Vertical subchunks per chunk column |
| `RENDER_DISTANCE` | `32` | Render radius in chunks |
| `SIMULATION_DISTANCE` | `16` | Simulation radius |
| `GENERATION_DISTANCE` | `34` | Generation/prefetch radius |
| `CHUNK_UNLOAD_DISTANCE` | `37` | Chunk eviction radius |
| `SEA_LEVEL` | `64` | Sea level |
| `MAX_CHUNKS_PER_FRAME` | `8` | Chunk generation request budget |
| `MAX_CHUNK_COMMITS_PER_FRAME` | `2` | Completed chunk commit budget |
| `MAX_MESH_BUILDS_PER_FRAME` | `8` | Mesh request budget |
| `MAX_MESH_COMMITS_PER_FRAME` | `2` | Finished mesh/GPU commit budget |

Worker counts are selected from the available CPU count and clamped to keep the main
thread responsive.

---

## 🧠 Architecture

### High-level data flow

```text
Player / Camera
      │
      ▼
World Streaming
      │
      ├── Missing chunk detection
      ├── Distance-priority generation queue
      └── Background ChunkGenerator workers
      │
      ▼
World / Chunk / SubChunk data
      │
      ├── Dirty mesh tracking
      ├── Mesh versioning
      └── Background mesh workers
      │
      ▼
PackedQuad mesh streams
      │
      ▼
Terrain / Water shared GPU arenas
      │
      ▼
Compute culling
      ├── Frustum
      └── Hi-Z occlusion
      │
      ▼
Indirect draw command buffers
      │
      ▼
Opaque terrain pass
      │
      ├── Depth resolve
      └── Hi-Z pyramid generation
      │
      ▼
Transparent water pass
      │
      ▼
Composite / post-processing
      │
      ▼
UI + text
      │
      ▼
Present
```

### Chunk streaming

`ChunkLoader` uses background worker threads and a shared priority heap. Lower squared
distance to the player means higher urgency. Duplicate in-flight requests are prevented
with a pending set, and the queue is bounded to avoid unbounded memory growth.

### Mesh streaming

Mesh creation is decoupled from chunk generation.

A subchunk carries a mesh revision/version. When a worker finishes, the result is accepted
only if its revision still matches the current world state. This prevents a mesh generated
from stale neighbor/block data from overwriting a newer version.

### GPU geometry storage

Terrain and water use persistent packed-quad arenas instead of creating a GPU buffer for
every chunk.

The allocator supports:

- sub-allocation,
- free blocks,
- arena growth,
- compaction,
- culling-metadata refresh after relocation.

This keeps the renderer suitable for continuous load/unload cycles while the player moves
through the world.

### GPU-driven visibility

Per-subchunk metadata contains:

- world-space AABB,
- terrain draw range,
- water draw range.

The compute culling pass tests the AABB and emits indirect draw commands only for visible
subchunks. On GPUs that support `MULTI_DRAW_INDIRECT_COUNT`, the GPU also controls how many
draws are executed.

---

## 📂 Project Structure

```text
Minerust/
├── assets/                     # Texture atlas, fonts, menu assets
├── assets_docs/                # Documentation assets
├── src/
│   ├── app/                    # Application state, game loop, rendering orchestration
│   │   ├── game.rs             # Event loop, CLI, controls, save/load hotkeys
│   │   ├── init.rs             # GPU/window/pipeline initialization
│   │   ├── input.rs            # Gameplay/menu/inventory input
│   │   ├── render.rs           # Frame render graph / passes
│   │   ├── update.rs           # Simulation, streaming and mesh commits
│   │   ├── state.rs            # Main runtime State
│   │   └── server.rs           # Dedicated server entry
│   ├── core/
│   │   ├── block.rs            # Block types and physical/render properties
│   │   ├── biome.rs            # Biome definitions
│   │   ├── chunk.rs            # Chunk/SubChunk storage and metadata
│   │   ├── item.rs             # Item registry, tools, food, loot
│   │   └── mobs/               # Early mob/AI scaffolding
│   ├── multiplayer/
│   │   ├── protocol.rs         # Binary packet protocol
│   │   ├── tcp.rs              # TCP transport implementation
│   │   ├── client.rs           # Multiplayer client
│   │   ├── server.rs           # Server-side network primitives
│   │   └── network.rs          # Game/network integration
│   ├── player/
│   │   ├── camera.rs           # Camera, movement and collision
│   │   ├── inventory/          # Inventory/container transaction model
│   │   └── player_stats.rs     # Player statistics/state
│   ├── render/
│   │   ├── indirect.rs         # GPU arenas, metadata, culling and indirect draws
│   │   ├── mesh_loader.rs      # Background mesh workers
│   │   ├── mesh.rs             # Mesh helpers and models
│   │   ├── quad.rs             # PackedQuad representation
│   │   ├── frustum.rs          # AABB/frustum math
│   │   ├── texture.rs          # Texture handling
│   │   └── atlas_map.rs        # Texture atlas mapping
│   ├── shaders/
│   │   ├── terrain.wgsl
│   │   ├── water.wgsl
│   │   ├── cull.wgsl
│   │   ├── hiz.wgsl
│   │   ├── depth_resolve.wgsl
│   │   ├── sky.wgsl
│   │   ├── sun.wgsl
│   │   ├── composite.wgsl
│   │   ├── outline.wgsl
│   │   └── ui.wgsl
│   ├── ui/                     # Menu, HUD, inventory and text UI
│   ├── utils/                  # Settings and GPU/system helpers
│   ├── world/
│   │   ├── generator.rs        # Procedural terrain generator
│   │   ├── loader.rs           # Async priority chunk loader
│   │   ├── terrain.rs          # World data + meshing snapshots/building
│   │   ├── spline.rs           # Terrain interpolation/splines
│   │   ├── item_entity.rs      # Dropped item entities
│   │   └── structures/         # Structure framework/basic structures
│   ├── save.rs                 # Postcard world/inventory persistence
│   ├── constants.rs
│   ├── lib.rs
│   └── main.rs
├── Cargo.toml
├── DEVELOPMENT.md
├── DOCUMENTATION_MAP.md
├── FOLDER_STRUCTURE.md
└── README.md
```

---

## 🧰 Main Dependencies

| Crate | Version | Purpose |
|---|---:|---|
| `wgpu` | `30.0.1` | Cross-platform GPU API |
| `winit` | `0.30.13` | Windowing and input |
| `glam` | `0.33.6` | Vector/matrix math |
| `glyphon` | `0.12.0` | GPU text rendering |
| `tokio` | `1.50` | Async networking/runtime |
| `fastnoise-lite` | `1.1` | Procedural noise |
| `postcard` | `1.1` | Compact save serialization |
| `crossbeam-channel` | `0.5` | Worker communication |
| `parking_lot` | `0.12` | Synchronization |
| `rustc-hash` | `2` | Fast hash collections |
| `clap` | `4.4` | CLI argument parsing |

See [`Cargo.toml`](Cargo.toml) for the complete dependency list.

---

## 📊 Profiling and performance work

Minerust is designed around profiling rather than fixed performance claims.

The engine contains instrumentation for:

- overall frame time,
- CPU-side update/render sections,
- process CPU time on Windows,
- GPU timestamps,
- chunk-generation backlog,
- mesh streaming work,
- visibility/culling statistics.

Use a **release build** for meaningful performance measurements:

```bash
cargo run --release
```

The current architecture is specifically intended to reduce:

- per-chunk CPU draw submission,
- unnecessary hidden geometry rendering,
- synchronous terrain generation stalls,
- per-chunk GPU buffer allocation overhead.

Actual FPS, RAM use, and VRAM use depend heavily on render distance, resolution,
GPU, world complexity, and current development state.

---

## 🧪 Experimental / work in progress

Several parts of the repository are intentionally incomplete or experimental:

- `gpu_mesher.rs` contains an experimental compute-based face extraction implementation,
  but it is not currently wired into the public `render` module/main rendering path.
- Mob classes and passive AI scaffolding exist, but there is no complete mob gameplay loop yet.
- Furnace/container slot rules exist, but furnace processing is not implemented yet.
- Multiplayer is functional as a transport/synchronization foundation, but server-authoritative
  gameplay/world simulation is not complete.
- Streaming, meshing prioritization, and Hi-Z behavior are actively being optimized.
- Some older module-level documentation may lag behind the current code; the source code is the
  authoritative reference.

---

## 🗺️ Roadmap

Near-term areas that fit the current architecture:

- [ ] Priority-based mesh streaming to favor missing/near-camera subchunks
- [ ] More robust temporal Hi-Z occlusion handling
- [ ] Temporal anti-aliasing (TAA)
- [ ] Dynamic/local light sources
- [ ] Expanded global illumination / ambient lighting
- [ ] Crafting and furnace simulation
- [ ] Complete tool/food gameplay behavior
- [ ] Mob spawning and AI
- [ ] More structures and world-generation features
- [ ] More authoritative multiplayer world state
- [ ] Better network delta/state synchronization
- [ ] Continued RAM/VRAM reduction and streaming optimization

---

## 📚 Documentation

Additional documentation is available in:

- [`DEVELOPMENT.md`](DEVELOPMENT.md)
- [`DOCUMENTATION_MAP.md`](DOCUMENTATION_MAP.md)
- [`FOLDER_STRUCTURE.md`](FOLDER_STRUCTURE.md)
- [`src/app/README.md`](src/app/README.md)
- [`src/core/README.md`](src/core/README.md)
- [`src/render/README.md`](src/render/README.md)
- [`src/world/README.md`](src/world/README.md)
- [`src/multiplayer/README.md`](src/multiplayer/README.md)
- [`src/player/README.md`](src/player/README.md)
- [`src/ui/README.md`](src/ui/README.md)

Because the project changes quickly, some detailed module documentation can become stale.
When documentation and implementation disagree, prefer the current source code.

---

## 🤝 Contributing

Contributions are welcome.

Recommended workflow:

```bash
git checkout -b feature/my-change

cargo fmt
cargo clippy --all-targets
cargo test

git commit -m "feat: describe the change"
```

Please keep changes focused and update relevant documentation when changing public behavior
or architecture.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for additional project guidelines.

---

## 📦 Releases

The repository includes a GitHub Actions workflow that builds a Windows release when a
version tag matching `v*` is pushed. The workflow packages the executable together with
the `assets/` directory into a ZIP archive.

---

## 📄 License

Minerust is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.
