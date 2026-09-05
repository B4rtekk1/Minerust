# Minerust

Minerust is an experimental high-performance voxel sandbox and rendering engine written in Rust.

The project focuses on large procedural worlds, low CPU rendering overhead, asynchronous world streaming, and modern GPU-driven rendering techniques.

## Demo

https://github.com/user-attachments/assets/3f86d46e-7a33-4144-ae3d-f78887f2b1a7

## Features

### Rendering

* GPU-driven terrain rendering with `wgpu`
* Indirect and multi-draw indirect rendering
* GPU frustum and Hi-Z occlusion culling
* GPU face extraction for voxel terrain
* Packed quad vertex pulling
* Shared GPU geometry arenas
* Asynchronous mesh generation
* Procedural sky and sun rendering
* Transparent water with reflections, refraction and Fresnel effects
* Post-processing and GPU performance profiling

### World

* Procedural, seed-based terrain generation
* Infinite chunk-based world streaming
* Multi-threaded prioritized chunk generation
* Multiple biomes, caves, vegetation and water
* Automatic chunk loading and unloading
* Incremental remeshing after world modifications

### Gameplay

* Block breaking and placement
* Inventory and hotbar
* Item drops and pickups
* Tools and durability
* Player movement and collision
* World save/load system
* Early-stage TCP multiplayer and dedicated server

## Technology

Minerust is built primarily with:

* Rust 2024
* wgpu
* WGSL
* winit
* glam
* Tokio

The renderer is designed around minimizing CPU submission overhead by moving visibility determination and terrain processing toward the GPU.

## Running

Requirements:

* Rust toolchain
* GPU supported by `wgpu`
* Windows is currently the primary development platform

Clone the repository:

```bash
git clone https://github.com/B4rtekk1/Minerust.git
cd Minerust
```

Run the game:

```bash
cargo run --release
```

Run the dedicated server:

```bash
cargo run --release -- --server
```

Release mode is recommended when testing rendering and world generation performance.

## Controls

| Input        | Action              |
| ------------ | ------------------- |
| `W A S D`    | Move                |
| `Space`      | Jump                |
| `Left Shift` | Sprint              |
| Mouse        | Look around         |
| `LMB`        | Break block         |
| `RMB`        | Place block         |
| `E`          | Inventory           |
| `1-9`        | Select hotbar slot  |
| `F5`         | Save world          |
| `F9`         | Load world          |
| `C`          | Performance overlay |

## Status

Minerust is under active development. Rendering architecture, gameplay systems and world generation are still evolving.

The long-term goal is to build a scalable voxel sandbox while experimenting with modern real-time rendering and GPU-driven engine architecture.

## License

Licensed under the MIT License.
