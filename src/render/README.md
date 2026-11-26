# src/render/ - Rendering Pipeline Module

## Overview

The `render/` module provides packed terrain geometry, background mesh workers, GPU arena allocation, texture loading, and culling helpers. The app module owns and dispatches the frame pipelines; WGSL sources live in `src/shaders/`.

## Module Structure

```
render/
├── mod.rs              ← Module declarations and re-exports
├── depth.rs            ← Depth helpers
├── frustum.rs          ← Frustum planes and AABBs
├── indirect.rs         ← GPU arenas, subchunk metadata, indirect draws
├── mesh.rs             ← Block/item/player geometry helpers
├── mesh_loader.rs      ← Background mesh workers
├── quad.rs             ← Packed quad streams
├── shadow_grid.rs      ← Shadow-grid helper
├── texture.rs          ← Atlas texture loading
└── water_waves.rs      ← Water animation helper
```

## Current renderer behavior

- Terrain quads are packed and read by vertex-pulling shaders. GPU culling builds
  indirect draw lists for visible subchunks; the renderer keeps opaque/cutout and
  transparent water paths separate.
- Near and merged far terrain variants are stored in GPU memory. Culling selects the far
  variant from 160 blocks; it keeps block resolution and preserves outer-corner AO/light.
- A 2×2×2 terrain mesh is generated and uploaded, but culling does not select it because
  neighboring LOD boundaries are not stitched. It is an inactive experiment, not an
  enabled distance tier.
- The far mesh uses the same vertex-lighting and fog shader path as near terrain; there is
  no separate distant-lighting approximation.
- The detailed sections below include historical design sketches and should not be read
  as exact current Rust types, APIs, memory budgets, or measured performance figures.

## File Documentation

### `mod.rs` - Module Root
Declares the rendering helpers and re-exports `MeshLoader`, packed quad types, atlas
loading functions, frustum helpers, and indirect manager/budget types. The frame renderer
itself is coordinated by `src/app/render.rs`.

### `mesh.rs` - Model Geometry Helpers
`mesh.rs` contains geometry helpers for block outlines and item/player models. Terrain
geometry uses packed quads (`quad.rs`) and GPU-managed ranges (`indirect.rs`), not the
`MeshPool`/unified vertex-and-index-buffer design described in older notes.

### `mesh_loader.rs` - Mesh Generation
`MeshLoader` schedules bounded, deduplicated requests on background worker threads and
returns versioned `MeshResult`s. Results contain packed near terrain, merged far terrain,
stored 2×2×2 terrain LOD, detailed water, and merged far water quads; the app validates
versions and handles GPU allocation/upload on the main/render side.

### `atlas_map.rs` - Texture Atlas
This module contains a private helper for packing individual images into a square atlas;
it is not used by the current runtime texture-array path and exposes no public atlas API.

### `texture.rs` - Texture Management
`load_texture_atlas_from_file()` decodes the grid image and returns row-major RGBA texture
layers; `generate_texture_atlas()` provides procedural fallback layers. GPU upload, mipmap
generation, and the on-disk cache are handled in `src/app/texture_cache.rs`.

### `frustum.rs` - Frustum Culling
Provides `AABB` and `extract_frustum_planes()` helpers. The active GPU visibility tests,
including frustum and Hi-Z culling, are implemented in `src/shaders/cull.wgsl`.

### `indirect.rs` - Indirect GPU Drawing
`IndirectManager` owns growable packed-quad arenas and subchunk draw metadata; `DrawIndirect`,
`IndirectBufferBudget`, and `SubchunkKey` are the public data/configuration types. The
compute shader selects visible ranges and writes indirect commands. Actual dispatch and
the fallback for devices without indirect-count support are coordinated by the app renderer.

## Frame dispatch

There is no `src/render/passes/` directory. Frame setup and pass dispatch are in
`src/app/render.rs`; compute culling, terrain, and water shader code is in `src/shaders/`.

## Data Flow

```
Subchunk snapshot → worker builds packed near/far/2×2×2 and water quads
    ↓
App accepts current mesh result → allocate/upload to growable quad arenas
    ↓
Compute culling selects visible near/far and water ranges
    ↓
Indirect draws → terrain/water shaders → frame output
```

## Integration with Other Modules

```
render/ ←→ app/       (Dispatch render from game loop)
render/ ←→ core/      (Block data, chunks)
render/ ←→ player/    (Camera frustum)
render/ ←→ assets/    (Textures, fonts)
```

## Mesh benchmark

`cargo run --release --example mesh_bench -- 12345` reports CPU mesh time and quad counts;
it does not measure GPU frame time. One recorded run reported 122,507 far quads (-45.6%)
and a 32,141-quad LOD selection estimate (-85.7%, with fallback when LOD is unavailable).
Storing both variants added 154,363 quads (+68.6% versus near); CPU meshing was 0.614 ms
per non-empty subchunk. This is a seed- and machine-specific measurement. The LOD result
is not drawn by the current culler.

## Optimization Techniques

1. **Packed quad arenas** - Growable shared storage for terrain and water
2. **Indirect drawing** - GPU-generated visible draw lists
3. **Asynchronous workers** - Build chunk meshes away from the render loop
4. **Merged far terrain** - Fewer quads at the same block resolution from 160 blocks
5. **Stored 2×2×2 mesh** - Experimental; generated but not selected for drawing

---

**Key Takeaway:** The `render/` module transforms voxel data into GPU-rendered visuals using modern rendering techniques like GPU-driven rendering and frustum culling. It's optimized for both visual quality and performance.

