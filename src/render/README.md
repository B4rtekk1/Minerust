# src/render/ - Rendering Pipeline Module

## Overview

The `render/` module implements the complete GPU rendering pipeline using wgpu. It handles mesh generation from voxel data, texture management, GPU buffer allocation, and render pass execution. This module bridges the gap between CPU world data and GPU-accelerated drawing.

## Module Structure

```
render/
├── mod.rs              ← Module declaration and public API
├── mesh.rs             ← Mesh data structures and GPU buffers
├── mesh_loader.rs      ← Converts voxel chunks to meshes
├── atlas_map.rs        ← Texture atlas management
├── texture.rs          ← Texture loading and GPU binding
├── frustum.rs          ← Frustum culling and AABB structures
├── indirect.rs         ← Indirect GPU drawing (instancing)
└── passes/             ← Individual render passes
    ├── shadow.rs       ← Shadow map rendering (CSM)
    ├── terrain.rs      ← Main terrain rendering
    ├── water.rs        ← Water rendering with effects
    ├── composite.rs    ← Post-processing and composition
    └── ui.rs           ← UI element rendering
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides the rendering system API.

**Key Types:**
- `RenderContext` - Main rendering system container
- `Renderer` - High-level rendering interface

**Key Functions:**
- `new() → RenderContext` - Initialize rendering system
- `render_frame() → RenderResult` - Execute one frame

### `mesh.rs` - Mesh Data Structures
**Purpose:** Defines GPU mesh data and buffer management.

**Key Types:**

```rust
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer      // GPU vertex buffer
    pub index_buffer: wgpu::Buffer       // GPU index buffer
    pub num_indices: u32                 // Number of indices
    pub vertex_offset: u32               // Offset in unified buffer
    pub index_offset: u32                // Offset in unified buffer
    pub aabb: AABB                       // Bounding box
    pub is_dirty: bool                   // Needs update
}

pub struct MeshPool {
    pub vertex_buffer: wgpu::Buffer      // ~560MB unified buffer
    pub index_buffer: wgpu::Buffer       // ~256MB unified buffer
    pub allocations: Vec<MeshAllocation> // Track sub-allocations
}
```

**Buffer Architecture:**

Instead of allocating separate buffers per chunk, Render3D uses **large pre-allocated buffers**:

```
Unified Vertex Buffer (560MB)
├── Chunk 0: vertices [0..1000]
├── Chunk 1: vertices [1000..2500]
├── Chunk 2: vertices [2500..4200]
└── ... (up to ~10M vertices)

Unified Index Buffer (256MB)
├── Chunk 0: indices [0..3000]
├── Chunk 1: indices [3000..7500]
├── Chunk 2: indices [7500..12600]
└── ... (up to ~80M indices)
```

**Advantages:**
- Single draw call per subchunk type (terrain, water)
- Reduced CPU overhead
- Better GPU cache locality
- Easier batch optimization

**Key Functions:**
- `allocate(size) → Allocation` - Allocate space in unified buffer
- `free(allocation)` - Deallocate space
- `update_vertices(data)` - Upload vertex data
- `update_indices(data)` - Upload index data

**Mesh Constants:**
```rust
MAX_VERTICES = 10_000_000        // ~560MB at 28 bytes/vertex
MAX_INDICES = 80_000_000         // ~256MB at 4 bytes/index
```

### `mesh_loader.rs` - Mesh Generation
**Purpose:** Converts voxel chunk data into renderable mesh vertices and indices.

**Rendering Pipeline:**
```
SubChunk blocks
    ↓
Voxel-to-Face Conversion
    ↓
Face Culling (hidden surfaces)
    ↓
Normal Calculation & AO Baking
    ↓
Vertex Generation
    ↓
Mesh Data (vertices + indices)
    ↓
GPU Buffer Upload
```

**Key Algorithms:**

#### Greedy Meshing (Optional Optimization)
Combines adjacent same-type faces into larger quads for fewer vertices.

```
# Instead of:
■ ■ ■  (9 vertices for 3 cubes)

# Generate:
═══    (4 vertices for 1 large quad)
```

#### Face Culling
Only generates faces between different block types:

```
Stone | Air     → Generate face (visible)
Stone | Stone   → Skip face (hidden)
Stone | Water   → Generate face (different rendering)
```

#### Ambient Occlusion
Darkens corners where multiple blocks meet:

```
  Air         Stone
    ╱╲           ╱╲
   ╱  ╲  →      ╱  ╲  (darker corner)
  Stone  Stone Stone  Stone
```

**Key Functions:**
- `generate_mesh(subchunk) → MeshData` - Create mesh from blocks
- `calculate_ao(surrounding_blocks) → f32` - AO value (0.0 to 1.0)
- `cull_face(adjacent_block) → bool` - Should this face render?

**Mesh Output:**
- Vertices with position, UV, texture index, AO
- Indices for triangle rendering (6 per quad face)

### `atlas_map.rs` - Texture Atlas
**Purpose:** Manages texture atlasing and UV coordinate mapping.

**Texture Atlas Concept:**

Instead of binding many textures, combine all into one large texture:

```
Texture Atlas (2048×2048 or larger)
┌─────────┬─────────┬─────────┐
│ Grass   │ Dirt    │ Stone   │  Row 0
├─────────┼─────────┼─────────┤
│ Sand    │ Water   │ Wood    │  Row 1
├─────────┼─────────┼─────────┤
│ Leaves  │ Snow    │ Gravel  │  Row 2
└─────────┴─────────┴─────────┘

Each block in mesh gets:
- Texture index (0-60)
- UV coordinates within atlas
```

**Key Types:**
```rust
pub struct AtlasMap {
    pub texture: wgpu::Texture
    pub sampler: wgpu::Sampler
    pub mappings: HashMap<String, AtlasEntry>
}

pub struct AtlasEntry {
    pub uv_offset: [f32; 2]      // (0.0-1.0, 0.0-1.0)
    pub uv_scale: [f32; 2]       // Size in atlas
    pub texture_index: f32        // Layer in atlas
}
```

**Key Functions:**
- `new() → AtlasMap` - Load and create atlas
- `get_uv(block_name) → AtlasEntry` - Look up UV coordinates
- `bind_to_pass(pass) → BindGroup` - Create GPU binding

**Atlas Metadata (atlas_map_structure.json):**
```json
{
  "textures": {
    "grass_top": {
      "x": 0,
      "y": 0,
      "width": 16,
      "height": 16
    },
    "grass_side": {
      "x": 16,
      "y": 0,
      "width": 16,
      "height": 16
    }
  }
}
```

### `texture.rs` - Texture Management
**Purpose:** Loads texture files and manages GPU texture resources.

**Key Types:**
```rust
pub struct TextureManager {
    pub atlas: wgpu::Texture
    pub bind_group: wgpu::BindGroup
    pub sampler: wgpu::Sampler
}
```

**Supported Formats:**
- PNG (primary format)
- JPEG
- TARGA (.tga)
- Any format supported by `image` crate

**Texture Loading Pipeline:**
```
PNG File
    ↓
image::open()
    ↓
RGBA conversion
    ↓
wgpu::Texture creation
    ↓
GPU upload
    ↓
Sampler binding
```

**Mipmap Generation:**
- Auto-generates mipmaps for distant textures
- Improves performance and reduces aliasing
- Transparent blocks get special handling

**Key Functions:**
- `load(path) → Texture` - Load texture from file
- `create_bind_group() → BindGroup` - Create GPU binding
- `get_sampler() → Sampler` - Texture sampling parameters

### `frustum.rs` - Frustum Culling
**Purpose:** Defines view frustum and AABB structures for visibility culling.

**Frustum Culling:**

Only render chunks inside the camera's view pyramid:

```
       Eye
        ╱│╲
╱ │ ╲  FOV (90°)
      ╱  │  ╲
     ╱───┼───╲
    │   visible   │
    │   chunks    │  ← Only these render
    │             │
    ╱─────────────╲
   far plane
```

**Key Types:**
```rust
pub struct Frustum {
    pub planes: [Plane; 6]  // Near, Far, Left, Right, Top, Bottom
}

pub struct AABB {
    pub min: Vec3,          // Minimum corner
    pub max: Vec3           // Maximum corner
}

pub struct Plane {
    pub normal: Vec3
    pub distance: f32       // Distance from origin
}
```

**Culling Test:**
```rust
if frustum.intersects(&subchunk.aabb) {
    render_subchunk();  // Visible
} else {
    skip_subchunk();    // Outside frustum
}
```

**Key Functions:**
- `from_matrices(view, proj) → Frustum` - Create frustum from camera matrices
- `intersects(aabb) → bool` - Is AABB visible?
- `contains_point(point) → bool` - Is point visible?

**Optimization:** Frustum culling removes ~80-90% of subchunks from rendering.

### `indirect.rs` - Indirect GPU Drawing
**Purpose:** Manages GPU-driven indirect drawing (commands generated on GPU).

**Indirect Drawing Pattern:**

Normal CPU approach:
```cpp
for (chunk in visible_chunks) {
    draw_call(chunk);  // Many CPU draw calls
}
```

GPU-driven approach:
```
Compute Shader
    ├── Frustum cull all chunks (GPU)
    ├── Generate draw commands (GPU)
    └── Write to indirect buffer (GPU)
    
GPU Render Pass
    └── Execute indirect buffer (single draw call!)
```

**Key Types:**
```rust
pub struct IndirectBuffer {
    pub buffer: wgpu::Buffer
    pub commands: Vec<DrawIndirectCommand>
    pub count: u32
}

#[repr(C)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32
    pub instance_count: u32
    pub base_vertex: u32
    pub base_instance: u32
}
```

**Advantages:**
- GPU generates draw commands (culling happens on GPU)
- Single CPU draw call handles all visible chunks
- Dramatically reduces CPU→GPU communication
- Better for large numbers of objects

**Key Functions:**
- `new() → IndirectBuffer` - Create indirect buffer
- `update_commands(results) → ()` - Rebuild command buffer
- `dispatch_draw(pass) → ()` - Execute on GPU

## Render Passes Directory (`passes/`)

### `shadow.rs` - Shadow Mapping
Renders scene to shadow maps for lighting:

```
Directional Light (Sun)
    ↓
View from light direction
    ↓
4 Cascaded Shadow Maps (CSM)
    - Cascade 0: Close (16m) - 2048x2048
    - Cascade 1: Medium (48m) - 2048x2048
    - Cascade 2: Far (128m) - 2048x2048
    - Cascade 3: Very Far (300m) - 2048x2048
    ↓
Percentage-Closer Filtering (PCF)
    ↓
Soft shadows in terrain pass
```

### `terrain.rs` - Main Terrain Pass
Renders solid opaque blocks with:
- Frustum culling
- Shadow mapping
- Ambient occlusion
- Dynamic lighting

### `water.rs` - Water Rendering
Special effects for water blocks:
- Vertex displacement (wave animation)
- Fresnel reflections
- Specular highlights
- Refraction distortion

### `composite.rs` - Post-Processing
Combines all passes and applies post-effects:
- Bloom/God rays
- Color correction
- Atmospheric fog
- FXAA anti-aliasing

### `ui.rs` - UI Rendering
Renders 2D UI elements:
- Text rendering (via `glyphon`)
- HUD elements
- Menus and overlays

## Data Flow

```
SubChunk blocks
    ↓
mesh_loader.rs (generate_mesh)
    ↓
Vertex/Index data
    ↓
mesh.rs (allocate + upload)
    ↓
GPU Buffers
    ↓
Culling pass (frustum.rs)
    ↓
Indirect commands (indirect.rs)
    ↓
Render passes (passes/*.rs)
    ↓
Screen output
```

## Integration with Other Modules

```
render/ ←→ app/       (Dispatch render from game loop)
render/ ←→ core/      (Block data, chunks)
render/ ←→ player/    (Camera frustum)
render/ ←→ assets/    (Textures, fonts)
```

## Performance Characteristics

### Memory Usage
- **Vertex Buffer**: ~560MB for ~10M vertices
- **Index Buffer**: ~256MB for ~80M indices
- **Texture Atlas**: 2-4GB depending on quality
- **Shadow Maps**: ~64MB (4 cascades × 2048²)

### GPU Draw Efficiency
- **Visible Subchunks**: 100-500 (depends on render distance)
- **Total Draw Calls**: 3-6 (one per render pass)
- **Triangles Per Frame**: 1-50 million
- **Frame Time Budget**: 16ms for 60 FPS

### Culling Efficiency
- **Frustum Culling**: Removes ~80% of subchunks
- **Occlusion Culling**: Further reduces draw cost
- **Mesh Generation**: ~4-6 subchunks per frame (frame budget)

## Optimization Techniques

1. **Unified Buffers** - Minimize per-object overhead
2. **Indirect Drawing** - GPU-driven culling
3. **Mesh Caching** - Only regenerate dirty chunks
4. **Texture Atlasing** - Single texture bind
5. **LOD System** - Distant chunks at lower quality
6. **Early Depth Testing** - Z-prepass for efficiency

---

**Key Takeaway:** The `render/` module transforms voxel data into stunning GPU-rendered visuals using modern rendering techniques like GPU-driven rendering, frustum culling, and shadow mapping. It's optimized for both visual quality and performance.

