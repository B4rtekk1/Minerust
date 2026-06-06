# src/core/ - Core World Systems Module

## Overview

The `core/` module defines the fundamental data structures that represent the Minerust world. It includes block types, chunk storage, biome definitions, and GPU data structures (vertices, uniforms). This is the low-level foundation upon which all world simulation and rendering is built.

## Module Structure

```
core/
├── mod.rs         ← Module declaration and public API
├── block.rs       ← Block type definitions and properties
├── chunk.rs       ← Chunk and SubChunk data structures
├── biome.rs       ← Biome definitions and generation parameters
├── vertex.rs      ← Vertex data structures for GPU
└── uniforms.rs    ← GPU uniform buffer structures
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and re-exports the public API.

**Key Re-exports:**
- `BlockType` - All block variants
- `Chunk`, `SubChunk` - World storage structures
- `Biome` - Biome definitions
- `Vertex` - GPU vertex data
- `Uniforms` - GPU uniform data

### `block.rs` - Block Type System
**Purpose:** Defines all block types and their properties.

**Key Types:**
```rust
pub enum BlockType {
    Air,              // Fully transparent, not solid
    Grass,            // Grass block (varied top/side)
    Dirt,             // Dirt block
    Stone,            // Stone block
    Sand,             // Sand/desert block
    Water,            // Water - liquid block
    Wood,             // Wood/log block
    Leaves,           // Foliage - semi-transparent
    Bedrock,          // Bedrock - unbreakable
    Snow,             // Snow block
    Gravel,           // Gravel block
    Clay,             // Clay block
    Ice,              // Ice - transparent solid
    Cactus,           // Cactus - desert plant
    DeadBush,         // Dead vegetation
    WoodStairs,       // Stairs variant
}
```

**Key Methods:**

- `color(&self) -> [f32; 3]` - RGB color for block (for debugging/diagnostics)
- `top_color(&self) -> [f32; 3]` - Color variation for top face
- `is_solid(&self) -> bool` - Can be walked on / blocks movement
- `is_opaque(&self) -> bool` - Blocks light and vision
- `is_solid_opaque(&self) -> bool` - Both solid AND opaque
- `is_liquid(&self) -> bool` - Is water/lava
- `get_texture_index(&self, face: Face) -> f32` - Texture index for rendering

**Block Categories:**

| Category | Examples | Properties |
|----------|----------|------------|
| Solid Opaque | Stone, Dirt, Wood | Blocks light, blocks movement |
| Solid Transparent | Ice, Leaves | Transparent but solid |
| Liquid | Water | Flows, transparent |
| Non-solid | Air, DeadBush | Can walk through |

**Texture Indices:**
```rust
// Defined in constants.rs
TEX_GRASS_TOP = 0.0
TEX_GRASS_SIDE = 1.0
TEX_DIRT = 2.0
TEX_STONE = 3.0
// ... etc
```

### `chunk.rs` - World Storage (SubChunk & Chunk)
**Purpose:** Data structures for storing voxel data and chunk metadata.

**Key Types:**

#### SubChunk
Represents a 16×16×16 section of voxels (one "vertical slice" of a chunk).

```rust
pub struct SubChunk {
    pub blocks: [[[BlockType; CHUNK_SIZE]; SUBCHUNK_HEIGHT]; CHUNK_SIZE]
    pub is_empty: bool              // Optimization: entire subchunk is air
    pub mesh_dirty: bool            // Optimization: needs remeshing
    pub num_indices: u32            // GPU mesh data - terrain
    pub num_water_indices: u32      // GPU mesh data - water
    pub aabb: AABB                  // Bounding box for frustum culling
    pub is_fully_opaque: bool       // Optimization: no transparency
}
```

**SubChunk Methods:**

- `new(chunk_x, subchunk_y, chunk_z) → SubChunk` - Create empty subchunk
- `get_block(x, y, z) → BlockType` - Read block at position (with bounds checking)
- `set_block(x, y, z, block)` - Write block and mark mesh dirty
- `check_empty()` - Scan entire subchunk to determine if empty
- `check_fully_opaque()` - Scan entire subchunk to determine opacity

**Block Access Pattern:**
```
World Coordinates → Subchunk Index → Local Coordinates → Array Access

y=100 → subchunk_idx = 100/16 = 6
        local_y = 100 % 16 = 4
```

#### Chunk
Represents a 16×256×16 column (one full height "pillar" of subchunks).

```rust
pub struct Chunk {
    pub subchunks: Vec<SubChunk>    // 16 subchunks vertically
    pub player_modified: bool       // Track if player built here
}
```

**Chunk Methods:**

- `new(x, z) → Chunk` - Create new chunk with empty subchunks
- `get_block(x, y, z) → BlockType` - Get block at world coordinates
- `set_block(x, y, z, block)` - Set block at world coordinates

**Chunk Structure:**
```
One Chunk = 16 SubChunks stacked vertically

Height 256 ┌─────────────────┐
           │  SubChunk 15    │  y: 240-255
           │                 │
Height 240 ├─────────────────┤
           │  SubChunk 14    │  y: 224-239
           │                 │
    ...    ├─────────────────┤
           │      ...        │
Height 16  ├─────────────────┤
           │  SubChunk 1     │  y: 16-31
           │                 │
Height 0   ├─────────────────┤
           │  SubChunk 0     │  y: 0-15  (Bedrock layer)
           │                 │
           └─────────────────┘
           16×16 blocks horizontally
```

**World Constants:**
```rust
CHUNK_SIZE = 16              // Horizontal size
SUBCHUNK_HEIGHT = 16         // Vertical size of one subchunk
WORLD_HEIGHT = 256           // Total world height
NUM_SUBCHUNKS = 16           // Number of subchunks per column
```

### `biome.rs` - Biome System
**Purpose:** Defines biome characteristics and generation parameters.

**Key Types:**
```rust
pub enum BiomeType {
    Plains,
    Forest,
    Desert,
    Mountain,
    Swamp,
    Ocean,
    // ... 11 total biomes
}
```

**Biome Properties:**
- Temperature (affects water/ice)
- Humidity (affects vegetation)
- Height scale (mountain vs flat)
- Terrain roughness
- Base block color
- Vegetation types

**Biome Generation Parameters:**
- Noise scale for terrain
- Height multiplier
- Cave frequency
- Tree spawn rates
- Ore distribution

**Usage in Generation:**
```
Noise Sample → Biome Lookup → Generation Rules → Block Placement
```

### `vertex.rs` - GPU Vertex Data
**Purpose:** Defines vertex data structures sent to GPU shaders.

**Key Types:**

```rust
#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3]           // XYZ position
    pub tex_coord: [f32; 2]          // UV texture coordinates
    pub tex_index: f32               // Which texture in atlas
    pub ambient_occlusion: f32       // AO baking for shading
    pub normal: [f32; 3]             // Surface normal (optional)
}
```

**Vertex Layout:**
- Position: 3 floats (12 bytes)
- TexCoord: 2 floats (8 bytes)
- TexIndex: 1 float (4 bytes)
- AO: 1 float (4 bytes)
- **Total: 28 bytes per vertex**

**Used in:**
- Mesh generation from voxel data
- Uploading to GPU vertex buffers
- Shader input assembly

### `uniforms.rs` - GPU Uniform Buffers
**Purpose:** Defines data structures for GPU uniform buffers (constants for all vertices/pixels).

**Key Types:**

```rust
#[repr(C)]
pub struct CameraUniforms {
    pub view_matrix: [[f32; 4]; 4]
    pub proj_matrix: [[f32; 4]; 4]
    pub view_pos: [f32; 3]
}

#[repr(C)]
pub struct LightingUniforms {
    pub sun_direction: [f32; 3]
    pub sun_color: [f32; 3]
    pub ambient_light: [f32; 3]
}
```

**Usage:**
- Passed to shaders as bind group
- Updated once per frame
- Used for camera and lighting

**Update Frequency:**
- Camera uniforms: Every frame
- Lighting uniforms: Every frame or less

## Data Relationships

```
BlockType (enum)
    ↓
    ├→ Properties (solid, opaque, texture)
    
SubChunk (16×16×16 array)
    ↓
    ├→ blocks[x][y][z]: BlockType
    ├→ GPU data: num_indices, num_water_indices
    ├→ Metadata: is_empty, mesh_dirty, aabb
    
Chunk (Column of SubChunks)
    ↓
    ├→ subchunks: Vec<SubChunk>
    ├→ player_modified: bool
    
Vertex (GPU data)
    ↓
    ├→ Generated from chunk/subchunk block data
    ├→ Sent to GPU vertex buffer
    
Uniforms (GPU constants)
    ↓
    ├→ Camera data
    ├→ Lighting data
```

## Performance Considerations

### Memory Layout
- **Block Array**: Stored as `[x][y][z]` for cache locality
- **Subchunk Size**: 16×16×16 chosen for balance of:
  - L3 cache fit
  - Mesh generation granularity
  - Network transmission efficiency

### Optimizations

1. **Empty SubChunk Tracking**
   ```rust
   if subchunk.is_empty {
       skip_rendering();  // Don't generate mesh
   }
   ```

2. **Fully Opaque Tracking**
   ```rust
   if subchunk.is_fully_opaque {
       enable_back_face_culling();  // Skip internal faces
   }
   ```

3. **Mesh Dirty Flag**
   ```rust
   if subchunk.mesh_dirty {
       regenerate_mesh();  // Only when blocks change
   }
   ```

4. **AABB Caching**
   ```rust
   if camera_frustum.intersects(subchunk.aabb) {
       render_subchunk();  // Only visible chunks
   }
   ```

## Integration with Other Modules

```
core/ ←→ world/       (Generate blocks)
core/ ←→ render/      (Convert to vertices)
core/ ←→ app/         (Update blocks, track changes)
core/ ←→ multiplayer/ (Serialize/deserialize chunks)
```

## Typical Usage Flow

### World Generation:
```
BiomeType → Biome params → Generate BlockType → Store in SubChunk
```

### Rendering:
```
SubChunk blocks → Mesh Generation → Vertices → GPU Buffer → Render
```

### Block Modification:
```
Player action → set_block() → mark mesh_dirty → Remesh → Reupload GPU
```

## Type Safety

The module uses Rust's type system extensively:
- `BlockType` is a `Copy` enum (efficient)
- `Chunk` and `SubChunk` use bounds checking in getters
- GPU structs use `#[repr(C)]` for memory safety
- No unsafe code in public API

---

**Key Takeaway:** The `core/` module provides the essential building blocks for the voxel world: block types with properties, efficient chunk storage, biome definitions, and GPU-compatible data structures. It's designed for both CPU simulation efficiency and GPU rendering performance.

