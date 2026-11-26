# src/world/ - World Generation & Terrain Module

## Overview

The `world/` module handles seeded generation, chunk streaming, world state, terrain meshing, sky light, structures, and dropped item entities. See `generator.rs`, `loader.rs`, `terrain.rs`, and `skylight.rs` for the current implementation.

## Module Structure

```
world/
├── mod.rs              ← Module declaration and public API
├── generator.rs        ← Noise-based procedural generation
├── terrain.rs          ← World state, chunk access, terrain/water meshing
├── loader.rs           ← Chunk loading/unloading system
├── skylight.rs         ← Sky-light propagation
├── item_entity.rs      ← Dropped item entities
└── structures/         ← Procedural structure generation
    ├── mod.rs
    ├── structure.rs
    └── house.rs
```

## Current terrain mesh variants

The mesher can produce a detailed near mesh, a merged far mesh, and a 2×2×2 LOD mesh.
GPU culling selects the far mesh at 160 blocks; both use the same block resolution, and
the far merge retains its outer-corner AO/light. The 2×2×2 mesh is generated and stored,
but is not selected for drawing because subchunk boundaries are not stitched. It is not
an active LOD tier.

The remaining sections below contain older design sketches, including proposed APIs and
generation settings that may not exist in the current implementation. Consult the Rust
source for exact behavior.

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides the world management API.

**Key Types:**
- `World` - Main world container
- `WorldGenerator` - Generation system

**Key Functions:**
- `new() → World` - Create new world
- `generate_chunk(x, z) → Chunk` - Generate single chunk
- `load_chunk(x, z) → Chunk` - Load or generate chunk

### `generator.rs` - Procedural Generation
**Purpose:** Core noise-based world generation using FastNoise.

**Generation Pipeline:**
```
Seed (deterministic)
    ↓
Chunk Coordinates (x, z)
    ↓
Noise Sampling (multiple scales)
    ↓
Biome Lookup
    ↓
Terrain Height Calculation
    ↓
Block Placement
    ↓
Feature Placement (trees, ores, etc.)
    ↓
Final Chunk Data
```

**Noise Types Used:**

#### 1. **Terrain Noise** (FBM - Fractional Brownian Motion)
```
Multiple octaves of Perlin noise combined:
Octave 1: Large-scale terrain (mountains, plains)
Octave 2: Medium-scale features (hills)
Octave 3: Small-scale details (texture)

Result: Smooth, natural-looking terrain
```

#### 2. **Biome Noise**
Determines which of 11 biomes occupies each block:
- Plains, Forest, Desert, Mountain, Swamp, Ocean, etc.

#### 3. **Cave Noise** (Multiple types)
- **Cheese Caves**: Large organic caverns
- **Spaghetti Caves**: Thin winding tunnels

**Key Types:**
```rust
pub struct WorldGenerator {
    pub seed: u32                       // Deterministic seed
    pub noise: FastNoise               // FastNoise instance
    pub biome_data: HashMap<(i32, i32), BiomeType>
}

pub struct GenerationOptions {
    pub chunk_size: i32
    pub world_height: i32
    pub sea_level: i32
    pub cave_frequency: f32
    pub ore_frequency: f32
}
```

**Key Functions:**
- `new(seed) → WorldGenerator` - Create generator with seed
- `generate_height(x, z) → i32` - Get terrain height at position
- `get_biome(x, z) → BiomeType` - Get biome at position
- `generate_chunk(chunk_x, chunk_z) → Chunk` - Generate full chunk

**Noise Parameters:**
```rust
// Terrain generation
TERRAIN_SCALE = 100.0              // Noise scale
HEIGHT_MULTIPLIER = 50.0           // Height variation
OCTAVE_COUNT = 3                   // Noise octaves
FREQUENCY_MULTIPLIER = 2.0         // Octave frequency increase

// Caves
CAVE_THRESHOLD = 0.25              // Cutoff for carving
CAVE_FREQUENCY = 0.08              // How common caves are

// Ores
ORE_FREQUENCY = 0.04               // How common ore veins are
```

**Determinism:**
```
Same seed → Same world layout
seed = 12345
generate_chunk(0, 0) always produces same blocks
```

### `terrain.rs` - Terrain Features
**Purpose:** Implements specific terrain features like caves, mountains, and water.

**Feature Generation Order:**
```
1. Base terrain (height field from noise)
2. Caves (carve air from solid blocks)
3. Mountains (apply height variation)
4. Water (place water at sea level)
5. Sand/Gravel beaches
6. Snow at high altitudes
```

**Key Features:**

#### **Caves System**
```rust
pub enum CaveType {
    Cheese,     // Large organic caverns
    Spaghetti,  // Thin winding tunnels
}
```

Carves out caves from terrain:
```
Terrain noise: if value < CAVE_THRESHOLD { place Air }
Result: Natural-looking underground caverns
```

#### **Mountain System**
```
Base terrain
    ↓
Heightmap sampling
    ↓
Steepness calculation
    ↓
Apply mountain effects (cliffs, peaks)
    ↓
Result: Varied, dramatic terrain
```

#### **Water System**
```
Sea Level = 64
    ↓
At height ≤ 64: place Water block
At height 63-64: Sand beaches
At height > 64: Normal terrain
```

#### **Ore Distribution**
```
For each ore type:
  - Define height range (e.g., diamonds 0-16)
  - Sample noise to find vein locations
  - Generate vein shape (spherical or stretched)
  - Place ore blocks
```

**Ore Types (Minecraft-inspired):**
- **Coal**: Common, high elevation
- **Iron**: Medium, throughout
- **Gold**: Rare, low elevation
- **Diamonds**: Very rare, very low elevation
- **Emeralds**: Rare, mountain biomes
- **Lapis**: Rare, low elevation
- **Redstone**: Rare, low elevation

**Key Functions:**
- `generate_caves(chunk) → ()` - Carve out cave systems
- `apply_mountains(chunk) → ()` - Add mountain features
- `place_water(chunk) → ()` - Place water blocks
- `place_ores(chunk) → ()` - Distribute ore veins

### `loader.rs` - Chunk Loading/Unloading
**Purpose:** Manages chunk streaming based on player position.

**Chunk Loading System:**

```
Player Position
    ↓
Calculate Required Chunks
  (in radius = RENDER_DISTANCE)
    ↓
Unload Distant Chunks
    ↓
Load/Generate Missing Chunks
    ↓
Async Generation Thread
    ├── Generate terrain
    └── Build mesh result
    ↓
Main thread accepts current result and uploads geometry
```

**Key Constants:**
```rust
RENDER_DISTANCE = 32        // Chunk radius used by the active app
SIMULATION_DISTANCE = 16    // Derived from render distance
GENERATION_DISTANCE = 34    // Two chunks beyond render distance
CHUNK_UNLOAD_DISTANCE = 37  // Five chunks beyond render distance
```
These are constants from `src/constants.rs`. `src/utils/settings.rs` contains an
unwired prototype with different defaults; it does not override the active distances.

**Load Distance Calculation:**
```
Player at chunk (10, 10)
RENDER_DISTANCE = 32

The active app considers chunks within a circular radius, not a fixed 20×20 square.
The number actually loaded varies with generated/available chunks and streaming state.
```

**Key Types:**
```rust
pub struct ChunkLoader {
    pub loaded_chunks: HashMap<(i32, i32), Arc<Chunk>>
    pub generation_queue: VecDeque<ChunkGenRequest>
    pub workers: Vec<GenerationWorker>
}

pub struct ChunkGenRequest {
    pub x: i32
    pub z: i32
    pub priority: u32    // Closer chunks have higher priority
}
```

**Key Functions:**
- `update(player_pos) → ()` - Update loaded chunks based on position
- `load_chunk(x, z) → Option<Chunk>` - Get chunk (load if needed)
- `unload_chunk(x, z) → ()` - Unload chunk
- `queue_generation(x, z) → ()` - Add to generation queue

**Async Generation:**
```
Main Thread                 Worker Thread
    ├─ Update positions          ├─ Request: gen_chunk(0, 0)
    ├─ Queue chunks              ├─ Generate terrain
    │                            ├─ Place ores
    │                            ├─ Build mesh
    │ <── Completed chunk ←───────┤
    ├─ Upload to GPU
    └─ Next frame
```

**Memory Management:**
- Chunks are lazily loaded
- Distant chunks are unloaded automatically
- Save format (world.minerust) stores modified chunks
- Unmodified generated chunks can be regenerated

### `spline.rs` - Interpolation Utilities
**Purpose:** Smooth interpolation between noise values.

**Interpolation Methods:**
```rust
pub fn linear(a: f32, b: f32, t: f32) -> f32
pub fn smoothstep(a: f32, b: f32, t: f32) -> f32
pub fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32
```

**Usage:**
```
Noise sample at (x, z)
    ↓
Interpolate between noise values
    ↓
Smooth terrain (less blocky)
    ↓
Natural-looking heightmap
```

**Smoothstep Interpolation:**
```
Linear:          Smoothstep:
│     ╱           │    ╭─
│    ╱            │   ╱
│   ╱             │  ╱
│  ╱              │╱
```

### `structures/` - Procedural Structures
**Purpose:** Generate complex structures like trees, villages, and dungeons.

**Structure Types:**
- **Trees**: Different types per biome (oak, spruce, birch, etc.)
- **Villages**: Settlements with houses, paths
- **Dungeons**: Underground rooms with spawners
- **Desert Temples**: Pyramid structures
- **Ocean Monuments**: Underwater structures
- **Nether Fortresses**: Floating structures

**Typical Structure Generation:**
```
Chunk generation
    ↓
Sample structure noise
    ↓
If noise > SPAWN_THRESHOLD:
    ├─ Determine structure type (biome-dependent)
    ├─ Generate structure at location
    └─ Place blocks
    ↓
Next structure
```

**Tree Generation Example:**
```
Decision: Place oak tree at (100, 64, 50)?
    ↓
Check: Is ground solid? Is there space above?
    ↓
If yes:
    ├─ Generate trunk (3×3×8 blocks)
    ├─ Generate foliage (sphere of leaves)
    └─ Apply randomization
    ↓
Oak tree placed
```

**Key Function Pattern:**
```rust
pub fn generate_structure(chunk: &mut Chunk, structure_type: StructureType) {
    // Generate structure
    // Place blocks in chunk
}
```

## World Generation Biomes

The engine supports 11 distinct biomes:

| Biome | Climate | Height | Vegetation | Features |
|-------|---------|--------|-----------|----------|
| Plains | Temperate | Low | Grass, flowers | Flat terrain |
| Forest | Temperate | Low-Med | Trees, grass | Dense trees |
| Desert | Hot/Dry | Med | Sand, cactus | Sparse vegetation |
| Mountain | Cold | High | Sparse veg | Tall peaks, cliffs |
| Swamp | Wet | Low | Trees, water | Lots of water |
| Ocean | Aquatic | Underwater | Kelp, corals | Deep water |
| Jungle | Tropical | Med-High | Dense trees | Tall trees |
| Taiga | Cold | Low-Med | Spruce, snow | Boreal forest |
| Tundra | Frigid | Low | Snow, ice | Sparse veg |
| Badlands | Hot/Dry | High | Terracotta | Colorful cliffs |
| The Nether | Hellish | N/A | Lava, fungal | Flotsam |

**Biome Selection:**
```
Temperature noise + Humidity noise
    ↓
2D plot (Temp vs Humidity)
    ↓
Lookup biome from plot
    ↓
Apply biome-specific generation
```

## Data Flow

```
Player moves
    ↓
loader.rs: Update required chunks
    ↓
Queue new chunks for generation
    ↓
generator.rs + terrain.rs (on worker thread)
    ├── Generate noise
    ├── Place blocks
    ├── Apply terrain features
    └── Place structures
    ↓
structures/: Add trees, buildings, etc.
    ↓
Complete chunk
    ↓
render/: Build mesh
    ↓
GPU: Render chunk
```

## Performance Optimization

### Generation Budgeting
```rust
pub const MAX_CHUNKS_PER_FRAME: usize = 4;  // Limit generation rate
```
- Don't generate all chunks at once (would stutter)
- Generate 4 per frame (~60 FPS)
- Pre-generate beyond render distance

### Multithreading
```
Main Thread (game loop)
    ├─ Player input
    ├─ Rendering
    └─ Queue chunk gen
    
Worker Threads (4×)
    ├─ Generate chunk 0
    ├─ Generate chunk 1
    ├─ Generate chunk 2
    └─ Generate chunk 3
```

### Noise Caching
- Cache noise values to avoid recomputation
- Biome decisions are deterministic (stored)
- Chunk data persists in memory

### Seed System
```rust
Random seed: 12345
    ↓
All generation deterministic from seed
    ↓
Same world layout every time
    ↓
Can share seed with friends
```

## Integration with Other Modules

```
world/ ←→ core/       (Generate blocks, place in chunks)
world/ ←→ render/     (Mesh generation after chunk creation)
world/ ←→ app/        (Load chunks based on player position)
world/ ←→ save/       (Save/load modified chunks)
world/ ←→ multiplayer/(Sync chunks with clients)
```

## World Saving

Modified chunks are saved to `world.minerust`:
```
world.minerust (Binary format)
├── Header (version, world properties)
├── Modified chunk 0: x=10, z=5
├── Modified chunk 1: x=10, z=6
└── ... (only changed chunks stored)

Unmodified generated chunks: Regenerated from seed
```

**Advantages:**
- Small file size (only modified chunks)
- Reproducible generation
- Can update world generation without corrupting saves

## Common World Generation Parameters

```yaml
# settings.yaml
world:
  seed: 12345
  render_distance: 10
  simulation_distance: 5
  generation_distance: 12
  sea_level: 64
  terrain_scale: 100.0
  height_multiplier: 50.0
  cave_frequency: 0.08
  ore_frequency: 0.04
```

---

**Key Takeaway:** The `world/` module creates infinite, beautiful, deterministic worlds using noise-based procedural generation combined with targeted feature placement. It's designed for both visual variety and computational efficiency through async streaming and multithreaded generation. This is the foundation of the Minerust experience.

