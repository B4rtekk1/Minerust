# src/app/ - Application Logic Module

## Overview

The `app/` module is the heart of the Minerust application. It manages the main game loop, window initialization, GPU device management, input handling, and overall application state. This module orchestrates all other systems (rendering, world generation, networking, UI).

## Module Structure

```
app/
├── mod.rs              ← Module declaration and public API
├── init.rs             ← Initialization (window, GPU device)
├── game.rs             ← Main game instance
├── state.rs            ← Application state management
├── input.rs            ← Input event handling
├── update.rs           ← Game state updates
├── render.rs           ← Rendering pipeline dispatch
├── resize.rs           ← Window resize handling
├── server.rs           ← Server functionality
└── texture_cache.rs    ← Texture loading and caching
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides the main entry point.

**Key Functions:**
- `pub fn run_game()` - Main entry point that starts the application

**Responsibilities:**
- Creates the window and GPU device
- Initializes all subsystems
- Runs the main game loop
- Handles shutdown

### `init.rs` - Initialization
**Purpose:** Sets up the window, GPU device, and all rendering infrastructure.

**Key Types:**
- `InitConfig` - Configuration for initialization
- GPU device and queue setup
- Swapchain initialization

**Responsibilities:**
- Create `winit` window with proper settings
- Initialize `wgpu` device and queue
- Set up surface and swapchain
- Create render targets and pipelines
- Initialize texture atlas

### `game.rs` - Game Instance
**Purpose:** Main game state container and orchestrator.

**Key Types:**
- `Game` - Main game struct holding all subsystems
  - Player state
  - World chunks
  - Rendering data
  - UI state
  - Network connections

**Responsibilities:**
- Hold references to all game systems
- Coordinate system updates
- Manage game flow (paused, playing, menu)
- Track delta time and frame timing

### `state.rs` - Application State
**Purpose:** Manages overall application state and transitions.

**Key Types:**
- `AppState` - Enum for game state (Menu, Playing, Paused, Settings)
- `GameState` - Detailed gameplay state

**State Transitions:**
```
Menu → Playing → Paused → Playing → Exit
       ↓
    Settings (from any state)
```

**Responsibilities:**
- Track current application state
- Manage state transitions
- Handle pause/resume logic

### `input.rs` - Input Handling
**Purpose:** Processes keyboard, mouse, and controller input events.

**Key Systems:**
- Keyboard input (WASD movement, space jump, shift sprint)
- Mouse input (camera look)
- Window events (close, resize, focus)
- Input binding system

**Responsibilities:**
- Convert `winit` input events to game actions
- Update player movement state
- Handle special keys (pause, settings, screenshot)
- Track input state for next frame

**Typical Key Bindings:**
```
W/A/S/D       → Move (forward/left/back/right)
Space         → Jump
Shift         → Sprint
Mouse         → Look around
Esc           → Pause menu
F1            → Toggle crosshair + hotbar
F3            → Toggle debug info
F11           → Fullscreen
```

### `update.rs` - Game Updates
**Purpose:** Updates game state each frame.

**Update Order:**
1. Input processing
2. Player physics (movement, gravity, collision)
3. Chunk loading/unloading
4. Mesh generation for dirty chunks
5. Network synchronization
6. Entity updates
7. Prepare render data

**Responsibilities:**
- Apply player movement
- Update camera position
- Load chunks based on player position
- Trigger chunk mesh generation
- Update animations (water, leaves, etc.)
- Sync multiplayer state
- Update UI state

### `render.rs` - Rendering Dispatch
**Purpose:** Orchestrates the rendering pipeline.

**Render Passes (in order):**
1. **Terrain Pass** - Render solid blocks and write depth
2. **Depth Resolve + Hi-Z** - Build next-frame occlusion data from opaque depth
3. **Water Pass** - Render water with special effects
4. **Composite Pass** - Combine passes, apply post-effects
5. **UI Pass** - Render text, buttons, HUD

**Responsibilities:**
- Set up render passes
- Bind pipelines and buffers
- Execute culling (frustum, occlusion)
- Dispatch draw calls
- Present frame to display

**GPU-Driven Features:**
- Indirect drawing (commands generated on GPU)
- Compute shader culling
- Instancing for efficient batching

### `resize.rs` - Window Resize Handling
**Purpose:** Handles window resize events and viewport updates.

**Responsibilities:**
- Detect window size changes
- Recreate swapchain if needed
- Update projection matrices
- Adjust render targets
- Update HDR formats if applicable

**Key Functions:**
- `on_resize(new_size)` - Called when window size changes
- Recalculate aspect ratio
- Update camera projection

### `server.rs` - Server Functionality
**Purpose:** Manages integrated server for multiplayer hosting.

**Responsibilities:**
- Listen for incoming connections
- Manage connected clients
- Send world updates to clients
- Receive player inputs from clients
- Validate gameplay events
- Synchronize world state

**Server Loop:**
```
Accept connections → Update world → Send chunks to clients 
→ Receive inputs → Validate → Apply changes → Repeat
```

### `texture_cache.rs` - Texture Caching
**Purpose:** Manages texture loading, caching, and GPU binding.

**Key Types:**
- `TextureCache` - LRU cache of loaded textures
- `TextureAtlas` - Combined texture atlas

**Responsibilities:**
- Load texture files from disk
- Cache textures to avoid reloading
- Manage GPU texture memory
- Update texture bindings
- Handle texture atlas updates

**Performance Features:**
- LRU (Least Recently Used) eviction
- Async loading to avoid frame stutters
- Streaming texture loading
- Mipmap generation

## Data Flow

### Typical Frame:
```
Input Events
    ↓
Input Processing (input.rs)
    ↓
Game State Update (update.rs)
    ↓
Render Dispatch (render.rs)
    ↓
GPU Culling & Drawing
    ↓
Present to Screen
    ↓
Next Frame
```

### Initialization Flow:
```
Main Entry (main.rs)
    ↓
run_game() (mod.rs)
    ↓
Init Window (init.rs)
    ↓
Init GPU (init.rs)
    ↓
Create Game State (game.rs, state.rs)
    ↓
Load Chunks (world)
    ↓
Build Meshes (render)
    ↓
Main Loop
```

## Key Patterns

### Error Handling
- Uses Rust's `Result<T, E>` for fallible operations
- Panics are used only for truly unrecoverable errors
- Error messages include context for debugging

### Resource Management
- Uses RAII pattern for GPU resources
- Automatic cleanup when objects go out of scope
- No manual memory management needed

### Async Operations
- Chunk generation happens on background threads
- Mesh building is async with frame budgeting
- Rendering blocks until GPU completes previous frame

## Integration with Other Modules

```
app/ ←→ render/      (Dispatch rendering)
app/ ←→ world/       (Load chunks, generate terrain)
app/ ←→ player/      (Update camera, movement)
app/ ←→ ui/          (Render UI elements)
app/ ←→ multiplayer/ (Send/receive network data)
app/ ←→ core/        (Block, chunk data structures)
```

## Optimization Notes

1. **Batch Rendering** - Multiple chunks per draw call using indirect drawing
2. **Frustum Culling** - Skip chunks outside view frustum
3. **LOD System** - Load distant chunks at lower detail
4. **Frame Budgeting** - Limit mesh generation to prevent frame spikes
5. **Texture Streaming** - Load textures on-demand with caching

## Extension Points

To add new features to the app module:

1. **New Input Commands** - Add to `input.rs` input handler
2. **New Game State** - Extend `state.rs` AppState enum
3. **New Update Logic** - Add to `update.rs` frame updates
4. **New Render Pass** - Add to `render.rs` render dispatch
5. **New UI Elements** - Add to UI system in `ui/` module

## Performance Characteristics

- **Frame Time Budget** - 16ms for 60 FPS, 33ms for 30 FPS
- **GPU Memory** - ~2GB typical (chunks + textures + render targets)
- **CPU Usage** - Scales with chunk generation distance
- **Network Bandwidth** - Depends on player movement and world changes

---

**Key Takeaway:** The `app/` module acts as the conductor, orchestrating all other systems to create a cohesive, high-performance game experience. It manages the critical path from input → update → render, ensuring frame timing and responsiveness.

