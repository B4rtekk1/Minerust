# src/player/ - Player Character Module

## Overview

The `player/` module manages the player character, including camera control, input handling, physics, and movement mechanics. It handles first-person camera mathematics, collision detection, and player state management.

## Module Structure

```
player/
├── mod.rs              ← Module declaration and public API
├── camera.rs           ← Camera system and projection matrices
└── input.rs            ← Player input and movement handling
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides player system API.

**Key Types:**
- `Player` - Main player state container
- `PlayerState` - Enumeration of player states

**Key Functions:**
- `new() → Player` - Create new player
- `update(delta_time) → ()` - Update player physics
- `get_view_matrix() → Matrix4` - Get camera view matrix

### `camera.rs` - Camera System
**Purpose:** Implements first-person camera with projection matrices and frustum calculations.

**Camera Mathematics:**

#### **View Matrix**
Transforms world coordinates to camera/eye coordinates:

```rust
pub fn view_matrix(
    position: Vec3,    // Eye position
    target: Vec3,      // Look-at target
    up: Vec3          // "Up" direction (usually Y-axis)
) -> Matrix4 {
    // Creates camera transformation matrix
}
```

**Visualization:**
```
World Space                Camera Space
    │ Y                        │ Z (backward)
    │                         │
    ├─→ X          →          ├─→ X (right)
    │                         │
   / Z                        │ Y (up)
  /

Camera positioned at (0, 2, 5) looking at (0, 2, 0)
```

#### **Projection Matrix**
Transforms camera space to normalized device coordinates (NDC):

```rust
pub fn perspective_projection(
fov: f32,              // Field of view (90°)
    aspect_ratio: f32,     // Width / Height
    near_plane: f32,       // 0.1 (closest visible)
    far_plane: f32         // 1000.0 (farthest visible)
) -> Matrix4 {
    // Creates perspective projection matrix
}
```

**Field of View:**
```
fov = 90° (default)         fov = 120° (wide angle)
    ╱  ╲                       ╱     ╲
   ╱    ╲  (narrow)          ╱       ╲ (wide)
  ╱      ╲                  ╱         ╲
 ╱        ╲                ╱           ╲
```

**Key Types:**
```rust
pub struct Camera {
    pub position: Vec3           // Eye position
    pub yaw: f32                 // Horizontal rotation (degrees)
    pub pitch: f32               // Vertical rotation (degrees)
pub fov: f32                 // Field of view (90° default)
    pub aspect_ratio: f32        // Window width/height
    pub near_plane: f32          // Closest visible (0.1)
    pub far_plane: f32           // Farthest visible (1000.0)
}

pub struct CameraUniforms {
    pub view_matrix: [[f32; 4]; 4]
    pub proj_matrix: [[f32; 4]; 4]
    pub view_pos: [f32; 3]       // For lighting calculations
}
```

**Yaw and Pitch:**
```
Yaw (horizontal):          Pitch (vertical):
    N (0°)                     Up (-90°)
    │                          │
W ─ ┼ ─ E                      ├─────
    │                          │ Forward (0°)
    S (180°)                   │ Down (90°)

Yaw = 45° (NE direction)   Pitch = 30° (looking up)
```

#### **View-Projection Matrix**
Combined matrix for efficient transformation:

```rust
pub fn view_projection_matrix(
    view: Matrix4,
    projection: Matrix4
) -> Matrix4 {
    projection * view
}

// Used in shaders:
// projected_pos = view_proj_matrix * world_pos
```

**Key Functions:**
- `new(position, fov) → Camera` - Create camera
- `view_matrix() → Matrix4` - Get view matrix
- `projection_matrix() → Matrix4` - Get projection matrix
- `view_projection_matrix() → Matrix4` - Combined matrix
- `rotate(yaw_delta, pitch_delta) → ()` - Rotate camera
- `get_forward() → Vec3` - Get facing direction
- `get_right() → Vec3` - Get right direction
- `update_aspect_ratio(width, height) → ()` - Handle window resize

**Clipping Planes:**
```
Camera Setup:
  near_plane = 0.1 (10cm)
  far_plane = 1000.0 (1km)

Object too close (< 0.1):   Not rendered (inside camera)
Object in range:             Rendered ✓
Object too far (> 1000):     Not rendered (beyond horizon)
```

**Frustum Culling:**
Camera calculates view frustum (pyramid-shaped visible region):
```
       Eye
        │╲
        │ ╲  Top
        │  ╲╱
        ├──────  Right
        │╲    ╱
        │ ╲  ╱
        │  ╱
       Far
       Plane

Chunks inside: Render
Chunks outside: Skip
```

### `input.rs` - Input Handling
**Purpose:** Processes player input and updates movement state.

**Input Categories:**

#### **Keyboard Input**
```rust
pub enum KeyboardInput {
    MoveForward,    // W key
    MoveBackward,   // S key
    MoveLeft,       // A key
    MoveRight,      // D key
    Jump,           // Space
    Sprint,         // Left Shift
    Crouch,         // Ctrl (for future implementation)
    Inventory,      // I key
    Pause,          // Esc
}
```

**Key Bindings (Configurable):**
| Action | Default | Alt |
|--------|---------|-----|
| Forward | W | Arrow Up |
| Backward | S | Arrow Down |
| Left | A | Arrow Left |
| Right | D | Arrow Right |
| Jump | Space | - |
| Sprint | LShift | - |
| Pause | Esc | - |
| Screenshot | F2 | - |
| Debug Info | F3 | - |

#### **Mouse Input**
```rust
pub struct MouseInput {
    pub delta_x: f32      // Horizontal movement
    pub delta_y: f32      // Vertical movement
    pub sensitivity: f32  // Multiplier (0.0-1.0)
}
```

**Mouse Look:**
```
Move mouse right: delta_x > 0
    └─ Increase yaw (look right)

Move mouse up: delta_y < 0 (inverted)
    └─ Decrease pitch (look up)
```

**Sensitivity Calculation:**
```
Camera rotation = mouse_delta × sensitivity × 0.1
Typical: sensitivity = 1.0 (medium)
         sensitivity = 0.5 (slow, precise)
         sensitivity = 2.0 (fast, arcade)
```

#### **Player State**
```rust
pub struct PlayerState {
    pub position: Vec3           // World position
    pub velocity: Vec3           // Current movement speed
    pub is_jumping: bool         // In air?
    pub is_sprinting: bool       // Running fast?
    pub is_grounded: bool        // On solid block?
}
```

**Physics Constants:**
```rust
PLAYER_WIDTH = 0.35              // Collision box width
PLAYER_HEIGHT = 1.8              // Standing height
PLAYER_CROUCH_HEIGHT = 1.5       // Crouching height
PLAYER_SPEED = 4.5 m/s           // Walking speed
PLAYER_SPRINT_SPEED = 15.0 m/s   // Running speed
PLAYER_JUMP_HEIGHT = 1.0 m       // Jump distance
GRAVITY = 9.81 m/s²             // Gravity acceleration
```

**Movement Physics:**

#### **Acceleration**
```
Input: W (forward)
    ↓
Current velocity: (0, 0, 0)
Target velocity: (4.5, 0, 0) m/s
    ↓
Accelerate smoothly:
velocity += (target - velocity) × acceleration_rate × delta_time
    ↓
Smooth movement (not instant)
```

#### **Jumping**
```
Grounded on block?
    ├─ Yes: Allow jump
    │   └─ velocity.y = jump_velocity
    │
    └─ No: Already jumping
        └─ Prevent double jump

Apply gravity:
velocity.y -= GRAVITY × delta_time
```

**Jump Physics:**
```
Velocity Y     Height
    │            ╱╲
    │           ╱  ╲
    ├─ jump_v  ╱    ╲
    │         ╱      ╲
    ├─ 0     ╱        ╲
    │       ╱          ╲
    └─ -v  ╱────────────╲
          t1             t2
    Time in air ≈ 0.4s
```

#### **Sprinting**
```
Input: Hold Shift
    ├─ Check stamina (optional)
    ├─ Increase movement speed: ×3.3
├─ FOV: Slightly increase (90° → 100°)
    └─ Animation: Faster arm swing

Release Shift:
    └─ Return to normal speed
```

#### **Collision Detection**
```
New position = current_position + velocity × delta_time
    ↓
Check for solid blocks at new position
    ├─ No collision: Accept new position
    ├─ Collision detected:
    │   ├─ Stop movement in that direction
    │   ├─ Slide along surface (no penetration)
    │   └─ Update is_grounded
    └─ Apply damage if falling too fast
```

**AABB Collision:**
```
Player:        Block:
  ╱╲           ███
 ╱  ╲          ███
      (0.35w)  ███
  1.8h

Check if player box overlaps with solid blocks
If yes: Collision → Push player out
```

**Key Functions:**
- `handle_input(events) → ()` - Process input events
- `update_movement(delta_time) → ()` - Apply physics
- `apply_gravity(delta_time) → ()` - Gravity calculation
- `check_collisions() → ()` - Collision detection
- `jump() → ()` - Initiate jump
- `sprint(is_active) → ()` - Toggle sprinting

## Player Update Flow

```
Frame Start
    ↓
Process Input (input.rs)
  ├─ Keyboard: W/A/S/D/Space
  ├─ Mouse: delta_x, delta_y
  └─ Update movement state
    ↓
Update Camera (camera.rs)
  ├─ Apply mouse look
  │   └─ Update yaw, pitch
  ├─ Calculate view matrix
  └─ Calculate frustum
    ↓
Update Player Physics (input.rs)
  ├─ Apply acceleration
  ├─ Apply gravity
  ├─ Check collisions
  └─ Update position
    ↓
Send Network Update (multiplayer)
  └─ Broadcast new position to others
    ↓
Render Frame (render)
  ├─ Use view matrix for rendering
  ├─ Use frustum for culling
  └─ Display HUD elements
    ↓
Next Frame
```

## Integration with Other Modules

```
player/ ←→ app/       (Input from main loop)
player/ ←→ render/    (View/projection matrices, frustum)
player/ ←→ world/     (Collision with blocks)
player/ ←→ multiplayer/(Send position updates)
```

## Camera Perspective Modes

The system is designed for **first-person** view:

```
First-Person (implemented):
  ├─ Camera at player's eye height (1.6m)
  ├─ Directly controls where player looks
  ├─ Immersive experience
  └─ Ideal for exploration

Third-Person (future):
  ├─ Camera behind player
  ├─ Better awareness of surroundings
  └─ Different physics calculations

Top-Down (future):
  ├─ Overhead view
  ├─ Strategy game feel
  └─ Used in some cutscenes
```

## Advanced Features (Optional)

### Head Bobbing
```
Walking animation (automatic):
  Y position += sin(time × frequency) × amplitude
  Creates natural head motion from footsteps
```

### FOV Zoom
```
Sprinting: FOV 90° → 100° (tunnel vision reduces)
Looking Down Sight: FOV 90° → 20° (zoom in)
Swimming: FOV 90° → 60° (underwater distortion)
```

### Camera Shake
```
Landing from fall: Jitter camera position
Block breaking: Vibration effect
Explosion nearby: Knockback effect
```

## Performance Characteristics

### CPU Overhead
- Input processing: <1ms per frame
- Physics calculations: ~1-2ms per frame
- Collision detection: ~1-3ms depending on nearby blocks

### Memory Usage
- Camera state: ~120 bytes
- Player state: ~200 bytes
- Input state: ~64 bytes
- **Total**: <1 KB

### Network Bandwidth (per player)
- Position updates: 20 bytes × 60 fps = 1.2 KB/s per player
- Sent to server every ~16ms
- Broadcast to other players

---

**Key Takeaway:** The `player/` module provides smooth, responsive first-person camera control and physics-based movement, essential for immersive voxel exploration and gameplay.

