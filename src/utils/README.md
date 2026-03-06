# src/utils/ - Utilities & Helpers Module

## Overview

The `utils/` module contains utility functions and helper systems used throughout the project, including configuration management, GPU diagnostics, and server properties.

## Module Structure

```
utils/
├── mod.rs                      ← Module declaration and public API
├── settings.rs                 ← Settings management system
├── gpu_memory_info.rs          ← GPU memory tracking and diagnostics
├── server_properties.rs        ← Server configuration
├── settings/                   ← Settings configuration files
│   └── (default settings)
└── settings_deafult.yaml       ← Default settings template
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides utilities API.

**Key Types:**
- `Settings` - Configuration system
- `GPUMemoryInfo` - GPU memory tracking
- `ServerProperties` - Server configuration

**Key Functions:**
- `load_settings() → Settings` - Load configuration
- `save_settings(settings) → Result` - Save configuration
- `get_gpu_memory() → GPUMemoryInfo` - Query GPU memory

### `settings.rs` - Settings Management
**Purpose:** Load, validate, and manage game configuration.

**Configuration Categories:**

#### **Graphics Settings**
```yaml
graphics:
  render_distance: 10
  shadow_distance: 300
  shadow_quality: high
  bloom_enabled: true
  vsync_enabled: true
  max_fps: 144
  texture_filtering: linear
  anisotropic_filtering: 16
```

#### **Gameplay Settings**
```yaml
gameplay:
  difficulty: normal
  player_speed: 4.5
  sprint_multiplier: 3.33
  jump_height: 1.0
fov: 90
  mouse_sensitivity: 1.0
  invert_mouse_y: false
```

#### **Audio Settings**
```yaml
audio:
  master_volume: 1.0
  music_volume: 0.7
  effect_volume: 0.8
  ambient_volume: 0.6
  enable_audio: true
```

#### **Networking Settings**
```yaml
networking:
  server_port: 25565
  max_players: 64
  compression_enabled: true
  tick_rate: 60
  chunk_sync_frequency: 10
```

**Key Types:**
```rust
pub struct Settings {
    pub graphics: GraphicsSettings,
    pub gameplay: GameplaySettings,
    pub audio: AudioSettings,
    pub networking: NetworkingSettings,
    pub file_path: PathBuf,
}

pub struct GraphicsSettings {
    pub render_distance: i32        // 1-32 chunks
    pub shadow_distance: f32        // meters
    pub shadow_quality: ShadowQuality,
    pub bloom_enabled: bool
    pub vsync_enabled: bool
    pub max_fps: u32
    // ... more options
}

pub enum ShadowQuality {
    Off,
    Low,       // 512x512 per cascade
    Medium,    // 1024x1024 per cascade
    High,      // 2048x2048 per cascade
    Ultra,     // 4096x4096 per cascade
}
```

**Settings Validation:**
```rust
pub fn validate(&self) -> Result<(), ValidationError> {
    // Check ranges
    if self.graphics.render_distance < 1 || self.graphics.render_distance > 32 {
        return Err(ValidationError::RenderDistanceOutOfRange);
    }
    
    if self.gameplay.player_speed < 0.1 || self.gameplay.player_speed > 20.0 {
        return Err(ValidationError::PlayerSpeedOutOfRange);
    }
    
    Ok(())
}
```

**Default Values:**
```yaml
# settings_deafult.yaml (typo is in original, could be renamed to settings_default.yaml)
graphics:
  render_distance: 10
  shadow_quality: high
  
gameplay:
  difficulty: normal
  
audio:
  master_volume: 1.0
```

**Settings File Locations:**
```
Windows: %APPDATA%/Render3D/settings.yaml
Linux:   ~/.config/render3d/settings.yaml
macOS:   ~/Library/Application Support/Render3D/settings.yaml
```

**Key Functions:**
- `load() → Result<Settings>` - Load from file or use defaults
- `save(&self) → Result<()>` - Persist to file
- `validate(&self) → Result<()>` - Validate all settings
- `reset_to_defaults() → Settings` - Get default settings
- `merge(defaults, user_config) → Settings` - Merge configs
- `get_graphics() → &GraphicsSettings` - Access category

### `gpu_memory_info.rs` - GPU Memory Diagnostics
**Purpose:** Track and report GPU memory usage for optimization.

**Memory Categories:**

```rust
pub struct GPUMemoryInfo {
    pub total_memory: u64           // Total VRAM available
    pub used_memory: u64            // Currently used
    pub available_memory: u64       // Free VRAM
    pub breakdown: MemoryBreakdown,
}

pub struct MemoryBreakdown {
    pub vertex_buffers: u64         // Mesh vertex data
    pub index_buffers: u64          // Mesh index data
    pub textures: u64               // Texture atlases
    pub shadow_maps: u64            // Shadow map textures
    pub render_targets: u64         // Intermediate buffers
    pub uniform_buffers: u64        // Constant buffers
    pub other: u64                  // Misc allocations
}
```

**Memory Usage Example (8GB GPU):**
```
Total Available: 8192 MB

Usage Breakdown:
├── Vertex Buffers:     560 MB (6.8%)  ← Main geometry
├── Index Buffers:      256 MB (3.1%)  ← Indices
├── Textures:         2048 MB (25.0%)  ← Atlas & block textures
├── Shadow Maps:        128 MB (1.6%)  ← CSM cascades
├── Render Targets:     256 MB (3.1%)  ← HDR, GBuffer, etc.
├── Uniform Buffers:     32 MB (0.4%)  ← Camera, lighting
└── Driver/Other:       512 MB (6.3%)

Used: 3792 MB (46.3%)
Free: 4400 MB (53.7%)
```

**Performance Analysis:**

```rust
pub fn analyze_pressure(&self) -> MemoryPressure {
    let usage_ratio = self.used_memory as f32 / self.total_memory as f32;
    
    match usage_ratio {
        0.0..0.50 => MemoryPressure::Low,      // Plenty of room
        0.50..0.75 => MemoryPressure::Medium,  // Comfortable
        0.75..0.90 => MemoryPressure::High,    // Getting tight
        0.90..1.00 => MemoryPressure::Critical, // Out of VRAM soon!
        _ => MemoryPressure::OutOfMemory,
    }
}
```

**Diagnostic Output:**
```
GPU Memory Report:
═══════════════════════════════════════════════════════════
Device: NVIDIA GeForce RTX 3080
Total Memory: 10.00 GB (10240 MB)
Used Memory:  4.60 GB (4710 MB) - 46.0%
Free Memory:  5.40 GB (5530 MB) - 54.0%

Memory Pressure: MEDIUM (OK)

Breakdown:
  Vertex Buffers:  560.0 MB (54.2% of used)
  Index Buffers:   256.0 MB (24.8% of used)
  Textures:      2000.0 MB (19.4% of used)
  Shadow Maps:     128.0 MB (1.2% of used)
  Other:           166.0 MB (1.6% of used)

Recommendations:
  ✓ Plenty of VRAM available
  ✓ Can increase texture quality
  ✓ Consider increasing shadow resolution
═══════════════════════════════════════════════════════════
```

**Memory Optimization Suggestions:**

```rust
pub fn get_suggestions(&self) -> Vec<String> {
    let mut suggestions = Vec::new();
    
    if self.texture_usage_percent() > 80.0 {
        suggestions.push("Consider reducing render distance or texture quality".to_string());
    }
    
    if self.vertex_buffer_percent() > 90.0 {
        suggestions.push("Too many loaded chunks - reduce render distance".to_string());
    }
    
    if self.available_memory < 500_000_000 {  // Less than 500MB
        suggestions.push("GPU memory critically low - reducing draw distance".to_string());
    }
    
    suggestions
}
```

**Key Functions:**
- `query() → GPUMemoryInfo` - Get current memory status
- `analyze_pressure() → MemoryPressure` - Assess memory situation
- `print_report() → String` - Generate diagnostic text
- `get_suggestions() → Vec<String>` - Optimization tips

### `server_properties.rs` - Server Configuration
**Purpose:** Manage server-specific settings and properties.

**Server Configuration:**

```rust
pub struct ServerProperties {
    pub server_name: String            // Display name
    pub max_players: usize              // Max concurrent players
    pub port: u16                       // Listen port
    pub difficulty: Difficulty          // Game difficulty
    pub pvp_enabled: bool               // Player vs player allowed?
    pub gamemode: Gamemode              // Survival, creative, etc.
    pub world_seed: u64                 // World generation seed
    pub view_distance: i32              // Render distance
    pub simulation_distance: i32        // Simulation distance
    pub whitelist_enabled: bool         // Use whitelist?
    pub whitelist: Vec<String>          // Allowed players
    pub ops: Vec<String>                // Operator players
    pub motd: String                    // Message of the day
}

pub enum Difficulty {
    Peaceful,   // No mobs, no damage
    Easy,       // Reduced damage
    Normal,     // Standard damage
    Hard,       // Increased damage
}

pub enum Gamemode {
    Survival,   // Mine, craft, fight
    Creative,   // Fly, place blocks freely
    Adventure,  // Restricted building
    Spectator,  // Observe only
}
```

**Configuration File (server.properties):**
```properties
server.name=My Awesome Server
max-players=64
server-port=25565
difficulty=normal
pvp-enabled=true
gamemode=survival
world-seed=12345
view-distance=10
simulation-distance=5
whitelist-enabled=false
motd=Welcome to Render3D!
```

**Whitelist System:**
```yaml
whitelist:
  - player1
  - player2
  - player3

# Only these players can join
# Others get: "You are not whitelisted"
```

**Operator System:**
```yaml
ops:
  - admin
  - moderator

# Ops can execute commands:
/stop          - Stop server
/save-all      - Save world
/ban player    - Ban a player
/unban player  - Unban a player
/op player     - Make player op
/deop player   - Remove op status
```

**Key Functions:**
- `load() → Result<ServerProperties>` - Load server config
- `save(&self) → Result<()>` - Persist changes
- `is_whitelisted(username) → bool` - Check whitelist
- `is_op(username) → bool` - Check if operator
- `add_whitelist(player) → ()` - Add to whitelist
- `remove_whitelist(player) → ()` - Remove from whitelist

## Integration with Other Modules

```
utils/ ←→ app/       (Load settings on startup)
utils/ ←→ render/    (Apply graphics settings)
utils/ ←→ player/    (Apply player settings)
utils/ ←→ multiplayer/(Load server properties)
```

## Configuration Priority

When multiple configs exist, priority is:

```
1. Command-line arguments (highest priority)
   cargo run -- --render-distance 20

2. User settings file
   ~/.config/render3d/settings.yaml

3. Game defaults
   settings_deafult.yaml (lowest priority)
```

## Settings Editor (Future)

In-game settings menu:
```
┌─────────────────────────────────────┐
│           SETTINGS                  │
├─────────────────────────────────────┤
│ Graphics:                            │
│   Render Distance: [12     ] ◄─────┐ │
│   Shadow Quality: [High   ▼]        │ │
│   V-Sync: [✓]                       │ │
│                                     │ │
│ Gameplay:                            │ │
│   Difficulty: [Normal ▼]             │ │
│   FOV: [70    ]                      │ │
│                                     │ │
│ Audio:                               │ │
│   Master: [████████░░]              │ │
│                                     │ │
│ [ Apply ] [ Defaults ] [ Cancel ]   │ │
└─────────────────────────────────────┘

Real-time preview as settings change
```

---

**Key Takeaway:** The `utils/` module provides essential infrastructure for configuration management, GPU memory diagnostics, and server administration, enabling robust customization and optimization of the Render3D engine.

