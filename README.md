# 🎮 Render3D - High-Performance Voxel Rendering Engine

<div align="center">

![Render3D](https://img.shields.io/badge/Rust-CE422B?style=for-the-badge&logo=rust&logoColor=white)
![WGPU](https://img.shields.io/badge/WGPU-0.28-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A blazingly fast GPU-driven voxel rendering engine with procedurally generated worlds and multiplayer support**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-technical-architecture) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

**Render3D** is a cutting-edge voxel rendering engine written in Rust, designed for high-performance, scalable game development. It combines modern GPU rendering techniques with sophisticated procedural generation to deliver an infinite, dynamically generated world reminiscent of sandbox games like Minecraft.

Built on **wgpu** for cross-platform GPU support (Vulkan, DirectX 12, Metal), Render3D pushes the boundaries of what's possible with voxel rendering through:

- 🚀 **GPU-Driven Rendering Pipeline** - Indirect drawing with compute shader culling
- 🌍 **Infinite Procedural Worlds** - 11 biomes with caves, ore, and structures
- 🌐 **Multiplayer Architecture** - Authoritative server with client prediction
- ⚡ **Extreme Performance** - Optimized for high-end rendering loads
- 🎨 **Advanced Visuals** - CSM shadows, water simulation, atmospheric effects

---

## ✨ Features

### 🎨 Rendering System

| Feature | Details |
|---------|---------|
| **Graphics API** | [wgpu](https://wgpu.rs/) - Universal GPU abstraction layer |
| **Rendering Method** | GPU-driven with indirect dispatch and compute culling |
| **Vertex Buffers** | Unified buffers (560MB capacity) with sub-allocation |
| **Shadows** | 4-cascade CSM with PCF filtering (up to 2048×2048) |
| **Effects** | Water physics, bloom, god rays, atmospheric scattering |
| **Culling** | CPU AABB + GPU frustum culling for optimal performance |

**Technical Highlights:**
- **Indirect Drawing**: Single `draw_indirect` call for entire terrain
- **Greedy Meshing**: Automatic face merging reduces geometry by 75%+
- **Zero-Copy Uploads**: Lock-free async mesh generation to GPU
- **Compute Shaders**: Pre-draw visibility pass eliminates hidden geometry

### 🌎 World Generation

| Feature | Details |
|---------|---------|
| **Terrain** | FBM Perlin noise with dynamic height variation |
| **Biomes** | 11 unique biomes (Plains, Mountains, Deserts, Oceans, etc.) |
| **Caves** | "Cheese" and "Spaghetti" patterns for natural cave systems |
| **Ores & Blocks** | Deterministic procedural placement |
| **Structures** | Trees, vegetation, and procedural features |
| **Height** | 256 blocks (16 subchunks × 16 blocks) |

**Generation Pipeline:**
- Multi-threaded async generation (FastNoise-Lite)
- Deterministic seeding (same seed = same world)
- Mesh building decoupled from rendering loop

### 🌐 Multiplayer

| Feature | Details |
|---------|---------|
| **Architecture** | Authoritative server with client-side prediction |
| **Protocols** | QUIC (reliable) + UDP (real-time) hybrid stack |
| **State Sync** | Delta compression for chunks and entities |
| **Latency Handling** | Client movement prediction & reconciliation |

---

## 🚀 Quick Start

### Prerequisites

- **Rust 1.70+** - Install from [rustup.rs](https://rustup.rs/)
- **GPU Support** - Vulkan 1.2+, DirectX 12, or Metal
- **OS**: Windows, Linux, macOS

### Installation & Running

```bash
# Clone repository
git clone https://github.com/B4rtekk1/Render3D.git
cd render3d

# Build (release mode recommended for performance)
cargo build --release

# Run the engine
cargo run --release

# Run in debug mode (slower, useful for development)
cargo run
```

### First Launch

Upon startup, the engine will:
1. Initialize GPU device and surfaces
2. Generate initial world chunks
3. Load textures and shaders
4. Display main menu

Use the menu to create a new world or connect to a server.

---

## 🎮 Controls

```
WASD             → Move around
SPACE            → Jump
SHIFT            → Sprint (hold)
MOUSE            → Look around
LEFT CLICK       → Place/Destroy block
RIGHT CLICK      → Interact
ESC              → Pause menu
F1               → Toggle UI
F3               → Debug info
```

---

## ⚙️ Configuration

### Compile-Time Constants (`src/constants.rs`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `RENDER_DISTANCE` | 10 | Chunks to load around player |
| `WORLD_HEIGHT` | 256 | Maximum build height |
| `CHUNK_SIZE` | 16 | Horizontal chunk dimension |
| `SUBCHUNK_HEIGHT` | 16 | Vertical subchunk size |
| `CSM_CASCADE_COUNT` | 4 | Shadow cascades |
| `CSM_SHADOW_MAP_SIZE` | 2048 | Shadow texture resolution |
| `MAX_CHUNKS_PER_FRAME` | 4 | Mesh uploads/frame limit |

---

## 📐 Technical Architecture

### Rendering Pipeline

```
Input Processing
    ↓
Game State Update
    ↓
Chunk Generation/Loading
    ↓
Mesh Building (Worker Threads)
    ↓
GPU Buffer Uploads
    ↓
Frustum Culling (Compute Shader)
    ↓
[Shadow Pass] → [Terrain Pass] → [Water Pass] → [Composite] → [UI]
    ↓
Present to Screen
```

### Memory Layout

**Unified Vertex Buffer**
```
[Chunk 0] [Chunk 1] [Chunk 2] ... [Max 10M vertices]
```

**Unified Index Buffer**
```
[Chunk 0] [Chunk 1] [Chunk 2] ... [Max 256M indices]
```

**Indirect Buffer** (Draw Commands)
```
[DrawIndirectArgs × Num Active Chunks]
↓
[Compute Shader Culls Invisible Chunks]
↓
[GPU Executes Surviving Commands]
```

### World Structure

```
World (Infinite)
 └─ Chunk [16×16 horizontal]
     ├─ SubChunk 0-15 [16×16×16 voxels each]
     │   └─ Voxel (Block data)
     └─ Metadata (Mesh status, AABBs)
```

### Key Optimizations

| Technique | Benefit |
|-----------|---------|
| **Indirect Dispatch** | Single GPU call instead of thousands |
| **Greedy Meshing** | 75%+ reduction in triangle count |
| **Compute Culling** | Invisible geometry skipped entirely |
| **Async Generation** | No frame drops during world exploration |
| **Zero-Copy Uploads** | Direct memory mapping to GPU |

---

## 📚 Project Structure

For detailed folder documentation, see [FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md)

**Core Modules:**

- `src/app/` - Application loop, window, and input
- `src/core/` - Block/chunk/biome data structures
- `src/render/` - Rendering pipeline and GPU management
- `src/world/` - Procedural generation and chunk loading
- `src/multiplayer/` - Networking and server/client logic
- `src/player/` - Camera and character controller
- `src/ui/` - Menu and HUD rendering
- `assets/` - Textures, fonts, and configuration

---

## 🔧 Development

### Build Modes

```bash
# Debug build (fast compilation, slower runtime)
cargo build

# Release build (slow compilation, optimized runtime)
cargo build --release

# With verbose output
RUST_LOG=debug cargo run --release

# Run tests
cargo test --release
```

### Dependencies

Key libraries used:

- **wgpu** (28.0) - GPU rendering
- **winit** (0.29) - Window/input handling
- **tokio** (1.40) - Async runtime
- **quinn** (0.11) - QUIC networking
- **fastnoise-lite** (1.1) - Procedural generation
- **glyphon** (0.10) - Text rendering
- **cgmath** (0.18) - Linear algebra

See `Cargo.toml` for complete dependency list.

---

## 📖 Documentation

- **[FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md)** - Detailed project organization
- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Development guidelines
- **[DOCUMENTATION_MAP.md](./DOCUMENTATION_MAP.md)** - Doc index

**Key Topics:**

- Rendering system architecture
- Procedural generation algorithms
- Networking protocol specification
- Performance optimization techniques
- GPU memory management

---

## 🎯 Roadmap

### Current Version (0.1.0)

- ✅ GPU-driven rendering pipeline
- ✅ Procedural world generation
- ✅ Multiplayer networking framework
- ✅ Shadow mapping system

### Planned Features

- 🔄 Inventory and crafting systems
- 🔄 Advanced terrain features (rivers, biome blending)
- 🔄 Particle systems and effects
- 🔄 Performance profiling tools
- 🔄 Mod support framework

### Future Exploration

- 🎯 GPU path tracing for ray-traced lighting
- 🎯 Voxel cone tracing for global illumination
- 🎯 Streaming world format (larger than RAM)
- 🎯 Advanced network compression
- 🎯 Server administration tools

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes with clear messages
4. **Push** to your fork
5. **Submit** a Pull Request

### Development Tips

- Run `cargo fmt` before committing (code style)
- Use `cargo clippy --release` to check for issues
- Keep PRs focused on a single feature
- Update documentation when changing APIs
- Test on both debug and release builds

---

## 📊 Performance Metrics

**Target Specifications (Release Build)**

| Metric | Target |
|--------|--------|
| **FPS** | 100+ (RTX 3050 4GB @ 1080p) |
| **Memory** | < 1GB (with render distance 12) |
| **GPU Memory** | ~1.5GB |
| **Chunk Load Time** | < 16ms |
| **Frame Time** | 9-12ms |

*Performance varies by hardware and settings.*

## 💬 Support & Community

- **Issues**: Report bugs on [GitHub Issues](https://github.com/B4rtekk1/Render3D/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/B4rtekk1/Render3D/discussions)
- **Email**: Contact via project repository

---

## 🙏 Acknowledgments

- **wgpu team** - For excellent GPU abstraction
- **Minecraft** - For inspiring voxel-based design
- **Rust community** - For amazing ecosystem

---

<div align="center">

**Made with ❤️ in Rust**

[⬆ Back to Top](#-render3d---high-performance-voxel-rendering-engine)

</div>

## Roadmap

- [ ] **Dynamic Light sources**: Torch and lantern support with propagation.
- [ ] **Entity System**: Passive mobs and enemy AI.
- [ ] **Inventory UI**: Drag-and-drop item management.
- [ ] **Modding API**: Wasm-based plugin system.
- [ ] **Post-Processing**: TAA (Temporal Anti-Aliasing) and Motion Blur.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
