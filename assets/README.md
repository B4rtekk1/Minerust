# assets/ - Game Assets Directory

## Overview

The `assets/` directory contains all non-code resources for the Minerust engine: textures, fonts, and configuration files. These resources are loaded at runtime to customize the visual appearance of the game.

## Directory Structure

```
assets/
├── textures.png                    # Texture atlas (combined textures)
├── atlas_map_structure.json        # Atlas UV mapping metadata
├── fonts/                          # Typography resources
│   ├── GoogleSans_17pt-Regular.ttf
│   ├── OFL.txt                     # Font license (Open Font License)
│   └── README.txt
└── textures/                       # Minecraft-compatible texture pack
    ├── pack.mcmeta                 # Pack metadata (Minecraft format)
    ├── pack.png                    # Pack thumbnail
    ├── LICENSE.txt                 # Texture license
    └── block/                      # 100+ block textures (16×16 PNG)
        ├── acacia_*.png
        ├── amethyst_*.png
        ├── ancient_debris_*.png
        ├── andesite.png
        ├── anvil_*.png
        ├── azure_bluet.png
        ├── ... (many more)
        └── structure_void.png
```

## Key Files

### `textures.png` - Texture Atlas
**Purpose:** Combined image containing all block textures arranged in a grid.

> **Current layout:** 8 columns × 6 rows of 256×256 tiles (16×16 pixel art
> scaled up), 48 texture-array layers. Tile `i` is layer `i` and matches the
> `TEX_*` constants in `src/constants.rs`. Layers 0–15 are the original
> hand-made tiles; layers 16–41 (ores, deepslate, badlands terracotta,
> podzol, moss, mycelium, mushroom blocks, sandstone, birch/cherry wood,
> cherry leaves, magma, dripstone, amethyst, calcite) are generated from
> them by `python3 tools/generate_block_textures.py`, which can be re-run
> safely. Layers 42–47 are spare.

**Format & Size:**
- Format: PNG (RGBA 8-bit)
- Size: Typically 2048×2048 or 4096×4096 pixels
- Contains: ~60 block texture types + variants

**Texture Arrangement:**
```
Atlas Layout (schematic):
┌────────┬────────┬────────┐
│ Grass  │ Dirt   │ Stone  │  Row 0
├────────┼────────┼────────┤
│ Sand   │ Water  │ Wood   │  Row 1
├────────┼────────┼────────┤
│ Leaves │ Snow   │ Gravel │  Row 2
└────────┴────────┴────────┘

Each 16×16 cell is one block texture
```

**Advantages:**
- Single texture bind per frame (efficiency)
- Easier to manage than 60+ separate files
- Supports texture atlasing with padding
- Better GPU cache performance

**Creation Process:**
1. Individual textures created (textures/block/*.png)
2. Combined using texture packer (TexturePacker, AssetForge, etc.)
3. Saved as `textures.png`
4. Mapping exported to `atlas_map_structure.json`

### `atlas_map_structure.json` - Texture Mapping
**Purpose:** Maps block names to UV coordinates in the texture atlas.

**File Format Example:**
```json
{
  "version": 1,
  "texture_size": 16,
  "atlas_size": 128,
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
    },
    "dirt": {
      "x": 32,
      "y": 0,
      "width": 16,
      "height": 16
    },
    "stone": {
      "x": 48,
      "y": 0,
      "width": 16,
      "height": 16
    },
    // ... more textures
  }
}
```

**UV Coordinate Calculation:**
```
Block name: "grass_top"
File entry: x=0, y=0, width=16, height=16

Atlas dimensions: 2048×2048
Texture dimensions: 16×16 (in atlas)

UV coordinates (0.0-1.0 range):
  min_u = 0 / 2048 = 0.0
  min_v = 0 / 2048 = 0.0
  max_u = 16 / 2048 = 0.0078125
  max_v = 16 / 2048 = 0.0078125
```

**Loading in Code:**
```rust
let atlas_data = serde_json::from_str(json_str)?;
let grass_top = &atlas_data.textures["grass_top"];
let uv = (grass_top.x as f32 / 2048.0, grass_top.y as f32 / 2048.0);
```

### `fonts/` - Typography Resources

#### **GoogleSans_17pt-Regular.ttf**
**Purpose:** TrueType font for in-game text rendering.

**Specifications:**
- Font: Google Sans (modern, clean)
- Size: 17 points (pre-rendered for performance)
- Style: Regular (normal weight)
- Format: TTF (TrueType Font)

**Usage:**
```rust
let font = Font::from_file("assets/fonts/GoogleSans_17pt-Regular.ttf")?;
let text = "Health: 8/10";
text.render(font, position, size)?;
```

**Glyph Rendering:**
The `glyphon` library renders text:
1. Loads font metrics
2. Lays out text (handles line wrapping)
3. Rasterizes glyphs to texture
4. Renders glyphs with correct spacing

#### **OFL.txt**
**Purpose:** Open Font License text.

**License Summary:**
```
Open Font License (OFL) 1.1
Free to use, copy, modify, redistribute
Must include license with distribution
Can be bundled with software
```

**Notice:**
```
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Open Font License, Version 1.1.
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://scripts.siliconvalley.openstack.org/trac/openstack/raw-attachment/wiki/...
```

#### **README.txt**
**Purpose:** Font documentation and usage notes.

**Contents:**
```
Google Sans Font
================

Features:
- Clean, modern design
- Excellent readability
- Supports Latin character set
- Hinting for screen rendering

Usage in Minerust:
- Main UI text
- Chat messages
- Labels and buttons
- Debug information

Installation:
- Copy GoogleSans_17pt-Regular.ttf to assets/fonts/
- Engine loads automatically at startup
```

### `textures/` - Texture Pack Directory

**Purpose:** Organized texture files compatible with Minecraft format (for compatibility and portability).

**Structure:**
```
textures/
├── pack.mcmeta          ← Pack metadata (Minecraft format)
├── pack.png             ← Pack thumbnail image
├── LICENSE.txt          ← License for textures
└── block/               ← Block texture directory
    ├── acacia_door_bottom.png
    ├── acacia_door_top.png
    ├── acacia_leaves.png
    ├── acacia_leaves.png.mcmeta  ← Animation metadata
    ├── acacia_log_top.png
    ├── acacia_log.png
    ├── ... (100+ block textures)
    └── grass_block_side.png
```

#### **pack.mcmeta** - Pack Metadata
**Purpose:** Minecraft-compatible pack information.

**File Content:**
```json
{
  "pack": {
    "pack_format": 13,
    "description": "Minerust Default Textures",
    "pack_format_version": [
      13,
      13
    ]
  }
}
```

**Fields:**
- `pack_format`: Texture pack version (must match game version)
- `description`: Pack name and description
- `pack_format_version`: Compatible versions

#### **pack.png** - Pack Thumbnail
**Purpose:** Display image for texture pack selection.

**Specifications:**
- Format: PNG with transparency
- Size: 64×64 pixels (typical)
- Shows: Logo or representative texture
- Used in: Pack selection UI menu

#### **block/** - Individual Block Textures

**Block Texture Naming Convention:**
```
<block_name>[_<variant>][_<side>].png

Examples:
acacia_log.png             ← Main log texture
acacia_log_top.png         ← Top face of log
grass_block_side.png       ← Side of grass block
oak_leaves.png             ← Leaf blocks

Animation variants:
water_flowing_0.png        ← Frame 0
water_flowing_1.png        ← Frame 1
water_flowing_2.png        ← Frame 2
```

**Block Texture Specifications:**
- Format: PNG 24-bit RGB or 32-bit RGBA
- Size: 16×16 pixels (can be scaled up)
- Resolution: 4K resolution support available
- Quality: High quality photography or hand-drawn

**Common Block Types Included:**

| Category | Examples |
|----------|----------|
| Stone | Stone, Granite, Andesite, Diorite |
| Dirt | Dirt, Grass Block, Podzol, Mycelium |
| Wood | Oak, Spruce, Birch, Acacia, Dark Oak |
| Leaves | Various leaf textures |
| Ore | Coal, Iron, Gold, Diamond, Emerald |
| Minerals | Lapis, Redstone, Quartz |
| Sand | Sand, Red Sand, Gravel |
| Clay | Clay, Terracotta (colored variants) |
| Ice | Ice, Packed Ice, Blue Ice |
| Water | Water (flowing, still) |
| Lava | Lava (flowing, still) |
| Flora | Grass, Tall Grass, Flowers, Dead Bush |
| Crops | Wheat, Carrots, Potatoes |
| Structures | Bricks, Planks, Logs, Scaffolding |
| Utility | Furnace, Crafting Table, Anvil |
| Decorative | Stairs, Slabs, Doors, Trapdoors |

**Animated Textures (MCmeta):**
```
water_flowing_0.png
water_flowing.png.mcmeta

water_flowing.png.mcmeta content:
{
  "animation": {
    "frametime": 2,
    "frames": [
      0, 1, 2, 3, 4, 5,
      4, 3, 2, 1
    ]
  }
}
```

Cycles through frames for animated effect (water, lava).

## Asset Loading Pipeline

### **Startup Sequence:**
```
Engine Start
    ↓
Load atlas_map_structure.json
    ├─ Parse JSON
    ├─ Build UV lookup table
    └─ Validate all textures exist
    ↓
Load textures.png
    ├─ Read PNG file
    ├─ Create GPU texture
    ├─ Set up mipmap chain
    └─ Bind to rendering pipeline
    ↓
Load fonts/
    ├─ Load TrueType files
    ├─ Create glyph atlas
    └─ Ready for text rendering
    ↓
Asset Loading Complete ✓
    ↓
Game Ready
```

### **Runtime Texture Access:**
```
Rendering code needs grass block texture
    ↓
Look up "grass_top" in atlas_map_structure.json
    ├─ Get: x=0, y=0, width=16, height=16
    ├─ Calculate UV coordinates
    └─ Return texture index
    ↓
Shader applies texture to mesh
    ├─ Sample from textures.png at UV coords
    └─ Display on screen
```

## Memory Management

### **GPU Memory Usage:**
```
Texture Atlas (textures.png):
  2048×2048 RGBA: ~16 MB
  + Mipmaps: ~5 MB
  Total: ~21 MB per texture atlas

Font Cache:
  Glyph atlas: ~4-8 MB
  Per loaded font

Total Assets Memory: ~30-50 MB (modest)
```

### **Disk Space:**
```
Individual block textures: ~5-10 MB
Packed atlas: ~2-4 MB
Fonts: ~500 KB
Total on Disk: ~8-15 MB
```

## Customization

### **Replacing Textures:**
1. Create new textures (16×16 PNG files)
2. Place in `assets/textures/block/`
3. Update `atlas_map_structure.json` with new coordinates
4. Repack into `textures.png`
5. Restart game

### **Adding New Fonts:**
1. Add TTF file to `assets/fonts/`
2. Update code to load new font
3. Use in UI rendering

### **Creating Custom Texture Packs:**
1. Follow Minecraft texture pack format
2. Place in `assets/textures/`
3. Update `pack.mcmeta` metadata
4. Game automatically loads all textures

## Licensing

**Texture Pack License:**
Located in `assets/textures/LICENSE.txt`

Typical licensing:
- Free for use in this project
- May require attribution
- Check before redistributing
- Creative Commons (CC-BY, CC-BY-SA)

**Font License:**
Located in `assets/fonts/OFL.txt`

Google Sans:
- Open Font License (OFL) 1.1
- Free to use and modify
- Must include license copy

## Performance Tips

1. **Texture Atlasing** - Reduces state changes
2. **Mipmaps** - Reduces aliasing for distant blocks
3. **Compression** - Consider texture compression (BC7, ASTC)
4. **LOD System** - Use lower quality for distant chunks
5. **Streaming** - Load textures on-demand

## Asset Validation

At startup, engine checks:
```
✓ All referenced textures exist
✓ Texture dimensions valid (power of 2)
✓ Font files readable
✓ JSON files valid syntax
✓ Licenses present

If invalid:
✗ Refuse to start
✗ Report which asset failed
✗ Display helpful error message
```

---

**Key Takeaway:** The `assets/` directory provides all visual resources in an organized, efficient format. Textures are packed into a single atlas for performance, while maintaining Minecraft-compatible structure for portability and user customization.

