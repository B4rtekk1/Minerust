# assets/ - Game Assets Directory

## Overview

The `assets/` directory contains the runtime texture image, atlas metadata, menu background, and UI font. The current directory does not contain a separate Minecraft-style texture pack or per-block PNG directory.

## Directory Structure

```
assets/
├── textures.png                    # Runtime texture image (2048×1536 RGBA)
├── atlas_map_structure.json        # Texture mapping metadata
├── menu.png                        # Main-menu background (1918×1078 RGBA)
└── fonts/
    ├── GoogleSans_17pt-Regular.ttf
    ├── OFL.txt                     # Font license
    └── README.txt
```

The texture image and menu dimensions were checked from the files in the repository.
There is no `assets/textures/`, `pack.mcmeta`, `pack.png`, or `assets/shaders/` directory;
WGSL sources are in `src/shaders/`.

## Key Files

### `textures.png` - Texture Atlas
**Purpose:** Source image split into equal-sized layers for a GPU `Texture2DArray`.

> **Current layout:** 2048×1536 RGBA, arranged as 8 columns × 6 rows of 256×256 tiles.
> `src/render/texture.rs` extracts the 48 tiles in row-major order; `src/app/texture_cache.rs`
> uploads them as array layers with mipmaps. The defined block texture indices occupy
> layers 0–41; layers 42–47 are spare.

**Format & Size:**
- Format: PNG (RGBA 8-bit)
- Size: 2048×1536 pixels
- Tiles: 48 square RGBA8 layers, each 256×256 pixels

**Texture Arrangement:**
```
The image is read as 8 × 6 square tiles. Tile index `i` maps to column `i % 8`
and row `i / 8`; the tile index becomes the texture-array layer.
```

**Advantages:**
- Single texture bind per frame (efficiency)
- Easier to manage than 60+ separate files
- Supports texture atlasing with padding
- Better GPU cache performance

**Creation Process:**
The repository script `tools/generate_block_textures.py` generates the block tiles and
the packed image. The running renderer uses the numeric layer mapping in `src/constants.rs`
and block material code; `atlas_map_structure.json` is descriptive metadata, not a UV atlas
loaded by the shader.

### `atlas_map_structure.json` - Texture Mapping
**Purpose:** Human-readable block-to-texture-layer metadata, including per-face choices.

The runtime renderer's layer IDs are defined in `src/constants.rs`; this JSON is not
parsed as shader UV coordinates. Keep the two mappings consistent when adding textures.

**File Format Example:**
```json
{
  "blocks": {
    "Grass": { "top": 0, "side": 1, "bottom": 2 },
    "Dirt": { "all": 2 },
    "Stone": { "all": 3 }
  }
}
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

## Historical reference — texture-pack layout (not present or loaded)

The following legacy notes describe a proposed Minecraft-style texture pack. The current
repository has no such directory, metadata, or pack-selection runtime; these notes are
retained only as history and must not be used as asset instructions.

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

### Current startup texture path
```
Texture initialization
    ↓
Check `assets/texture_atlas.cache`
    ├─ Valid cache → use cached layers
    ├─ Invalid cache → procedural fallback
    └─ No cache
        ↓
        Decode `assets/textures.png` into 48 array layers
            ├─ Success → cache and use the layers
            └─ Missing/invalid image → procedural fallback
    ↓
Generate mip levels and upload as a D2 array texture
    ↓
Bind texture array to the renderer
```

Layer indices are packed with terrain quads and sampled by the WGSL shader. The JSON file
is not parsed by this runtime path. Font and menu resources are initialized separately by
the UI/client code.

## Memory Management

### **GPU Memory Usage:**
```
Texture array (48 × 256 × 256 × RGBA8):
  Base level: 12 MiB
  Full mip chain: about 16 MiB total, uncompressed

Font/UI allocations are managed separately and vary with the rendered glyph set.
```

### **Disk Space:**
The current runtime image and font files are stored directly under `assets/`; use the
filesystem for current file sizes rather than relying on estimates from the old texture-pack layout.

## Customization

### **Replacing Textures:**
1. Update the source tiles and regenerate `assets/textures.png` with
   `tools/generate_block_textures.py`.
2. Keep the 8×6, 256×256-per-tile layout expected by `src/render/texture.rs`.
3. Keep the layer IDs in `src/constants.rs`, block texture selection, and descriptive
   `atlas_map_structure.json` consistent.
4. Restart the client to reload the image and rebuild the texture cache.

### **Adding New Fonts:**
1. Add TTF file to `assets/fonts/`
2. Update code to load new font
3. Use in UI rendering

There is no texture-pack discovery or runtime pack-selection API at present.

## Licensing

**Texture image:** No separate texture license file is present in `assets/`; verify the
source and redistribution terms before reusing the image.

**Font License:**
Located in `assets/fonts/OFL.txt`

Google Sans:
- Open Font License (OFL) 1.1
- Free to use and modify
- Must include license copy

## Performance Tips

1. **Texture array** - Keeps block layers in one GPU resource.
2. **Mipmaps** - Generated per layer during texture upload.
3. **Far terrain** - Uses merged geometry at distance; it keeps the same texture resolution.
4. **Texture compression** - Not currently enabled by this asset-loading path.

## Asset Validation

At startup, texture loading:
```
✓ Accepts a PNG whose dimensions form the configured 8×6 grid of square tiles
✓ Builds a GPU texture array and mip levels
✓ Falls back to procedural layers if the image cannot be loaded
```

---

**Current state:** `assets/` contains the texture-array source image, descriptive layer
metadata, menu background and font resources. The renderer samples texture-array layers;
there is no Minecraft-compatible pack loader.

