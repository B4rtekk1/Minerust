# src/ui/ - User Interface Module

## Overview

The `ui/` module handles all user interface elements including menus, HUD (heads-up display), text rendering, and interactive UI components. It uses `glyphon` for efficient text rendering and `wgpu` for 2D drawing.

## Module Structure

```
ui/
├── mod.rs              ← Module declaration and public API
├── ui.rs               ← Core UI rendering system
└── menu.rs             ← Menu systems (main, pause, settings)
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides UI system API.

**Key Types:**
- `UIManager` - Main UI system
- `UIState` - Current UI state

**Key Functions:**
- `new() → UIManager` - Initialize UI system
- `render(renderer) → ()` - Render all UI elements
- `handle_input(event) → ()` - Process UI input

### `ui.rs` - UI Rendering System
**Purpose:** Core functionality for rendering 2D UI elements (text, buttons, panels).

**Key Features:**

#### **Text Rendering**
Uses `glyphon` library for efficient glyph rendering:

```rust
pub struct TextRenderer {
    pub font: Font                 // Loaded font
    pub font_size: u32             // Size in pixels
    pub color: [f32; 4]            // RGBA color
}
```

**Text Rendering Pipeline:**
```
Text String
    ↓
glyphon layout
    ├─ Break into lines
    ├─ Measure dimensions
    └─ Position glyphs
    ↓
GPU glyph atlas
    └─ Render glyphs
    ↓
Screen Output
```

**HUD Elements:**
```
┌────────────────────────────────────┐
│ Minerust v0.1.0        FPS: 120    │  Top-left/right
├────────────────────────────────────┤
│                                    │
│  O───────┤                         │  Crosshair (center)
│                                    │
└────────────────────────────────────┤
│ Health: ████████░░                 │  Bottom-left
│ Hunger:  ████████░░                │
└────────────────────────────────────┘
```

#### **UI Layout System**
```rust
pub struct UIRect {
    pub x: f32                       // Position X (0.0-1.0 or pixels)
    pub y: f32                       // Position Y
    pub width: f32                   // Width
    pub height: f32                  // Height
    pub anchor: Anchor               // Top-left, center, etc.
}

pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}
```

**Layout Example:**
```
Anchor: TopRight         Anchor: Center
┌──────┐                 ┌──────┐
│      │                 │      │
│      ├─ UI Rect        │  UI  │ ← Centered
│      │                 │      │
│ UI ──┘                 └──────┘
└──────┘
```

#### **Button System**
```rust
pub struct Button {
    pub rect: UIRect
    pub label: String
    pub color: [f32; 4]
    pub hover_color: [f32; 4]
    pub is_hovered: bool
    pub callback: Option<Box<dyn Fn()>>
}

impl Button {
    pub fn contains(&self, mouse_pos: [f32; 2]) -> bool {
        // Check if mouse over button
    }
    
    pub fn on_click(&self) {
        // Execute callback
    }
}
```

#### **Panel System**
```rust
pub struct Panel {
    pub rect: UIRect
    pub title: String
    pub background_color: [f32; 4]
    pub border_color: [f32; 4]
    pub children: Vec<UIElement>  // Buttons, text, etc.
}
```

**Key Functions:**
- `render_text(text, pos, size, color) → ()` - Render text string
- `render_button(button) → ()` - Render button with hover state
- `render_panel(panel) → ()` - Render panel and children
- `render_crosshair(center) → ()` - Draw aiming crosshair
- `render_hud() → ()` - Render all HUD elements
- `update(mouse_pos, delta_time) → ()` - Update UI state

### `menu.rs` - Menu Systems
**Purpose:** Main menu, pause menu, settings menu, and other screen menus.

**Menu States:**

#### **Main Menu**
```
╔══════════════════════════════════╗
║        MINERUST                  ║
║      Voxel Engine               ║
╠══════════════════════════════════╣
║                                  ║
║  [ New Game  ]                   ║
║  [ Continue  ]                   ║
║  [ Settings  ]                   ║
║  [ Exit      ]                   ║
║                                  ║
╚══════════════════════════════════╝
```

#### **Pause Menu**
```
╔══════════════════════════════════╗
║         PAUSED                   ║
╠══════════════════════════════════╣
║                                  ║
║  [ Resume Game ]                 ║
║  [ Settings    ]                 ║
║  [ Exit to Menu]                 ║
║                                  ║
╚══════════════════════════════════╝
```

#### **Settings Menu**
```
╔══════════════════════════════════╗
║        SETTINGS                  ║
╠══════════════════════════════════╣
║                                  ║
║ Graphics:                         ║
║   Render Distance: [10    ]       ║
║   Draw Distance:   [300   ]       ║
║                                  ║
║ Audio:                            ║
║   Master Volume:   [████████░]    ║
║   Music:           [███████░░]    ║
║   Effects:         [████████░]    ║
║                                  ║
║  [ Apply  ] [ Cancel ]            ║
║                                  ║
╚══════════════════════════════════╝
```

**Menu Types:**
```rust
pub enum MenuType {
    MainMenu,
    PauseMenu,
    SettingsMenu,
    WorldMenu,
    MultiplayerMenu,
    ChatMenu,
}
```

**Menu Transitions:**
```
MainMenu
  ├─ New Game → Playing
  ├─ Settings → SettingsMenu
  │              └─ Back → MainMenu
  ├─ Continue → Playing
  └─ Exit → Quit

PauseMenu (from Playing)
  ├─ Resume → Playing
  ├─ Settings → SettingsMenu
  └─ Exit to Menu → MainMenu
```

**Key Types:**
```rust
pub struct MenuManager {
    pub current_menu: Option<MenuType>
    pub menu_stack: Vec<MenuType>  // For back button
    pub buttons: Vec<Button>
    pub panels: Vec<Panel>
    pub input_fields: Vec<InputField>
}

pub struct InputField {
    pub text: String
    pub cursor_pos: usize
    pub is_focused: bool
    pub placeholder: String
}
```

**Key Functions:**
- `open_menu(menu_type) → ()` - Open menu
- `close_menu() → ()` - Close current menu
- `go_back() → ()` - Back to previous menu
- `handle_button_click(button_id) → ()` - Process button click
- `render_current_menu() → ()` - Draw active menu
- `is_menu_open() → bool` - Check if menu visible

## HUD Elements

### **Crosshair**
```
  │
  ├──
  │
```
Center screen aiming reticle.

### **Health Bar**
```
Health: ████████░░  8/10
```
Player health status.

### **Hunger Bar** (if applicable)
```
Food: ███████░░░  7/10
```
Player hunger/food level.

### **Hotbar** (Inventory Quick Access)
```
[1] [2] [3] [4] [5] [6] [7] [8] [9]
 █   □   □   □   □   □   □   □   □
Selected: Stone Block (×64)
```
Quick item selection.

### **Debug Info** (F3)
```
Position: 123.45 64.00 -456.78
Rotation: Yaw 45.2° Pitch -30.1°
Chunk: (7, -29)
FPS: 120 | Frame: 8.3ms
Memory: 2048 MB / 8192 MB
Loaded Chunks: 441
Triangles: 12,345,678
```
Developer information display.

### **Chat/Messages**
```
┌──────────────────────────┐
│ Player1: Hello world!    │
│ Player2: Hi there!       │
│ [Type message...]        │
└──────────────────────────┘
```
Player communication.

## UI Input Handling

### **Mouse Input**
```
Move mouse: Update UI hover states
Click: Trigger button callbacks
```

### **Keyboard Input**
```
Tab: Cycle through buttons
Enter: Activate focused button
Escape: Close menu / Resume game
Type: Input text in text fields
```

### **Gamepad Input** (Future)
```
D-pad: Navigate menu
A Button: Select
B Button: Cancel/Back
Triggers: Scroll lists
```

## Rendering Pipeline

### **2D Rendering Order**
```
1. Background panels (draw first, behind everything)
2. Buttons (interactive elements)
3. Text (labels, values)
4. Crosshair (on top, always visible)
5. Tooltips (floating help text)
```

**Depth Ordering:**
```
Z = 1.0  ┌────────────────────┐
         │   Crosshair (top)  │
Z = 0.5  ├────────────────────┤
         │   Text / Buttons   │
Z = 0.1  ├────────────────────┤
         │  Panels (back)     │
Z = 0.0  └────────────────────┘
```

## Integration with Other Modules

```
ui/ ←→ app/       (Input handling, render dispatch)
ui/ ←→ player/    (Display player stats)
ui/ ←→ world/     (World selection menu)
ui/ ←→ utils/     (Load/save settings)
ui/ ←→ assets/    (Load fonts)
```

## Localization (Future)

Support for multiple languages:
```
EN: "New Game"
PL: "Nowa Gra"
DE: "Neues Spiel"
FR: "Nouveau Jeu"
```

## Accessibility Features

### **High Contrast Mode**
```
Text Color: White on black
Buttons: Large, clear outlines
Font Size: Adjustable (small/normal/large)
```

### **Text-to-Speech** (Future)
```
Menu items read aloud
Chat messages spoken
Important notifications voiced
```

### **Colorblind Modes** (Future)
```
Deuteranopia (red-green blind)
Protanopia (red blind)
Tritanopia (blue-yellow blind)
Monochromacy (complete color blind)
```

## Performance Characteristics

### **Memory Usage**
- UI state: ~10 KB
- Font cache: ~5 MB (loaded glyphs)
- Text buffers: ~100 KB

### **GPU Rendering**
- UI pass: ~1-2 ms per frame
- Text rendering: ~0.5-1 ms per frame
- **Total UI overhead**: <3% of frame budget

### **Draw Calls**
- Background panels: 1 draw call
- Buttons/UI: 1 draw call
- Text: 1 draw call per font size

## UI Theme System (Future)

Customizable visual themes:
```
Dark Theme:
  Background: #1a1a1a
  Text: #ffffff
  Accent: #00ff00

Light Theme:
  Background: #ffffff
  Text: #000000
  Accent: #0066cc

Custom Theme:
  Load from config file
```

---

**Key Takeaway:** The `ui/` module provides a complete UI system for menus and HUD elements, with efficient text rendering and flexible layout system for player interaction and information display.

