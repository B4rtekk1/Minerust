use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use tracing::{error, warn};

/// Root settings structure containing all configurable game options.
///
/// Serialised to and deserialised from `settings.bin` via
/// [`save_settings`] and [`load_settings`]. Sub-sections with
/// `#[serde(default)]` are backwards-compatible: older save files that
/// predate a section will deserialise successfully with default values
/// for the missing fields.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameSettings {
    /// Player identity settings (nickname, ID).
    pub player: PlayerSettings,
    /// Rendering and display settings.
    pub graphics: GraphicsSettings,
    /// Volume and mute settings.
    pub audio: AudioSettings,
    /// Mouse, keyboard, and keybind settings.
    pub controls: ControlsSettings,
    /// Gameplay feel settings (view bobbing, camera smoothness).
    /// Defaults silently on load if absent from the save file.
    #[serde(default)]
    pub gameplay: GameplaySettings,
    /// Developer / diagnostic overlays.
    /// Defaults silently on load if absent from the save file.
    #[serde(default)]
    pub debug: DebugSettings,
}

impl Default for GameSettings {
    /// Returns a `GameSettings` composed entirely of sub-section defaults.
    fn default() -> Self {
        Self {
            player: PlayerSettings::default(),
            graphics: GraphicsSettings::default(),
            audio: AudioSettings::default(),
            controls: ControlsSettings::default(),
            gameplay: GameplaySettings::default(),
            debug: DebugSettings::default(),
        }
    }
}

/// Player identity settings persisted across sessions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlayerSettings {
    /// Display name shown to other players and in the UI. Defaults to `"Player"`.
    pub nickname: String,
    /// Locally generated player ID. `0` indicates an unassigned ID.
    pub id: u64,
}

impl Default for PlayerSettings {
    fn default() -> Self {
        Self {
            nickname: "Player".to_string(),
            id: 0,
        }
    }
}

/// All rendering and display-related settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphicsSettings {
    /// Number of chunks rendered around the player in each direction.
    pub render_distance: u32,
    /// Number of chunks kept ticked (entity/block updates) around the player.
    /// Typically smaller than `render_distance`.
    pub simulation_distance: u32,
    /// Whether vertical sync is enabled to cap FPS to the display refresh rate.
    pub vsync: bool,
    /// Whether the game runs in borderless or exclusive fullscreen.
    pub fullscreen: bool,
    /// Hard cap on rendered frames per second. `999` is effectively unlimited.
    pub max_fps: u32,
    /// Horizontal field of view in degrees. Defaults to `90.0`.
    #[serde(default)]
    pub fov: f32,
    /// Shadow map configuration. Defaults silently on load if absent.
    #[serde(default)]
    pub shadows: ShadowSettings,
    /// Tone-mapping and ambient occlusion settings. Defaults silently on load if absent.
    #[serde(default)]
    pub lighting: LightingSettings,
    /// Water rendering quality settings. Defaults silently on load if absent.
    #[serde(default)]
    pub water: WaterSettings,
}

impl Default for GraphicsSettings {
    /// Returns balanced defaults suitable for mid-range hardware:
    /// - 16-chunk render distance, 12-chunk simulation distance
    /// - VSync on, windowed, uncapped FPS (`999`), 90° FOV
    fn default() -> Self {
        Self {
            render_distance: 16,
            simulation_distance: 12,
            vsync: true,
            fullscreen: false,
            max_fps: 999,
            fov: 90.0,
            shadows: ShadowSettings::default(),
            lighting: LightingSettings::default(),
            water: WaterSettings::default(),
        }
    }
}

/// Shadow map configuration.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShadowSettings {
    /// Whether shadow casting is enabled at all.
    pub enabled: bool,
    /// Shadow map texture resolution in pixels (per cascade). Higher values
    /// give sharper shadows at the cost of VRAM and fill rate.
    pub resolution: u32,
    /// Number of cascaded shadow map splits. More cascades improve shadow
    /// quality at distance but increase draw calls.
    pub cascades: u32,
    /// Maximum distance in world units at which shadows are rendered.
    pub distance: f32,
    /// Filtering algorithm used for shadow edge softening.
    pub quality: ShadowQuality,
}

impl Default for ShadowSettings {
    /// Returns high-quality defaults: enabled, 2048 px resolution,
    /// 4 cascades, 150-unit distance, PCSS filtering.
    fn default() -> Self {
        Self {
            enabled: true,
            resolution: 2048,
            cascades: 4,
            distance: 150.0,
            quality: ShadowQuality::Pcss,
        }
    }
}

/// Shadow edge filtering quality.
///
/// Higher tiers are more expensive but produce softer, more realistic penumbra.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum ShadowQuality {
    /// No filtering — aliased hard shadow edges. Cheapest option.
    Hard,
    /// Percentage-closer filtering — slightly blurred edges at low cost.
    Pcf,
    /// Percentage-closer soft shadows — variable penumbra based on blocker
    /// distance. Most realistic but most expensive.
    Pcss,
}

/// Tone-mapping and ambient lighting settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LightingSettings {
    /// Gamma correction exponent applied during tone-mapping. `2.2` is the
    /// standard sRGB gamma.
    pub gamma: f32,
    /// Scene exposure multiplier before tone-mapping. `1.0` is neutral.
    pub exposure: f32,
    /// Ambient occlusion technique. Defaults to [`AoMode::Off`].
    pub ao_mode: AoMode,
    /// Bloom intensity. `0.0` disables bloom entirely.
    pub bloom_strength: f32,
}

impl Default for LightingSettings {
    /// Returns standard defaults: sRGB gamma (`2.2`), neutral exposure (`1.0`),
    /// AO off, no bloom.
    fn default() -> Self {
        Self {
            gamma: 2.2,
            exposure: 1.0,
            ao_mode: AoMode::Off,
            bloom_strength: 0.0,
        }
    }
}

/// Ambient occlusion rendering technique.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum AoMode {
    /// Ambient occlusion is disabled.
    Off,
    /// Ray-traced ambient occlusion — high quality but GPU-intensive.
    Rtao,
}

/// Water surface rendering quality settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WaterSettings {
    /// Whether screen-space or planar reflections are rendered on water surfaces.
    pub reflections: bool,
    /// Whether procedural wave displacement (Tesla waves) is applied to water.
    /// More visually dynamic but more expensive than flat water.
    pub tesla_waves: bool,
}

impl Default for WaterSettings {
    /// Enables reflections, disables tesla waves.
    fn default() -> Self {
        Self {
            reflections: true,
            tesla_waves: false,
        }
    }
}

/// Volume and mute settings for all audio channels.
///
/// All volume values are linear scalars in the range `[0.0, 1.0]`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioSettings {
    /// Global volume multiplier applied to all audio output.
    pub master_volume: f32,
    /// Volume multiplier for background music, relative to `master_volume`.
    pub music_volume: f32,
    /// Volume multiplier for sound effects, relative to `master_volume`.
    pub sfx_volume: f32,
    /// When `true`, all audio output is silenced regardless of volume levels.
    pub muted: bool,
}

impl Default for AudioSettings {
    /// Returns full master and SFX volume (`1.0`), slightly reduced music
    /// volume (`0.8`), and unmuted.
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            music_volume: 0.8,
            sfx_volume: 1.0,
            muted: false,
        }
    }
}

/// Mouse and keyboard input settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ControlsSettings {
    /// Mouse look sensitivity multiplier. `0.5` is the default balanced value.
    pub mouse_sensitivity: f32,
    /// When `true`, vertical mouse movement is inverted (push forward to look down).
    pub invert_mouse: bool,
    /// Key and button bindings for all player actions.
    pub keybinds: Keybinds,
}

impl Default for ControlsSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 0.5,
            invert_mouse: false,
            keybinds: Keybinds::default(),
        }
    }
}

/// Key and button bindings for all player actions.
///
/// Each field is a string token in the form `"device.Key"`, for example
/// `"keyboard.W"` or `"mouse.Right"`. Fields with `#[serde(default = …)]`
/// were added after the initial release and fall back to their default
/// functions when loading older save files that predate them.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Keybinds {
    /// Move forward. Default: `keyboard.W`.
    pub forward: String,
    /// Move backward. Default: `keyboard.S`.
    pub back: String,
    /// Strafe left. Default: `keyboard.A`.
    pub left: String,
    /// Strafe right. Default: `keyboard.D`.
    pub right: String,
    /// Jump. Default: `keyboard.Space`.
    pub jump: String,
    /// Sprint modifier. Default: `keyboard.LeftShift`.
    pub sprint: String,
    /// Sneak / crouch modifier. Default: `keyboard.LeftControl`.
    pub sneak: String,
    /// Open inventory screen. Default: `keyboard.E`.
    pub inventory: String,
    /// Open chat input. Default: `keyboard.T`.
    pub chat: String,
    /// Capture screenshot. Default: `keyboard.F2`.
    pub screenshot: String,
    /// Save the current world to disk. Default: `keyboard.F5`.
    pub save_world: String,
    /// Load the most recent world save. Default: `keyboard.F9`.
    pub load_world: String,
    /// Toggle reflection rendering debug mode. Default: `keyboard.R`.
    pub reflection_mode: String,
    /// Interact with a block or entity. Default: `mouse.Right`.
    /// Added after initial release; falls back to [`default_interact_key`].
    #[serde(default = "default_interact_key")]
    pub interact: String,
    /// Toggle creative-mode flight. Default: `keyboard.G`.
    /// Added after initial release; falls back to [`default_fly_key`].
    #[serde(default = "default_fly_key")]
    pub toggle_fly: String,
}

/// Returns the default binding for the interact action (`"mouse.Right"`).
///
/// Used as a `serde` default function so older save files without this
/// field deserialise correctly.
fn default_interact_key() -> String {
    "mouse.Right".to_string()
}

/// Returns the default binding for the toggle-fly action (`"keyboard.G"`).
///
/// Used as a `serde` default function so older save files without this
/// field deserialise correctly.
fn default_fly_key() -> String {
    "keyboard.G".to_string()
}

impl Default for Keybinds {
    fn default() -> Self {
        Self {
            forward: "keyboard.W".to_string(),
            back: "keyboard.S".to_string(),
            left: "keyboard.A".to_string(),
            right: "keyboard.D".to_string(),
            jump: "keyboard.Space".to_string(),
            sprint: "keyboard.LeftShift".to_string(),
            sneak: "keyboard.LeftControl".to_string(),
            inventory: "keyboard.E".to_string(),
            chat: "keyboard.T".to_string(),
            screenshot: "keyboard.F2".to_string(),
            save_world: "keyboard.F5".to_string(),
            load_world: "keyboard.F9".to_string(),
            reflection_mode: "keyboard.R".to_string(),
            interact: default_interact_key(),
            toggle_fly: default_fly_key(),
        }
    }
}

/// Gameplay feel and camera settings.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameplaySettings {
    /// Whether the camera bobs up and down while the player is walking.
    pub view_bobbing: bool,
    /// Controls how quickly the camera interpolates to the target orientation.
    /// Higher values feel snappier; lower values feel smoother/floatier.
    /// Expressed as a damping coefficient — `10.0` is the default balanced value.
    pub camera_smoothness: f32,
}

impl Default for GameplaySettings {
    fn default() -> Self {
        Self {
            view_bobbing: true,
            camera_smoothness: 10.0,
        }
    }
}

/// Developer and diagnostic overlay settings.
///
/// All fields default to `false` so debug overlays are never shown
/// unintentionally in production builds.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    /// Displays a frames-per-second counter in the corner of the screen.
    pub show_fps: bool,
    /// Displays the player's current world coordinates (X, Y, Z) on screen.
    pub show_coords: bool,
    /// Renders geometry as wireframe instead of solid surfaces.
    pub wireframe_mode: bool,
    /// Draws coloured borders around loaded chunk boundaries.
    pub show_chunk_borders: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            show_fps: false,
            show_coords: false,
            wireframe_mode: false,
            show_chunk_borders: false,
        }
    }
}

/// Serialises `settings` to `settings.bin` using `bincode`.
///
/// Creates or overwrites the file in the current working directory.
///
/// # Errors
///
/// Returns a boxed error if serialisation fails or the file cannot be
/// created or written.
pub fn save_settings(settings: &GameSettings) -> Result<(), Box<dyn std::error::Error>> {
    let encoded: Vec<u8> = bincode::serialize(settings)?;
    let mut file = File::create("settings.bin")?;
    file.write_all(&encoded)?;
    Ok(())
}

/// Loads settings from `settings.bin`, falling back to
/// [`GameSettings::default`] on any error.
///
/// Errors are logged at `WARN` level via `tracing` but are not propagated
/// to the caller, making this safe to call unconditionally at startup.
/// For explicit error handling use the private [`try_load_settings`].
pub fn load_settings() -> GameSettings {
    match try_load_settings() {
        Ok(settings) => settings,
        Err(e) => {
            warn!("Failed to load settings: {}. Using defaults.", e);
            GameSettings::default()
        }
    }
}

/// Attempts to open and deserialise `settings.bin`.
///
/// Separated from [`load_settings`] so the error path can be handled in
/// one place without duplicating file I/O logic.
///
/// # Errors
///
/// Returns a boxed error if the file cannot be opened, read, or
/// deserialised by `bincode`.
fn try_load_settings() -> Result<GameSettings, Box<dyn std::error::Error>> {
    let mut file = File::open("settings.bin")?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;
    Ok(bincode::deserialize(&encoded)?)
}