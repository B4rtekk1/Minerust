use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameSettings {
    pub player: PlayerSettings,
    pub graphics: GraphicsSettings,
    pub audio: AudioSettings,
    pub controls: ControlsSettings,
    #[serde(default)]
    pub gameplay: GameplaySettings,
    #[serde(default)]
    pub debug: DebugSettings,
}

impl Default for GameSettings {
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlayerSettings {
    pub nickname: String,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphicsSettings {
    pub render_distance: u32,
    pub simulation_distance: u32,
    pub vsync: bool,
    pub fullscreen: bool,
    pub max_fps: u32,
    #[serde(default)]
    pub fov: f32,
    #[serde(default)]
    pub shadows: ShadowSettings,
    #[serde(default)]
    pub lighting: LightingSettings,
    #[serde(default)]
    pub water: WaterSettings,
}

impl Default for GraphicsSettings {
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ShadowSettings {
    pub enabled: bool,
    pub resolution: u32, // np. 1024, 2048, 4096
    pub cascades: u32,   // 1-4
    pub distance: f32,
    pub quality: ShadowQuality, // Softness control
}

impl Default for ShadowSettings {
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum ShadowQuality {
    Hard,
    Pcf,  // Percentage Closer Filtering
    Pcss, // Percentage Closer Soft Shadows
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LightingSettings {
    pub gamma: f32,
    pub exposure: f32,
    pub ao_mode: AoMode,
    pub bloom_strength: f32,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            gamma: 2.2,
            exposure: 1.0,
            ao_mode: AoMode::Off,
            bloom_strength: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum AoMode {
    Off,
    Rtao,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WaterSettings {
    pub reflections: bool,
    pub tesla_waves: bool,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            reflections: true,
            tesla_waves: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AudioSettings {
    pub master_volume: f32,
    pub music_volume: f32,
    pub sfx_volume: f32,
    pub muted: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            master_volume: 1.0,
            music_volume: 0.8,
            sfx_volume: 1.0,
            muted: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ControlsSettings {
    pub mouse_sensitivity: f32,
    pub invert_mouse: bool,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Keybinds {
    pub forward: String,
    pub back: String,
    pub left: String,
    pub right: String,
    pub jump: String,
    pub sprint: String,
    pub sneak: String,
    pub inventory: String,
    pub chat: String,
    pub screenshot: String,
    pub save_world: String,
    pub load_world: String,
    pub reflection_mode: String,
    #[serde(default = "default_interact_key")]
    pub interact: String,
    #[serde(default = "default_fly_key")]
    pub toggle_fly: String,
}

fn default_interact_key() -> String {
    "mouse.Right".to_string()
}
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameplaySettings {
    pub view_bobbing: bool,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub show_fps: bool,
    pub show_coords: bool,
    pub wireframe_mode: bool,
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

pub fn save_settings(settings: &GameSettings) {
    let encoded: Vec<u8> = bincode::serialize(settings).unwrap();
    let mut file = File::create("settings.bin").unwrap();
    file.write_all(&encoded).unwrap();
    Ok(())
}

pub fn load_settings() -> GameSettings {
    let mut file = File::open("settings.bin").unwrap();
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded).unwrap();
    bincode::deserialize(&encoded).unwrap()
}
