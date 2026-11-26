//! Player-selectable graphics options, persisted between runs.

use std::fs;
use std::path::PathBuf;

use crate::logger::{LogLevel, log};
use crate::minerust_data::data::get_project_dirs;

/// How water surfaces are rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaterQuality {
    /// The original lightweight water: gentle ripples and a flat tint.
    #[default]
    Classic,
    /// Animated waves, depth-based colour and clarity, sun glitter, crest
    /// scattering and shoreline foam.
    Realistic,
}

impl WaterQuality {
    pub fn toggled(self) -> Self {
        match self {
            Self::Classic => Self::Realistic,
            Self::Realistic => Self::Classic,
        }
    }

    /// Value passed to the shaders in the `water_quality` uniform slot.
    pub fn shader_value(self) -> f32 {
        match self {
            Self::Classic => 0.0,
            Self::Realistic => 1.0,
        }
    }

    pub fn menu_label(self) -> &'static str {
        match self {
            Self::Classic => "WATER: CLASSIC",
            Self::Realistic => "WATER: REALISTIC",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Realistic => "realistic",
        }
    }
}

/// Edge anti-aliasing. Changing it rebuilds every render pipeline, so the
/// choice is saved and applied at the next start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AntiAliasing {
    /// 4× MSAA: smoothest edges, highest raster and memory cost.
    #[default]
    Msaa4,
    /// 2× MSAA: roughly half the multisample cost.
    Msaa2,
    /// No anti-aliasing: cheapest, visibly jagged edges.
    Off,
}

impl AntiAliasing {
    pub fn next(self) -> Self {
        match self {
            Self::Msaa4 => Self::Msaa2,
            Self::Msaa2 => Self::Off,
            Self::Off => Self::Msaa4,
        }
    }

    pub fn sample_count(self) -> u32 {
        match self {
            Self::Msaa4 => 4,
            Self::Msaa2 => 2,
            Self::Off => 1,
        }
    }

    pub fn menu_label(self) -> &'static str {
        match self {
            Self::Msaa4 => "ANTI-ALIASING: 4X",
            Self::Msaa2 => "ANTI-ALIASING: 2X",
            Self::Off => "ANTI-ALIASING: OFF",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Self::Msaa4 => "4x",
            Self::Msaa2 => "2x",
            Self::Off => "off",
        }
    }
}

/// Graphics options chosen in the main menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphicsOptions {
    pub water: WaterQuality,
    pub anti_aliasing: AntiAliasing,
    /// Ray-marched voxel sun shadows (toggled in game with F7).
    pub shadows: bool,
}

impl Default for GraphicsOptions {
    fn default() -> Self {
        Self {
            water: WaterQuality::default(),
            anti_aliasing: AntiAliasing::default(),
            shadows: true,
        }
    }
}

impl GraphicsOptions {
    fn path() -> Option<PathBuf> {
        get_project_dirs()
            .ok()
            .map(|dirs| dirs.config_dir().join("graphics.cfg"))
    }

    /// Loads the saved options; missing or unreadable files give defaults.
    pub fn load() -> Self {
        let Some(text) = Self::path().and_then(|path| fs::read_to_string(path).ok()) else {
            return Self::default();
        };
        Self::parse(&text)
    }

    fn parse(text: &str) -> Self {
        let mut options = Self::default();
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            match (key.trim(), value.trim()) {
                ("water", v) if v == WaterQuality::Realistic.key() => {
                    options.water = WaterQuality::Realistic;
                }
                ("shadows", v) => options.shadows = v != "off",
                ("aa", v) => {
                    for aa in [AntiAliasing::Msaa4, AntiAliasing::Msaa2, AntiAliasing::Off] {
                        if v == aa.key() {
                            options.anti_aliasing = aa;
                        }
                    }
                }
                _ => {}
            }
        }
        options
    }

    /// Writes the options; failures are logged and otherwise ignored.
    pub fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        let result = path
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|_| {
                fs::write(
                    &path,
                    format!(
                        "water={}\naa={}\nshadows={}\n",
                        self.water.key(),
                        self.anti_aliasing.key(),
                        if self.shadows { "on" } else { "off" }
                    ),
                )
            });
        if let Err(error) = result {
            log(
                LogLevel::Warning,
                &format!("Could not save graphics options to {}: {error}", path.display()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_round_trip_through_text() {
        let realistic = GraphicsOptions {
            water: WaterQuality::Realistic,
            ..Default::default()
        };
        assert_eq!(GraphicsOptions::parse("water=realistic\n"), realistic);
        let off = GraphicsOptions::parse("water=classic\naa=off\n");
        assert_eq!(off.anti_aliasing, AntiAliasing::Off);
        assert_eq!(off.anti_aliasing.sample_count(), 1);
        assert_eq!(AntiAliasing::Off.next(), AntiAliasing::Msaa4);
        assert_eq!(GraphicsOptions::parse("water=classic"), GraphicsOptions::default());
        assert_eq!(GraphicsOptions::parse("garbage"), GraphicsOptions::default());
        assert!(GraphicsOptions::default().shadows);
        assert!(!GraphicsOptions::parse("shadows=off").shadows);
        assert_eq!(WaterQuality::Classic.toggled(), WaterQuality::Realistic);
    }
}
