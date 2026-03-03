use serde::Deserialize;
use crate::utils::settings::Keybinds;

#[derive(Deserialize)]
struct Controls {
    keybinds: Keybinds,
}

#[derive(Deserialize)]
struct SettingsYaml {
    controls: Controls,
}

pub fn load_keybinds(path: &str) -> Result<Keybinds, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let settings: SettingsYaml = serde_yaml::from_str(&content)?;
    Ok(settings.controls.keybinds)
}