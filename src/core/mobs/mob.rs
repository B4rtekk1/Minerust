use once_cell::sync::Lazy;
use std::collections::HashMap;

pub enum MobType {
    Neutral,
    Hostile,
    Passive,
}

pub enum MobVariant {
    Normal,
    Baby,
    Elder,
}

pub struct Mob {
    pub id: &'static str,
    pub name: &'static str,
    pub health: f32,
    pub walk_speed: f32,
    pub run_speed: f32,
    pub jump_height: f32,
    pub damage: f32,
    pub mob_type: MobType,
    pub variant: MobVariant,
}

pub static MOBS: Lazy<HashMap<&'static str, &'static Mob>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for mob in self::super::mobs_registry::all_mobs() {
        map.insert(mob.id, mob);
    }
    map
});

pub fn get_mob(id: &str) -> Option<&'static Mob> {
    MOBS.get(id).copied()
}

pub fn mob_exists(id: &str) -> bool {
    MOBS.contains_key(id)
}
