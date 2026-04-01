use super::mob::{Mob, MobType, MobVariant};

pub static VILLAGER: &Mob = &Mob {
    id: "minerust:villager",
    name: "Villager",
    health: 20.0,
    walk_speed: 1.0,
    run_speed: 1.5,
    jump_height: 1.0,
    damage: 0.0,
    mob_type: MobType::Passive,
    variant: MobVariant::Normal,
};

pub static BABY_VILLAGER: &Mob = &Mob {
    id: "minerust:baby_villager",
    name: "Baby villager",
    health: 15.0,
    walk_speed: 1.5,
    run_speed: 2.5,
    jump_height: 1.0,
    damage: 0.0,
    mob_type: MobType::Passive,
    variant: MobVariant::Baby,
};
