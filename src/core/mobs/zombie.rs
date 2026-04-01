use super::mob::{Mob, MobType, MobVariant};

pub static ZOMBIE: &Mob = &Mob {
    id: "minerust:zombie",
    name: "Zombie",
    health: 20.0,
    walk_speed: 1.0,
    run_speed: 1.5,
    jump_height: 1.0,
    damage: 3.0,
    mob_type: MobType::Hostile,
    variant: MobVariant::Normal,
};

pub static BABY_ZOMBIE: &Mob = &Mob {
    id: "minerust:baby_zombie",
    name: "Baby zombie",
    health: 15.0,
    walk_speed: 1.5,
    run_speed: 2.5,
    jump_height: 1.0,
    damage: 4.0,
    mob_type: MobType::Hostile,
    variant: MobVariant::Baby,
};
