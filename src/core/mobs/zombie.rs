use super::mob::Mob;

pub static MOB: &Mob = &Mob {
    id: "minerust:zombie",
    name: "Zombie",
    health: 20.0,
    walk_speed: 1.0,
    run_speed: 1.5,
    jump_height: 1.0,
    damage: 3.0,
};