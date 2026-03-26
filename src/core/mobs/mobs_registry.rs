use super::mob::Mob;
use super::zombie;

pub fn all_mobs() -> Vec<&'static Mob> {
    vec![
        zombie::MOB,
    ]
}