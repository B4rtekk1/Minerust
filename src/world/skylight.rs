//! World-space sky light.
//!
//! Every non-opaque cell gets a sky light level in `0..=15` that depends only
//! on the world around it, never on where the camera is:
//!
//! 1. **Sky columns**: light falls straight down from the open sky at full
//!    strength until it hits an opaque block. Foliage dims it on the way
//!    (forest floors are shaded) and water dims it with depth.
//! 2. **Spreading**: light then flows sideways and around corners, losing a
//!    fixed amount per block, so overhangs, cave mouths and tunnels fade
//!    gradually into darkness.
//!
//! Light is solved per mesh job over a region padded by [`LIGHT_PAD`] blocks
//! around the subchunk, so faces near the subchunk's edges see the light
//! coming from their neighbours. Light from further than `LIGHT_PAD` blocks
//! away is ignored; with a falloff of [`SPREAD_COST`] per block it would have
//! decayed to (almost) nothing anyway.

use crate::constants::{CHUNK_SIZE, SUBCHUNK_HEIGHT};
use crate::core::block::BlockType;

/// Blocks of context around the subchunk used to solve its light.
pub const LIGHT_PAD: i32 = 8;
/// Region side along X and Z.
pub const LIGHT_SIZE: usize = (CHUNK_SIZE + LIGHT_PAD * 2) as usize;
/// Region height.
pub const LIGHT_HEIGHT: usize = (SUBCHUNK_HEIGHT + LIGHT_PAD * 2) as usize;
pub const LIGHT_LEN: usize = LIGHT_SIZE * LIGHT_HEIGHT * LIGHT_SIZE;
pub const MAX_SKY_LIGHT: u8 = 15;
/// Light lost per block when spreading through air.
pub const SPREAD_COST: u8 = 2;

/// How a cell transports light.
pub const MEDIUM_OPAQUE: u8 = 0;
pub const MEDIUM_AIR: u8 = 1;
pub const MEDIUM_FOLIAGE: u8 = 2;
pub const MEDIUM_WATER: u8 = 3;

pub fn medium(block: BlockType) -> u8 {
    match block {
        BlockType::Leaves | BlockType::Ice => MEDIUM_FOLIAGE,
        BlockType::Water => MEDIUM_WATER,
        b if b.is_solid_opaque() => MEDIUM_OPAQUE,
        _ => MEDIUM_AIR,
    }
}

/// Index of region cell `(x, y, z)` (all region-local, `0..LIGHT_SIZE` etc.).
#[inline]
pub fn light_index(x: usize, y: usize, z: usize) -> usize {
    (x * LIGHT_HEIGHT + y) * LIGHT_SIZE + z
}

/// Solves sky light for one region.
///
/// - `media`: [`medium`] of every cell, indexed by [`light_index`].
/// - `sky_height`: per column (`x * LIGHT_SIZE + z`), the highest opaque
///   block in the whole world column (world Y), or `-1` if none.
/// - `region_bottom`: world Y of region row `y = 0`.
pub fn solve(media: &[u8], sky_height: &[i16], region_bottom: i32) -> Vec<u8> {
    debug_assert_eq!(media.len(), LIGHT_LEN);
    let mut light = vec![0u8; LIGHT_LEN];
    let top = LIGHT_HEIGHT as i32 - 1;

    // 1. Sky columns.
    for x in 0..LIGHT_SIZE {
        for z in 0..LIGHT_SIZE {
            let highest = sky_height[x * LIGHT_SIZE + z] as i32;
            let mut level = if region_bottom + top > highest {
                MAX_SKY_LIGHT
            } else {
                0
            };
            let mut water_depth = 0u8;
            for y in (0..LIGHT_HEIGHT).rev() {
                if level == 0 {
                    break;
                }
                let i = light_index(x, y, z);
                match media[i] {
                    MEDIUM_OPAQUE => break,
                    MEDIUM_FOLIAGE => level = level.saturating_sub(2),
                    MEDIUM_WATER => {
                        water_depth += 1;
                        if water_depth % 2 == 0 {
                            level = level.saturating_sub(1);
                        }
                    }
                    _ => {}
                }
                light[i] = level;
            }
        }
    }

    // 2. Spread. Only cells that can still brighten a neighbour start in the
    // queue, so open terrain (all cells already at full light) is cheap.
    let mut queue: Vec<u32> = Vec::new();
    for x in 0..LIGHT_SIZE {
        for y in 0..LIGHT_HEIGHT {
            for z in 0..LIGHT_SIZE {
                let i = light_index(x, y, z);
                let level = light[i];
                if level <= SPREAD_COST {
                    continue;
                }
                let can_spread = neighbours(x, y, z).any(|n| {
                    media[n] != MEDIUM_OPAQUE && light[n] + cost(media[n]) < level
                });
                if can_spread {
                    queue.push(i as u32);
                }
            }
        }
    }
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head] as usize;
        head += 1;
        let level = light[i];
        let (x, y, z) = position(i);
        for n in neighbours(x, y, z) {
            let medium = media[n];
            if medium == MEDIUM_OPAQUE {
                continue;
            }
            let next = level.saturating_sub(cost(medium));
            if next > light[n] {
                light[n] = next;
                if next > SPREAD_COST {
                    queue.push(n as u32);
                }
            }
        }
    }
    light
}

#[inline]
fn cost(medium: u8) -> u8 {
    match medium {
        MEDIUM_AIR => SPREAD_COST,
        _ => SPREAD_COST + 1,
    }
}

#[inline]
fn position(i: usize) -> (usize, usize, usize) {
    let z = i % LIGHT_SIZE;
    let y = (i / LIGHT_SIZE) % LIGHT_HEIGHT;
    let x = i / (LIGHT_SIZE * LIGHT_HEIGHT);
    (x, y, z)
}

fn neighbours(x: usize, y: usize, z: usize) -> impl Iterator<Item = usize> {
    let candidates = [
        (x.wrapping_sub(1), y, z),
        (x + 1, y, z),
        (x, y.wrapping_sub(1), z),
        (x, y + 1, z),
        (x, y, z.wrapping_sub(1)),
        (x, y, z + 1),
    ];
    candidates.into_iter().filter_map(|(x, y, z)| {
        (x < LIGHT_SIZE && y < LIGHT_HEIGHT && z < LIGHT_SIZE).then(|| light_index(x, y, z))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(fill: impl Fn(usize, usize, usize) -> u8) -> Vec<u8> {
        let mut media = vec![MEDIUM_AIR; LIGHT_LEN];
        for x in 0..LIGHT_SIZE {
            for y in 0..LIGHT_HEIGHT {
                for z in 0..LIGHT_SIZE {
                    media[light_index(x, y, z)] = fill(x, y, z);
                }
            }
        }
        media
    }

    fn sky_heights(media: &[u8], region_bottom: i32) -> Vec<i16> {
        (0..LIGHT_SIZE * LIGHT_SIZE)
            .map(|c| {
                let (x, z) = (c / LIGHT_SIZE, c % LIGHT_SIZE);
                (0..LIGHT_HEIGHT)
                    .rev()
                    .find(|&y| media[light_index(x, y, z)] == MEDIUM_OPAQUE)
                    .map_or(-1, |y| (region_bottom + y as i32) as i16)
            })
            .collect()
    }

    #[test]
    fn open_sky_is_fully_lit_and_cave_is_dark() {
        // Solid ground up to y = 10 with a sealed pocket at y = 4.
        let media = region(|x, y, z| {
            if y == 4 && (10..14).contains(&x) && (10..14).contains(&z) {
                MEDIUM_AIR
            } else if y <= 10 {
                MEDIUM_OPAQUE
            } else {
                MEDIUM_AIR
            }
        });
        let light = solve(&media, &sky_heights(&media, 0), 0);
        assert_eq!(light[light_index(5, 20, 5)], MAX_SKY_LIGHT);
        assert_eq!(light[light_index(5, 11, 5)], MAX_SKY_LIGHT);
        assert_eq!(light[light_index(11, 4, 11)], 0);
    }

    #[test]
    fn light_fades_under_an_overhang() {
        // A roof at y = 20 over x < 16; open sky for x >= 16.
        let media = region(|x, y, _| {
            if y <= 5 || (y == 20 && x < 16) {
                MEDIUM_OPAQUE
            } else {
                MEDIUM_AIR
            }
        });
        let light = solve(&media, &sky_heights(&media, 0), 0);
        let under = |x: usize| light[light_index(x, 10, 16)];
        assert_eq!(under(20), MAX_SKY_LIGHT);
        assert_eq!(under(15), MAX_SKY_LIGHT - SPREAD_COST);
        assert_eq!(under(12), MAX_SKY_LIGHT - SPREAD_COST * 4);
        assert!(under(15) > under(12) && under(12) > under(8));
    }

    #[test]
    fn foliage_and_water_dim_light() {
        let media = region(|_, y, _| match y {
            0..=5 => MEDIUM_OPAQUE,
            6..=13 => MEDIUM_WATER,
            25 => MEDIUM_FOLIAGE,
            _ => MEDIUM_AIR,
        });
        let light = solve(&media, &sky_heights(&media, 0), 0);
        assert_eq!(light[light_index(16, 24, 16)], MAX_SKY_LIGHT - 2);
        assert!(light[light_index(16, 6, 16)] < light[light_index(16, 13, 16)]);
    }
}
