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
/// Light lost per block when block light spreads through the world.
pub const BLOCK_SPREAD_COST: u8 = 2;
const BLOCK_LIGHT_RADIUS: usize = (MAX_SKY_LIGHT as usize / BLOCK_SPREAD_COST as usize) + 1;

/// How a cell transports light.
pub const MEDIUM_OPAQUE: u8 = 0;
pub const MEDIUM_AIR: u8 = 1;
pub const MEDIUM_FOLIAGE: u8 = 2;
pub const MEDIUM_WATER: u8 = 3;

pub fn medium(block: BlockType) -> u8 {
    match block {
        BlockType::Leaves | BlockType::CherryLeaves | BlockType::Ice => MEDIUM_FOLIAGE,
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
                let can_spread = neighbours(x, y, z)
                    .any(|n| media[n] != MEDIUM_OPAQUE && light[n] + cost(media[n]) < level);
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

/// Solves block light from emitted levels over one padded region.
///
/// Opaque source cells emit through each exposed face, and each emission is
/// restricted to the half-space in front of that face. Light cannot enter or
/// pass through other opaque cells.
pub fn solve_block(media: &[u8], sources: &[u8]) -> Vec<u8> {
    debug_assert_eq!(media.len(), LIGHT_LEN);
    debug_assert_eq!(sources.len(), LIGHT_LEN);
    let mut light = vec![0u8; LIGHT_LEN];
    for (index, &source) in sources.iter().enumerate() {
        if source > 0 {
            light[index] = light[index].max(source.min(MAX_SKY_LIGHT));
        }
    }

    let directions = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    for (index, &source) in sources.iter().enumerate() {
        if source == 0 {
            continue;
        }
        let source_position = position(index);
        for direction in directions {
            spread_block_face(
                media,
                &mut light,
                source_position,
                direction,
                source.min(MAX_SKY_LIGHT),
            );
        }
    }
    light
}

fn spread_block_face(
    media: &[u8],
    light: &mut [u8],
    source: (usize, usize, usize),
    direction: (i32, i32, i32),
    source_level: u8,
) {
    let Some((start_x, start_y, start_z)) = offset_position(source, direction) else {
        return;
    };
    let start = light_index(start_x, start_y, start_z);
    if media[start] == MEDIUM_OPAQUE {
        return;
    }

    let first_level = source_level.saturating_sub(BLOCK_SPREAD_COST);
    if first_level == 0 {
        return;
    }

    let local_side = BLOCK_LIGHT_RADIUS * 2 + 1;
    let mut local_light = vec![0u8; local_side * local_side * local_side];
    let mut queue = Vec::new();
    let mut touched = Vec::new();
    let start_local = block_local_index((start_x, start_y, start_z), source, local_side);
    local_light[start_local] = first_level;
    touched.push((start, start_local));
    if first_level > BLOCK_SPREAD_COST {
        queue.push((start, first_level));
    }

    let mut head = 0;
    while head < queue.len() {
        let (index, level) = queue[head];
        head += 1;
        if level <= BLOCK_SPREAD_COST {
            continue;
        }
        let next = level.saturating_sub(BLOCK_SPREAD_COST);
        let (x, y, z) = position(index);
        for neighbour in neighbours(x, y, z) {
            let neighbour_position = position(neighbour);
            if !in_face_half_space(neighbour_position, source, direction)
                || !within_block_radius(neighbour_position, source)
                || media[neighbour] == MEDIUM_OPAQUE
            {
                continue;
            }

            let local = block_local_index(neighbour_position, source, local_side);
            if next > local_light[local] {
                if local_light[local] == 0 {
                    touched.push((neighbour, local));
                }
                local_light[local] = next;
                if next > BLOCK_SPREAD_COST {
                    queue.push((neighbour, next));
                }
            }
        }
    }

    for (index, local) in touched {
        light[index] = light[index].max(local_light[local]);
    }
}

#[inline]
fn offset_position(
    (x, y, z): (usize, usize, usize),
    (dx, dy, dz): (i32, i32, i32),
) -> Option<(usize, usize, usize)> {
    let (x, y, z) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
    (x >= 0
        && x < LIGHT_SIZE as i32
        && y >= 0
        && y < LIGHT_HEIGHT as i32
        && z >= 0
        && z < LIGHT_SIZE as i32)
        .then_some((x as usize, y as usize, z as usize))
}

#[inline]
fn in_face_half_space(
    (x, y, z): (usize, usize, usize),
    (source_x, source_y, source_z): (usize, usize, usize),
    direction: (i32, i32, i32),
) -> bool {
    match direction {
        (-1, 0, 0) => x < source_x,
        (1, 0, 0) => x > source_x,
        (0, -1, 0) => y < source_y,
        (0, 1, 0) => y > source_y,
        (0, 0, -1) => z < source_z,
        (0, 0, 1) => z > source_z,
        _ => false,
    }
}

#[inline]
fn within_block_radius(
    (x, y, z): (usize, usize, usize),
    (source_x, source_y, source_z): (usize, usize, usize),
) -> bool {
    x.abs_diff(source_x) <= BLOCK_LIGHT_RADIUS
        && y.abs_diff(source_y) <= BLOCK_LIGHT_RADIUS
        && z.abs_diff(source_z) <= BLOCK_LIGHT_RADIUS
}

#[inline]
fn block_local_index(
    (x, y, z): (usize, usize, usize),
    (source_x, source_y, source_z): (usize, usize, usize),
    local_side: usize,
) -> usize {
    let radius = BLOCK_LIGHT_RADIUS as isize;
    let (x, y, z) = (
        x as isize - source_x as isize + radius,
        y as isize - source_y as isize + radius,
        z as isize - source_z as isize + radius,
    );
    (x as usize * local_side + y as usize) * local_side + z as usize
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

    #[test]
    fn block_light_spreads_from_opaque_sources_and_stops_at_walls() {
        let source = light_index(16, 8, 16);
        let mut media = vec![MEDIUM_AIR; LIGHT_LEN];
        let mut sources = vec![0u8; LIGHT_LEN];
        media[source] = MEDIUM_OPAQUE;
        sources[source] = MAX_SKY_LIGHT;
        for z in 0..LIGHT_SIZE {
            for y in 0..LIGHT_HEIGHT {
                media[light_index(20, y, z)] = MEDIUM_OPAQUE;
            }
        }

        let light = solve_block(&media, &sources);
        assert_eq!(light[source], 15);
        assert_eq!(light[light_index(15, 8, 16)], 13);
        assert_eq!(light[light_index(9, 8, 16)], 1);
        assert_eq!(light[light_index(8, 8, 16)], 0);
        assert_eq!(light[light_index(21, 8, 16)], 0);
    }

    #[test]
    fn enclosed_block_light_source_lights_nothing() {
        let source = light_index(16, 8, 16);
        let media = vec![MEDIUM_OPAQUE; LIGHT_LEN];
        let mut sources = vec![0u8; LIGHT_LEN];
        sources[source] = MAX_SKY_LIGHT;

        let light = solve_block(&media, &sources);
        for (index, &level) in light.iter().enumerate() {
            assert_eq!(level, if index == source { MAX_SKY_LIGHT } else { 0 });
        }
    }

    #[test]
    fn block_light_only_leaves_through_exposed_faces() {
        let source = light_index(16, 8, 16);
        let media = region(|x, y, z| {
            if y <= 8 && (10..=20).contains(&x) && (10..=20).contains(&z) {
                MEDIUM_OPAQUE
            } else {
                MEDIUM_AIR
            }
        });
        let mut sources = vec![0u8; LIGHT_LEN];
        sources[source] = MAX_SKY_LIGHT;

        let light = solve_block(&media, &sources);
        assert_eq!(light[light_index(16, 9, 16)], 13);
        assert_eq!(light[light_index(17, 9, 16)], 11);
        assert_eq!(light[light_index(21, 8, 16)], 0);
        assert_eq!(light[light_index(21, 7, 16)], 0);
    }
}
