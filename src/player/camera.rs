use glam::{Mat4, Vec3};

use crate::constants::*;
use crate::core::block::BlockType;
use crate::player::input::InputState;
use crate::world::World;

/// First-person camera that doubles as the player's physical body.
///
/// Owns the player's world-space position, look angles, and physics state.
/// Movement, collision detection, and water interaction are all handled in
/// [`Camera::update`]. The eye point is offset `+1.8` units above
/// [`Self::position`] to simulate head height.
pub struct Camera {
    /// Foot-level world-space position of the player.
    ///
    /// The Y component represents the bottom of the player's AABB. Use
    /// [`Camera::eye_position`] to get the rendering origin.
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: Vec3,

    /// `true` when the player is resting on a solid surface.
    ///
    /// Jumping is only allowed when this flag is set. Cleared when the player
    /// moves upward or away from the ground, set when downward collision is detected.
    pub on_ground: bool,

    /// `true` when at least one block overlapping the player's body is [`BlockType::Water`].
    ///
    /// Switches the physics constants to underwater values (reduced gravity,
    /// lower speed, swim controls).
    pub in_water: bool,
}

impl Camera {
    /// Creates a new camera at the given world-space spawn position.
    ///
    /// Yaw and pitch are initialized to `0.0` (looking toward +X).
    /// Velocity is zero and both `on_ground` and `in_water` are `false`.
    pub fn new(spawn: (f32, f32, f32)) -> Self {
        Camera {
            position: Vec3::new(spawn.0, spawn.1, spawn.2),
            yaw: 0.0,
            pitch: 0.0,
            velocity: Vec3::ZERO,
            on_ground: false,
            in_water: false,
        }
    }

    /// Returns the horizontal forward unit vector based on the current yaw.
    ///
    /// Y is always `0.0`; use [`Camera::look_direction`] for the full 3-D
    /// gaze vector including pitch.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    pub fn right(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    pub fn look_direction(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn eye_position(&self) -> Vec3 {
        Vec3::new(self.position.x, self.position.y + 1.8, self.position.z)
    }

    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_position();
        let target = eye + self.look_direction();
        Mat4::look_at_rh(eye, target, Vec3::Y)
    }

    /// Returns `true` if the block at the player's feet or mid-body is [`BlockType::Water`].
    ///
    /// Checks two sample points: the foot block (`position.y`) and a mid-body
    /// block (`position.y + 0.9`) to handle partial submersion.
    fn check_in_water(&self, world: &World) -> bool {
        let feet_block = world.get_block(
            self.position.x.floor() as i32,
            self.position.y.floor() as i32,
            self.position.z.floor() as i32,
        );
        let body_block = world.get_block(
            self.position.x.floor() as i32,
            (self.position.y + 0.9).floor() as i32,
            self.position.z.floor() as i32,
        );
        feet_block == BlockType::Water || body_block == BlockType::Water
    }

    /// Returns `true` if the block at the eye position is [`BlockType::Water`].
    ///
    /// Used by the renderer to apply underwater post-processing effects.
    pub fn is_head_underwater(&self, world: &World) -> bool {
        let eye = self.eye_position();
        let block = world.get_block(
            eye.x.floor() as i32,
            eye.y.floor() as i32,
            eye.z.floor() as i32,
        );
        block == BlockType::Water
    }

    /// Advances the player simulation by one frame.
    ///
    /// Each call performs the following steps in order:
    /// 1. Detects water submersion via [`Camera::check_in_water`].
    /// 2. Select physics constants (speed, gravity, drag) based on water state and sprint input.
    /// 3. Accumulates a movement direction from `input` and scales it to `base_speed`.
    /// 4. Applies gravity, jump impulse, and drag.
    /// 5. Resolves collisions on each axis independently using [`Camera::check_collision`].
    /// 6. Clamps Y to a minimum of `1.0` to prevent falling out of the world.
    ///
    /// # Parameters
    /// - `world` — used for block queries during collision and water detection.
    /// - `dt` — delta time in seconds since the last frame.
    /// - `input` — current frame's digital input state.
    pub fn update(&mut self, world: &World, dt: f32, input: &InputState) {
        self.in_water = self.check_in_water(world);

        let (base_speed, gravity, max_fall_speed, jump_velocity, horizontal_drag, vertical_drag) =
            if self.in_water {
                let speed = if input.sprint {
                    PLAYER_SPRINT_SPEED * 0.331
                } else {
                    PLAYER_BASE_SPEED * 0.331
                };
                (speed, 6.0, 3.0, 4.0, 0.9, 0.95)
            } else {
                let speed = if input.sprint {
                    PLAYER_SPRINT_SPEED
                } else {
                    PLAYER_BASE_SPEED
                };
                (speed, 25.0, 50.0, 8.0, 1.0, 1.0)
            };

        let mut move_dir = Vec3::ZERO;

        if input.forward {
            move_dir += self.forward();
        }
        if input.backward {
            move_dir -= self.forward();
        }
        if input.left {
            move_dir -= self.right();
        }
        if input.right {
            move_dir += self.right();
        }

        if move_dir.length_squared() > 0.0 {
            move_dir = move_dir.normalize() * base_speed;
        }

        self.velocity.x = move_dir.x * horizontal_drag;
        self.velocity.z = move_dir.z * horizontal_drag;

        if self.in_water {
            if input.jump {
                self.velocity.y = jump_velocity;
            } else if input.sprint {
                self.velocity.y = -jump_velocity;
            } else {
                self.velocity.y -= gravity * dt;
                self.velocity.y *= vertical_drag;
            }
            self.velocity.y = self.velocity.y.clamp(-max_fall_speed * 2.0, jump_velocity);
        } else {
            if input.jump && self.on_ground {
                self.velocity.y = jump_velocity;
                self.on_ground = false;
            }
            self.velocity.y -= gravity * dt;
            self.velocity.y = self.velocity.y.max(-max_fall_speed);
        }

        let new_pos = self.position + self.velocity * dt;

        if !self.check_collision(world, new_pos.x, self.position.y, self.position.z) {
            self.position.x = new_pos.x;
        } else {
            self.velocity.x = 0.0;
        }

        if !self.check_collision(world, self.position.x, self.position.y, new_pos.z) {
            self.position.z = new_pos.z;
        } else {
            self.velocity.z = 0.0;
        }

        if !self.check_collision(world, self.position.x, new_pos.y, self.position.z) {
            self.position.y = new_pos.y;
            if !self.in_water {
                self.on_ground = false;
            }
        } else {
            if self.velocity.y < 0.0 {
                self.on_ground = true;
            }
            self.velocity.y = 0.0;
        }

        self.position.y = self.position.y.max(1.0);
    }

    /// Returns `true` if the player AABB centered at `(x, y, z)` overlaps any solid block.
    ///
    /// Iterates over all blocks within the bounding box defined by
    /// [`PLAYER_WIDTH`] and [`PLAYER_HEIGHT`] and delegates intersection
    /// testing to [`check_intersection`].
    ///
    /// Used by [`Camera::update`] for per-axis collision resolution.
    pub fn check_collision(&self, world: &World, x: f32, y: f32, z: f32) -> bool {
        let player_width = PLAYER_WIDTH;
        let player_height = PLAYER_HEIGHT;

        let min_x = (x - player_width).floor() as i32;
        let max_x = (x + player_width).floor() as i32;
        let min_y = y.floor() as i32;
        let max_y = (y + player_height).floor() as i32;
        let min_z = (z - player_width).floor() as i32;
        let max_z = (z + player_width).floor() as i32;

        for bx in min_x..=max_x {
            for by in min_y..=max_y {
                for bz in min_z..=max_z {
                    if world.is_solid(bx, by, bz) {
                        if check_intersection(Vec3::new(x, y, z), bx, by, bz) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Returns `true` if the player's current AABB intersects the block at `(bx, by, bz)`.
    ///
    /// Convenience wrapper around [`check_intersection`] using [`Self::position`].
    pub fn intersects_block(&self, bx: i32, by: i32, bz: i32) -> bool {
        check_intersection(self.position, bx, by, bz)
    }

    /// Casts a ray from the eye position along the look direction and returns
    /// the first solid block hit within `max_dist` world units.
    ///
    /// Steps along the ray in increments of `0.1` units. Returns
    /// `Some((hit_x, hit_y, hit_z, prev_x, prev_y, prev_z))` where the first
    /// three components are the coordinates of the block that was hit and the
    /// last three are the coordinates of the last empty block before the hit
    /// (used for block placement). Returns `None` if no solid block is found
    /// within `max_dist`.
    pub fn raycast(&self, world: &World, max_dist: f32) -> Option<(i32, i32, i32, i32, i32, i32)> {
        let dir = self.look_direction();
        let eye = self.eye_position();
        let mut pos = Vec3::new(eye.x, eye.y, eye.z);
        let step = 0.1;
        let mut prev = (
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        );

        for _ in 0..(max_dist / step) as i32 {
            pos += dir * step;
            let current = (
                pos.x.floor() as i32,
                pos.y.floor() as i32,
                pos.z.floor() as i32,
            );
            if current != prev {
                if world.is_solid(current.0, current.1, current.2) {
                    return Some((current.0, current.1, current.2, prev.0, prev.1, prev.2));
                }
                prev = current;
            }
        }
        None
    }
}

/// Returns `true` if the player AABB rooted at `pos` overlaps the unit block at `(bx, by, bz)`.
///
/// The player AABB extends [`PLAYER_WIDTH`] units in ±X and ±Z from `pos`,
/// and [`PLAYER_HEIGHT`] units upward from `pos.y`. Uses a standard
/// axis-aligned box vs. box intersection test.
pub fn check_intersection(pos: Vec3, bx: i32, by: i32, bz: i32) -> bool {
    let player_width = PLAYER_WIDTH;
    let player_height = PLAYER_HEIGHT;

    let block_min_x = bx as f32;
    let block_max_x = (bx + 1) as f32;
    let block_min_y = by as f32;
    let block_max_y = (by + 1) as f32;
    let block_min_z = bz as f32;
    let block_max_z = (bz + 1) as f32;

    let player_min_x = pos.x - player_width;
    let player_max_x = pos.x + player_width;
    let player_min_y = pos.y;
    let player_max_y = pos.y + player_height;
    let player_min_z = pos.z - player_width;
    let player_max_z = pos.z + player_width;

    player_max_x > block_min_x
        && player_min_x < block_max_x
        && player_max_y > block_min_y
        && player_min_y < block_max_y
        && player_max_z > block_min_z
        && player_min_z < block_max_z
}
