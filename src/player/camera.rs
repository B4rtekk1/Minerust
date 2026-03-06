use cgmath::{Matrix4, Point3, Vector3, prelude::*};

use crate::constants::*;
use crate::core::block::BlockType;
use crate::player::input::InputState;
use crate::world::World;

pub struct Camera {
    pub position: Point3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: Vector3<f32>,
    pub on_ground: bool,
    pub in_water: bool,
}

impl Camera {
    pub fn new(spawn: (f32, f32, f32)) -> Self {
        Camera {
            position: Point3::new(spawn.0, spawn.1, spawn.2),
            yaw: 0.0,
            pitch: 0.0,
            velocity: Vector3::zero(),
            on_ground: false,
            in_water: false,
        }
    }

    pub fn forward(&self) -> Vector3<f32> {
        Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    pub fn right(&self) -> Vector3<f32> {
        Vector3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    pub fn look_direction(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn eye_position(&self) -> Point3<f32> {
        Point3::new(self.position.x, self.position.y + 1.8, self.position.z)
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        let eye = self.eye_position();
        let target = eye + self.look_direction();
        Matrix4::look_at_rh(eye, target, Vector3::unit_y())
    }

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

    pub fn is_head_underwater(&self, world: &World) -> bool {
        let eye = self.eye_position();
        let block = world.get_block(
            eye.x.floor() as i32,
            eye.y.floor() as i32,
            eye.z.floor() as i32,
        );
        block == BlockType::Water
    }

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

        let mut move_dir = Vector3::zero();

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

        if move_dir.magnitude2() > 0.0 {
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
                        if check_intersection(Point3::new(x, y, z), bx, by, bz) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn intersects_block(&self, bx: i32, by: i32, bz: i32) -> bool {
        check_intersection(self.position, bx, by, bz)
    }

    pub fn raycast(&self, world: &World, max_dist: f32) -> Option<(i32, i32, i32, i32, i32, i32)> {
        let dir = self.look_direction();
        let eye = self.eye_position();
        let mut pos = Vector3::new(eye.x, eye.y, eye.z);
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

pub fn check_intersection(pos: Point3<f32>, bx: i32, by: i32, bz: i32) -> bool {
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
