use crate::{BlockType, ItemId, ItemStack, item_registry, World};
use glam::Vec3;

pub type EntityId = u64;

/// A physical item stack in the world, kept independent from its renderer.
#[derive(Debug, Clone)]
pub struct ItemEntity {
    pub id: EntityId,
    pub stack: ItemStack,
    pub position: Vec3,
    pub velocity: Vec3,
    pub pickup_delay: f32,
    pub lifetime: f32,
}

impl ItemEntity {
    pub const RADIUS: f32 = 0.125;
    pub const GRAVITY: f32 = 20.0;
    /// Horizontal drag prevents a spawn impulse from making an item slide
    /// forever over flat terrain.
    pub const HORIZONTAL_DRAG: f32 = 8.0;

    pub fn new(id: EntityId, stack: ItemStack, position: Vec3) -> Self {
        Self {
            id,
            stack,
            position,
            velocity: Vec3::new(0.0, 2.0, 0.0),
            pickup_delay: 0.25,
            lifetime: 300.0,
        }
    }

    /// Advances gravity and terrain collision; `false` means the item expired.
    pub fn update(&mut self, world: &World, dt: f32) -> bool {
        self.pickup_delay = (self.pickup_delay - dt).max(0.0);
        self.lifetime -= dt;
        if self.lifetime <= 0.0 {
            return false;
        }
        self.velocity.y -= Self::GRAVITY * dt;
        let drag = (-Self::HORIZONTAL_DRAG * dt).exp();
        self.velocity.x *= drag;
        self.velocity.z *= drag;
        if self.velocity.x.abs() < 0.01 {
            self.velocity.x = 0.0;
        }
        if self.velocity.z.abs() < 0.01 {
            self.velocity.z = 0.0;
        }
        self.move_axis(world, Vec3::X, self.velocity.x * dt);
        self.move_axis(world, Vec3::Y, self.velocity.y * dt);
        self.move_axis(world, Vec3::Z, self.velocity.z * dt);
        true
    }

    fn move_axis(&mut self, world: &World, axis: Vec3, distance: f32) {
        if distance == 0.0 {
            return;
        }
        let next = self.position + axis * distance;
        if collides(world, next, Self::RADIUS) {
            if axis.y != 0.0 {
                self.velocity.y = 0.0;
            } else if axis.x != 0.0 {
                self.velocity.x = 0.0;
            } else {
                self.velocity.z = 0.0;
            }
        } else {
            self.position = next;
        }
    }
}

fn collides(world: &World, position: Vec3, radius: f32) -> bool {
    let min = position - Vec3::splat(radius);
    let max = position + Vec3::splat(radius);
    for x in min.x.floor() as i32..=max.x.floor() as i32 {
        for y in min.y.floor() as i32..=max.y.floor() as i32 {
            for z in min.z.floor() as i32..=max.z.floor() as i32 {
                if world.get_block(x, y, z).is_solid() {
                    return true;
                }
            }
        }
    }
    false
}

/// Maps a broken block to a registered pickup item. Bedrock never drops.
pub fn drop_for_block(block: BlockType) -> Option<ItemId> {
    let key = match block {
        BlockType::Air | BlockType::Bedrock | BlockType::DeadBush => return None,
        BlockType::Grass => "minerust:grass", BlockType::Dirt => "minerust:dirt",
        BlockType::Stone => "minerust:stone", BlockType::Sand => "minerust:sand",
        BlockType::Water => "minerust:water", BlockType::Wood | BlockType::WoodLogX | BlockType::WoodLogZ => "minerust:wood",
        BlockType::Leaves => "minerust:leaves", BlockType::Snow => "minerust:snow",
        BlockType::Gravel => "minerust:gravel", BlockType::Clay => "minecraft:clay",
        BlockType::Ice => "minecraft:ice", BlockType::Cactus => "minerust:cactus",
        BlockType::WoodStairs => "minerust:wood_stairs",
    };
    item_registry().resolve(key)
}

/// Returns the block appearance used to render a registered block item.
pub fn block_for_item(item_id: ItemId) -> Option<BlockType> {
    item_registry().get(item_id).placeable_block()
}
