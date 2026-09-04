use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::block::BlockType;
use crate::constants::*;
use crate::{Inventory, ItemRegistry, ItemStack, ItemState, PlayerSlot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedItemStack {
    /// Stable resource key, never a runtime `ItemId`.
    pub item: String,
    pub count: u16,
    pub durability: Option<u16>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SavedInventory {
    pub slots: Vec<Option<SavedItemStack>>,
    pub selected_hotbar: u8,
}

impl SavedInventory {
    pub fn from_inventory(inventory: &Inventory, registry: &ItemRegistry) -> Self {
        let slots = (0..36).map(|index| inventory.get_flat(index).map(|stack| SavedItemStack {
            item: registry.get(stack.item).key.to_owned(),
            count: stack.count,
            durability: stack.state.durability,
        })).collect();
        Self { slots, selected_hotbar: inventory.selected_hotbar }
    }

    pub fn into_inventory(self, registry: &ItemRegistry) -> Inventory {
        let mut inventory = Inventory::default();
        inventory.select_hotbar(self.selected_hotbar);
        for (index, saved) in self.slots.into_iter().enumerate().take(36) {
            let Some(saved) = saved else { continue };
            let Some(item) = registry.resolve(&saved.item) else { continue };
            let count = saved.count.min(registry.get(item).max_stack);
            if count == 0 { continue; }
            if let Some(slot) = PlayerSlot::from_flat(index) {
                inventory.set(slot, Some(ItemStack { item, count, state: ItemState { durability: saved.durability } }));
            }
        }
        inventory
    }
}

#[derive(Serialize, Deserialize)]
pub struct SavedChunk {
    pub cx: i32,
    pub cz: i32,
    pub subchunks: HashMap<u8, Vec<BlockType>>, // sy -> block data
}

#[derive(Serialize, Deserialize)]
pub struct SavedWorld {
    pub seed: u32,
    pub player_x: f32,
    pub player_y: f32,
    pub player_z: f32,
    pub player_yaw: f32,
    pub player_pitch: f32,
    #[serde(default)]
    pub inventory: SavedInventory,
    pub chunks: Vec<SavedChunk>,
}

impl SavedWorld {
    pub fn from_world<S: std::hash::BuildHasher>(
        chunks: &HashMap<(i32, i32), crate::chunk::Chunk, S>,
        seed: u32,
        player_pos: (f32, f32, f32),
        player_rot: (f32, f32),
        inventory: &Inventory,
    ) -> Self {
        let mut saved_chunks = Vec::new();

        for (&(cx, cz), chunk) in chunks.iter() {
            if !chunk.player_modified {
                continue;
            }

            let mut saved_subchunks = HashMap::new();
            for (sy, subchunk) in chunk.subchunks.iter().enumerate() {
                // Check if subchunk is actually modified or just empty
                if subchunk.is_empty {
                    continue;
                }

                let mut blocks = Vec::with_capacity(
                    CHUNK_SIZE as usize * SUBCHUNK_HEIGHT as usize * CHUNK_SIZE as usize,
                );
                for lx in 0..CHUNK_SIZE as usize {
                    for ly in 0..SUBCHUNK_HEIGHT as usize {
                        for lz in 0..CHUNK_SIZE as usize {
                            blocks.push(subchunk.get_block(lx as i32, ly as i32, lz as i32));
                        }
                    }
                }
                saved_subchunks.insert(sy as u8, blocks);
            }

            saved_chunks.push(SavedChunk {
                cx,
                cz,
                subchunks: saved_subchunks,
            });
        }

        SavedWorld {
            seed,
            player_x: player_pos.0,
            player_y: player_pos.1,
            player_z: player_pos.2,
            player_yaw: player_rot.0,
            player_pitch: player_rot.1,
            inventory: SavedInventory::from_inventory(inventory, crate::item_registry()),
            chunks: saved_chunks,
        }
    }
}

pub fn save_world<P: AsRef<Path>>(path: P, world: &SavedWorld) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Could not create file: {}", e))?;
    let encoded = postcard::to_stdvec(world)
        .map_err(|e| format!("Serialization error: {}", e))?;
    file.write_all(&encoded)
        .map_err(|e| format!("Could not write save file: {}", e))
}

pub fn load_world<P: AsRef<Path>>(path: P) -> Result<SavedWorld, String> {
    let mut file = File::open(path).map_err(|e| format!("Could not open file: {}", e))?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)
        .map_err(|e| format!("Could not read save file: {}", e))?;
    postcard::from_bytes(&encoded).map_err(|e| format!("Deserialization error: {}", e))
}

pub const WORLD_FILE_EXTENSION: &str = "minerust";
pub const DEFAULT_WORLD_FILE: &str = "world.minerust";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{item_registry, ItemStack};

    #[test]
    fn inventory_round_trip_uses_resource_keys() {
        let registry = item_registry();
        let stone = registry.resolve("minerust:stone").unwrap();
        let mut inventory = Inventory::default();
        inventory.select_hotbar(4);
        inventory.set(PlayerSlot::Hotbar(4), Some(ItemStack::new(stone, 23)));

        let saved = SavedInventory::from_inventory(&inventory, registry);
        assert_eq!(saved.slots[31].as_ref().unwrap().item, "minerust:stone");
        let restored = saved.into_inventory(registry);
        assert_eq!(restored.selected_hotbar, 4);
        assert_eq!(restored.selected_stack().unwrap().count, 23);
    }
}
