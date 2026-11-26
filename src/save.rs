use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::block::BlockType;
use crate::constants::*;
use crate::world::GeneratorVersion;
use crate::{Inventory, ItemRegistry, ItemStack, ItemState, PlayerSlot};

const SAVE_MAGIC: &[u8; 8] = b"MINERUST";

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
    #[serde(skip)]
    pub generator_version: GeneratorVersion,
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
        Self::from_world_with_generator_version(
            chunks,
            seed,
            player_pos,
            player_rot,
            inventory,
            GeneratorVersion::Current,
        )
    }

    pub fn from_world_with_generator_version<S: std::hash::BuildHasher>(
        chunks: &HashMap<(i32, i32), crate::chunk::Chunk, S>,
        seed: u32,
        player_pos: (f32, f32, f32),
        player_rot: (f32, f32),
        inventory: &Inventory,
        generator_version: GeneratorVersion,
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
            generator_version,
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
    let mut output = Vec::with_capacity(SAVE_MAGIC.len() + 1 + encoded.len());
    output.extend_from_slice(SAVE_MAGIC);
    output.push(world.generator_version as u8);
    output.extend_from_slice(&encoded);
    file.write_all(&output)
        .map_err(|e| format!("Could not write save file: {}", e))
}

pub fn load_world<P: AsRef<Path>>(path: P) -> Result<SavedWorld, String> {
    let mut file = File::open(path).map_err(|e| format!("Could not open file: {}", e))?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)
        .map_err(|e| format!("Could not read save file: {}", e))?;
    let (generator_version, payload) = if encoded.starts_with(SAVE_MAGIC) {
        if encoded.len() <= SAVE_MAGIC.len() {
            return Err("Save file header is truncated".to_owned());
        }
        let version = GeneratorVersion::from_u8(encoded[SAVE_MAGIC.len()])
            .ok_or_else(|| format!("Unsupported generator version {}", encoded[SAVE_MAGIC.len()]))?;
        (version, &encoded[SAVE_MAGIC.len() + 1..])
    } else {
        (GeneratorVersion::Legacy, encoded.as_slice())
    };
    let mut world: SavedWorld = postcard::from_bytes(payload)
        .map_err(|e| format!("Deserialization error: {}", e))?;
    world.generator_version = generator_version;
    Ok(world)
}

pub const WORLD_FILE_EXTENSION: &str = "minerust";
pub const DEFAULT_WORLD_FILE: &str = "world.minerust";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{item_registry, ItemStack};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temporary_save_path() -> std::path::PathBuf {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "minerust-save-test-{}-{}.minerust",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn sample_saved_world(generator_version: GeneratorVersion) -> SavedWorld {
        SavedWorld {
            seed: 42,
            generator_version,
            player_x: 1.0,
            player_y: 70.0,
            player_z: -3.0,
            player_yaw: 0.25,
            player_pitch: -0.1,
            inventory: SavedInventory::default(),
            chunks: Vec::new(),
        }
    }

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

    #[test]
    fn unversioned_save_loads_with_legacy_generator() {
        let path = temporary_save_path();
        let old_format = postcard::to_stdvec(&sample_saved_world(GeneratorVersion::Legacy)).unwrap();
        std::fs::write(&path, old_format).unwrap();

        let loaded = load_world(&path).unwrap();
        assert_eq!(loaded.seed, 42);
        assert_eq!(loaded.generator_version, GeneratorVersion::Legacy);
        save_world(&path, &loaded).unwrap();
        assert_eq!(
            load_world(&path).unwrap().generator_version,
            GeneratorVersion::Legacy
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn versioned_saves_round_trip_generator_version() {
        for version in [GeneratorVersion::Legacy, GeneratorVersion::Current] {
            let path = temporary_save_path();
            save_world(&path, &sample_saved_world(version)).unwrap();
            let loaded = load_world(&path).unwrap();
            assert_eq!(loaded.generator_version, version);
            assert_eq!(loaded.seed, 42);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn unknown_generator_version_is_rejected() {
        let path = temporary_save_path();
        let mut bytes = SAVE_MAGIC.to_vec();
        bytes.push(255);
        bytes.extend(postcard::to_stdvec(&sample_saved_world(GeneratorVersion::Current)).unwrap());
        std::fs::write(&path, bytes).unwrap();

        let error = load_world(&path).err().unwrap();
        assert!(error.contains("Unsupported generator version"));
        std::fs::remove_file(path).unwrap();
    }
}
