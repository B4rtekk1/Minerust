use std::collections::HashMap;

use once_cell::sync::Lazy;

use crate::BlockType;

/// Compact runtime identifier. Saves and network protocols use [`ItemDefinition::key`],
/// never this registration-order-dependent value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ItemState {
    /// Remaining durability for tools. `None` means that this stack has no
    /// per-instance state and can therefore stack with an identical item.
    pub durability: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemStack {
    pub item: ItemId,
    pub count: u16,
    pub state: ItemState,
}

impl ItemStack {
    pub fn new(item: ItemId, count: u16) -> Self {
        Self { item, count, state: ItemState::default() }
    }

    pub fn can_stack_with(&self, other: &Self) -> bool {
        self.item == other.item && self.state == other.state
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Block { block: BlockType },
    Material,
    Generic,
}

#[derive(Debug, Clone)]
pub struct ItemDefinition {
    pub id: ItemId,
    pub key: &'static str,
    pub display_name: &'static str,
    pub max_stack: u16,
    pub kind: ItemKind,
}

impl ItemDefinition {
    pub fn placeable_block(&self) -> Option<BlockType> {
        match self.kind { ItemKind::Block { block } => Some(block), _ => None }
    }
}

#[derive(Debug)]
pub struct ItemRegistry {
    definitions: Vec<ItemDefinition>,
    by_key: HashMap<&'static str, ItemId>,
    by_block: HashMap<BlockType, ItemId>,
}

impl ItemRegistry {
    fn new(entries: &[(&'static str, &'static str, u16, ItemKind)]) -> Self {
        let mut definitions = Vec::with_capacity(entries.len());
        let mut by_key = HashMap::with_capacity(entries.len());
        let mut by_block = HashMap::with_capacity(entries.len());
        for &(key, display_name, max_stack, kind) in entries {
            let id = ItemId(definitions.len() as u32);
            definitions.push(ItemDefinition { id, key, display_name, max_stack, kind });
            by_key.insert(key, id);
            if let ItemKind::Block { block } = kind { by_block.insert(block, id); }
        }
        Self { definitions, by_key, by_block }
    }

    pub fn get(&self, id: ItemId) -> &ItemDefinition {
        &self.definitions[id.0 as usize]
    }

    pub fn resolve(&self, key: &str) -> Option<ItemId> {
        self.by_key.get(key).copied()
    }

    pub fn item_for_block(&self, block: BlockType) -> Option<ItemId> { self.by_block.get(&block).copied() }
}

const BLOCK_ITEMS: &[(&str, &str, u16, ItemKind)] = &[
    ("minerust:grass", "Grass", 64, ItemKind::Block { block: BlockType::Grass }),
    ("minerust:dirt", "Dirt", 64, ItemKind::Block { block: BlockType::Dirt }),
    ("minerust:stone", "Stone", 64, ItemKind::Block { block: BlockType::Stone }),
    ("minerust:sand", "Sand", 64, ItemKind::Block { block: BlockType::Sand }),
    ("minerust:water", "Water", 1, ItemKind::Block { block: BlockType::Water }),
    ("minerust:wood", "Wood", 64, ItemKind::Block { block: BlockType::Wood }),
    ("minerust:leaves", "Leaves", 64, ItemKind::Block { block: BlockType::Leaves }),
    ("minerust:snow", "Snow", 64, ItemKind::Block { block: BlockType::Snow }),
    ("minerust:gravel", "Gravel", 64, ItemKind::Block { block: BlockType::Gravel }),
    ("minecraft:clay", "Clay", 64, ItemKind::Block { block: BlockType::Clay }),
    ("minecraft:ice", "Ice", 64, ItemKind::Block { block: BlockType::Ice }),
    ("minerust:cactus", "Cactus", 64, ItemKind::Block { block: BlockType::Cactus }),
    ("minerust:wood_stairs", "Wood Stairs", 64, ItemKind::Block { block: BlockType::WoodStairs }),
];

static REGISTRY: Lazy<ItemRegistry> = Lazy::new(|| ItemRegistry::new(BLOCK_ITEMS));

pub fn item_registry() -> &'static ItemRegistry { &REGISTRY }
