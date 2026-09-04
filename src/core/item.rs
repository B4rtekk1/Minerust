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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ItemKind {
    Block { block: BlockType },
    Tool(ToolItem),
    Food(FoodItem),
    Material,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolType { Pickaxe, Axe, Shovel }

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToolItem {
    pub tool_type: ToolType,
    pub mining_speed: f32,
    pub harvest_level: u8,
    pub attack_damage: f32,
    pub max_durability: u16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodItem { pub hunger: f32, pub saturation: f32 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LootEntry { pub item: ItemId, pub min: u16, pub max: u16, pub chance_percent: u8 }

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LootTable { pub entries: Vec<LootEntry> }

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

    pub fn max_durability(&self) -> Option<u16> {
        match self.kind { ItemKind::Tool(tool) => Some(tool.max_durability), _ => None }
    }
}

#[derive(Debug)]
pub struct ItemRegistry {
    definitions: Vec<ItemDefinition>,
    by_key: HashMap<&'static str, ItemId>,
    by_block: HashMap<BlockType, ItemId>,
    loot_by_block: HashMap<BlockType, LootTable>,
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
        let mut loot_by_block = HashMap::with_capacity(by_block.len());
        for (&block, &item) in &by_block {
            loot_by_block.insert(block, LootTable { entries: vec![LootEntry { item, min: 1, max: 1, chance_percent: 100 }] });
        }
        Self { definitions, by_key, by_block, loot_by_block }
    }

    pub fn get(&self, id: ItemId) -> &ItemDefinition {
        &self.definitions[id.0 as usize]
    }

    pub fn resolve(&self, key: &str) -> Option<ItemId> {
        self.by_key.get(key).copied()
    }

    pub fn item_for_block(&self, block: BlockType) -> Option<ItemId> { self.by_block.get(&block).copied() }

    pub fn new_stack(&self, item: ItemId, count: u16) -> ItemStack {
        let definition = self.get(item);
        ItemStack { item, count: count.min(definition.max_stack), state: ItemState { durability: definition.max_durability() } }
    }

    pub fn loot_for_block(&self, block: BlockType) -> Option<&LootTable> { self.loot_by_block.get(&block) }

    pub fn roll_block_loot(&self, block: BlockType) -> Vec<ItemStack> {
        self.loot_for_block(block).into_iter().flat_map(|table| &table.entries).filter_map(|entry| {
            (rand::random::<u8>() % 100 < entry.chance_percent).then(|| self.new_stack(entry.item, entry.min))
        }).collect()
    }
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
    ("minerust:iron_pickaxe", "Iron Pickaxe", 1, ItemKind::Tool(ToolItem { tool_type: ToolType::Pickaxe, mining_speed: 6.0, harvest_level: 2, attack_damage: 4.0, max_durability: 250 })),
    ("minerust:apple", "Apple", 16, ItemKind::Food(FoodItem { hunger: 4.0, saturation: 2.4 })),
];

static REGISTRY: Lazy<ItemRegistry> = Lazy::new(|| ItemRegistry::new(BLOCK_ITEMS));

pub fn item_registry() -> &'static ItemRegistry { &REGISTRY }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_stacks_get_per_instance_durability() {
        let registry = item_registry();
        let pickaxe = registry.resolve("minerust:iron_pickaxe").unwrap();
        let stack = registry.new_stack(pickaxe, 99);
        assert_eq!(stack.count, 1);
        assert_eq!(stack.state.durability, Some(250));
    }

    #[test]
    fn block_loot_is_defined_by_the_registry() {
        let drops = item_registry().roll_block_loot(BlockType::Stone);
        assert_eq!(drops.len(), 1);
        assert_eq!(item_registry().get(drops[0].item).key, "minerust:stone");
    }
}
