use crate::{ItemRegistry, ItemStack};

pub const MAIN_SLOT_COUNT: usize = 27;
pub const HOTBAR_SLOT_COUNT: usize = 9;
pub const INVENTORY_SLOT_COUNT: usize = MAIN_SLOT_COUNT + HOTBAR_SLOT_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerSlot { Main(u8), Hotbar(u8) }

impl PlayerSlot {
    pub fn from_flat(index: usize) -> Option<Self> {
        match index { 0..MAIN_SLOT_COUNT => Some(Self::Main(index as u8)), MAIN_SLOT_COUNT..INVENTORY_SLOT_COUNT => Some(Self::Hotbar((index - MAIN_SLOT_COUNT) as u8)), _ => None }
    }
}

/// Generic fixed-size storage. It owns no UI state.
#[derive(Debug, Clone)]
pub struct Container<const N: usize> { slots: [Option<ItemStack>; N], revision: u64 }
impl<const N: usize> Default for Container<N> { fn default() -> Self { Self { slots: std::array::from_fn(|_| None), revision: 0 } } }
impl<const N: usize> Container<N> {
    pub fn get(&self, index: usize) -> Option<&ItemStack> { self.slots.get(index)?.as_ref() }
    pub fn take(&mut self, index: usize) -> Option<ItemStack> { let stack = self.slots.get_mut(index)?.take(); if stack.is_some() { self.revision += 1; } stack }
    pub fn set(&mut self, index: usize, stack: Option<ItemStack>) -> bool { let Some(slot) = self.slots.get_mut(index) else { return false }; *slot = stack; self.revision += 1; true }
    pub fn revision(&self) -> u64 { self.revision }
}

/// Physical player-owned storage. This is the only gameplay source of truth.
#[derive(Debug, Clone, Default)]
pub struct PlayerInventory {
    pub main: Container<MAIN_SLOT_COUNT>,
    pub hotbar: Container<HOTBAR_SLOT_COUNT>,
    pub selected_hotbar: u8,
}

/// UI-only state; it can later be shared by player, chest and crafting views.
#[derive(Debug, Clone, Default)]
pub struct InventoryUiState { pub cursor_stack: Option<ItemStack> }

/// Transitional public name retained while callers migrate to PlayerInventory.
pub type Inventory = PlayerInventory;

impl PlayerInventory {
    pub fn selected_stack(&self) -> Option<&ItemStack> { self.hotbar.get(self.selected_hotbar as usize) }
    pub fn select_hotbar(&mut self, slot: u8) { self.selected_hotbar = slot.min((HOTBAR_SLOT_COUNT - 1) as u8); }
    pub fn get(&self, slot: PlayerSlot) -> Option<&ItemStack> { match slot { PlayerSlot::Main(i) => self.main.get(i as usize), PlayerSlot::Hotbar(i) => self.hotbar.get(i as usize) } }
    pub fn get_flat(&self, index: usize) -> Option<&ItemStack> { PlayerSlot::from_flat(index).and_then(|slot| self.get(slot)) }
    pub fn take(&mut self, slot: PlayerSlot) -> Option<ItemStack> { match slot { PlayerSlot::Main(i) => self.main.take(i as usize), PlayerSlot::Hotbar(i) => self.hotbar.take(i as usize) } }
    pub fn set(&mut self, slot: PlayerSlot, stack: Option<ItemStack>) -> bool { match slot { PlayerSlot::Main(i) => self.main.set(i as usize, stack), PlayerSlot::Hotbar(i) => self.hotbar.set(i as usize, stack) } }
    pub fn revision(&self) -> u64 { self.main.revision().wrapping_add(self.hotbar.revision()) }

    /// Inserts into compatible stacks first, then empty slots; overflow is returned.
    pub fn insert(&mut self, stack: ItemStack, registry: &ItemRegistry) -> Option<ItemStack> {
        self.insert_into(stack, registry, (0..INVENTORY_SLOT_COUNT).filter_map(PlayerSlot::from_flat))
    }

    pub fn quick_move(&mut self, source: PlayerSlot, registry: &ItemRegistry) -> bool {
        let Some(stack) = self.take(source) else { return false };
        let targets = match source {
            PlayerSlot::Main(_) => (0..HOTBAR_SLOT_COUNT).map(|i| PlayerSlot::Hotbar(i as u8)).collect::<Vec<_>>(),
            PlayerSlot::Hotbar(_) => (0..MAIN_SLOT_COUNT).map(|i| PlayerSlot::Main(i as u8)).collect::<Vec<_>>(),
        };
        let remainder = self.insert_into(stack, registry, targets.into_iter());
        if let Some(stack) = remainder { self.set(source, Some(stack)); }
        true
    }

    pub fn consume_selected(&mut self, count: u16) -> bool {
        let slot = PlayerSlot::Hotbar(self.selected_hotbar);
        let Some(mut stack) = self.take(slot) else { return false };
        if stack.count < count { self.set(slot, Some(stack)); return false; }
        stack.count -= count;
        if stack.count > 0 { self.set(slot, Some(stack)); }
        true
    }

    fn insert_into<I>(&mut self, mut moving: ItemStack, registry: &ItemRegistry, targets: I) -> Option<ItemStack>
    where I: Iterator<Item = PlayerSlot> {
        let targets: Vec<_> = targets.collect();
        // Pass 1: merge compatible existing stacks before considering empties.
        for slot in &targets {
            let Some(destination) = self.get(*slot).cloned() else { continue };
            if !destination.can_stack_with(&moving) { continue; }
            let room = registry.get(destination.item).max_stack.saturating_sub(destination.count);
            let added = room.min(moving.count);
            if added == 0 { continue; }
            let mut updated = destination;
            updated.count += added;
            moving.count -= added;
            self.set(*slot, Some(updated));
            if moving.count == 0 { return None; }
        }
        // Pass 2: create new stacks in empty slots.
        for slot in targets {
            if self.get(slot).is_some() { continue; }
            let added = registry.get(moving.item).max_stack.min(moving.count);
            self.set(slot, Some(ItemStack { count: added, ..moving.clone() }));
            moving.count -= added;
            if moving.count == 0 { return None; }
        }
        Some(moving)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item_registry;
    fn stone(count: u16) -> ItemStack { ItemStack::new(item_registry().resolve("minerust:stone").unwrap(), count) }

    #[test]
    fn insert_merges_before_empty_slots() {
        let mut inventory = PlayerInventory::default();
        inventory.set(PlayerSlot::Main(1), Some(stone(60)));
        assert_eq!(inventory.insert(stone(8), item_registry()), None);
        assert_eq!(inventory.get(PlayerSlot::Main(1)).unwrap().count, 64);
        assert_eq!(inventory.get(PlayerSlot::Main(0)).unwrap().count, 4);
    }

    #[test]
    fn full_inventory_returns_remainder() {
        let mut inventory = PlayerInventory::default();
        for i in 0..MAIN_SLOT_COUNT { inventory.set(PlayerSlot::Main(i as u8), Some(stone(64))); }
        for i in 0..HOTBAR_SLOT_COUNT { inventory.set(PlayerSlot::Hotbar(i as u8), Some(stone(64))); }
        assert_eq!(inventory.insert(stone(3), item_registry()).unwrap().count, 3);
    }
}
