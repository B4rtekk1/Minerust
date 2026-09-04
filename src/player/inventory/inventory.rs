use crate::{ItemKind, ItemRegistry, ItemStack};

pub const MAIN_SLOT_COUNT: usize = 27;
pub const HOTBAR_SLOT_COUNT: usize = 9;
pub const INVENTORY_SLOT_COUNT: usize = MAIN_SLOT_COUNT + HOTBAR_SLOT_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerSlot {
    Main(u8),
    Hotbar(u8),
}

impl PlayerSlot {
    pub fn from_flat(index: usize) -> Option<Self> {
        match index {
            0..MAIN_SLOT_COUNT => Some(Self::Main(index as u8)),
            MAIN_SLOT_COUNT..INVENTORY_SLOT_COUNT => {
                Some(Self::Hotbar((index - MAIN_SLOT_COUNT) as u8))
            }
            _ => None,
        }
    }
}

/// Generic fixed-size storage. It owns no UI state.
#[derive(Debug, Clone)]
pub struct Container<const N: usize> {
    slots: [Option<ItemStack>; N],
    rules: [SlotRule; N],
    revision: u64,
}
impl<const N: usize> Default for Container<N> {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            rules: [SlotRule::Any; N],
            revision: 0,
        }
    }
}
impl<const N: usize> Container<N> {
    pub fn with_rules(rules: [SlotRule; N]) -> Self { Self { slots: std::array::from_fn(|_| None), rules, revision: 0 } }
    pub fn rule(&self, index: usize) -> Option<SlotRule> { self.rules.get(index).copied() }
    pub fn accepts(&self, index: usize, stack: &ItemStack, registry: &ItemRegistry) -> bool { self.rule(index).is_some_and(|rule| rule.accepts(stack, registry)) }
    pub fn get(&self, index: usize) -> Option<&ItemStack> {
        self.slots.get(index)?.as_ref()
    }
    pub fn take(&mut self, index: usize) -> Option<ItemStack> {
        let stack = self.slots.get_mut(index)?.take();
        if stack.is_some() {
            self.revision += 1;
        }
        stack
    }
    pub fn set(&mut self, index: usize, stack: Option<ItemStack>) -> bool {
        let Some(slot) = self.slots.get_mut(index) else {
            return false;
        };
        *slot = stack;
        self.revision += 1;
        true
    }
    pub fn set_checked(&mut self, index: usize, stack: Option<ItemStack>, registry: &ItemRegistry) -> bool {
        if stack.as_ref().is_some_and(|stack| !self.accepts(index, stack, registry)) { return false; }
        self.set(index, stack)
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
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
pub struct InventoryUiState {
    pub cursor_stack: Option<ItemStack>,
    /// Original slot of the cursor stack, used to restore it predictably when
    /// an inventory screen closes.
    pub cursor_origin: Option<PlayerSlot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryAction {
    LeftClick(PlayerSlot),
    RightClick(PlayerSlot),
    QuickMove(PlayerSlot),
    DropOne(PlayerSlot),
    DropStack(PlayerSlot),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct InventoryTransactionResult {
    pub changed: bool,
    pub dropped: Option<ItemStack>,
}

/// Per-slot admission rule for containers such as furnaces, armor and output slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRule { Any, OutputOnly, Fuel, Smeltable, Tool }

impl SlotRule {
    pub fn accepts(self, stack: &ItemStack, registry: &ItemRegistry) -> bool {
        match self {
            Self::Any => true,
            Self::OutputOnly => false,
            Self::Tool => matches!(registry.get(stack.item).kind, ItemKind::Tool(_)),
            Self::Fuel | Self::Smeltable => false,
        }
    }
}

/// Constrained storage layout used by a future furnace UI and simulation.
#[derive(Debug, Clone)]
pub struct FurnaceInventory {
    pub input: Container<1>,
    pub fuel: Container<1>,
    pub output: Container<1>,
}

impl Default for FurnaceInventory {
    fn default() -> Self {
        Self {
            input: Container::with_rules([SlotRule::Smeltable]),
            fuel: Container::with_rules([SlotRule::Fuel]),
            output: Container::with_rules([SlotRule::OutputOnly]),
        }
    }
}

/// Transitional public name retained while callers migrate to PlayerInventory.
pub type Inventory = PlayerInventory;

impl PlayerInventory {
    pub fn selected_stack(&self) -> Option<&ItemStack> {
        self.hotbar.get(self.selected_hotbar as usize)
    }
    pub fn select_hotbar(&mut self, slot: u8) {
        self.selected_hotbar = slot.min((HOTBAR_SLOT_COUNT - 1) as u8);
    }
    pub fn get(&self, slot: PlayerSlot) -> Option<&ItemStack> {
        match slot {
            PlayerSlot::Main(i) => self.main.get(i as usize),
            PlayerSlot::Hotbar(i) => self.hotbar.get(i as usize),
        }
    }
    pub fn get_flat(&self, index: usize) -> Option<&ItemStack> {
        PlayerSlot::from_flat(index).and_then(|slot| self.get(slot))
    }
    pub fn take(&mut self, slot: PlayerSlot) -> Option<ItemStack> {
        match slot {
            PlayerSlot::Main(i) => self.main.take(i as usize),
            PlayerSlot::Hotbar(i) => self.hotbar.take(i as usize),
        }
    }
    pub fn set(&mut self, slot: PlayerSlot, stack: Option<ItemStack>) -> bool {
        match slot {
            PlayerSlot::Main(i) => self.main.set(i as usize, stack),
            PlayerSlot::Hotbar(i) => self.hotbar.set(i as usize, stack),
        }
    }
    pub fn revision(&self) -> u64 {
        self.main.revision().wrapping_add(self.hotbar.revision())
    }

    /// Inserts into compatible stacks first, then empty slots; overflow is returned.
    pub fn insert(&mut self, stack: ItemStack, registry: &ItemRegistry) -> Option<ItemStack> {
        self.insert_into(
            stack,
            registry,
            (0..INVENTORY_SLOT_COUNT).filter_map(PlayerSlot::from_flat),
        )
    }

    /// Inserts a stack into the hotbar before using the main inventory.
    ///
    /// This is used for freshly mined block drops so they become immediately
    /// available for placement without disturbing the normal inventory sort
    /// order used by UI transfers and pickups.
    pub fn insert_hotbar_first(
        &mut self,
        stack: ItemStack,
        registry: &ItemRegistry,
    ) -> Option<ItemStack> {
        self.insert_into(
            stack,
            registry,
            (0..HOTBAR_SLOT_COUNT)
                .map(|i| PlayerSlot::Hotbar(i as u8))
                .chain((0..MAIN_SLOT_COUNT).map(|i| PlayerSlot::Main(i as u8))),
        )
    }

    pub fn quick_move(&mut self, source: PlayerSlot, registry: &ItemRegistry) -> bool {
        let Some(stack) = self.take(source) else {
            return false;
        };
        let targets = match source {
            PlayerSlot::Main(_) => (0..HOTBAR_SLOT_COUNT)
                .map(|i| PlayerSlot::Hotbar(i as u8))
                .collect::<Vec<_>>(),
            PlayerSlot::Hotbar(_) => (0..MAIN_SLOT_COUNT)
                .map(|i| PlayerSlot::Main(i as u8))
                .collect::<Vec<_>>(),
        };
        let remainder = self.insert_into(stack, registry, targets.into_iter());
        if let Some(stack) = remainder {
            self.set(source, Some(stack));
        }
        true
    }

    /// Returns a cursor stack to its original slot when possible, then yields
    /// the remainder for normal inventory insertion or a world drop.
    pub fn return_to_slot(&mut self, slot: PlayerSlot, mut stack: ItemStack, registry: &ItemRegistry) -> Option<ItemStack> {
        match self.get(slot).cloned() {
            None => { self.set(slot, Some(stack)); None }
            Some(mut destination) if destination.can_stack_with(&stack) => {
                let moved = registry.get(destination.item).max_stack.saturating_sub(destination.count).min(stack.count);
                destination.count += moved;
                stack.count -= moved;
                self.set(slot, Some(destination));
                (stack.count > 0).then_some(stack)
            }
            Some(_) => Some(stack),
        }
    }

    /// Applies UI semantics without exposing container storage to input code.
    pub fn apply_action(
        &mut self,
        ui: &mut InventoryUiState,
        action: InventoryAction,
        registry: &ItemRegistry,
    ) -> InventoryTransactionResult {
        match action {
            InventoryAction::DropOne(slot) => {
                let Some(mut stack) = self.take(slot) else { return InventoryTransactionResult::default() };
                let dropped = ItemStack { count: 1, ..stack.clone() };
                stack.count -= 1;
                if stack.count > 0 { self.set(slot, Some(stack)); }
                InventoryTransactionResult { changed: true, dropped: Some(dropped) }
            }
            InventoryAction::DropStack(slot) => {
                let dropped = self.take(slot);
                InventoryTransactionResult { changed: dropped.is_some(), dropped }
            }
            InventoryAction::QuickMove(slot) => InventoryTransactionResult {
                changed: self.quick_move(slot, registry),
                dropped: None,
            },
            InventoryAction::LeftClick(slot) => {
                match ui.cursor_stack.take() {
                    None => {
                        ui.cursor_stack = self.take(slot);
                        ui.cursor_origin = ui.cursor_stack.as_ref().map(|_| slot);
                    }
                    Some(mut cursor) => match self.get(slot).cloned() {
                        None => {
                            self.set(slot, Some(cursor));
                            ui.cursor_origin = None;
                        }
                        Some(mut target) if target.can_stack_with(&cursor) => {
                            let moved = registry
                                .get(target.item)
                                .max_stack
                                .saturating_sub(target.count)
                                .min(cursor.count);
                            target.count += moved;
                            cursor.count -= moved;
                            self.set(slot, Some(target));
                            if cursor.count > 0 {
                                ui.cursor_stack = Some(cursor);
                            } else { ui.cursor_origin = None; }
                        }
                        Some(target) => {
                            self.set(slot, Some(cursor));
                            ui.cursor_stack = Some(target);
                            ui.cursor_origin = Some(slot);
                        }
                    },
                }
                InventoryTransactionResult { changed: true, dropped: None }
            }
            InventoryAction::RightClick(slot) => {
                if ui.cursor_stack.is_none() {
                    let Some(mut stack) = self.take(slot) else {
                        return InventoryTransactionResult::default();
                    };
                    let amount = stack.count.div_ceil(2);
                    stack.count -= amount; 
                    if stack.count > 0 {
                        self.set(slot, Some(stack.clone()));
                    }
                    ui.cursor_stack = Some(ItemStack {
                        count: amount,
                        ..stack
                    });
                    ui.cursor_origin = Some(slot);
                    return InventoryTransactionResult { changed: true,  dropped: None };
                }
                let mut cursor = ui.cursor_stack.take().expect("cursor checked above");
                match self.get(slot).cloned() {
                    None => {
                        self.set(
                            slot,
                            Some(ItemStack {
                                count: 1,
                                ..cursor.clone()
                            }),
                        );
                        cursor.count -= 1;
                    }
                    Some(mut target)
                        if target.can_stack_with(&cursor)
                            && target.count < registry.get(target.item).max_stack =>
                    {
                        target.count += 1;
                        cursor.count -= 1;
                        self.set(slot, Some(target));
                    }
                    _ => {}
                }
                if cursor.count > 0 {
                    ui.cursor_stack = Some(cursor);
                } else { ui.cursor_origin = None; }
                InventoryTransactionResult { changed: true, dropped: None }
            }
        }
    }

    pub fn consume_selected(&mut self, count: u16) -> bool {
        let slot = PlayerSlot::Hotbar(self.selected_hotbar);
        let Some(mut stack) = self.take(slot) else {
            return false;
        };
        if stack.count < count {
            self.set(slot, Some(stack));
            return false;
        }
        stack.count -= count;
        if stack.count > 0 {
            self.set(slot, Some(stack));
        }
        true
    }

    /// Damages the selected tool once. A tool at zero durability disappears.
    pub fn damage_selected_tool(&mut self, registry: &ItemRegistry) -> bool {
        let slot = PlayerSlot::Hotbar(self.selected_hotbar);
        let Some(mut stack) = self.take(slot) else { return false };
        if !matches!(registry.get(stack.item).kind, ItemKind::Tool(_)) { self.set(slot, Some(stack)); return false; }
        let Some(durability) = stack.state.durability.as_mut() else { self.set(slot, Some(stack)); return false; };
        *durability = durability.saturating_sub(1);
        if *durability > 0 { self.set(slot, Some(stack)); }
        true
    }

    fn insert_into<I>(
        &mut self,
        mut moving: ItemStack,
        registry: &ItemRegistry,
        targets: I,
    ) -> Option<ItemStack>
    where
        I: Iterator<Item = PlayerSlot>,
    {
        let targets: Vec<_> = targets.collect();
        // Pass 1: merge compatible existing stacks before considering empties.
        for slot in &targets {
            let Some(destination) = self.get(*slot).cloned() else {
                continue;
            };
            if !destination.can_stack_with(&moving) {
                continue;
            }
            let room = registry
                .get(destination.item)
                .max_stack
                .saturating_sub(destination.count);
            let added = room.min(moving.count);
            if added == 0 {
                continue;
            }
            let mut updated = destination;
            updated.count += added;
            moving.count -= added;
            self.set(*slot, Some(updated));
            if moving.count == 0 {
                return None;
            }
        }
        // Pass 2: create new stacks in empty slots.
        for slot in targets {
            if self.get(slot).is_some() {
                continue;
            }
            let added = registry.get(moving.item).max_stack.min(moving.count);
            self.set(
                slot,
                Some(ItemStack {
                    count: added,
                    ..moving.clone()
                }),
            );
            moving.count -= added;
            if moving.count == 0 {
                return None;
            }
        }
        Some(moving)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item_registry;
    fn stone(count: u16) -> ItemStack {
        ItemStack::new(item_registry().resolve("minerust:stone").unwrap(), count)
    }

    #[test]
    fn insert_merges_before_empty_slots() {
        let mut inventory = PlayerInventory::default();
        inventory.set(PlayerSlot::Main(1), Some(stone(60)));
        assert_eq!(inventory.insert(stone(8), item_registry()), None);
        assert_eq!(inventory.get(PlayerSlot::Main(1)).unwrap().count, 64);
        assert_eq!(inventory.get(PlayerSlot::Main(0)).unwrap().count, 4);
    }

    #[test]
    fn insert_hotbar_first_prefers_hotbar_over_empty_main_slot() {
        let mut inventory = PlayerInventory::default();
        assert_eq!(inventory.insert_hotbar_first(stone(1), item_registry()), None);
        assert_eq!(inventory.get(PlayerSlot::Hotbar(0)).unwrap().count, 1);
        assert!(inventory.get(PlayerSlot::Main(0)).is_none());
    }

    #[test]
    fn full_inventory_returns_remainder() {
        let mut inventory = PlayerInventory::default();
        for i in 0..MAIN_SLOT_COUNT {
            inventory.set(PlayerSlot::Main(i as u8), Some(stone(64)));
        }
        for i in 0..HOTBAR_SLOT_COUNT {
            inventory.set(PlayerSlot::Hotbar(i as u8), Some(stone(64)));
        }
        assert_eq!(
            inventory.insert(stone(3), item_registry()).unwrap().count,
            3
        );
    }

    #[test]
    fn quick_move_merges_before_using_an_empty_hotbar_slot() {
        let mut inventory = PlayerInventory::default();
        inventory.set(PlayerSlot::Main(0), Some(stone(20)));
        inventory.set(PlayerSlot::Hotbar(1), Some(stone(50)));
        inventory.apply_action(
            &mut InventoryUiState::default(),
            InventoryAction::QuickMove(PlayerSlot::Main(0)),
            item_registry(),
        );
        assert_eq!(inventory.get(PlayerSlot::Hotbar(1)).unwrap().count, 64);
        assert_eq!(inventory.get(PlayerSlot::Hotbar(0)).unwrap().count, 6);
    }

    #[test]
    fn tool_slot_rule_rejects_blocks_and_accepts_tools() {
        let registry = item_registry();
        let stone_stack = stone(1);
        let tool_stack = registry.new_stack(registry.resolve("minerust:iron_pickaxe").unwrap(), 1);
        assert!(!SlotRule::Tool.accepts(&stone_stack, registry));
        assert!(SlotRule::Tool.accepts(&tool_stack, registry));
    }

    #[test]
    fn constrained_container_refuses_invalid_stack() {
        let mut container = Container::<1>::with_rules([SlotRule::Tool]);
        assert!(!container.set_checked(0, Some(stone(1)), item_registry()));
        let pickaxe = item_registry().new_stack(item_registry().resolve("minerust:iron_pickaxe").unwrap(), 1);
        assert!(container.set_checked(0, Some(pickaxe), item_registry()));
    }

    #[test]
    fn drop_one_keeps_the_remainder_in_the_slot() {
        let mut inventory = PlayerInventory::default();
        inventory.set(PlayerSlot::Hotbar(0), Some(stone(3)));
        let result = inventory.apply_action(&mut InventoryUiState::default(), InventoryAction::DropOne(PlayerSlot::Hotbar(0)), item_registry());
        assert_eq!(result.dropped.unwrap().count, 1);
        assert_eq!(inventory.get(PlayerSlot::Hotbar(0)).unwrap().count, 2);
    }

    #[test]
    fn selected_tool_loses_durability() {
        let registry = item_registry();
        let mut inventory = PlayerInventory::default();
        let pickaxe = registry.new_stack(registry.resolve("minerust:iron_pickaxe").unwrap(), 1);
        inventory.set(PlayerSlot::Hotbar(0), Some(pickaxe));
        assert!(inventory.damage_selected_tool(registry));
        assert_eq!(inventory.selected_stack().unwrap().state.durability, Some(249));
    }

    #[test]
    fn cursor_recovery_prefers_its_origin_slot() {
        let mut inventory = PlayerInventory::default();
        inventory.set(PlayerSlot::Main(2), Some(stone(12)));
        let mut ui = InventoryUiState::default();
        inventory.apply_action(&mut ui, InventoryAction::LeftClick(PlayerSlot::Main(2)), item_registry());
        let remainder = inventory.return_to_slot(ui.cursor_origin.unwrap(), ui.cursor_stack.take().unwrap(), item_registry());
        assert!(remainder.is_none());
        assert_eq!(inventory.get(PlayerSlot::Main(2)).unwrap().count, 12);
    }
}
