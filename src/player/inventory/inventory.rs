use crate::GameItem;

pub const MAIN_SLOT_COUNT: usize = 27;
pub const HOTBAR_SLOT_COUNT: usize = 9;
pub const INVENTORY_SLOT_COUNT: usize = MAIN_SLOT_COUNT + HOTBAR_SLOT_COUNT;
pub const MAX_STACK_SIZE: u32 = 64;

/// A stack stored in an inventory slot or carried by the cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryItem {
    pub item: GameItem,
    pub item_name: String,
    pub quantity: u32,
}

impl InventoryItem {
    pub fn new(item: GameItem, quantity: u32) -> Self {
        Self {
            item_name: item.name.to_owned(),
            item,
            quantity,
        }
    }

    pub fn item_id(&self) -> &str {
        self.item.id
    }

    pub fn max_quantity(&self) -> u32 {
        self.item.max_stack_size.clamp(1, MAX_STACK_SIZE)
    }
}

/// Player storage: 27 main slots, followed by the nine hotbar slots.
#[derive(Debug, Clone)]
pub struct Inventory {
    pub slots: [Option<InventoryItem>; INVENTORY_SLOT_COUNT],
    /// Item currently attached to the inventory UI cursor.
    pub cursor_item: Option<InventoryItem>,
    /// Index in the hotbar range `0..9`, not an absolute slot index.
    pub selected_slot: usize,
}

impl Default for Inventory {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            cursor_item: None,
            selected_slot: 0,
        }
    }
}

impl Inventory {
    pub const MAIN_SLOTS: std::ops::Range<usize> = 0..MAIN_SLOT_COUNT;
    pub const HOTBAR_SLOTS: std::ops::Range<usize> = MAIN_SLOT_COUNT..INVENTORY_SLOT_COUNT;

    /// Adds as much of a stack as possible and returns the quantity accepted.
    pub fn add_item(&mut self, item: GameItem, quantity: u32) -> u32 {
        let mut remaining = quantity;
        if remaining == 0 {
            return 0;
        }
        let max = if item.stackable {
            item.max_stack_size.clamp(1, MAX_STACK_SIZE)
        } else {
            1
        };
        if item.stackable {
            for slot in self.slots.iter_mut().flatten() {
                if slot.item.id == item.id && slot.quantity < max {
                    let added = remaining.min(max - slot.quantity);
                    slot.quantity += added;
                    remaining -= added;
                    if remaining == 0 {
                        return quantity;
                    }
                }
            }
        }
        for slot in &mut self.slots {
            if slot.is_none() {
                let added = remaining.min(max);
                *slot = Some(InventoryItem::new(item.clone(), added));
                remaining -= added;
                if remaining == 0 {
                    break;
                }
            }
        }
        quantity - remaining
    }

    pub fn can_add(&self, item: &GameItem, quantity: u32) -> bool {
        let mut copy = self.clone();
        copy.add_item(item.clone(), quantity) == quantity
    }

    /// Removes up to `quantity` matching items and returns the quantity removed.
    pub fn remove_item(&mut self, item_id: &str, quantity: u32) -> u32 {
        let mut remaining = quantity;
        for slot in &mut self.slots {
            let Some(stack) = slot else {
                continue;
            };
            if stack.item.id != item_id {
                continue;
            }
            let removed = remaining.min(stack.quantity);
            stack.quantity -= removed;
            remaining -= removed;
            if stack.quantity == 0 {
                *slot = None;
            }
            if remaining == 0 {
                break;
            }
        }
        quantity - remaining
    }

    /// Moves `quantity` from a slot to the cursor (or half the stack if omitted).
    pub fn split_stack(&mut self, source: usize, quantity: Option<u32>) -> bool {
        if source >= INVENTORY_SLOT_COUNT || self.cursor_item.is_some() {
            return false;
        }
        let Some(stack) = self.slots[source].as_mut() else {
            return false;
        };
        let amount = quantity.unwrap_or(stack.quantity.div_ceil(2));
        if amount == 0 || amount >= stack.quantity {
            return false;
        }
        stack.quantity -= amount;
        self.cursor_item = Some(InventoryItem::new(stack.item.clone(), amount));
        true
    }

    /// Moves a stack to another slot, merging compatible stacks first.
    pub fn move_stack(&mut self, from: usize, to: usize) -> bool {
        if from >= INVENTORY_SLOT_COUNT || to >= INVENTORY_SLOT_COUNT || from == to {
            return false;
        }
        let Some(mut moving) = self.slots[from].take() else {
            return false;
        };
        match self.slots[to].as_mut() {
            None => {
                self.slots[to] = Some(moving);
                true
            }
            Some(destination) if destination.item.id == moving.item.id && moving.item.stackable => {
                let room = moving.max_quantity().saturating_sub(destination.quantity);
                let added = room.min(moving.quantity);
                destination.quantity += added;
                moving.quantity -= added;
                if moving.quantity > 0 {
                    self.slots[from] = Some(moving);
                }
                added > 0
            }
            Some(_) => {
                self.slots[from] = Some(moving);
                false
            }
        }
    }

    pub fn swap_slots(&mut self, left: usize, right: usize) -> bool {
        if left >= INVENTORY_SLOT_COUNT || right >= INVENTORY_SLOT_COUNT {
            return false;
        }
        self.slots.swap(left, right);
        true
    }

    /// Moves a stack between main inventory and hotbar, like shift-click.
    pub fn shift_click(&mut self, source: usize) -> bool {
        if source >= INVENTORY_SLOT_COUNT || self.slots[source].is_none() {
            return false;
        }
        let targets = if source < MAIN_SLOT_COUNT {
            Self::HOTBAR_SLOTS
        } else {
            Self::MAIN_SLOTS
        };
        let mut moved = false;
        for target in targets.clone() {
            let transferred = self.move_stack(source, target);
            moved |= transferred;
            if transferred && self.slots[source].is_none() {
                return true;
            }
        }
        for target in targets {
            if self.slots[target].is_none() {
                return self.move_stack(source, target);
            }
        }
        moved
    }

    /// Removes a stack (or a requested part of it) for spawning as a world drop.
    pub fn drop_item(&mut self, slot: usize, quantity: Option<u32>) -> Option<InventoryItem> {
        let stack = self.slots.get_mut(slot)?.as_mut()?;
        let amount = quantity.unwrap_or(stack.quantity).min(stack.quantity);
        if amount == 0 {
            return None;
        }
        let dropped = InventoryItem::new(stack.item.clone(), amount);
        stack.quantity -= amount;
        if stack.quantity == 0 {
            self.slots[slot] = None;
        }
        Some(dropped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::game_item::get_item;

    fn stone() -> GameItem {
        get_item("minerust:stone").unwrap().clone()
    }

    #[test]
    fn add_merges_before_using_a_new_slot() {
        let mut inventory = Inventory::default();
        assert_eq!(inventory.add_item(stone(), 60), 60);
        assert_eq!(inventory.add_item(stone(), 8), 8);
        assert_eq!(inventory.slots[0].as_ref().unwrap().quantity, 64);
        assert_eq!(inventory.slots[1].as_ref().unwrap().quantity, 4);
    }
}
