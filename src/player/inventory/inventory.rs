use crate::GameItem;
pub struct InventoryItem {
    pub item: GameItem,
    pub item_name: String,
    pub quantity: u32,
}

pub struct Inventory {
    pub slots: [Option<InventoryItem>; 36], // 3 * 9 + 9 hotbar
    pub selected_slot: usize,
}