use std::collections::HashMap;
use once_cell::sync::Lazy;

/// A single game item type with its static properties.
///
/// All string fields carry a [`'static`] lifetime — they must be string literals
/// declared at compile time via the [`register_items!`] macro.
///
/// # Example
///
/// ```rust
/// let item = get_item("minerust:stone").unwrap();
/// assert_eq!(item.name, "Stone");
/// assert!(item.stackable);
/// assert_eq!(item.durability, None); // indestructible
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GameItem {
    /// Unique namespaced identifier, e.g. `"minerust:stone"`.
    /// Used as the key in [`ITEMS`].
    pub id: &'static str,

    /// Human-readable display name shown in the UI.
    pub name: &'static str,

    /// Whether multiple instances can occupy a single inventory slot.
    pub stackable: bool,

    /// Maximum number of items per stack.
    /// Ignored when `stackable` is `false`.
    pub max_stack_size: u32,

    /// Maximum number of uses before the item breaks.
    /// `None` means the item is indestructible.
    pub durability: Option<u32>,

    /// Item weight in grams, used for inventory capacity calculations.
    pub weight: u32,
}

/// Declares static game items and registers them in the global [`ITEMS`] map.
///
/// Each item is allocated once via [`Box::leak`], producing a `&'static GameItem`
/// with no duplicate-name conflicts between items. This macro must be invoked
/// **exactly once** per crate; calling it more than once will cause a compile
/// error due to duplicate `ITEMS` definitions.
///
/// # Syntax
///
/// ```text
/// register_items! {
///     "namespace:id" => "Name", stackable, max_stack, durability, weight;
/// }
/// ```
///
/// # Parameters
///
/// | Position | Pattern       | Type           | Description                     |
/// |----------|---------------|----------------|---------------------------------|
/// | 1        | `$id`         | `&'static str` | Namespaced key                  |
/// | 2        | `$name`       | `&'static str` | Display name                    |
/// | 3        | `$stackable`  | `bool`         | Whether the item can be stacked |
/// | 4        | `$max_stack`  | `u32`          | Maximum stack size              |
/// | 5        | `$durability` | `Option<u32>`  | `None` or `Some(n)` uses        |
/// | 6        | `$weight`     | `u32`          | Weight in grams                 |
///
/// # Example
///
/// ```rust
/// register_items! {
///     "minerust:stone"        => "Stone",        true,  64, None,       1;
///     "minerust:iron_pickaxe" => "Iron Pickaxe", false,  1, Some(250), 800;
/// }
/// ```
macro_rules! register_items {
    ($($id:literal => $name:literal, $stackable:literal, $max_stack:literal, $durability:expr, $weight:literal;)*) => {
        /// Global item registry.
        ///
        /// Lazily initialized on first access via [`once_cell::sync::Lazy`].
        /// The registry is populated by the [`register_items!`] macro and is safe
        /// for concurrent read access from multiple threads.
        ///
        /// Prefer the helper functions [`get_item`] and [`item_exists`] over
        /// accessing this directly.
        pub static ITEMS: Lazy<HashMap<&'static str, &'static GameItem>> = Lazy::new(|| {
            let mut map = HashMap::new();
            $(
                let item: &'static GameItem = Box::leak(Box::new(GameItem {
                    id: $id,
                    name: $name,
                    stackable: $stackable,
                    max_stack_size: $max_stack,
                    durability: $durability,
                    weight: $weight as u32,
                }));
                map.insert($id, item);
            )*
            map
        });
    };
}

/// Looks up an item in the global registry by its namespaced ID.
///
/// Returns `Some(&GameItem)` if found, or `None` if the ID is not registered.
/// The returned reference is `'static` and safe to store or pass freely.
///
/// # Example
///
/// ```rust
/// match get_item("minerust:stone") {
///     Some(item) => println!("Found: {} ({}g)", item.name, item.weight),
///     None       => println!("Unknown item"),
/// }
/// ```
pub fn get_item(id: &str) -> Option<&'static GameItem> {
    ITEMS.get(id).copied()
}

/// Returns `true` if an item with the given ID is registered, `false` otherwise.
///
/// Prefer this over `get_item(id).is_some()` when you only need to check
/// existence without borrowing the item data.
///
/// # Example
///
/// ```rust
/// // Validate item ID from save data before deserialising
/// if !item_exists(&save.item_id) {
///     return Err(SaveError::UnknownItem(save.item_id.clone()));
/// }
/// ```
pub fn item_exists(id: &str) -> bool {
    ITEMS.contains_key(id)
}

register_items! {
    "minerust:grass" => "Grass", true, 64, None, 1;
    "minerust:dirt"  => "Dirt",  true, 64, None, 1;
    "minerust:stone" => "Stone", true, 64, None, 5;
    "minerust:sand"  => "Sand",  true, 64, None, 1;
    "minerust:water" => "Water", false,  1, None, 100;
    "minerust:wood"  => "Wood",  true, 64, None, 5;
    "minerust:leaves" => "Leaves", true, 64, None,1;
    "minerust:bedrock" => "Bedrock", false, 1, None, 999999;
    "minerust:snow" => "Snow", true, 64, None, 1;
    "minerust:gravel" => "Gravel", true, 64, None, 1;
    "minecraft:clay" => "Clay", true, 64, None, 1;
    "minecraft:ice" => "Ice", true, 64, None, 100;
    "minerust:cactus" => "Cactus", true, 64, None, 1;
    "minerust:WoodStairs" => "Wood Stairs", true, 64, None, 5;
}


