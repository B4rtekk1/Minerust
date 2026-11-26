use minerust::Vertex;

/// The top-level game mode, used to drive which systems are active each frame.
///
/// Transitions flow: `Menu` → `Connecting` → `Playing`, and back to `Menu`
/// on disconnect or error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    /// The main menu is visible and the player has not yet joined a server.
    Menu,
    /// The player is in an active game session.
    Playing,
    /// A connection attempt is in progress; the menu is still visible but
    /// input is typically disabled.
    Connecting,
}

impl Default for GameState {
    /// Returns [`GameState::Menu`], the initial state on launch.
    fn default() -> Self {
        GameState::Menu
    }
}

/// Identifies which text input field in the main menu currently has focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuField {
    /// The server address field (e.g. `"127.0.0.1:25565"`).
    ServerAddress,
    /// The player username field.
    Username,
    /// No field is focused; keyboard input is ignored.
    None,
}

impl Default for MenuField {
    /// Returns [`MenuField::None`]; no field is focused by default.
    fn default() -> Self {
        MenuField::None
    }
}

/// Runtime state for menus, including text field contents and transient
/// feedback messages.
///
/// Input events should be forwarded to [`MenuState::handle_char`] and
/// [`MenuState::handle_backspace`]. Field focus is managed through
/// [`MenuState::select_field`] and [`MenuState::next_field`].
#[derive(Debug, Clone)]
pub struct MenuState {
    /// Currently displayed menu screen.
    pub screen: MenuScreen,
    /// Index of the selected actionable menu item for keyboard navigation.
    pub selected_item: usize,
    /// Set when the player chooses to quit; the event loop performs the exit.
    pub quit_requested: bool,
    /// Text entered in the server address field. Capped at 50 characters.
    pub server_address: String,
    /// Text entered in the username field. Capped at 16 characters.
    pub username: String,
    /// The field that currently receives keyboard input.
    pub selected_field: MenuField,
    /// `true` when the whole active text field is selected via Ctrl+A.
    pub selected_all: bool,
    /// An error message to display to the player (e.g. connection refused).
    /// `None` when no error is active. Cleared by [`MenuState::clear_error`].
    pub error_message: Option<String>,
    /// A transient status message (e.g. "Connecting…"). `None` when idle.
    pub status_message: Option<String>,
}

impl Default for MenuState {
    /// Returns a `MenuState` pre-filled with sensible defaults:
    /// - Server address: `"127.0.0.1:25565"`
    /// - Username: `"Player"`
    /// - No focused field, no messages.
    fn default() -> Self {
        Self {
            screen: MenuScreen::Main,
            selected_item: 0,
            quit_requested: false,
            server_address: "127.0.0.1:25565".to_string(),
            username: "Player".to_string(),
            selected_field: MenuField::None,
            selected_all: false,
            error_message: None,
            status_message: None,
        }
    }
}

#[allow(dead_code)]
impl MenuState {
    /// Creates a new `MenuState` with default values. Equivalent to
    /// [`MenuState::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `ch` to the currently focused field, if any.
    ///
    /// ASCII control characters (e.g. backspace, escape) are silently ignored
    /// — use [`MenuState::handle_backspace`] for deletion. Field-specific
    /// length limits are enforced:
    /// - Server address: 50 characters.
    /// - Username: 16 characters.
    pub fn handle_char(&mut self, ch: char) {
        if !ch.is_ascii_control() {
            let replace_selection = self.selected_all;
            match self.selected_field {
                MenuField::ServerAddress => {
                    if replace_selection {
                        self.server_address.clear();
                    }
                    if self.server_address.len() < 50 {
                        self.server_address.push(ch);
                    }
                }
                MenuField::Username => {
                    if replace_selection {
                        self.username.clear();
                    }
                    if self.username.len() < 16 {
                        self.username.push(ch);
                    }
                }
                MenuField::None => {}
            }
            self.selected_all = false;
        }
    }

    /// Appends pasted text to the focused field using the same filtering and
    /// length limits as normal typed input.
    pub fn handle_paste(&mut self, text: &str) {
        if self.selected_all {
            match self.selected_field {
                MenuField::ServerAddress => self.server_address.clear(),
                MenuField::Username => self.username.clear(),
                MenuField::None => {}
            }
            self.selected_all = false;
        }

        for ch in text.chars() {
            self.handle_char(ch);
        }
    }

    /// Removes the last character from the currently focused field.
    ///
    /// No-op when [`MenuField::None`] is selected or the field is already empty.
    pub fn handle_backspace(&mut self) {
        if self.selected_all {
            match self.selected_field {
                MenuField::ServerAddress => self.server_address.clear(),
                MenuField::Username => self.username.clear(),
                MenuField::None => {}
            }
            self.selected_all = false;
            return;
        }

        match self.selected_field {
            MenuField::ServerAddress => {
                self.server_address.pop();
            }
            MenuField::Username => {
                self.username.pop();
            }
            MenuField::None => {}
        }
    }

    /// Advances focus to the next field in tab order.
    ///
    /// Cycles: `None` → `ServerAddress` → `Username` → `None`.
    pub fn next_field(&mut self) {
        self.selected_field = match self.selected_field {
            MenuField::None => MenuField::ServerAddress,
            MenuField::ServerAddress => MenuField::Username,
            MenuField::Username => MenuField::None,
        };
        self.selected_all = false;
    }

    /// Directly sets keyboard focus to `field`.
    ///
    /// Pass [`MenuField::None`] to remove focus from all fields.
    pub fn select_field(&mut self, field: MenuField) {
        self.selected_field = field;
        self.selected_all = false;
    }

    /// Changes screens and resets keyboard selection to the first enabled item.
    pub fn open_screen(&mut self, screen: MenuScreen) {
        self.screen = screen;
        self.selected_item = 0;
        self.select_field(MenuField::None);
    }

    /// Returns to the active world's pause menu, or the main menu before play.
    pub fn back(&mut self, has_world: bool) {
        self.open_screen(if has_world {
            MenuScreen::Pause
        } else {
            MenuScreen::Main
        });
    }

    /// Moves selection through enabled, actionable items, wrapping at either end.
    pub fn move_selection(&mut self, delta: isize, layout: &MenuLayout) {
        let selectable: Vec<usize> = layout
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| (item.enabled && item.action.is_some()).then_some(index))
            .collect();
        if selectable.is_empty() {
            self.selected_item = 0;
            return;
        }
        let current = selectable
            .iter()
            .position(|&index| index == self.selected_item)
            .unwrap_or(0);
        let next = (current as isize + delta).rem_euclid(selectable.len() as isize) as usize;
        self.selected_item = selectable[next];
        self.select_field(MenuField::None);
    }

    /// Returns the currently selected enabled action, if any.
    pub fn selected_action(&self, layout: &MenuLayout) -> Option<MenuAction> {
        layout
            .items
            .get(self.selected_item)
            .filter(|item| item.enabled)
            .and_then(|item| item.action)
    }

    /// Selects the entire active field, matching Ctrl+A behavior in text boxes.
    pub fn select_all_current_field(&mut self) {
        self.selected_all = match self.selected_field {
            MenuField::ServerAddress => !self.server_address.is_empty(),
            MenuField::Username => !self.username.is_empty(),
            MenuField::None => false,
        };
    }

    /// Returns the active selected text for clipboard copy operations.
    pub fn selected_text(&self) -> Option<&str> {
        if !self.selected_all {
            return None;
        }

        match self.selected_field {
            MenuField::ServerAddress => Some(&self.server_address),
            MenuField::Username => Some(&self.username),
            MenuField::None => None,
        }
    }

    /// Clears any active error message, hiding the error display in the UI.
    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    /// Sets the error message displayed to the player (e.g. `"Connection refused"`).
    ///
    /// Replaces any previously set error. Call [`MenuState::clear_error`] to
    /// dismiss it.
    pub fn set_error(&mut self, msg: &str) {
        self.error_message = Some(msg.to_string());
    }

    /// Sets a transient status message (e.g. `"Connecting…"`).
    ///
    /// Intended for non-error feedback such as connection progress. Replaces
    /// any previously set status.
    pub fn set_status(&mut self, msg: &str) {
        self.status_message = Some(msg.to_string());
    }

    /// Returns `true` if any text field currently has keyboard focus.
    ///
    /// Useful for suppressing game hotkeys while the player is typing.
    pub fn is_editing(&self) -> bool {
        self.selected_field != MenuField::None
    }
}

/// A distinct screen in the in-game menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuScreen {
    Main,
    Pause,
    Multiplayer,
    Settings,
    Controls,
}

/// Action emitted by a menu item hit test or keyboard activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuAction {
    Continue,
    Resume,
    NewWorld,
    LoadWorld,
    SaveWorld,
    OpenMultiplayer,
    OpenSettings,
    OpenControls,
    Back,
    MainMenu,
    Connect,
    Quit,
    ToggleRenderMode,
    ToggleWater,
    CycleAntiAliasing,
    ToggleShadows,
    FocusUsername,
    FocusServerAddress,
}

/// The help content shown on the controls screen.
pub const CONTROLS: &[&str] = &[
    "W A S D     Move",
    "SPACE       Jump",
    "SHIFT       Sprint",
    "E           Inventory",
    "MOUSE       Look / mine / place",
    "1–9         Select hotbar slot",
    "ESC         Pause / return",
];

#[derive(Debug, Clone, Copy)]
pub struct MenuContext {
    pub has_world: bool,
    pub save_available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuItemKind {
    Button,
    TextField,
    Text,
}

#[derive(Debug, Clone)]
pub struct MenuItem {
    pub rect: Rect,
    pub action: Option<MenuAction>,
    pub kind: MenuItemKind,
    pub enabled: bool,
    pub label: String,
}

/// An axis-aligned rectangle in screen-space pixels.
///
/// Used throughout [`MenuLayout`] to define the bounds of every UI element.
/// The origin `(x, y)` is the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// X coordinate of the left edge, in pixels from the left of the window.
    pub x: f32,
    /// Y coordinate of the top edge, in pixels from the top of the window.
    pub y: f32,
    /// Width of the rectangle in pixels.
    pub w: f32,
    /// Height of the rectangle in pixels.
    pub h: f32,
}

impl Rect {
    /// Returns `true` if the point `(px, py)` lies within the rectangle
    /// (inclusive on all edges).
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

/// Computed pixel-space bounds for the full menu screen.
///
/// Constructed once per resize via [`MenuLayout::new`] and then queried by
/// the renderer and [`MenuLayout::hit_test`]. All coordinates are in pixels
/// with the origin at the top-left of the window.
#[derive(Debug, Clone)]
pub struct MenuLayout {
    pub panel: Rect,
    pub title: Rect,
    pub items: Vec<MenuItem>,
    pub message: Rect,
    pub scale: f32,
}

impl MenuLayout {
    pub fn new(width: u32, height: u32, screen: MenuScreen, context: MenuContext) -> Self {
        let w = width as f32;
        let h = height as f32;
        let labels: Vec<(Option<MenuAction>, MenuItemKind, bool, String)> = match screen {
            MenuScreen::Main => {
                let mut items = Vec::new();
                if context.has_world {
                    items.push((
                        Some(MenuAction::Continue),
                        MenuItemKind::Button,
                        true,
                        "CONTINUE".into(),
                    ));
                }
                items.extend([
                    (
                        Some(MenuAction::NewWorld),
                        MenuItemKind::Button,
                        true,
                        "NEW WORLD".into(),
                    ),
                    (
                        Some(MenuAction::LoadWorld),
                        MenuItemKind::Button,
                        context.save_available,
                        "LOAD WORLD".into(),
                    ),
                    (
                        Some(MenuAction::OpenMultiplayer),
                        MenuItemKind::Button,
                        true,
                        "MULTIPLAYER".into(),
                    ),
                    (
                        Some(MenuAction::OpenSettings),
                        MenuItemKind::Button,
                        true,
                        "SETTINGS".into(),
                    ),
                    (
                        Some(MenuAction::OpenControls),
                        MenuItemKind::Button,
                        true,
                        "CONTROLS".into(),
                    ),
                    (
                        Some(MenuAction::Quit),
                        MenuItemKind::Button,
                        true,
                        "QUIT".into(),
                    ),
                ]);
                items
            }
            MenuScreen::Pause => vec![
                (
                    Some(MenuAction::Resume),
                    MenuItemKind::Button,
                    true,
                    "RESUME".into(),
                ),
                (
                    Some(MenuAction::SaveWorld),
                    MenuItemKind::Button,
                    context.has_world,
                    "SAVE WORLD".into(),
                ),
                (
                    Some(MenuAction::LoadWorld),
                    MenuItemKind::Button,
                    context.save_available,
                    "LOAD WORLD".into(),
                ),
                (
                    Some(MenuAction::OpenSettings),
                    MenuItemKind::Button,
                    true,
                    "SETTINGS".into(),
                ),
                (
                    Some(MenuAction::OpenControls),
                    MenuItemKind::Button,
                    true,
                    "CONTROLS".into(),
                ),
                (
                    Some(MenuAction::MainMenu),
                    MenuItemKind::Button,
                    true,
                    "MAIN MENU".into(),
                ),
            ],
            MenuScreen::Multiplayer => vec![
                (
                    Some(MenuAction::FocusServerAddress),
                    MenuItemKind::TextField,
                    true,
                    "SERVER ADDRESS".into(),
                ),
                (
                    Some(MenuAction::FocusUsername),
                    MenuItemKind::TextField,
                    true,
                    "USERNAME".into(),
                ),
                (
                    Some(MenuAction::Connect),
                    MenuItemKind::Button,
                    true,
                    "CONNECT".into(),
                ),
                (
                    Some(MenuAction::Back),
                    MenuItemKind::Button,
                    true,
                    "BACK".into(),
                ),
            ],
            MenuScreen::Settings => vec![
                (
                    Some(MenuAction::ToggleRenderMode),
                    MenuItemKind::Button,
                    true,
                    "RENDER MODE".into(),
                ),
                (
                    Some(MenuAction::ToggleWater),
                    MenuItemKind::Button,
                    true,
                    "WATER QUALITY".into(),
                ),
                (
                    Some(MenuAction::CycleAntiAliasing),
                    MenuItemKind::Button,
                    true,
                    "ANTI-ALIASING".into(),
                ),
                (
                    Some(MenuAction::ToggleShadows),
                    MenuItemKind::Button,
                    true,
                    "SUN SHADOWS".into(),
                ),
                (
                    Some(MenuAction::Back),
                    MenuItemKind::Button,
                    true,
                    "BACK".into(),
                ),
            ],
            MenuScreen::Controls => {
                let mut items: Vec<_> = CONTROLS
                    .iter()
                    .map(|line| (None, MenuItemKind::Text, false, (*line).into()))
                    .collect();
                items.push((
                    Some(MenuAction::Back),
                    MenuItemKind::Button,
                    true,
                    "BACK".into(),
                ));
                items
            }
        };
        let base_height = labels.len() as f32 * 56.0 + 116.0;
        let height_scale = (((h - 32.0).max(1.0) / base_height).min(1.3)).max(0.68);
        let width_scale = (((w - 48.0).max(1.0) / 360.0).min(1.3)).max(0.58);
        let scale = height_scale.min(width_scale);
        let panel_w = (h * 0.58).clamp(420.0, 640.0).min(w.max(1.0));
        let padding = (24.0 * scale).min(panel_w * 0.08);
        let row_h = 48.0 * scale;
        let gap = 8.0 * scale;
        let panel_h = base_height * scale;
        let panel = Rect {
            x: (w - panel_w) * 0.5,
            y: ((h - panel_h) * 0.5).max(8.0),
            w: panel_w,
            h: panel_h.min((h - 16.0).max(1.0)),
        };
        let title = Rect {
            x: panel.x + padding,
            y: panel.y + 22.0 * scale,
            w: (panel.w - padding * 2.0).max(1.0),
            h: 48.0 * scale,
        };
        let row_x = panel.x + padding;
        let row_w = (panel.w - padding * 2.0).min((w - 48.0).max(1.0)).max(1.0);
        let first_y = title.y + title.h + 12.0 * scale;
        let items: Vec<_> = labels
            .into_iter()
            .enumerate()
            .map(|(index, (action, kind, enabled, label))| MenuItem {
                rect: Rect {
                    x: row_x,
                    y: first_y + index as f32 * (row_h + gap),
                    w: row_w,
                    h: row_h,
                },
                action,
                kind,
                enabled,
                label,
            })
            .collect();
        let message = Rect {
            x: panel.x + padding,
            y: panel.y + panel.h - 34.0 * scale,
            w: (panel.w - padding * 2.0).max(1.0),
            h: 24.0 * scale,
        };
        Self {
            panel,
            title,
            items,
            message,
            scale,
        }
    }

    pub fn hit_test(&self, px: f32, py: f32) -> Option<MenuAction> {
        self.items
            .iter()
            .find(|item| item.enabled && item.action.is_some() && item.rect.contains(px, py))
            .and_then(|item| item.action)
    }
}

/// Builds the layered menu panel and button geometry in screen-space pixels.
pub fn build_menu_geometry(
    width: u32,
    height: u32,
    layout: &MenuLayout,
    hovered: Option<MenuAction>,
    focused: MenuField,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut add_rect = |rect: Rect, color: [f32; 4]| {
        if width == 0 || height == 0 {
            return;
        }
        let x0 = rect.x / width as f32 * 2.0 - 1.0;
        let x1 = (rect.x + rect.w) / width as f32 * 2.0 - 1.0;
        let y0 = 1.0 - (rect.y + rect.h) / height as f32 * 2.0;
        let y1 = 1.0 - rect.y / height as f32 * 2.0;
        let base = vertices.len() as u32;
        let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);
        for (i, (x, y)) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            .into_iter()
            .enumerate()
        {
            vertices.push(Vertex {
                position: [x, y, 0.0],
                packed: Vertex::pack_ui(normal, color, 0, i as u8),
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };
    add_rect(
        Rect {
            x: 0.0,
            y: 0.0,
            w: width as f32,
            h: height as f32,
        },
        [0.02, 0.03, 0.05, 0.48],
    );
    add_rect(
        Rect {
            x: layout.panel.x + 7.0,
            y: layout.panel.y + 9.0,
            w: layout.panel.w,
            h: layout.panel.h,
        },
        [0.01, 0.02, 0.03, 0.78],
    );
    add_rect(layout.panel, [0.07, 0.09, 0.12, 0.96]);
    let border = layout.scale.max(0.7) * 3.0;
    add_rect(
        Rect {
            x: layout.panel.x,
            y: layout.panel.y,
            w: layout.panel.w,
            h: border,
        },
        [0.44, 0.70, 0.82, 1.0],
    );
    add_rect(
        Rect {
            x: layout.title.x,
            y: layout.title.y + layout.title.h + 5.0 * layout.scale,
            w: layout.title.w,
            h: 2.0 * layout.scale,
        },
        [0.25, 0.55, 0.66, 0.95],
    );
    for item in &layout.items {
        if item.kind == MenuItemKind::Text {
            continue;
        }
        let action = item.action;
        let is_focused = matches!(
            (action, focused),
            (Some(MenuAction::FocusUsername), MenuField::Username)
                | (
                    Some(MenuAction::FocusServerAddress),
                    MenuField::ServerAddress
                )
        );
        let fill = if !item.enabled {
            [0.10, 0.12, 0.15, 0.62]
        } else if is_focused {
            [0.16, 0.33, 0.40, 0.98]
        } else if action == hovered {
            [0.22, 0.38, 0.44, 0.98]
        } else {
            [0.12, 0.16, 0.20, 0.94]
        };
        add_rect(
            Rect {
                x: item.rect.x,
                y: item.rect.y + 3.0 * layout.scale,
                w: item.rect.w,
                h: item.rect.h,
            },
            [0.02, 0.03, 0.04, 0.95],
        );
        add_rect(
            item.rect,
            if !item.enabled {
                [0.18, 0.21, 0.25, 0.9]
            } else {
                [0.35, 0.47, 0.52, 0.96]
            },
        );
        add_rect(
            Rect {
                x: item.rect.x + border,
                y: item.rect.y + border,
                w: (item.rect.w - border * 2.0).max(1.0),
                h: (item.rect.h - border * 2.0).max(1.0),
            },
            fill,
        );
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(screen: MenuScreen, has_world: bool, save_available: bool) -> MenuLayout {
        MenuLayout::new(
            800,
            600,
            screen,
            MenuContext {
                has_world,
                save_available,
            },
        )
    }

    #[test]
    fn hit_test_returns_each_enabled_action_on_every_screen() {
        for screen in [
            MenuScreen::Main,
            MenuScreen::Pause,
            MenuScreen::Multiplayer,
            MenuScreen::Settings,
            MenuScreen::Controls,
        ] {
            let layout = layout(screen, true, true);
            for item in layout
                .items
                .iter()
                .filter(|item| item.enabled && item.action.is_some())
            {
                assert_eq!(
                    layout.hit_test(
                        item.rect.x + item.rect.w * 0.5,
                        item.rect.y + item.rect.h * 0.5
                    ),
                    item.action,
                    "{screen:?}: {}",
                    item.label
                );
            }
        }
    }

    #[test]
    fn continue_is_only_present_when_a_world_exists() {
        assert!(
            layout(MenuScreen::Main, false, false)
                .items
                .iter()
                .all(|item| item.action != Some(MenuAction::Continue))
        );
        assert!(
            layout(MenuScreen::Main, true, false)
                .items
                .iter()
                .any(|item| item.action == Some(MenuAction::Continue))
        );
    }

    #[test]
    fn disabled_load_world_does_not_respond_to_hits() {
        let layout = layout(MenuScreen::Main, false, false);
        let item = layout
            .items
            .iter()
            .find(|item| item.action == Some(MenuAction::LoadWorld))
            .unwrap();
        assert!(!item.enabled);
        assert_eq!(layout.hit_test(item.rect.x + 5.0, item.rect.y + 5.0), None);
    }

    #[test]
    fn back_returns_to_parent_menu() {
        let mut menu = MenuState::default();
        menu.open_screen(MenuScreen::Settings);
        menu.back(false);
        assert_eq!(menu.screen, MenuScreen::Main);
        menu.open_screen(MenuScreen::Controls);
        menu.back(true);
        assert_eq!(menu.screen, MenuScreen::Pause);
    }

    #[test]
    fn keyboard_selection_wraps_over_enabled_actions() {
        let layout = layout(MenuScreen::Main, false, false);
        let mut menu = MenuState::default();
        menu.move_selection(-1, &layout);
        assert_eq!(menu.selected_action(&layout), Some(MenuAction::Quit));
        menu.move_selection(1, &layout);
        assert_eq!(menu.selected_action(&layout), Some(MenuAction::NewWorld));
    }

    #[test]
    fn layout_fits_common_and_wide_windows() {
        for (width, height) in [(384, 600), (800, 600), (3840, 2160)] {
            for screen in [
                MenuScreen::Main,
                MenuScreen::Pause,
                MenuScreen::Multiplayer,
                MenuScreen::Settings,
                MenuScreen::Controls,
            ] {
                let layout = MenuLayout::new(
                    width,
                    height,
                    screen,
                    MenuContext {
                        has_world: true,
                        save_available: true,
                    },
                );
                assert!(layout.panel.x >= 0.0 && layout.panel.y >= 0.0);
                assert!(layout.panel.x + layout.panel.w <= width as f32 + 0.1);
                assert!(layout.panel.y + layout.panel.h <= height as f32 + 0.1);
                for item in &layout.items {
                    assert!(item.rect.w <= width as f32 - 48.0 + 0.1);
                    assert!(
                        item.rect.x >= layout.panel.x
                            && item.rect.x + item.rect.w <= layout.panel.x + layout.panel.w + 0.1
                    );
                    assert!(
                        item.rect.y >= layout.panel.y
                            && item.rect.y + item.rect.h <= layout.panel.y + layout.panel.h + 0.1
                    );
                }
            }
        }
    }
}
