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

/// Runtime state for the main menu, including text field contents and
/// transient feedback messages.
///
/// Input events should be forwarded to [`MenuState::handle_char`] and
/// [`MenuState::handle_backspace`]. Field focus is managed through
/// [`MenuState::select_field`] and [`MenuState::next_field`].
#[derive(Debug, Clone)]
pub struct MenuState {
    /// Text entered in the server address field. Capped at 50 characters.
    pub server_address: String,
    /// Text entered in the username field. Capped at 16 characters.
    pub username: String,
    /// Whether the server-address input should be drawn next to "MULTIPLAYER".
    pub server_address_input_visible: bool,
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
            server_address: "127.0.0.1:25565".to_string(),
            username: "Player".to_string(),
            server_address_input_visible: false,
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

    /// Shows and focuses the server-address input box.
    pub fn show_server_address_input(&mut self) {
        self.server_address_input_visible = true;
        self.select_field(MenuField::ServerAddress);
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

/// An interactive element in the main menu that a mouse click can target.
///
/// Returned by [`MenuLayout::hit_test`] to let the caller dispatch the
/// appropriate action without needing to know the layout geometry directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuHit {
    /// The render presentation mode toggle was clicked.
    RenderMode,
    /// The "new world" menu label was clicked.
    NewWorld,
    /// The "multiplayer" menu label was clicked.
    Multiplayer,
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

/// Computed pixel-space bounds for the clickable main-menu labels.
///
/// Constructed once per resize via [`MenuLayout::new`] and then queried by
/// the renderer and [`MenuLayout::hit_test`]. All coordinates are in pixels
/// with the origin at the top-left of the window.
#[derive(Debug, Clone, Copy)]
pub struct MenuLayout {
    /// Clickable bounds for the render mode toggle.
    pub render_mode_text: Rect,
    /// Clickable bounds for the "new world" label.
    pub new_world_text: Rect,
    /// Clickable bounds for the "multiplayer" label.
    pub multiplayer_text: Rect,
    /// Bounds for the server-address input shown next to "multiplayer".
    pub server_address_input: Rect,
}

impl MenuLayout {
    /// Computes the layout for a window of `width × height` pixels.
    ///
    /// The labels sit in a left-side column and their hit regions are wider
    /// than the glyphs so clicking feels natural without drawing button boxes.
    ///
    /// # Arguments
    ///
    /// * `width`  - Window width in physical pixels.
    /// * `height` - Window height in physical pixels.
    pub fn new(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;

        let hit_w = (w * 0.22).clamp(180.0, 300.0).min((w - 32.0).max(160.0));
        let hit_h = 48.0;
        let gap = 14.0;
        let total_h = hit_h * 3.0 + gap * 2.0;
        let x = (w * 0.045).clamp(20.0, 72.0);
        let bottom_margin = (h * 0.14).clamp(48.0, 120.0);
        let y = (h - total_h - bottom_margin).max(24.0);
        let new_world_y = y + hit_h + gap;
        let multiplayer_y = new_world_y + hit_h + gap;

        let input_gap = 18.0;
        let right_margin = x;
        let input_x = x + hit_w + input_gap;
        let desired_input_w = (w * 0.36).clamp(220.0, 420.0);
        let min_input_w = 140.0_f32.min((w - x * 2.0).max(96.0));
        let available_input_w = w - input_x - right_margin;
        let (input_x, input_y, input_w) = if available_input_w >= min_input_w {
            (
                input_x,
                multiplayer_y + 2.0,
                available_input_w.min(desired_input_w),
            )
        } else {
            let below_y = (multiplayer_y + hit_h + 8.0)
                .min(h - hit_h - 16.0)
                .max(16.0);
            (x, below_y, (w - x * 2.0).clamp(96.0, desired_input_w))
        };

        Self {
            render_mode_text: Rect {
                x,
                y,
                w: hit_w,
                h: hit_h,
            },
            new_world_text: Rect {
                x,
                y: new_world_y,
                w: hit_w,
                h: hit_h,
            },
            multiplayer_text: Rect {
                x,
                y: multiplayer_y,
                w: hit_w,
                h: hit_h,
            },
            server_address_input: Rect {
                x: input_x,
                y: input_y,
                w: input_w,
                h: hit_h - 4.0,
            },
        }
    }

    /// Tests whether the point `(px, py)` intersects any interactive element
    /// and returns the corresponding [`MenuHit`].
    ///
    /// Elements are tested from top to bottom.
    /// Returns `None` if the point does not fall inside any interactive region.
    ///
    /// # Arguments
    ///
    /// * `px` - Cursor X position in pixels from the left edge of the window.
    /// * `py` - Cursor Y position in pixels from the top edge of the window.
    pub fn hit_test(&self, px: f32, py: f32) -> Option<MenuHit> {
        if self.render_mode_text.contains(px, py) {
            return Some(MenuHit::RenderMode);
        }
        if self.new_world_text.contains(px, py) {
            return Some(MenuHit::NewWorld);
        }
        if self.multiplayer_text.contains(px, py) {
            return Some(MenuHit::Multiplayer);
        }
        None
    }
}
