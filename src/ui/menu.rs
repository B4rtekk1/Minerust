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
    /// The field that currently receives keyboard input.
    pub selected_field: MenuField,
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
            selected_field: MenuField::None,
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
            match self.selected_field {
                MenuField::ServerAddress => {
                    if self.server_address.len() < 50 {
                        self.server_address.push(ch);
                    }
                }
                MenuField::Username => {
                    if self.username.len() < 16 {
                        self.username.push(ch);
                    }
                }
                MenuField::None => {}
            }
        }
    }

    /// Removes the last character from the currently focused field.
    ///
    /// No-op when [`MenuField::None`] is selected or the field is already empty.
    pub fn handle_backspace(&mut self) {
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
    }

    /// Directly sets keyboard focus to `field`.
    ///
    /// Pass [`MenuField::None`] to remove focus from all fields.
    pub fn select_field(&mut self, field: MenuField) {
        self.selected_field = field;
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
    /// The server address text field was clicked.
    ServerAddress,
    /// The username text field was clicked.
    Username,
    /// The "Connect" button was clicked.
    Connect,
    /// The "Singleplayer" button was clicked.
    Singleplayer,
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

/// Computed pixel-space bounds for every element in the main menu.
///
/// Constructed once per resize via [`MenuLayout::new`] and then queried by
/// the renderer (for drawing) and [`MenuLayout::hit_test`] (for input). All
/// coordinates are in pixels with the origin at the top-left of the window.
///
/// Layout overview (top → bottom within the center panel):
/// ```text
/// ┌─────────────────────────────────┐
/// │ header                          │
/// ├──────────────────┬──────────────┤
/// │ server_label     │              │
/// │ server_field     │  quick_card  │
/// │ username_label   │              │
/// │ username_field   │              │
/// ├──────────────────┴──────────────┤
/// │ status_pill                     │
/// │ [connect_button] [singleplayer] │
/// └─────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MenuLayout {
    /// Outer panel rectangle; the visual backdrop for all other elements.
    pub panel: Rect,
    /// Title / branding area at the top of the panel.
    pub header: Rect,
    /// Label above the server address input field.
    pub server_label: Rect,
    /// Server address text input field.
    pub server_field: Rect,
    /// Label above the username input field.
    pub username_label: Rect,
    /// Username text input field.
    pub username_field: Rect,
    /// Right-hand card area (quick-connect history or tips).
    pub quick_card: Rect,
    /// "Connect" action button.
    pub connect_button: Rect,
    /// "Singleplayer" action button.
    pub singleplayer_button: Rect,
    /// Pill-shaped status / error message bar above the buttons.
    pub status_pill: Rect,
}

impl MenuLayout {
    /// Computes the layout for a window of `width × height` pixels.
    ///
    /// The panel is centred in the window and its dimensions are clamped so
    /// it remains usable at both small and large resolutions. All child
    /// element positions are derived relative to the panel, so the entire
    /// layout scales coherently with the window.
    ///
    /// # Arguments
    ///
    /// * `width`  - Window width in physical pixels.
    /// * `height` - Window height in physical pixels.
    pub fn new(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;

        // Panel: centred, clamped to [560, 820] wide and [520, 640] tall,
        // with a minimum 24 px margin on each side.
        let panel_w = (w * 0.62).clamp(560.0, 820.0).min(w - 48.0);
        let panel_h = (h * 0.76).clamp(520.0, 640.0).min(h - 48.0);
        let panel_x = (w - panel_w) * 0.5;
        let panel_y = (h - panel_h) * 0.5;

        let header_h = 112.0;
        let content_top = panel_y + header_h;
        let content_left = panel_x + 40.0;
        let content_right = panel_x + panel_w - 40.0;

        // Left column takes ~56% of interior width; right column fills the rest.
        let left_width = (panel_w * 0.56).clamp(320.0, 420.0);
        let gap = 18.0;
        let left_card_w = left_width;
        let right_card_w = (content_right - content_left - left_card_w - gap).max(200.0);

        let server_label_y = content_top + 14.0;
        let server_field_y = server_label_y + 26.0;
        let username_label_y = server_field_y + 78.0;
        let username_field_y = username_label_y + 26.0;
        let field_h = 52.0;
        let field_w = left_card_w;

        // Buttons sit 118 px above the panel bottom, split equally with a gap.
        let button_y = panel_y + panel_h - 118.0;
        let button_h = 54.0;
        let button_gap = 14.0;
        let button_w = ((panel_w - 80.0) - button_gap) * 0.5;
        let button_x = panel_x + 40.0;
        let quick_card_h = 230.0;

        Self {
            panel: Rect {
                x: panel_x,
                y: panel_y,
                w: panel_w,
                h: panel_h,
            },
            header: Rect {
                x: panel_x + 24.0,
                y: panel_y + 20.0,
                w: panel_w - 48.0,
                h: header_h - 24.0,
            },
            server_label: Rect {
                x: content_left,
                y: server_label_y,
                w: field_w,
                h: 22.0,
            },
            server_field: Rect {
                x: content_left,
                y: server_field_y,
                w: field_w,
                h: field_h,
            },
            username_label: Rect {
                x: content_left,
                y: username_label_y,
                w: field_w,
                h: 22.0,
            },
            username_field: Rect {
                x: content_left,
                y: username_field_y,
                w: field_w,
                h: field_h,
            },
            quick_card: Rect {
                x: content_left + field_w + gap,
                y: content_top + 14.0,
                w: right_card_w,
                h: quick_card_h,
            },
            connect_button: Rect {
                x: button_x,
                y: button_y,
                w: button_w,
                h: button_h,
            },
            singleplayer_button: Rect {
                x: button_x + button_w + button_gap,
                y: button_y,
                w: button_w,
                h: button_h,
            },
            status_pill: Rect {
                x: content_left,
                y: panel_y + panel_h - 168.0,
                w: panel_w - 80.0,
                h: 36.0,
            },
        }
    }

    /// Tests whether the point `(px, py)` intersects any interactive element
    /// and returns the corresponding [`MenuHit`].
    ///
    /// Elements are tested in priority order: input fields before buttons.
    /// Returns `None` if the point does not fall inside any interactive region.
    ///
    /// # Arguments
    ///
    /// * `px` - Cursor X position in pixels from the left edge of the window.
    /// * `py` - Cursor Y position in pixels from the top edge of the window.
    pub fn hit_test(&self, px: f32, py: f32) -> Option<MenuHit> {
        if self.server_field.contains(px, py) {
            return Some(MenuHit::ServerAddress);
        }
        if self.username_field.contains(px, py) {
            return Some(MenuHit::Username);
        }
        if self.connect_button.contains(px, py) {
            return Some(MenuHit::Connect);
        }
        if self.singleplayer_button.contains(px, py) {
            return Some(MenuHit::Singleplayer);
        }
        None
    }
}