#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameState {
    Menu,
    Playing,
    Connecting,
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Menu
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuField {
    ServerAddress,
    Username,
    None,
}

impl Default for MenuField {
    fn default() -> Self {
        MenuField::None
    }
}

#[derive(Debug, Clone)]
pub struct MenuState {
    pub server_address: String,
    pub username: String,
    pub selected_field: MenuField,
    pub error_message: Option<String>,
    pub status_message: Option<String>,
}

impl Default for MenuState {
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
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn next_field(&mut self) {
        self.selected_field = match self.selected_field {
            MenuField::None => MenuField::ServerAddress,
            MenuField::ServerAddress => MenuField::Username,
            MenuField::Username => MenuField::None,
        };
    }

    pub fn select_field(&mut self, field: MenuField) {
        self.selected_field = field;
    }

    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    pub fn set_error(&mut self, msg: &str) {
        self.error_message = Some(msg.to_string());
    }

    pub fn set_status(&mut self, msg: &str) {
        self.status_message = Some(msg.to_string());
    }

    pub fn is_editing(&self) -> bool {
        self.selected_field != MenuField::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuHit {
    ServerAddress,
    Username,
    Connect,
    Singleplayer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MenuLayout {
    pub panel: Rect,
    pub header: Rect,
    pub server_label: Rect,
    pub server_field: Rect,
    pub username_label: Rect,
    pub username_field: Rect,
    pub quick_card: Rect,
    pub connect_button: Rect,
    pub singleplayer_button: Rect,
    pub status_pill: Rect,
}

impl MenuLayout {
    pub fn new(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;
        let panel_w = (w * 0.62).clamp(560.0, 820.0).min(w - 48.0);
        let panel_h = (h * 0.76).clamp(520.0, 640.0).min(h - 48.0);
        let panel_x = (w - panel_w) * 0.5;
        let panel_y = (h - panel_h) * 0.5;

        let header_h = 112.0;
        let content_top = panel_y + header_h;
        let content_left = panel_x + 40.0;
        let content_right = panel_x + panel_w - 40.0;
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
