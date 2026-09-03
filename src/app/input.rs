use minerust::camera::check_intersection;
use minerust::{CHUNK_SIZE, RENDER_DISTANCE, World};
use winit::event::MouseButton;
use winit::window::CursorGrabMode;

use crate::logger::{LogLevel, log};
use crate::ui::menu::{MenuField, MenuHit, MenuLayout};
use crate::ui::ui::HOTBAR_SLOTS;

use super::state::State;

const BLOCK_PLACE_REPEAT_INTERVAL: f32 = 0.4;

impl State {
    /// Translates a raw mouse-click position into a menu action.
    ///
    /// Called whenever the player clicks while the game is in
    /// [`GameState::Menu`].  A [`MenuLayout`] is constructed from the current
    /// surface dimensions so that hit regions scale correctly at any
    /// resolution, then a point-in-rect hit test determines which (if any)
    /// interactive element was clicked.
    ///
    /// # Behavior per hit result
    /// | `MenuHit` variant       | Effect                                                  |
    /// |-------------------------|---------------------------------------------------------|
    /// | `NewWorld`              | Transitions directly to `GameState::Playing`.           |
    /// | `Multiplayer`           | Shows and focuses the server-address input box.         |
    /// | `None` (missed all UI)  | Clears the active field so keyboard input is ignored.   |
    ///
    /// # Parameters
    /// - `x` – Horizontal cursor position in physical pixels (origin = top-left).
    /// - `y` – Vertical cursor position in physical pixels.
    pub fn handle_menu_click(&mut self, x: f32, y: f32) {
        let layout = MenuLayout::new(self.config.width, self.config.height);

        if self.menu_state.server_address_input_visible
            && layout.server_address_input.contains(x, y)
        {
            self.menu_state.select_field(MenuField::ServerAddress);
            return;
        }

        match layout.hit_test(x, y) {
            Some(MenuHit::RenderMode) => self.toggle_present_mode(),
            Some(MenuHit::NewWorld) => self.start_new_world(),
            Some(MenuHit::Multiplayer) => self.menu_state.show_server_address_input(),
            None => self.menu_state.select_field(MenuField::None),
        }
    }

    /// Switches immediately between uncapped presentation and display-synced
    /// presentation. Both modes are supported by the active surface because
    /// the initial configuration uses `Immediate` and `Fifo` is mandatory.
    fn toggle_present_mode(&mut self) {
        self.config.present_mode = match self.config.present_mode {
            wgpu::PresentMode::Fifo => wgpu::PresentMode::Immediate,
            _ => wgpu::PresentMode::Fifo,
        };
        self.surface.configure(&self.device, &self.config);
    }

    fn start_new_world(&mut self) {
        let mut world = World::new();
        // Spawn selection needs the actual blocks (trees, ceilings, etc.),
        // not only the height-map. Generate the initial area before searching.
        world.generate_chunks_in_radius(0, 0, 2);
        let seed = world.seed;
        let spawn = world.find_spawn_point();
        let spawn_cx = (spawn.0 / CHUNK_SIZE as f32).floor() as i32;
        let spawn_cz = (spawn.2 / CHUNK_SIZE as f32).floor() as i32;

        {
            let mut world_lock = self.world.write();
            *world_lock = world;
            world_lock.generate_chunks_in_radius(spawn_cx, spawn_cz, 2);
        }
        World::spawn_chunks_in_ring_async(
            self.world.clone(),
            spawn_cx,
            spawn_cz,
            2,
            RENDER_DISTANCE,
        );

        self.chunk_loader = minerust::ChunkLoader::new(seed);
        self.mesh_loader =
            minerust::MeshLoader::new(self.world.clone(), minerust::get_mesh_worker_count());
        self.chunks_rendered = 0;
        self.subchunks_rendered = 0;
        self.enqueue_all_dirty_meshes();
        self.indirect_manager.clear();
        self.water_indirect_manager.clear();
        self.visible_chunk_columns.clear();
        self.visible_chunk_cache_center = (i32::MIN, i32::MIN);
        self.visible_chunk_columns_dirty = true;
        self.last_gen_player_cx = i32::MIN;
        self.last_gen_player_cz = i32::MIN;
        self.highlighted_block = None;
        self.input = Default::default();
        self.digging = Default::default();
        self.placement = Default::default();
        self.inventory = Default::default();
        self.item_entities.clear();
        self.next_entity_id = 1;
        self.camera = minerust::Camera::new(spawn);
        self.game_start_time = std::time::Instant::now();
        self.last_frame = std::time::Instant::now();
        self.game_state = crate::ui::menu::GameState::Playing;
        self.has_entered_world = true;
        self.mouse_captured = true;
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
        self.window.set_cursor_visible(false);

        log(
            LogLevel::Info,
            &format!(
                "New world generated from menu click (seed: {}, spawn: {:?})",
                seed, spawn
            ),
        );
    }

    /// Processes a mouse-button press or release event.
    ///
    /// This method has two responsibilities:
    ///
    /// 1. **Always**: mirror the button state into [`InputState`] so that
    ///    continuous per-frame logic (e.g., left-click mining) can poll it
    ///    without re-processing events.
    ///
    /// 2. Reset right-click drag placement when RMB is pressed or released.
    ///    Actual block placement is polled from [`State::update`] so holding
    ///    RMB can place a continuous straight line.
    ///
    /// # Parameters
    /// - `button`  – Which mouse button changed state.
    /// - `pressed` – `true` on press, `false` on release.
    pub fn handle_mouse_input(&mut self, button: MouseButton, pressed: bool) {
        // Always update raw input state so per-frame polling sees current buttons.
        match button {
            MouseButton::Left => self.input.left_mouse = pressed,
            MouseButton::Right => {
                self.input.right_mouse = pressed;
                self.placement.reset();
            }
            _ => {}
        }
    }

    /// Returns a block placement to apply this frame when RMB is held.
    pub fn update_held_block_placement(
        &mut self,
        raycast: Option<(i32, i32, i32, i32, i32, i32)>,
        dt: f32,
    ) -> Option<(i32, i32, i32, minerust::BlockType)> {
        if !self.mouse_captured || !self.input.right_mouse {
            self.placement.reset();
            return None;
        }

        self.placement.cooldown = (self.placement.cooldown - dt).max(0.0);
        if self.placement.cooldown > 0.0 {
            return None;
        }

        let (_, _, _, px, py, pz) = raycast?;
        let place_pos = (px, py, pz);

        if self.camera.intersects_block(px, py, pz) {
            return None;
        }

        for player in self.remote_players.values() {
            let player_pos = glam::Vec3::new(player.x, player.y, player.z);
            if check_intersection(player_pos, px, py, pz) {
                return None;
            }
        }

        if !self.accept_line_placement(place_pos) {
            return None;
        }

        self.record_line_placement(place_pos);
        self.placement.cooldown = BLOCK_PLACE_REPEAT_INTERVAL;

        Some((px, py, pz, HOTBAR_SLOTS[self.hotbar_slot]))
    }

    fn accept_line_placement(&self, pos: (i32, i32, i32)) -> bool {
        let Some(last) = self.placement.last else {
            return true;
        };
        if pos == last {
            return false;
        }

        let Some(axis) = adjacent_axis(last, pos) else {
            return false;
        };

        if let Some(required_axis) = self.placement.axis {
            axis == required_axis && same_line(self.placement.anchor.unwrap_or(last), pos, axis)
        } else {
            true
        }
    }

    fn record_line_placement(&mut self, pos: (i32, i32, i32)) {
        if self.placement.anchor.is_none() {
            self.placement.anchor = Some(pos);
            self.placement.last = Some(pos);
            return;
        }

        if self.placement.axis.is_none() {
            if let Some(last) = self.placement.last {
                self.placement.axis = adjacent_axis(last, pos);
            }
        }

        self.placement.last = Some(pos);
    }

    /// Initiates an asynchronous connection to the multiplayer server.
    ///
    /// Reads the server address and username from [`MenuState`] and spawns
    /// the networking tasks on the shared Tokio runtime.  On success the
    /// `network_rx` / `network_tx` channels are populated so that
    /// `update_network` can exchange packets with the server each frame.
    ///
    /// The [`MenuState`] status message is updated in-place by
    /// `connect_to_server` to reflect connection progress or any error that
    /// occurs (e.g., DNS failure, refused connection).
    ///
    /// # Note
    /// This method is a thin forwarding shim that exists so menu click
    /// handling (`handle_menu_click`) does not need to borrow individual
    /// fields of `self` separately — `&mut self` here satisfies the borrow
    /// checker cleanly.
    pub fn connect_to_server(&mut self) {
        use crate::multiplayer::network::connect_to_server;
        connect_to_server(
            &mut self.menu_state,
            &mut self.game_state,
            &self.network_runtime,
            &mut self.network_rx,
            &mut self.network_tx,
        );
    }
}

fn adjacent_axis(a: (i32, i32, i32), b: (i32, i32, i32)) -> Option<usize> {
    let delta = [b.0 - a.0, b.1 - a.1, b.2 - a.2];
    let mut axis = None;

    for (idx, value) in delta.into_iter().enumerate() {
        if value == 0 {
            continue;
        }
        if value.abs() != 1 || axis.is_some() {
            return None;
        }
        axis = Some(idx);
    }

    axis
}

fn same_line(anchor: (i32, i32, i32), pos: (i32, i32, i32), axis: usize) -> bool {
    match axis {
        0 => anchor.1 == pos.1 && anchor.2 == pos.2,
        1 => anchor.0 == pos.0 && anchor.2 == pos.2,
        2 => anchor.0 == pos.0 && anchor.1 == pos.1,
        _ => false,
    }
}
