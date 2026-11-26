use minerust::camera::check_intersection;
use minerust::{CHUNK_SIZE, DEFAULT_WORLD_FILE, SUBCHUNK_HEIGHT, World, load_world, save_world};
use winit::event::MouseButton;
use winit::window::CursorGrabMode;

use crate::logger::{LogLevel, log};
use crate::ui::menu::{MenuAction, MenuContext, MenuField, MenuLayout, MenuScreen};

use super::state::State;

const BLOCK_PLACE_REPEAT_INTERVAL: f32 = 0.4;

impl State {
    fn resolve_inventory_cursor(&mut self) {
        let Some(stack) = self.inventory_ui.cursor_stack.take() else {
            return;
        };
        let remainder = match self.inventory_ui.cursor_origin.take() {
            Some(origin) => self
                .inventory
                .return_to_slot(origin, stack, minerust::item_registry()),
            None => Some(stack),
        };
        if let Some(remainder) =
            remainder.and_then(|stack| self.inventory.insert(stack, minerust::item_registry()))
        {
            self.spawn_item_stack(
                remainder,
                self.camera.position + glam::Vec3::new(0.0, -0.4, 0.0),
            );
        }
        self.hotbar_dirty = true;
    }

    pub fn toggle_inventory(&mut self) {
        if self.inventory_open {
            self.resolve_inventory_cursor();
        }
        self.inventory_open = !self.inventory_open;
        self.input = Default::default();
        self.digging = Default::default();
        self.placement.reset();
        self.mouse_captured = !self.inventory_open;
        if self.inventory_open {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        } else {
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
            self.window.set_cursor_visible(false);
        }
    }

    pub fn handle_inventory_click(&mut self, x: f32, y: f32, right_click: bool) {
        let layout =
            crate::ui::inventory::InventoryLayout::new(self.config.width, self.config.height);
        let Some(slot) = layout.slot_at(x, y) else {
            return;
        };
        let Some(slot) = minerust::PlayerSlot::from_flat(slot) else {
            return;
        };
        let action = if self.modifiers.shift_key() && !right_click {
            minerust::InventoryAction::QuickMove(slot)
        } else if right_click {
            minerust::InventoryAction::RightClick(slot)
        } else {
            minerust::InventoryAction::LeftClick(slot)
        };
        let result =
            self.inventory
                .apply_action(&mut self.inventory_ui, action, minerust::item_registry());
        if result.changed {
            self.hotbar_dirty = true;
        }
    }
    pub fn handle_menu_click(&mut self, x: f32, y: f32) {
        if self.game_state == crate::ui::menu::GameState::Connecting {
            return;
        }
        let context = MenuContext {
            has_world: self.has_entered_world,
            save_available: std::path::Path::new(DEFAULT_WORLD_FILE).exists(),
        };
        let layout = MenuLayout::new(
            self.config.width,
            self.config.height,
            self.menu_state.screen,
            context,
        );
        if let Some(action) = layout.hit_test(x, y) {
            if let Some((index, _)) = layout
                .items
                .iter()
                .enumerate()
                .find(|(_, item)| item.rect.contains(x, y) && item.action == Some(action))
            {
                self.menu_state.selected_item = index;
            }
            self.activate_menu_action(action);
        } else {
            self.menu_state.select_field(MenuField::None);
        }
    }

    pub fn activate_menu_action(&mut self, action: MenuAction) {
        use crate::ui::menu::GameState;
        if self.game_state == GameState::Connecting {
            return;
        }
        match action {
            MenuAction::Continue | MenuAction::Resume => self.resume_game(),
            MenuAction::NewWorld => self.start_new_world(),
            MenuAction::LoadWorld => {
                self.load_world_from_disk();
            }
            MenuAction::SaveWorld => {
                self.save_world_to_disk();
            }
            MenuAction::OpenMultiplayer => {
                self.menu_state.open_screen(MenuScreen::Multiplayer);
                self.menu_state.select_field(MenuField::ServerAddress);
            }
            MenuAction::OpenSettings => self.menu_state.open_screen(MenuScreen::Settings),
            MenuAction::OpenControls => self.menu_state.open_screen(MenuScreen::Controls),
            MenuAction::Back => self.menu_state.back(self.has_entered_world),
            MenuAction::MainMenu => {
                self.menu_state.open_screen(MenuScreen::Main);
                self.game_state = GameState::Menu;
                self.release_game_cursor();
            }
            MenuAction::Connect => self.connect_to_server(),
            MenuAction::Quit => self.menu_state.quit_requested = true,
            MenuAction::ToggleRenderMode => self.toggle_present_mode(),
            MenuAction::ToggleWater => {
                self.graphics.water = self.graphics.water.toggled();
                self.graphics.save();
            }
            MenuAction::CycleAntiAliasing => {
                self.graphics.anti_aliasing = self.graphics.anti_aliasing.next();
                self.graphics.save();
            }
            MenuAction::ToggleShadows => {
                self.graphics.shadows = !self.graphics.shadows;
                self.graphics.save();
            }
            MenuAction::FocusUsername => self.menu_state.select_field(MenuField::Username),
            MenuAction::FocusServerAddress => {
                self.menu_state.select_field(MenuField::ServerAddress)
            }
        }
    }

    fn release_game_cursor(&mut self) {
        self.mouse_captured = false;
        self.input = Default::default();
        self.digging = Default::default();
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    fn resume_game(&mut self) {
        use crate::ui::menu::GameState;
        if !self.has_entered_world {
            return;
        }
        self.game_state = GameState::Playing;
        self.mouse_captured = true;
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
        self.window.set_cursor_visible(false);
    }

    pub fn save_world_to_disk(&mut self) -> bool {
        let world = self.world.read();
        let saved = minerust::SavedWorld::from_world_with_generator_version(
            &world.chunks,
            world.seed,
            (
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
            ),
            (self.camera.yaw, self.camera.pitch),
            &self.inventory,
            world.generator_version,
        );
        match save_world(DEFAULT_WORLD_FILE, &saved) {
            Ok(()) => {
                log(
                    LogLevel::Info,
                    &format!("World saved to {}", DEFAULT_WORLD_FILE),
                );
                self.menu_state.set_status("World saved successfully");
                self.menu_state.clear_error();
                true
            }
            Err(error) => {
                log(LogLevel::Error, &format!("Failed to save world: {}", error));
                self.menu_state
                    .set_error(&format!("Save failed: {}", error));
                false
            }
        }
    }

    pub fn load_world_from_disk(&mut self) -> bool {
        let saved = match load_world(DEFAULT_WORLD_FILE) {
            Ok(saved) => saved,
            Err(error) => {
                log(LogLevel::Error, &format!("Error loading world: {}", error));
                self.menu_state
                    .set_error(&format!("Load failed: {}", error));
                return false;
            }
        };

        log(
            LogLevel::Info,
            &format!("Regenerating world with seed {}...", saved.seed),
        );
        {
            let mut world = self.world.write();
            *world = World::new_with_seed_and_version(saved.seed, saved.generator_version);
        }
        self.indirect_manager.clear();
        self.water_indirect_manager.clear();
        self.camera.position.x = saved.player_x;
        self.camera.position.y = saved.player_y;
        self.camera.position.z = saved.player_z;
        self.camera.yaw = saved.player_yaw;
        self.camera.pitch = saved.player_pitch;
        self.inventory = saved.inventory.into_inventory(minerust::item_registry());
        super::state::ensure_test_lamp(&mut self.inventory);
        self.inventory_ui = Default::default();
        self.hotbar_dirty = true;

        {
            let mut world = self.world.write();
            for chunk_data in &saved.chunks {
                if let Some(chunk) = world.chunks.get_mut(&(chunk_data.cx, chunk_data.cz)) {
                    for (&sy, block_data) in &chunk_data.subchunks {
                        if let Some(subchunk) = chunk.subchunks.get_mut(sy as usize) {
                            let mut n = 0;
                            for lx in 0..CHUNK_SIZE as usize {
                                for ly in 0..SUBCHUNK_HEIGHT as usize {
                                    for lz in 0..CHUNK_SIZE as usize {
                                        if let Some(&block) = block_data.get(n) {
                                            subchunk.set_block_raw(
                                                lx as i32, ly as i32, lz as i32, block,
                                            );
                                            n += 1;
                                        }
                                    }
                                }
                            }
                            subchunk.is_empty = false;
                            subchunk.mark_mesh_dirty();
                        }
                    }
                    chunk.player_modified = true;
                    chunk.rebuild_metadata();
                }
            }
            for chunk in world.chunks.values_mut() {
                for subchunk in &mut chunk.subchunks {
                    subchunk.mark_mesh_dirty();
                }
            }
        }

        self.chunk_loader =
            minerust::ChunkLoader::new_with_version(saved.seed, saved.generator_version);
        self.mesh_loader =
            minerust::MeshLoader::new(self.world.clone(), minerust::get_mesh_worker_count());
        self.chunks_rendered = 0;
        self.subchunks_rendered = 0;
        self.hiz_valid = false;
        self.visible_chunk_columns.clear();
        self.visible_chunk_cache_center = (i32::MIN, i32::MIN);
        self.visible_chunk_columns_dirty = true;
        self.enqueue_all_dirty_meshes();
        self.has_entered_world = true;
        self.game_state = crate::ui::menu::GameState::Playing;
        self.menu_state.clear_error();
        self.menu_state.set_status("World loaded");
        self.mouse_captured = true;
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
        self.window.set_cursor_visible(false);
        log(
            LogLevel::Info,
            &format!(
                "World loaded from {} (seed: {})",
                DEFAULT_WORLD_FILE, saved.seed
            ),
        );
        true
    }

    /// Switches between automatic low-latency and display-synced presentation.
    /// AutoNoVsync falls back to a supported mode; Fifo is mandatory.
    fn toggle_present_mode(&mut self) {
        self.config.present_mode = match self.config.present_mode {
            wgpu::PresentMode::Fifo => minerust::DEFAULT_PRESENT_MODE,
            _ => wgpu::PresentMode::Fifo,
        };
        self.surface.configure(&self.device, &self.config);
    }

    pub(super) fn start_new_world(&mut self) {
        self.start_new_world_with_seed(None);
    }

    /// Starts a new world; `seed` fixes the world for reproducible runs
    /// (profiling, benchmarks), otherwise a random seed is used.
    pub(super) fn start_new_world_with_seed(&mut self, seed: Option<u32>) {
        let mut world = match seed {
            Some(seed) => World::new_empty_with_seed(seed),
            None => World::new(),
        };
        // Spawn selection needs the actual blocks (trees, ceilings, etc.),
        // not only the height-map. Generate the initial area before searching.
        let (spawn_x, spawn_z) = world.spawn_column();
        world.generate_chunks_in_radius(
            spawn_x.div_euclid(CHUNK_SIZE),
            spawn_z.div_euclid(CHUNK_SIZE),
            2,
        );
        let seed = world.seed;
        let spawn = world.find_spawn_point();
        let spawn_cx = (spawn.0 / CHUNK_SIZE as f32).floor() as i32;
        let spawn_cz = (spawn.2 / CHUNK_SIZE as f32).floor() as i32;

        {
            let mut world_lock = self.world.write();
            *world_lock = world;
            world_lock.generate_chunks_in_radius(spawn_cx, spawn_cz, 2);
        }
        self.chunk_loader = minerust::ChunkLoader::new(seed);
        self.mesh_loader =
            minerust::MeshLoader::new(self.world.clone(), minerust::get_mesh_worker_count());
        self.chunks_rendered = 0;
        self.subchunks_rendered = 0;
        self.enqueue_all_dirty_meshes();
        self.indirect_manager.clear();
        self.water_indirect_manager.clear();
        self.hiz_valid = false;
        self.visible_chunk_columns.clear();
        self.visible_chunk_cache_center = (i32::MIN, i32::MIN);
        self.visible_chunk_columns_dirty = true;
        self.highlighted_block = None;
        self.input = Default::default();
        self.digging = Default::default();
        self.placement = Default::default();
        self.inventory_open = false;
        self.inventory = Default::default();
        super::state::ensure_test_lamp(&mut self.inventory);
        self.hotbar_dirty = true;
        self.inventory_ui = Default::default();
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

        let stack = self.inventory.selected_stack()?;
        let block = minerust::item_registry()
            .get(stack.item)
            .placeable_block()?;

        self.record_line_placement(place_pos);
        self.placement.cooldown = BLOCK_PLACE_REPEAT_INTERVAL;

        Some((px, py, pz, block))
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
