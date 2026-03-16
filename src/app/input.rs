use render3d::camera::check_intersection;
use winit::event::MouseButton;

use crate::ui::ui::HOTBAR_SLOTS;

use super::state::State;

impl State {
    pub fn handle_mouse_input(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.input.left_mouse = pressed,
            MouseButton::Right => self.input.right_mouse = pressed,
            _ => {}
        }

        if !self.mouse_captured {
            return;
        }

        if button == MouseButton::Right && pressed {
            let target = self.camera.raycast(&*self.world.read(), 5.0);
            if let Some((_, _, _, px, py, pz)) = target {
                if self.camera.intersects_block(px, py, pz) {
                    return;
                }

                for player in self.remote_players.values() {
                    let player_pos = cgmath::Point3::new(player.x, player.y, player.z);
                    if check_intersection(player_pos, px, py, pz) {
                        return;
                    }
                }

                let block_to_place = HOTBAR_SLOTS[self.hotbar_slot];
                self.world
                    .write()
                    .set_block_player(px, py, pz, block_to_place);
                self.mark_chunk_dirty(px, py, pz);
            }
        }
    }

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
