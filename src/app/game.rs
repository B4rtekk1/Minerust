use std::time::Instant;

use clap::Parser;
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, WindowBuilder},
};

use minerust::{
    CHUNK_SIZE, DEFAULT_WORLD_FILE, SUBCHUNK_HEIGHT, SavedWorld, World, load_world, save_world,
};

use crate::logger::{LogLevel, log};
use crate::ui::menu::GameState;

use super::server::run_dedicated_server;
use super::state::State;

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Command-line arguments parsed by [`clap`] at startup.
///
/// Running with `--server` skips the windowed game entirely and starts a
/// headless TCP server instead.  The port can be overridden with `--port`.
///
/// # Examples
/// ```text
/// # Start a headless server on the default port
/// minerust --server
///
/// # Start a headless server on a custom port
/// minerust --server --port 12345
///
/// # Start the windowed game (default when no flags are given)
/// minerust
/// ```
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run as a headless dedicated server instead of opening a game window.
    #[arg(long, default_value_t = false)]
    server: bool,

    /// TCP port the dedicated server listens on.
    #[arg(long, default_value_t = 25565)]
    port: u16,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parses command-line arguments and either:
///
/// - **Dedicated server mode** (`--server`): blocks the calling thread on a
///   Tokio runtime that runs [`run_dedicated_server`] until the process is
///   killed.  No window is created.
///
/// - **Client / game mode** (default): creates a winit window, initializes
///   the wgpu [`State`] synchronously via `pollster::block_on`, then drives
///   the winit event loop until the window is closed.
///
/// # Event loop overview
///
/// | Event | Action |
/// |---|---|
/// | `Resized` | Rebuilds all resolution-dependent GPU resources. |
/// | `RedrawRequested` | Measures frame time, runs `update`, calls `render`. |
/// | `KeyboardInput` | Dispatches to menu or in-game key handlers (see below). |
/// | `MouseWheel` | Scrolls the hotbar slot selection. |
/// | `MouseInput` | Captures cursor on first in-game click; dispatches block actions. |
/// | `CursorMoved` | Tracks cursor position for menu hover/hit-testing. |
/// | `DeviceEvent::MouseMotion` | Rotates the camera when the cursor is captured. |
/// | `AboutToWait` | Switches to 30 fps throttle after 30 s of inactivity. |
/// | `CloseRequested` | Exits the event loop cleanly. |
///
/// # Key bindings (in-game)
///
/// | Key | Action |
/// |---|---|
/// | W / A / S / D | Move forward / left / backward / right. |
/// | Space | Jump. |
/// | Left Shift | Sprint. |
/// | 1–9 | Select hotbar slot. |
/// | Escape (mouse captured) | Release cursor without leaving the game. |
/// | Escape (mouse free) | Open the main menu. |
/// | F5 | Save world to disk. |
/// | F9 | Load world from disk. |
/// | F11 | Toggle borderless fullscreen. |
/// | R | Cycle water reflection mode (Off → SSR). |
///
/// # Key bindings (menu)
///
/// | Key | Action |
/// |---|---|
/// | Tab | Move focus to the next text field. |
/// | Enter | Attempt to connect to the server. |
/// | Escape | Dismiss the menu and resume play. |
/// | Backspace | Delete the last character in the active field. |
/// | F11 | Toggle borderless fullscreen. |
/// | Any printable character | Appended to the active text field. |
///
/// # Panics
/// Panics if the winit event loop or window cannot be created, or if the
/// Tokio runtime for the server cannot be initialized.
pub fn run_game() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // ── Dedicated server mode ─────────────────────────────────────────────── //
    if args.server {
        let addr = format!("0.0.0.0:{}", args.port);
        log(
            LogLevel::Info,
            &format!("Starting headless server on {}...", addr),
        );
        log(
            LogLevel::Info,
            "Note: This is a console-only server. No game window will appear.",
        );
        log(
            LogLevel::Info,
            "To play the game, run the application without --server.",
        );
        log(LogLevel::Info, "Press Ctrl+C to stop the server.");
        // Flush so the operator sees the startup messages immediately even
        // when stdout is piped (e.g., `minerust --server | tee server.log`).
        use std::io::Write;
        let _ = std::io::stdout().flush();

        // Block the main thread on the async server; `run_dedicated_server`
        // runs an infinite accept loop so this never returns normally.
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_dedicated_server(&addr));
        return Ok(());
    }

    // ── Windowed game mode ────────────────────────────────────────────────── //

    // Create the event loop. If creation can fail in this winit version,
    // log the error and return it; otherwise this simply constructs the
    // event loop value.
    let event_loop = match EventLoop::new() {
        // If EventLoop::new returns a Result in this winit version
        // handle Ok/Err.
        Ok(ev) => ev,
        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Failed to create event loop: {}", e),
            );
            return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    };

    // Build the window. On failure, log the error and return it instead
    // of panicking so the caller can handle it gracefully.
    let window = match WindowBuilder::new()
        .with_title("Minerust")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .with_transparent(true)
        // Start the window in borderless fullscreen on the current monitor.
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build(&event_loop)
    {
        Ok(w) => w,
        Err(e) => {
            log(LogLevel::Error, &format!("Failed to create window: {}", e));
            return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    };

    // `State::new` is async (wgpu adapter/device requests are futures), but
    // the rest of the game is synchronous; `pollster::block_on` bridges them
    // without pulling in a full async runtime for the client path.
    let mut state = pollster::block_on(State::new(window));

    event_loop
        .run(move |event, elwt| {
            // `Poll` by default so `AboutToWait` fires immediately after each
            // batch of OS events and we can request a redraw every frame.
            // Switched to `WaitUntil` when idle (see `AboutToWait` handler).
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                // ── Window resized ────────────────────────────────────────── //
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    state.resize(size);
                    state.window.request_redraw();
                }

                // ── Render frame ──────────────────────────────────────────── //
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let now = Instant::now();
                    // Wall-clock time between the last two redraws, used for
                    // the frame-time display in the debug HUD.
                    state.frame_time_ms =
                        now.duration_since(state.last_redraw).as_secs_f32() * 1000.0;
                    state.last_redraw = now;
                    state.frame_count += 1;

                    // Recompute FPS every 500 ms to avoid flickering.
                    let elapsed = now.duration_since(state.last_fps_update).as_secs_f32();
                    if elapsed >= 0.5 {
                        state.current_fps = state.frame_count as f32 / elapsed;
                        state.frame_count = 0;
                        state.last_fps_update = now;
                    }

                    // Run game logic (camera, physics, chunk uploads, networking).
                    let update_start = Instant::now();
                    state.update();
                    state.cpu_update_ms = update_start.elapsed().as_secs_f32() * 1000.0;

                    match state.render() {
                        Ok(_) => {}
                        // Surface lost (e.g., window un-minimized on some
                        // platforms): trigger a resize to reconfigure the
                        // swap-chain with the current window dimensions.
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size()),
                        // GPU out of memory: nothing reasonable to do here,
                        // so exit cleanly rather than panic.
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => log(LogLevel::Error, &format!("Render error: {:?}", e)),
                    }

                    // Request the next frame immediately (uncapped frame rate).
                    state.window.request_redraw();
                }

                // ── Keyboard input ────────────────────────────────────────── //
                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    physical_key: PhysicalKey::Code(key),
                                    state: key_state,
                                    text,
                                    ..
                                },
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = key_state == ElementState::Pressed;

                    // ---- Menu text-field input --------------------------------
                    // `text` carries the OS-processed character (respecting the
                    // current keyboard layout and dead-key composition) rather
                    // than the raw key code, so it handles accented characters
                    // and IME input transparently.
                    if state.game_state == GameState::Menu && pressed {
                        if let Some(ref txt) = text {
                            for ch in txt.chars() {
                                state.menu_state.handle_char(ch);
                            }
                        }
                    }

                    if state.game_state == GameState::Menu {
                        // ---- Menu navigation hotkeys -------------------------
                        if pressed {
                            match key {
                                KeyCode::Tab => {
                                    // Cycle focus: ServerAddress → Username → ServerAddress.
                                    state.menu_state.next_field();
                                }
                                KeyCode::Enter => {
                                    state.connect_to_server();
                                }
                                KeyCode::Escape => {
                                    // Dismiss the menu and return to the game
                                    // without disconnecting, and immediately
                                    // recapture the cursor for gameplay.
                                    state.game_state = GameState::Playing;
                                    state.mouse_captured = true;
                                    let _ = state
                                        .window
                                        .set_cursor_grab(CursorGrabMode::Confined)
                                        .or_else(|_| {
                                            state.window.set_cursor_grab(CursorGrabMode::Locked)
                                        });
                                    state.window.set_cursor_visible(false);
                                }
                                KeyCode::Backspace => {
                                    state.menu_state.handle_backspace();
                                }
                                KeyCode::F11 => {
                                    // Toggle borderless fullscreen.  Passing
                                    // `None` as the monitor lets winit pick the
                                    // monitor the window is currently on.
                                    if state.window.fullscreen().is_some() {
                                        state.window.set_fullscreen(None);
                                    } else {
                                        state.window.set_fullscreen(Some(
                                            winit::window::Fullscreen::Borderless(None),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                        // ---- In-game key bindings ----------------------------
                        match key {
                            // Movement – held state polled each frame by `update`.
                            KeyCode::KeyW => state.input.forward = pressed,
                            KeyCode::KeyS => state.input.backward = pressed,
                            KeyCode::KeyA => state.input.left = pressed,
                            KeyCode::KeyD => state.input.right = pressed,
                            KeyCode::Space => state.input.jump = pressed,
                            KeyCode::ShiftLeft => state.input.sprint = pressed,

                            KeyCode::Escape if pressed => {
                                // Escape always returns to the menu from gameplay.
                                // Release the cursor at the same time so the UI is
                                // immediately interactive.
                                state.game_state = GameState::Menu;
                                state.mouse_captured = false;
                                state.input = Default::default();
                                state.digging = Default::default();
                                let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                                state.window.set_cursor_visible(true);
                            }

                            KeyCode::F11 if pressed => {
                                if state.window.fullscreen().is_some() {
                                    state.window.set_fullscreen(None);
                                } else {
                                    state.window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                }
                            }

                            KeyCode::KeyR if pressed => {
                                // Cycle: 0 = Off, 1 = SSR.  Wraps with modulo
                                // so adding more modes in the future only
                                // requires extending the match arm below.
                                state.reflection_mode = (state.reflection_mode + 1) % 2;
                                let mode_name = match state.reflection_mode {
                                    0 => "Off",
                                    1 => "SSR",
                                    _ => "Unknown",
                                };
                                log(LogLevel::Info, &format!("Reflection mode: {}", mode_name));
                            }

                            // ---- F5: Save world to disk ---------------------
                            KeyCode::F5 if pressed => {
                                let world = state.world.read();
                                // `SavedWorld::from_world` serializes only the
                                // chunks that contain player-modified blocks, so
                                // procedurally-generated terrain can be
                                // regenerated from the seed on the next load.
                                let saved = SavedWorld::from_world(
                                    &world.chunks,
                                    world.seed,
                                    (
                                        state.camera.position.x,
                                        state.camera.position.y,
                                        state.camera.position.z,
                                    ),
                                    (state.camera.yaw, state.camera.pitch),
                                );
                                if let Err(e) = save_world(DEFAULT_WORLD_FILE, &saved) {
                                    log(LogLevel::Error, &format!("Failed to save world: {}", e));
                                } else {
                                    log(
                                        LogLevel::Info,
                                        &format!("World saved to {}", DEFAULT_WORLD_FILE),
                                    );
                                }
                            }

                            // ---- F9: Load world from disk -------------------
                            KeyCode::F9 if pressed => match load_world(DEFAULT_WORLD_FILE) {
                                Ok(saved) => {
                                    log(
                                        LogLevel::Info,
                                        &format!("Regenerating world with seed {}...", saved.seed),
                                    );

                                    // Reinitialize the world from the saved seed
                                    // so procedurally-generated terrain is
                                    // recreated, then overwrite individual blocks
                                    // with the serialized player edits below.
                                    {
                                        let mut world = state.world.write();
                                        *world = World::new_with_seed(saved.seed);
                                    }

                                    // Clear the indirect draw managers so they
                                    // don't hold stale GPU buffer references from
                                    // the previous world.
                                    state.indirect_manager.clear();
                                    state.water_indirect_manager.clear();

                                    // Restore camera transform.
                                    state.camera.position.x = saved.player_x;
                                    state.camera.position.y = saved.player_y;
                                    state.camera.position.z = saved.player_z;
                                    state.camera.yaw = saved.player_yaw;
                                    state.camera.pitch = saved.player_pitch;

                                    // Overwrite sub-chunk block data with the
                                    // serialized player edits.  Block data is
                                    // stored flat (x-major, then y, then z) in
                                    // the save file and must be unpacked in the
                                    // same order here.
                                    {
                                        let mut world = state.world.write();
                                        for chunk_data in &saved.chunks {
                                            let cx = chunk_data.cx;
                                            let cz = chunk_data.cz;
                                            for (&sy, block_data) in &chunk_data.subchunks {
                                                if let Some(chunk) = world.chunks.get_mut(&(cx, cz))
                                                {
                                                    if (sy as usize) < chunk.subchunks.len() {
                                                        let subchunk =
                                                            &mut chunk.subchunks[sy as usize];
                                                        // Fill blocks in x→y→z order to match
                                                        // the serialization order in save_world.
                                                        let mut n = 0;
                                                        for lx in 0..CHUNK_SIZE as usize {
                                                            for ly in 0..SUBCHUNK_HEIGHT as usize {
                                                                for lz in 0..CHUNK_SIZE as usize {
                                                                    if n < block_data.len() {
                                                                        subchunk.blocks[lx][ly]
                                                                            [lz] = block_data[n];
                                                                        n += 1;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        subchunk.is_empty = false;
                                                        subchunk.mesh_dirty = true;
                                                    }
                                                    chunk.player_modified = true;
                                                }
                                            }
                                        }
                                    }

                                    // Mark every sub-chunk dirty so the mesh
                                    // loader rebuilds all GPU geometry on the
                                    // next few frames (not just the edited ones).
                                    {
                                        let mut world = state.world.write();
                                        for chunk in world.chunks.values_mut() {
                                            for subchunk in &mut chunk.subchunks {
                                                subchunk.mesh_dirty = true;
                                            }
                                        }
                                    }
                                    log(
                                        LogLevel::Info,
                                        &format!(
                                            "World loaded from {} (seed: {})",
                                            DEFAULT_WORLD_FILE, saved.seed
                                        ),
                                    );
                                }
                                Err(e) => log(LogLevel::Error, &format!("Error loading: {}", e)),
                            },

                            // ---- Hotbar slot selection (1–9) ----------------
                            // Setting `hotbar_dirty` triggers a mesh rebuild of
                            // the hotbar UI on the next frame.
                            KeyCode::Digit1 if pressed => {
                                state.hotbar_slot = 0;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit2 if pressed => {
                                state.hotbar_slot = 1;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit3 if pressed => {
                                state.hotbar_slot = 2;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit4 if pressed => {
                                state.hotbar_slot = 3;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit5 if pressed => {
                                state.hotbar_slot = 4;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit6 if pressed => {
                                state.hotbar_slot = 5;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit7 if pressed => {
                                state.hotbar_slot = 6;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit8 if pressed => {
                                state.hotbar_slot = 7;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit9 if pressed => {
                                state.hotbar_slot = 8;
                                state.hotbar_dirty = true;
                            }
                            _ => {}
                        }
                    }
                }

                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta, .. },
                    ..
                } => {
                    if state.game_state != GameState::Menu {
                        // Normalize both scroll variants to a signed `f32`:
                        // `LineDelta` gives lines directly; `PixelDelta` (used
                        // by high-DPI trackpads) is divided by 20 to produce a
                        // comparable magnitude.
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                        };
                        let slots = crate::ui::ui::HOTBAR_SLOTS.len() as i32;
                        // `rem_euclid` wraps correctly for negative values
                        // (scrolling backward past slot 0 lands on the last slot).
                        let new_slot = (state.hotbar_slot as i32 - scroll.signum() as i32)
                            .rem_euclid(slots) as usize;
                        if new_slot != state.hotbar_slot {
                            state.hotbar_slot = new_slot;
                            state.hotbar_dirty = true;
                        }
                    }
                }

                // ── Mouse button input ────────────────────────────────────── //
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            state: btn_state,
                            button,
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = btn_state == ElementState::Pressed;

                    if state.game_state == GameState::Menu {
                        // In the menu, only left-click is handled; it is forwarded
                        // to the hit-test system which maps the cursor position to
                        // a `MenuHit` variant.
                        if pressed && button == winit::event::MouseButton::Left {
                            if let Some((x, y)) = state.cursor_position {
                                state.handle_menu_click(x, y);
                            }
                        }
                    } else if pressed && !state.mouse_captured {
                        // First click in-game captures the cursor so subsequent
                        // mouse motion is routed to camera rotation.
                        // `Confined` is tried first (keeps cursor inside the
                        // window), falling back to `Locked` (OS-level cursor
                        // lock) on platforms that don't support `Confined`.
                        state.mouse_captured = true;
                        let _ = state
                            .window
                            .set_cursor_grab(CursorGrabMode::Confined)
                            .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                        state.window.set_cursor_visible(false);
                        // Re-center the cursor so the camera doesn't jump on
                        // the first motion event after capture.
                        let _ = state.window.set_cursor_position(PhysicalPosition::new(
                            state.config.width / 2,
                            state.config.height / 2,
                        ));
                    } else {
                        // Cursor already captured: forward to the block
                        // interaction handler (left = dig, right = place).
                        state.handle_mouse_input(button, pressed);
                    }
                }

                // ── Cursor position tracking ──────────────────────────────── //
                // Updated unconditionally so menu hover effects and hit-testing
                // always have an up-to-date position even while the menu is open.
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    state.cursor_position = Some((position.x as f32, position.y as f32));
                }

                // ── Raw mouse motion → camera rotation ────────────────────── //
                // `DeviceEvent::MouseMotion` reports raw (un-accelerated) delta
                // values that are unaffected by OS cursor acceleration curves,
                // giving consistent feel across different pointer speed settings.
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    if state.mouse_captured {
                        let sensitivity = 0.002; // radians per pixel
                        state.camera.yaw += delta.0 as f32 * sensitivity;
                        // Subtract because a downward mouse movement (positive Y
                        // on most OS conventions) should pitch the camera down
                        // (decreasing pitch in our coordinate system).
                        state.camera.pitch -= delta.1 as f32 * sensitivity;
                        // Clamp slightly inside ±π/2 to avoid gimbal lock and
                        // the degenerate case where `look_at` produces a zero
                        // vector when the camera faces straight up or down.
                        state.camera.pitch = state.camera.pitch.clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.1,
                            std::f32::consts::FRAC_PI_2 - 0.1,
                        );
                    }
                }

                // ── Idle throttle ─────────────────────────────────────────── //
                // When no input has been received for 30 seconds the event loop
                // switches from `Poll` (busy-wait, uncapped FPS) to `WaitUntil`
                // (wake at most every ~33 ms, ~30 fps) to reduce CPU and GPU
                // load while the window is open but unused.  Any input event
                // resets `last_input_time`, causing the next `AboutToWait` to
                // switch back to `Poll` automatically.
                Event::AboutToWait => {
                    let is_idle = state.last_input_time.elapsed().as_secs() >= 30;
                    if is_idle {
                        let next_frame = Instant::now() + std::time::Duration::from_millis(33);
                        elwt.set_control_flow(ControlFlow::WaitUntil(next_frame));
                    } else {
                        elwt.set_control_flow(ControlFlow::Poll);
                    }
                    state.window.request_redraw();
                }

                // ── Window close button ───────────────────────────────────── //
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => elwt.exit(),

                _ => {}
            }
        })
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok(())
}
