use crate::multiplayer::player::RemotePlayer;
use crate::multiplayer::protocol::{Packet, decode_pitch, decode_yaw};
use crate::multiplayer::tcp::TcpClient;
use crate::ui::menu::{GameState, MenuState};
use std::time::Instant;
use winit::window::Window;

// ─────────────────────────────────────────────────────────────────────────────
// connect_to_server
// ─────────────────────────────────────────────────────────────────────────────

/// Initiates an asynchronous connection to the multiplayer server and wires
/// up the two mpsc channels that the rest of the game uses to exchange packets.
///
/// # Architecture
///
/// After a successful TCP handshake, two Tokio tasks are spawned on the
/// shared `network_runtime`:
///
/// ```text
///  Tokio runtime
///  ┌─────────────────────────────────────────────────────────┐
///  │  recv task: client_rx.recv() ──→ rx_tx (network_rx)     │
///  │  send task: tx_rx (network_tx) ──→ client_tx.send()     │
///  └─────────────────────────────────────────────────────────┘
///
///  Game loop (main thread)
///    network_rx ← packets arriving from server
///    network_tx → packets to send to server
/// ```
///
/// Both tasks exit when their channel endpoint is closed, which happens
/// automatically when `State` (and thus `network_tx`/`network_rx`) is dropped.
///
/// Once the channels are in place, an initial `Connect` packet is queued so
/// the server receives the player's username and assigns a `player_id`.  The
/// resulting `ConnectAck` is processed by [`update_network`] on the next frame.
///
/// # Menu state effects
/// - On success: `menu_state` shows a "Connecting…" status; `game_state` is
///   set to `GameState::Connecting` so the render loop continues without
///   blocking.
/// - On failure: `menu_state` shows the OS error string and `game_state` is
///   left unchanged (stays in `GameState::Menu`).
///
/// # Parameters
/// - `menu_state`       – Source of the server address and username; also
///                        receives status/error messages.
/// - `game_state`       – Transitioned to `Connecting` on a successful TCP
///                        handshake.
/// - `network_runtime`  – The shared Tokio runtime on which the two I/O tasks
///                        are spawned.  If `None` the function is a no-op.
/// - `network_rx`       – Written with the game-loop-facing receive channel on
///                        success, replacing any previous value.
/// - `network_tx`       – Written with the game-loop-facing send channel on
///                        success, replacing any previous value.
pub fn connect_to_server(
    menu_state: &mut MenuState,
    game_state: &mut GameState,
    network_runtime: &Option<tokio::runtime::Runtime>,
    network_rx: &mut Option<tokio::sync::mpsc::UnboundedReceiver<Packet>>,
    network_tx: &mut Option<tokio::sync::mpsc::UnboundedSender<Packet>>,
) {
    // Clone the strings up front so they can be moved into the async block
    // without creating a borrow conflict with `menu_state`.
    let addr     = menu_state.server_address.clone();
    let username = menu_state.username.clone();

    menu_state.set_status(&format!("Connecting to {}...", addr));
    *game_state = GameState::Connecting;

    if let Some(rt) = network_runtime {
        // `block_on` runs the TCP handshake synchronously on the calling
        // thread.  This is acceptable here because `connect_to_server` is
        // called from a menu button click, not from the render loop.
        let result = rt.block_on(async {
            let mut client = TcpClient::new();
            client.connect(&addr).await?;
            Ok::<TcpClient, std::io::Error>(client)
        });

        match result {
            Ok(client) => {
                println!("Connected to server: {}", addr);

                // ── Channel setup ─────────────────────────────────────────── //
                // `rx_rx` → polled each frame by `update_network` to drain
                //           arriving packets.
                // `tx_tx` → written by `update_network` and the typed send
                //           helpers to queue outgoing packets.
                let (rx_tx, rx_rx) = tokio::sync::mpsc::unbounded_channel();
                let (tx_tx, mut tx_rx) = tokio::sync::mpsc::unbounded_channel();
                *network_rx = Some(rx_rx);
                *network_tx = Some(tx_tx);

                // ── Receive task ──────────────────────────────────────────── //
                // Reads packets from the TCP stream and forwards them to the
                // game loop via `rx_tx`.  Exits when the TCP stream closes or
                // the game loop drops `network_rx` (closing `rx_tx`).
                let client_rx = client.clone();
                rt.spawn(async move {
                    while let Ok(packet) = client_rx.recv().await {
                        if rx_tx.send(packet).is_err() {
                            break; // game loop has shut down; exit cleanly
                        }
                    }
                });

                // ── Send task ─────────────────────────────────────────────── //
                // Drains `tx_rx` and writes each packet to the TCP stream.
                // Exits when the game loop drops `network_tx` (closing `tx_rx`)
                // or when the TCP write fails.
                let client_tx = client.clone();
                rt.spawn(async move {
                    while let Some(packet) = tx_rx.recv().await {
                        if client_tx.send(&packet).await.is_err() {
                            break; // server disconnected or stream broken
                        }
                    }
                });

                // Send the initial Connect packet.  The server will overwrite
                // `player_id: 0` with the real assigned ID and reply with
                // `ConnectAck`, which `update_network` processes.
                let connect_packet = Packet::Connect {
                    player_id: 0,
                    username,
                };
                if let Some(tx) = network_tx {
                    let _ = tx.send(connect_packet);
                }
            }
            Err(e) => {
                eprintln!("Failed to connect: {}", e);
                menu_state.set_error(&format!("Connection failed: {}", e));
                // `game_state` is left in its previous value (Menu) so the
                // player can retry after correcting the address.
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// update_network
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame network update: sends the local player's position/rotation and
/// processes all packets that have arrived since the last frame.
///
/// This function is called once per `RedrawRequested` event from the game loop.
/// It is intentionally synchronous and non-blocking — it only calls
/// `try_recv`, never `await`, so it cannot stall the render loop.
///
/// # Position/rotation throttle
///
/// Position and rotation packets are sent at most once every 50 ms (20 Hz).
/// This rate provides smooth remote-player movement without flooding the
/// server or saturating the uplink on slow connections.  The throttle is
/// controlled by `last_position_send`.
///
/// Rotation values are quantized to `u8` before sending
/// (`encode_yaw`/`encode_pitch`) and decoded back to `f32` when received
/// (`decode_yaw`/`decode_pitch`), trading a small angular precision loss
/// (~1.4°) for a 75% reduction in per-packet rotation size.
///
/// # Packet handling
///
/// | Packet | Action |
/// |---|---|
/// | `ConnectAck { success: true }` | Store assigned `player_id`, transition to `Playing`, capture mouse cursor. |
/// | `ConnectAck { success: false }` | Transition back to `Menu`. |
/// | `Position` | Update or insert the remote player's position (self is filtered out). |
/// | `Rotation` | Update the remote player's yaw/pitch after decoding. |
/// | `Connect` | Insert or update the remote player's username (used as "player joined" event). |
/// | `Disconnect` | Remove the remote player from the map. |
/// | All other packets | Silently ignored (`_ => {}`). |
///
/// # Parameters
/// - `my_player_id`        – The server-assigned local player ID; updated on `ConnectAck`.
/// - `camera_pos`          – Current camera world position, sent as the local player's position.
/// - `camera_yaw`          – Current camera yaw in radians.
/// - `camera_pitch`        – Current camera pitch in radians.
/// - `last_position_send`  – Timestamp of the last position/rotation send; reset to `Instant::now()` after each send.
/// - `network_tx`          – Send channel to the Tokio send task; `None` when not connected.
/// - `network_rx`          – Receive channel from the Tokio receive task; `None` when not connected.
/// - `remote_players`      – Live map of all known remote players; mutated by Position, Rotation, Connect, Disconnect packets.
/// - `game_state`          – Transitioned to `Playing` on `ConnectAck { success: true }` or back to `Menu` on failure.
/// - `mouse_captured`      – Set to `true` when the game transitions to `Playing` so mouse delta drives camera rotation.
/// - `window`              – Used to lock the OS cursor when transitioning to `Playing`.
pub fn update_network(
    my_player_id: &mut u32,
    camera_pos: &glam::Vec3,
    camera_yaw: f32,
    camera_pitch: f32,
    last_position_send: &mut Instant,
    network_tx: &Option<tokio::sync::mpsc::UnboundedSender<Packet>>,
    network_rx: &mut Option<tokio::sync::mpsc::UnboundedReceiver<Packet>>,
    remote_players: &mut std::collections::HashMap<u32, RemotePlayer>,
    game_state: &mut GameState,
    mouse_captured: &mut bool,
    window: &Window,
) {
    // ── Outgoing: position and rotation (throttled to 20 Hz) ─────────────── //
    if last_position_send.elapsed().as_millis() > 50 {
        *last_position_send = Instant::now();

        let pos_packet = Packet::Position {
            player_id: *my_player_id,
            x: camera_pos.x,
            y: camera_pos.y,
            z: camera_pos.z,
        };

        // Rotation is quantized to u8 before sending; the server echoes the
        // quantized values verbatim to other clients, who decode them back.
        let rot_packet = Packet::Rotation {
            player_id: *my_player_id,
            yaw:   crate::multiplayer::protocol::encode_yaw(camera_yaw),
            pitch: crate::multiplayer::protocol::encode_pitch(camera_pitch),
        };

        if let Some(tx) = network_tx {
            // Send errors are non-fatal: if the channel is closed the
            // disconnect will be detected on the receive side via EOF.
            let _ = tx.send(pos_packet);
            let _ = tx.send(rot_packet);
        }
    }

    // ── Incoming: drain all packets that arrived this frame ───────────────── //
    // `try_recv` is non-blocking; the loop exits immediately when the queue
    // is empty rather than waiting for the next packet.
    if let Some(rx) = network_rx.as_mut() {
        while let Ok(packet) = rx.try_recv() {
            match packet {
                // ---- ConnectAck: server accepted or rejected our Connect ---- //
                Packet::ConnectAck { success, player_id } => {
                    if success {
                        *my_player_id = player_id;
                        println!("Joined as Player ID: {}", player_id);
                        *game_state = GameState::Playing;

                        // Capture the cursor so mouse motion drives the camera.
                        // Try `Confined` first (keeps cursor within window bounds)
                        // and fall back to `Locked` on platforms that don't
                        // support confinement.
                        *mouse_captured = true;
                        let _ = window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .or_else(|_| {
                                window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                            });
                        window.set_cursor_visible(false);
                    } else {
                        // Server rejected the connection (e.g., username taken,
                        // server full); return to the menu so the player can retry.
                        *game_state = GameState::Menu;
                    }
                }

                // ---- Position: a remote player moved ----------------------- //
                Packet::Position { player_id, x, y, z } => {
                    // Filter out echoed packets for the local player.
                    if player_id != *my_player_id {
                        if let Some(player) = remote_players.get_mut(&player_id) {
                            player.x = x;
                            player.y = y;
                            player.z = z;
                        } else {
                            // First position packet for a player we haven't seen
                            // yet — create a placeholder entry.  The username will
                            // be updated when the corresponding `Connect` packet
                            // arrives (which may already be queued behind this one).
                            remote_players.insert(
                                player_id,
                                RemotePlayer {
                                    x,
                                    y,
                                    z,
                                    yaw:   0.0,
                                    pitch: 0.0,
                                    username: format!("Player{}", player_id),
                                },
                            );
                        }
                    }
                }

                // ---- Rotation: a remote player turned ---------------------- //
                Packet::Rotation { player_id, yaw, pitch } => {
                    if player_id != *my_player_id {
                        if let Some(player) = remote_players.get_mut(&player_id) {
                            player.yaw   = decode_yaw(yaw);
                            player.pitch = decode_pitch(pitch);
                        }
                        // If the player is unknown we silently drop the rotation
                        // packet; the next Position packet will create the entry.
                    }
                }

                // ---- Connect: a remote player (or us) joined --------------- //
                // The server broadcasts a `Connect` packet to all peers when a
                // new player joins.  We use this as a "player joined" event to
                // attach the real username to the remote player's entry.
                Packet::Connect { player_id, username } => {
                    println!("Player joined: {} (ID: {})", username, player_id);
                    if let Some(player) = remote_players.get_mut(&player_id) {
                        player.username = username;
                    } else {
                        // Player may not have sent a Position yet; create an
                        // entry at a sensible default world position.
                        remote_players.insert(
                            player_id,
                            RemotePlayer {
                                x:     0.0,
                                y:     70.0, // above ground level, not inside terrain
                                z:     0.0,
                                yaw:   0.0,
                                pitch: 0.0,
                                username,
                            },
                        );
                    }
                }

                // ---- Disconnect: a remote player left ---------------------- //
                Packet::Disconnect { player_id } => {
                    remote_players.remove(&player_id);
                    println!("Player left: ID {}", player_id);
                }

                // Other packet types (BlockChange, Chat, Pong, etc.) are not
                // yet handled in this path; they can be added here as needed.
                _ => {}
            }
        }
    }
}