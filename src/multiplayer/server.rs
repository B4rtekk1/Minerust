#![allow(dead_code)]

use crate::multiplayer::protocol::{Packet, PlayerId};
use crate::multiplayer::transport::TransportType;
use std::collections::HashMap;
use std::io::Result;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};

/// Server-side snapshot of a connected player's state.
///
/// Kept in memory for the lifetime of the player's session and updated
/// as [`Packet::Position`] and [`Packet::Rotation`] packets arrive.
#[derive(Debug, Clone)]
pub struct PlayerInfo {
    /// Unique ID assigned to this player by the server.
    pub id: PlayerId,
    /// Player's chosen display name, received in [`Packet::Connect`].
    pub username: String,
    /// World-space X coordinate.
    pub x: f32,
    /// World-space Y coordinate (vertical axis).
    pub y: f32,
    /// World-space Z coordinate.
    pub z: f32,
    /// Compressed yaw angle. `0`–`255` maps to `0°`–`360°`.
    /// See `encode_yaw` / `decode_yaw` in the protocol module.
    pub yaw: u8,
    /// Compressed pitch angle. `0`–`255` maps to `-90°`–`+90°`.
    /// See `encode_pitch` / `decode_pitch` in the protocol module.
    pub pitch: u8,
}

/// High-level game events emitted by [`GameServer::handle_packet`].
///
/// Consumers (e.g. a game loop or relay broadcaster) obtain these by
/// calling [`GameServer::take_event_receiver`] and awaiting the channel.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// A new player has joined. Carries their [`PlayerId`] and username.
    PlayerConnected(PlayerId, String),
    /// A player has left or been forcibly removed. Carries their [`PlayerId`].
    PlayerDisconnected(PlayerId),
    /// A player's world position has changed. Fields: `(id, x, y, z)`.
    PlayerMoved(PlayerId, f32, f32, f32),
    /// A player's look direction has changed. Fields: `(id, yaw, pitch)`.
    /// Both angles use the compressed byte encoding from the protocol module.
    PlayerRotated(PlayerId, u8, u8),
    /// A block in the world has been placed or broken.
    /// Fields: `(x, y, z, block_type)`.
    BlockChanged(i32, i32, i32, u8),
    /// A player has sent a chat message. Fields: `(id, message)`.
    ChatMessage(PlayerId, String),
}

/// Core multiplayer server: maintains connected player state and converts
/// incoming [`Packet`]s into [`ServerEvent`]s.
///
/// `GameServer` is intentionally transport-agnostic — it receives already-
/// decoded packets and emits events over an unbounded channel, leaving
/// network I/O to the transport layer. Shared player state is protected by
/// an [`RwLock`] so multiple async tasks can read concurrently while writes
/// remain exclusive.
pub struct GameServer {
    /// The network transport this server is running on (TCP, UDP, …).
    transport_type: TransportType,
    /// Thread-safe map of all currently connected players, keyed by player ID.
    players: Arc<RwLock<HashMap<PlayerId, PlayerInfo>>>,
    /// Sender half of the server-event channel. Cloned into async tasks as needed.
    event_tx: mpsc::UnboundedSender<ServerEvent>,
    /// Receiver half of the server-event channel.
    /// Stored as `Option` so it can be moved out exactly once via
    /// [`GameServer::take_event_receiver`].
    event_rx: Option<mpsc::UnboundedReceiver<ServerEvent>>,
}

impl GameServer {
    /// Creates a new `GameServer` that will accept connections over
    /// `transport_type`.
    ///
    /// Initialises the internal event channel and an empty player map.
    /// Call [`GameServer::take_event_receiver`] to obtain the event stream
    /// before starting to process packets.
    pub fn new(transport_type: TransportType) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            transport_type,
            players: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Some(event_rx),
        }
    }

    /// Moves the [`ServerEvent`] receiver out of the server.
    ///
    /// Returns `Some(rx)` on the first call and `None` on every subsequent
    /// call. The caller is responsible for driving the receiver (e.g. in a
    /// dedicated async task) to avoid the internal channel buffer filling up.
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<ServerEvent>> {
        self.event_rx.take()
    }

    /// Returns the [`TransportType`] this server was configured with.
    pub fn transport_type(&self) -> TransportType {
        self.transport_type
    }

    /// Processes a single decoded packet received from `player_id`.
    ///
    /// Updates the in-memory player state where relevant and publishes a
    /// corresponding [`ServerEvent`] to the event channel. Unrecognised
    /// packet variants (e.g. `Ping`/`Pong`) are silently ignored.
    ///
    /// | Packet variant   | State mutation          | Event emitted             |
    /// |------------------|-------------------------|---------------------------|
    /// | `Connect`        | Inserts player at spawn | `PlayerConnected`         |
    /// | `Position`       | Updates `x`, `y`, `z`  | `PlayerMoved`             |
    /// | `Rotation`       | Updates `yaw`, `pitch`  | `PlayerRotated`           |
    /// | `BlockChange`    | —                       | `BlockChanged`            |
    /// | `Chat`           | —                       | `ChatMessage`             |
    /// | `Disconnect`     | Removes player          | `PlayerDisconnected`      |
    ///
    /// New players spawned by `Connect` are placed at `(0, 64, 0)` facing
    /// north (`yaw = 0`, `pitch = 128` ≈ horizontal).
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok(())`. The `Result` return type is
    /// reserved for future transport-level error propagation.
    pub async fn handle_packet(&self, player_id: PlayerId, packet: Packet) -> Result<()> {
        match packet {
            Packet::Connect { username, .. } => {
                let player = PlayerInfo {
                    id: player_id,
                    username: username.clone(),
                    x: 0.0,
                    y: 64.0,
                    z: 0.0,
                    yaw: 0,
                    pitch: 128, // ~horizontal: maps to 0° pitch
                };

                {
                    let mut players = self.players.write().await;
                    players.insert(player_id, player);
                }

                let _ = self
                    .event_tx
                    .send(ServerEvent::PlayerConnected(player_id, username));
            }

            Packet::Position { x, y, z, .. } => {
                {
                    let mut players = self.players.write().await;
                    if let Some(player) = players.get_mut(&player_id) {
                        player.x = x;
                        player.y = y;
                        player.z = z;
                    }
                }

                let _ = self
                    .event_tx
                    .send(ServerEvent::PlayerMoved(player_id, x, y, z));
            }

            Packet::Rotation { yaw, pitch, .. } => {
                {
                    let mut players = self.players.write().await;
                    if let Some(player) = players.get_mut(&player_id) {
                        player.yaw = yaw;
                        player.pitch = pitch;
                    }
                }

                let _ = self
                    .event_tx
                    .send(ServerEvent::PlayerRotated(player_id, yaw, pitch));
            }

            Packet::BlockChange {
                x,
                y,
                z,
                block_type,
            } => {
                let _ = self
                    .event_tx
                    .send(ServerEvent::BlockChanged(x, y, z, block_type));
            }

            Packet::Chat { message, .. } => {
                let _ = self
                    .event_tx
                    .send(ServerEvent::ChatMessage(player_id, message));
            }

            Packet::Disconnect { .. } => {
                {
                    let mut players = self.players.write().await;
                    players.remove(&player_id);
                }

                let _ = self
                    .event_tx
                    .send(ServerEvent::PlayerDisconnected(player_id));
            }

            // Ping, Pong, ConnectAck and any future variants are
            // handled at the transport layer or are server-originated;
            // nothing to do here.
            _ => {}
        }

        Ok(())
    }

    /// Forcibly removes a player from the server without requiring a
    /// [`Packet::Disconnect`] from the client.
    ///
    /// Emits a [`ServerEvent::PlayerDisconnected`] event. Intended for use
    /// when a transport-level error or timeout is detected by the network
    /// layer.
    pub async fn remove_player(&self, player_id: PlayerId) {
        let mut players = self.players.write().await;
        players.remove(&player_id);
        let _ = self
            .event_tx
            .send(ServerEvent::PlayerDisconnected(player_id));
    }

    /// Returns a snapshot of all currently connected players.
    ///
    /// Acquires a read lock, clones every [`PlayerInfo`], and releases the
    /// lock. The returned `Vec` is a point-in-time copy and may be stale by
    /// the time the caller uses it.
    pub async fn get_players(&self) -> Vec<PlayerInfo> {
        let players = self.players.read().await;
        players.values().cloned().collect()
    }

    /// Returns the number of currently connected players.
    ///
    /// Acquires a brief read lock on the player map.
    pub async fn player_count(&self) -> usize {
        self.players.read().await.len()
    }
}

/// Configuration used to bind and start a [`GameServer`].
///
/// Construct directly or via [`ServerConfig::default`], then pass to the
/// server startup function in the transport layer.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// IP address to listen on. Use `"0.0.0.0"` to accept connections on
    /// all interfaces, or `"127.0.0.1"` for loopback-only.
    pub address: String,
    /// TCP/UDP port to listen on.
    pub port: u16,
    /// Network transport to use (TCP, UDP, …).
    pub transport: TransportType,
    /// Maximum number of simultaneously connected players. Connections beyond
    /// this limit should be rejected with a failed [`Packet::ConnectAck`].
    pub max_players: usize,
}

impl Default for ServerConfig {
    /// Returns a sensible default configuration:
    /// - Binds to all interfaces (`0.0.0.0`)
    /// - Port `25565` (conventional Minecraft-style game port)
    /// - TCP transport
    /// - Up to `100` concurrent players
    fn default() -> Self {
        Self {
            address: "0.0.0.0".to_string(),
            port: 25565,
            transport: TransportType::Tcp,
            max_players: 100,
        }
    }
}

impl ServerConfig {
    /// Formats the configured address and port as a single `"host:port"` string
    /// suitable for passing to [`tokio::net::TcpListener::bind`] or equivalent.
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}