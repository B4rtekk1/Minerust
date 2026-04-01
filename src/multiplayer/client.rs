#![allow(dead_code)]

use crate::multiplayer::protocol::{Packet, PlayerId};
use crate::multiplayer::tcp::TcpClient;
use crate::multiplayer::transport::TransportType;
use std::io::{Error, ErrorKind, Result};
use tokio::sync::mpsc;

// ─────────────────────────────────────────────────────────────────────────────
// ClientEvent
// ─────────────────────────────────────────────────────────────────────────────

/// High-level events produced by [`GameClient::handle_packet`] and consumed by
/// the game loop via the unbounded channel returned from
/// [`GameClient::take_event_receiver`].
///
/// Events are a decoded, game-facing view of the raw [`Packet`] stream.
/// Separating them from packets lets the game loop remain independent of
/// protocol details and makes it easy to add new transports without changing
/// the event-consumption code.
#[derive(Debug, Clone)]
pub enum ClientEvent {
    /// The local player successfully connected and was assigned `player_id` and world `seed`.
    Connected(PlayerId, u32),
    /// The local player's connection was closed (graceful or error).
    Disconnected,
    /// A remote player joined the session with the given username.
    PlayerJoined(PlayerId, String),
    /// A remote player left the session.
    PlayerLeft(PlayerId),
    /// A remote player moved to world position `(x, y, z)`.
    PlayerMoved(PlayerId, f32, f32, f32),
    /// A remote player rotated; yaw and pitch are quantized to `u8`
    /// (0 = 0°, 255 ≈ 359°) to reduce bandwidth.
    PlayerRotated(PlayerId, u8, u8),
    /// A block at `(x, y, z)` was changed to `block_type`.
    BlockChanged(i32, i32, i32, u8),
    /// A chat message arrived from `player_id`.
    ChatMessage(PlayerId, String),
    /// A pong response arrived; `timestamp` is the Unix-millisecond value
    /// echoed from the matching `Ping` packet for round-trip time calculation.
    Pong(u64),
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionState
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks the lifecycle of a [`GameClient`] connection.
///
/// State transitions:
/// ```text
/// Disconnected → Connecting → Connected
///                           ↘ Disconnected (on rejection / error)
/// Connected    → Reconnecting → Connected
///                             ↘ Disconnected
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// No active connection; the client has never connected or has fully shut down.
    Disconnected,
    /// TCP handshake and `Connect`/`ConnectAck` exchange are in progress.
    Connecting,
    /// `ConnectAck { success: true }` was received; packets can be sent and received.
    Connected,
    /// The connection was lost and the client is attempting to re-establish it.
    Reconnecting,
}

// ─────────────────────────────────────────────────────────────────────────────
// GameClient
// ─────────────────────────────────────────────────────────────────────────────

/// Async multiplayer client that connects to a [`run_dedicated_server`] instance.
///
/// # Architecture
///
/// `GameClient` wraps a transport-specific connection (currently only
/// [`TcpClient`]) behind a uniform API.  The `TransportType` enum allows
/// future UDP or WebSocket transports to be added without changing the callers.
///
/// Decoded packets are forwarded as [`ClientEvent`] values through an internal
/// unbounded `tokio::mpsc` channel.  The receiving end is returned once via
/// [`take_event_receiver`] and then owned by the game loop, which drains it
/// each frame.  This decoupling means the game loop never blocks on network I/O.
///
/// # Typical usage
///
/// ```rust
/// let mut client = GameClient::new(TransportType::Tcp);
/// let event_rx = client.take_event_receiver().unwrap();
///
/// client.connect("127.0.0.1:25565", "Alice").await?;
///
/// // In the game loop:
/// while let Ok(event) = event_rx.try_recv() { /* handle event */ }
/// client.send_position(x, y, z).await?;
/// ```
pub struct GameClient {
    /// Which transport layer to use for all send/receive operations.
    transport_type: TransportType,
    /// Current lifecycle state of the connection.
    state: ConnectionState,
    /// The server-assigned player ID, populated after a successful `ConnectAck`.
    player_id: Option<PlayerId>,
    /// Active TCP connection, present only while `state == Connected`.
    tcp_client: Option<TcpClient>,
    /// Sender half of the event channel; kept on `GameClient` so
    /// `handle_packet` can push events without access to the receiver.
    event_tx: mpsc::UnboundedSender<ClientEvent>,
    /// Receiver half of the event channel.  Wrapped in `Option` so it can be
    /// moved out exactly once via [`take_event_receiver`].
    event_rx: Option<mpsc::UnboundedReceiver<ClientEvent>>,
}

impl GameClient {
    /// Creates a new, disconnected client configured to use `transport_type`.
    ///
    /// The internal event channel is created here; call [`take_event_receiver`]
    /// immediately after construction to obtain the receiving end before
    /// starting the connection.
    pub fn new(transport_type: TransportType) -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            transport_type,
            state: ConnectionState::Disconnected,
            player_id: None,
            tcp_client: None,
            event_tx,
            event_rx: Some(event_rx),
        }
    }

    /// Returns the event receiver, consuming it from `self`.
    ///
    /// This can only succeed once per `GameClient` instance.  The returned
    /// receiver should be handed to the game loop before [`connect`] is called
    /// so that no events are lost.  Returns `None` on subsequent calls.
    pub fn take_event_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<ClientEvent>> {
        self.event_rx.take()
    }

    /// Establishes a connection to the server at `address` and completes the
    /// `Connect` / `ConnectAck` handshake.
    ///
    /// # Handshake protocol
    ///
    /// 1. A TCP connection is opened to `address`.
    /// 2. A `Packet::Connect { player_id: 0, username }` is sent (the server
    ///    ignores the client-supplied `player_id` and assigns its own).
    /// 3. The server responds with `Packet::ConnectAck { success, player_id }`.
    ///    - On `success = true`: the assigned `player_id` is stored and a
    ///      `ClientEvent::Connected` is pushed to the event channel.
    ///    - On `success = false`: returns `Err(PermissionDenied)`.
    ///    - On any other packet: returns `Err(InvalidData)`.
    ///
    /// # Errors
    /// Propagates I/O errors from TCP connection or send/receive operations,
    /// and returns `PermissionDenied` / `InvalidData` for protocol failures.
    pub async fn connect(&mut self, address: &str, username: &str) -> Result<()> {
        self.state = ConnectionState::Connecting;

        match self.transport_type {
            TransportType::Tcp => {
                let mut client = TcpClient::new();
                client.connect(address).await?;

                // The server will overwrite `player_id: 0` with the real ID;
                // we send 0 as a placeholder to satisfy the protocol schema.
                let connect_packet = Packet::Connect {
                    player_id: 0,
                    username: username.to_string(),
                };
                client.send(&connect_packet).await?;

                let response = client.recv().await?;
                match response {
                    Packet::ConnectAck { success, player_id, seed } => {
                        if success {
                            self.tcp_client = Some(client);
                            self.state = ConnectionState::Connected;
                            self.player_id = Some(player_id);
                            // Non-fatal if the receiver was already dropped.
                            let _ = self.event_tx.send(ClientEvent::Connected(player_id, seed));
                            Ok(())
                        } else {
                            Err(Error::new(
                                ErrorKind::PermissionDenied,
                                "Connection rejected",
                            ))
                        }
                    }
                    // Any packet other than ConnectAck at this point is a
                    // protocol error; disconnect and report.
                    _ => Err(Error::new(ErrorKind::InvalidData, "Unexpected response")),
                }
            }
        }
    }

    /// Sends a raw packet to the server.
    ///
    /// Prefer the typed helpers (`send_position`, `send_rotation`, etc.)
    /// for common operations; use this for packets without a dedicated helper.
    ///
    /// # Errors
    /// Returns `NotConnected` if no transport is active, or propagates I/O
    /// errors from the underlying transport.
    pub async fn send(&self, packet: &Packet) -> Result<()> {
        match self.transport_type {
            TransportType::Tcp => {
                if let Some(client) = &self.tcp_client {
                    client.send(packet).await
                } else {
                    Err(Error::new(ErrorKind::NotConnected, "Not connected"))
                }
            }
        }
    }

    /// Receives the next packet from the server, blocking until one arrives.
    ///
    /// In the normal game-loop architecture the receive loop runs in a
    /// dedicated Tokio task; direct calls to this method are mainly useful for
    /// the initial handshake and testing.
    ///
    /// # Errors
    /// Returns `NotConnected` if no transport is active, or propagates I/O /
    /// decode errors from the underlying transport.
    pub async fn recv(&self) -> Result<Packet> {
        match self.transport_type {
            TransportType::Tcp => {
                if let Some(client) = &self.tcp_client {
                    client.recv().await
                } else {
                    Err(Error::new(ErrorKind::NotConnected, "Not connected"))
                }
            }
        }
    }

    /// Translates a received [`Packet`] into a [`ClientEvent`] and pushes it
    /// to the event channel.
    ///
    /// This method is called by the background receive task for every packet
    /// that arrives from the server.  Packets without a corresponding event
    /// (e.g., `ConnectAck`, which is already consumed in [`connect`]) are
    /// silently ignored via the `_ => {}` arm.
    ///
    /// Send errors on the channel are discarded (`let _ = ...`) because the
    /// only failure mode is a dropped receiver, which means the game loop has
    /// already shut down and there is nothing useful to do.
    pub fn handle_packet(&self, packet: Packet) {
        match packet {
            Packet::Position { player_id, x, y, z } => {
                let _ = self
                    .event_tx
                    .send(ClientEvent::PlayerMoved(player_id, x, y, z));
            }
            Packet::Rotation {
                player_id,
                yaw,
                pitch,
            } => {
                let _ = self
                    .event_tx
                    .send(ClientEvent::PlayerRotated(player_id, yaw, pitch));
            }
            Packet::BlockChange {
                x,
                y,
                z,
                block_type,
            } => {
                let _ = self
                    .event_tx
                    .send(ClientEvent::BlockChanged(x, y, z, block_type));
            }
            Packet::Chat { player_id, message } => {
                let _ = self
                    .event_tx
                    .send(ClientEvent::ChatMessage(player_id, message));
            }
            // A `Connect` packet arriving after the handshake means another
            // player joined the session; map it to `PlayerJoined`.
            Packet::Connect {
                player_id,
                username,
            } => {
                let _ = self
                    .event_tx
                    .send(ClientEvent::PlayerJoined(player_id, username));
            }
            Packet::Disconnect { player_id } => {
                let _ = self.event_tx.send(ClientEvent::PlayerLeft(player_id));
            }
            Packet::Pong { timestamp } => {
                let _ = self.event_tx.send(ClientEvent::Pong(timestamp));
            }
            // ConnectAck and Ping are handled elsewhere or are server-only.
            _ => {}
        }
    }

    /// Sends a `Disconnect` packet to the server and shuts down the transport.
    ///
    /// After this call `state` is `Disconnected` and `player_id` is `None`.
    /// A `ClientEvent::Disconnected` is pushed to the event channel so the
    /// game loop can clean up remote-player state.
    ///
    /// The send of the `Disconnect` packet is best-effort: if the transport
    /// is already broken the error is swallowed so the local cleanup still
    /// completes cleanly.
    pub async fn disconnect(&mut self) -> Result<()> {
        match self.transport_type {
            TransportType::Tcp => {
                if let Some(mut client) = self.tcp_client.take() {
                    // Best-effort: notify the server we are leaving.
                    if let Some(id) = self.player_id {
                        let _ = client.send(&Packet::Disconnect { player_id: id }).await;
                    }
                    client.disconnect().await?;
                }
            }
        }

        self.state = ConnectionState::Disconnected;
        self.player_id = None;
        let _ = self.event_tx.send(ClientEvent::Disconnected);
        Ok(())
    }

    // ── State accessors ───────────────────────────────────────────────────── //

    /// Returns the current connection lifecycle state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Returns `true` when the `ConnectAck` handshake has completed
    /// successfully and the connection is ready to send and receive packets.
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }

    /// Returns the server-assigned player ID, or `None` when not connected.
    pub fn player_id(&self) -> Option<PlayerId> {
        self.player_id
    }

    /// Returns the transport type this client was constructed with.
    pub fn transport_type(&self) -> TransportType {
        self.transport_type
    }

    // ── Typed send helpers ────────────────────────────────────────────────── //

    /// Sends the local player's current world position to the server.
    ///
    /// The `player_id` field is filled from the stored server-assigned ID.
    ///
    /// # Errors
    /// Returns `NotConnected` if `player_id` is not yet set.
    pub async fn send_position(&self, x: f32, y: f32, z: f32) -> Result<()> {
        if let Some(id) = self.player_id {
            self.send(&Packet::Position {
                player_id: id,
                x,
                y,
                z,
            })
            .await
        } else {
            Err(Error::new(ErrorKind::NotConnected, "Not connected"))
        }
    }

    /// Sends the local player's current rotation to the server.
    ///
    /// Yaw and pitch are quantized to `u8` (0 = 0°, 255 ≈ 359°) to reduce
    /// bandwidth; the caller is responsible for quantizing before calling.
    ///
    /// # Errors
    /// Returns `NotConnected` if `player_id` is not yet set.
    pub async fn send_rotation(&self, yaw: u8, pitch: u8) -> Result<()> {
        if let Some(id) = self.player_id {
            self.send(&Packet::Rotation {
                player_id: id,
                yaw,
                pitch,
            })
            .await
        } else {
            Err(Error::new(ErrorKind::NotConnected, "Not connected"))
        }
    }

    /// Notifies the server that block `(x, y, z)` was changed to `block_type`.
    ///
    /// Unlike position and rotation, block changes do not require a connected
    /// `player_id` in the packet (the server identifies the sender from the
    /// connection), so this helper does not gate on `player_id`.
    pub async fn send_block_change(&self, x: i32, y: i32, z: i32, block_type: u8) -> Result<()> {
        self.send(&Packet::BlockChange {
            x,
            y,
            z,
            block_type,
        })
        .await
    }

    /// Sends a chat message to the server, which broadcasts it to all peers.
    ///
    /// # Errors
    /// Returns `NotConnected` if `player_id` is not yet set.
    pub async fn send_chat(&self, message: &str) -> Result<()> {
        if let Some(id) = self.player_id {
            self.send(&Packet::Chat {
                player_id: id,
                message: message.to_string(),
            })
            .await
        } else {
            Err(Error::new(ErrorKind::NotConnected, "Not connected"))
        }
    }

    /// Sends a `Ping` packet stamped with the current Unix millisecond time.
    ///
    /// The server echoes the timestamp back in a `Pong` packet, which
    /// [`handle_packet`] converts to `ClientEvent::Pong`.  The caller can
    /// compute round-trip latency by comparing the echoed timestamp with the
    /// current time.
    pub async fn send_ping(&self) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.send(&Packet::Ping { timestamp }).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClientConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Connection parameters used to create and configure a [`GameClient`].
///
/// Stored separately from `GameClient` so the UI can edit and persist the
/// configuration without holding a live connection object.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Hostname or IP address of the server (without port).
    pub server_address: String,
    /// TCP port the server is listening on.
    pub server_port: u16,
    /// Display name sent to the server in the `Connect` packet.
    pub username: String,
    /// Which transport layer to use when connecting.
    pub transport: TransportType,
}

impl Default for ClientConfig {
    /// Returns a configuration pointing at `localhost:25565` with username
    /// `"Player"` over TCP — the conventional Minecraft-compatible defaults.
    fn default() -> Self {
        Self {
            server_address: "127.0.0.1".to_string(),
            server_port: 25565,
            username: "Player".to_string(),
            transport: TransportType::Tcp,
        }
    }
}

impl ClientConfig {
    /// Formats `server_address` and `server_port` into a single `"host:port"`
    /// string suitable for passing to [`GameClient::connect`].
    pub fn full_address(&self) -> String {
        format!("{}:{}", self.server_address, self.server_port)
    }
}
