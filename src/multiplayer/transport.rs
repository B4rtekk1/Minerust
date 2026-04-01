#![allow(dead_code)]

use crate::multiplayer::protocol::Packet;
use std::io::Result;

/// Abstraction over a bidirectional, packet-oriented network connection.
///
/// Implementors provide the actual transport mechanism (TCP, UDP, WebSocket,
/// etc.) while the rest of the multiplayer stack works exclusively against
/// this trait, keeping game logic decoupled from network I/O details.
///
/// # Implementing `Transport`
///
/// All methods are async and must be `Send` so implementations can be driven
/// from a Tokio task. The connection is assumed to be stateful: `send` and
/// `recv` operate on the same underlying stream, and `close` permanently
/// terminates it.
///
/// ```ignore
/// struct MyTransport { /* ... */ }
///
/// impl Transport for MyTransport {
///     async fn send(&self, packet: &Packet) -> Result<()> { /* ... */ }
///     async fn recv(&self) -> Result<Packet> { /* ... */ }
///     async fn close(&self) -> Result<()> { /* ... */ }
///     fn is_connected(&self) -> bool { /* ... */ }
/// }
/// ```
pub trait Transport: Send + Sync {
    /// Serialises and sends `packet` to the remote peer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the connection is closed or the write fails.
    fn send(&self, packet: &Packet) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Waits for and deserialises the next packet from the remote peer.
    ///
    /// Blocks until a complete packet is available on the stream.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the connection is closed, the read fails,
    /// or the received bytes cannot be deserialised into a [`Packet`].
    fn recv(&self) -> impl std::future::Future<Output = Result<Packet>> + Send;

    /// Closes the connection and releases any associated resources.
    ///
    /// After this call [`Transport::is_connected`] should return `false`
    /// and subsequent [`send`](Transport::send) / [`recv`](Transport::recv)
    /// calls should return [`std::io::ErrorKind::NotConnected`].
    ///
    /// Calling `close` on an already-closed connection should be a no-op.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the underlying shutdown fails.
    fn close(&self) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Returns `true` if the connection is currently open.
    ///
    /// Note: this typically reflects a local flag. A peer-initiated
    /// disconnect may not be detected until the next failed `send` or `recv`.
    fn is_connected(&self) -> bool;
}

/// The network transport protocol used for a connection or server.
///
/// Passed to [`GameServer`](crate::multiplayer::server::GameServer) and
/// related types to select or identify the underlying transport implementation.
/// New variants should be added here as additional transports are supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    /// Transmission Control Protocol — reliable, ordered, connection-oriented.
    /// The only currently supported transport.
    Tcp,
}

impl Default for TransportType {
    /// Returns [`TransportType::Tcp`], the only currently available transport.
    fn default() -> Self {
        TransportType::Tcp
    }
}

impl std::fmt::Display for TransportType {
    /// Formats the transport type as a human-readable uppercase abbreviation
    /// (e.g. `"TCP"`), suitable for log output and UI display.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportType::Tcp => write!(f, "TCP"),
        }
    }
}
