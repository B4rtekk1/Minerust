#![allow(dead_code)]

use crate::multiplayer::protocol::Packet;
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Result};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::RwLock;

/// Maximum number of bytes accepted for a single packet payload.
///
/// Packets whose length header exceeds this value are rejected with
/// [`ErrorKind::InvalidData`] to guard against memory exhaustion from
/// malformed or malicious clients.
const READ_BUFFER_SIZE: usize = 4096;

/// An established, framed TCP connection to a single peer.
///
/// The underlying [`TcpStream`] is split into independent read and write
/// halves so that sends and receives can be driven concurrently from
/// separate tasks without holding the same lock. Each half is wrapped in
/// an [`RwLock`] — callers always acquire a write lock because the async
/// I/O operations require exclusive access.
///
/// Connection liveness is tracked by an [`AtomicBool`] flag; once
/// [`TcpConnection::close`] is called, all subsequent [`send`](TcpConnection::send)
/// and [`recv`](TcpConnection::recv) calls immediately return
/// [`ErrorKind::NotConnected`].
pub struct TcpConnection {
    /// Exclusive write access to the TCP write half for sending bytes.
    writer: Arc<RwLock<tokio::net::tcp::OwnedWriteHalf>>,
    /// Exclusive write access to the TCP read half for receiving bytes.
    reader: Arc<RwLock<tokio::net::tcp::OwnedReadHalf>>,
    /// `true` while the connection is open; set to `false` by [`TcpConnection::close`].
    connected: AtomicBool,
    /// Remote peer address, captured at construction time.
    addr: SocketAddr,
}

impl TcpConnection {
    /// Wraps an accepted or connected [`TcpStream`] into a `TcpConnection`.
    ///
    /// Splits the stream into owned read/write halves and marks the
    /// connection as live. `addr` should be the remote peer's socket address.
    pub fn new(stream: TcpStream, addr: SocketAddr) -> Self {
        let (reader, writer) = stream.into_split();
        Self {
            writer: Arc::new(RwLock::new(writer)),
            reader: Arc::new(RwLock::new(reader)),
            connected: AtomicBool::new(true),
            addr,
        }
    }

    /// Returns the remote peer's [`SocketAddr`].
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Serialises `packet` and writes it to the TCP stream.
    ///
    /// Serialisation uses [`Packet::to_bytes`], which prepends a 2-byte
    /// little-endian length header. The write is followed by a flush to
    /// ensure the bytes are handed off to the OS immediately.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::NotConnected`] if [`TcpConnection::close`] has already been called.
    /// - Any I/O error propagated from the underlying write or flush.
    pub async fn send(&self, packet: &Packet) -> Result<()> {
        if !self.connected.load(Ordering::Relaxed) {
            return Err(Error::new(ErrorKind::NotConnected, "Connection closed"));
        }

        let bytes = packet.to_bytes();
        let mut writer = self.writer.write().await;
        writer.write_all(&bytes).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Reads exactly one framed packet from the TCP stream.
    ///
    /// Reads the 2-byte little-endian length header first, then reads that
    /// many additional bytes to form the complete packet buffer, and finally
    /// delegates to [`Packet::from_bytes`] for deserialisation.
    ///
    /// This method blocks until a full packet is available or an error occurs.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::NotConnected`] if the connection has been closed.
    /// - [`ErrorKind::InvalidData`] if the declared packet length exceeds
    ///   [`READ_BUFFER_SIZE`], guarding against oversized or malformed frames.
    /// - Any I/O error from the underlying socket reads (e.g.
    ///   [`ErrorKind::UnexpectedEof`] if the peer closed the connection mid-packet).
    /// - Any deserialisation error returned by [`Packet::from_bytes`].
    pub async fn recv(&self) -> Result<Packet> {
        if !self.connected.load(Ordering::Relaxed) {
            return Err(Error::new(ErrorKind::NotConnected, "Connection closed"));
        }

        let mut reader = self.reader.write().await;

        // Read the 2-byte length prefix that precedes every packet.
        let mut len_buf = [0u8; 2];
        reader.read_exact(&mut len_buf).await?;
        let len = u16::from_le_bytes(len_buf) as usize;

        if len > READ_BUFFER_SIZE {
            return Err(Error::new(ErrorKind::InvalidData, "Packet too large"));
        }

        // Re-include the length bytes so from_bytes receives the full framed buffer.
        let mut data = vec![0u8; len + 2];
        data[0..2].copy_from_slice(&len_buf);
        reader.read_exact(&mut data[2..]).await?;

        Packet::from_bytes(&data)
    }

    /// Marks the connection as closed and shuts down the write half of the socket.
    ///
    /// After this call, [`send`](TcpConnection::send) and
    /// [`recv`](TcpConnection::recv) will immediately return
    /// [`ErrorKind::NotConnected`]. The TCP shutdown signals EOF to the remote
    /// peer.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from the underlying shutdown call.
    pub async fn close(&self) -> Result<()> {
        self.connected.store(false, Ordering::Relaxed);
        let mut writer = self.writer.write().await;
        writer.shutdown().await?;
        Ok(())
    }

    /// Returns `true` if the connection has not yet been closed via
    /// [`TcpConnection::close`].
    ///
    /// Note: this reflects the local flag only. A peer-initiated disconnect
    /// will not set this flag to `false` until the next failed read or write.
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }
}

/// A TCP server that accepts client connections and broadcasts [`Packet`]s.
///
/// Each accepted connection is stored in a shared map keyed by a
/// monotonically increasing `u32` connection ID. This ID is independent of
/// the game-level [`PlayerId`](crate::multiplayer::protocol::PlayerId) and is
/// assigned solely by the transport layer.
///
/// # Lifecycle
///
/// 1. Create with [`TcpServer::bind`].
/// 2. Loop on [`TcpServer::accept`] to receive incoming connections.
/// 3. Spawn a read task per connection to call [`TcpConnection::recv`].
/// 4. Use [`TcpServer::broadcast`] or [`TcpServer::broadcast_except`] to fan
///    out packets to all connected clients.
/// 5. Call [`TcpServer::remove_client`] when a connection closes.
/// 6. Call [`TcpServer::stop`] to signal the accept loop to exit.
pub struct TcpServer {
    /// Bound TCP listener. `None` after the server is stopped.
    listener: Option<TcpListener>,
    /// All currently tracked client connections, keyed by connection ID.
    connections: Arc<RwLock<HashMap<u32, Arc<TcpConnection>>>>,
    /// Monotonically increasing source for connection IDs. Starts at `1`.
    next_id: AtomicU32,
    /// Set to `false` by [`TcpServer::stop`] to signal the accept loop to exit.
    running: AtomicBool,
}

impl TcpServer {
    /// Binds a TCP listener to `addr` and returns a ready-to-accept server.
    ///
    /// # Arguments
    ///
    /// * `addr` - A `"host:port"` string such as `"0.0.0.0:25565"`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the address cannot be bound (e.g. port already
    /// in use, permission denied).
    pub async fn bind(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        tracing::info!("[TCP Server] Listening on {}", addr);

        Ok(Self {
            listener: Some(listener),
            connections: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU32::new(1),
            running: AtomicBool::new(true),
        })
    }

    /// Waits for the next incoming TCP connection and registers it.
    ///
    /// Assigns a unique connection ID, enables `TCP_NODELAY` to reduce
    /// latency for small packets, wraps the stream in a [`TcpConnection`],
    /// and inserts it into the internal connection map.
    ///
    /// # Returns
    ///
    /// A `(connection_id, connection)` pair. The caller should spawn an async
    /// task that loops on [`TcpConnection::recv`] for the returned connection.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::NotConnected`] if the listener has been taken (server stopped).
    /// - Any I/O error from the underlying `accept` or `set_nodelay` calls.
    pub async fn accept(&self) -> Result<(u32, Arc<TcpConnection>)> {
        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::NotConnected, "Server not running"))?;

        let (stream, addr) = listener.accept().await?;
        // Disable Nagle's algorithm so small game packets are sent immediately.
        stream.set_nodelay(true)?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let connection = Arc::new(TcpConnection::new(stream, addr));

        {
            let mut conns = self.connections.write().await;
            conns.insert(id, connection.clone());
        }

        tracing::info!("[TCP Server] Client {} connected from {}", id, addr);
        Ok((id, connection))
    }

    /// Sends `packet` to every currently registered client connection.
    ///
    /// Send failures for individual clients are logged as warnings but do
    /// not abort the broadcast — all other clients still receive the packet.
    ///
    /// # Errors
    ///
    /// Always returns `Ok(())`. Per-client errors are traced, not propagated.
    pub async fn broadcast(&self, packet: &Packet) -> Result<()> {
        let conns = self.connections.read().await;
        for (id, conn) in conns.iter() {
            if let Err(e) = conn.send(packet).await {
                tracing::warn!("[TCP Server] Failed to send to client {}: {}", id, e);
            }
        }
        Ok(())
    }

    /// Sends `packet` to all registered clients except the one with `except_id`.
    ///
    /// Useful for relaying a player's own action to all other players without
    /// echoing it back to the originator. Like [`TcpServer::broadcast`],
    /// per-client errors are logged but do not abort the fanout.
    ///
    /// # Errors
    ///
    /// Always returns `Ok(())`. Per-client errors are traced, not propagated.
    pub async fn broadcast_except(&self, packet: &Packet, except_id: u32) -> Result<()> {
        let conns = self.connections.read().await;
        for (id, conn) in conns.iter() {
            if *id != except_id {
                if let Err(e) = conn.send(packet).await {
                    tracing::warn!("[TCP Server] Failed to send to client {}: {}", id, e);
                }
            }
        }
        Ok(())
    }

    /// Removes the client with `id` from the connection map.
    ///
    /// Does not close the underlying socket — the caller is responsible for
    /// calling [`TcpConnection::close`] before or after removal if a clean
    /// TCP shutdown is required. Logs at `INFO` level when a client is found
    /// and removed.
    pub async fn remove_client(&self, id: u32) {
        let mut conns = self.connections.write().await;
        if conns.remove(&id).is_some() {
            tracing::info!("[TCP Server] Client {} disconnected", id);
        }
    }

    /// Returns the number of currently registered client connections.
    pub async fn client_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Signals the server to stop accepting new connections.
    ///
    /// Sets the internal `running` flag to `false`. Does not close the
    /// listener socket or drop existing connections; those must be cleaned
    /// up by the caller.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Returns `true` while the server has not been stopped via [`TcpServer::stop`].
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// A TCP client that manages a single outbound connection to a game server.
///
/// Wraps an optional [`TcpConnection`]; methods that require an active
/// connection return [`ErrorKind::NotConnected`] when called before
/// [`TcpClient::connect`] or after [`TcpClient::disconnect`].
///
/// `TcpClient` is [`Clone`] because the inner connection is reference-counted.
/// Both the original and the clone refer to the same underlying socket.
#[derive(Clone)]
pub struct TcpClient {
    /// The active connection, or `None` if not yet connected or disconnected.
    connection: Option<Arc<TcpConnection>>,
}

impl TcpClient {
    /// Creates an unconnected `TcpClient`.
    ///
    /// Call [`TcpClient::connect`] before sending or receiving packets.
    pub fn new() -> Self {
        Self { connection: None }
    }

    /// Opens a TCP connection to `addr` and enables `TCP_NODELAY`.
    ///
    /// # Arguments
    ///
    /// * `addr` - Remote address in `"host:port"` form, e.g. `"127.0.0.1:25565"`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the connection attempt or `set_nodelay` call fails.
    pub async fn connect(&mut self, addr: &str) -> Result<()> {
        println!("[TCP Client] Connecting to {}...", addr);
        let stream = TcpStream::connect(addr).await?;
        // Disable Nagle's algorithm so small game packets are sent immediately.
        stream.set_nodelay(true)?;

        let socket_addr = stream.peer_addr()?;
        self.connection = Some(Arc::new(TcpConnection::new(stream, socket_addr)));

        println!("[TCP Client] Connected to {}", addr);
        Ok(())
    }

    /// Returns a reference to the underlying [`TcpConnection`], if connected.
    ///
    /// Returns `None` when the client is not connected. Prefer the higher-level
    /// [`TcpClient::send`] and [`TcpClient::recv`] methods for most use cases.
    pub fn connection(&self) -> Option<&Arc<TcpConnection>> {
        self.connection.as_ref()
    }

    /// Serialises and sends `packet` to the server.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::NotConnected`] if not connected.
    /// - Any I/O error from the underlying [`TcpConnection::send`].
    pub async fn send(&self, packet: &Packet) -> Result<()> {
        match &self.connection {
            Some(conn) => conn.send(packet).await,
            None => Err(Error::new(ErrorKind::NotConnected, "Not connected")),
        }
    }

    /// Reads and deserialises the next packet from the server.
    ///
    /// Blocks until a complete packet is available.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::NotConnected`] if not connected.
    /// - Any I/O or deserialisation error from the underlying [`TcpConnection::recv`].
    pub async fn recv(&self) -> Result<Packet> {
        match &self.connection {
            Some(conn) => conn.recv().await,
            None => Err(Error::new(ErrorKind::NotConnected, "Not connected")),
        }
    }

    /// Closes the connection and moves it out of `self`.
    ///
    /// After this call [`TcpClient::is_connected`] returns `false` and
    /// [`send`](TcpClient::send) / [`recv`](TcpClient::recv) return
    /// [`ErrorKind::NotConnected`]. Calling `disconnect` on an already
    /// disconnected client is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates any I/O error from [`TcpConnection::close`].
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(conn) = self.connection.take() {
            conn.close().await?;
        }
        println!("[TCP Client] Disconnected");
        Ok(())
    }

    /// Returns `true` if the client holds an open connection.
    ///
    /// Delegates to [`TcpConnection::is_connected`]; returns `false` if
    /// there is no connection at all.
    pub fn is_connected(&self) -> bool {
        self.connection
            .as_ref()
            .map(|c| c.is_connected())
            .unwrap_or(false)
    }
}

impl Default for TcpClient {
    /// Equivalent to [`TcpClient::new`]; creates an unconnected client.
    fn default() -> Self {
        Self::new()
    }
}