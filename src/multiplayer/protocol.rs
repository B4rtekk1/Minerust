use std::io::{Cursor, Error, ErrorKind, Read, Result};

/// A unique identifier for a connected player, assigned by the server.
pub type PlayerId = u32;

/// All packet types exchanged between client and server over the network.
///
/// Every variant maps to a fixed packet ID byte (see [`Packet::packet_id`]).
/// Wire format for every packet:
///
/// ```text
/// [ 2 bytes: total length (u16 LE) ][ 1 byte: packet ID ][ payload ]
/// ```
///
/// Strings in the payload are length-prefixed:
/// ```text
/// [ 2 bytes: byte length (u16 LE) ][ N bytes: UTF-8 data ]
/// ```
///
/// All multi-byte integers are little-endian.
#[derive(Debug, Clone)]
pub enum Packet {
    /// Sent by the client to announce itself when joining the server.
    ///
    /// Packet ID: `0x01`
    Connect {
        /// The player ID the client wishes to use (may be overridden by the server).
        player_id: PlayerId,
        /// The player's chosen display name.
        username: String,
    },

    /// Sent by the server in response to a [`Packet::Connect`].
    ///
    /// Packet ID: `0x02`
    ConnectAck {
        /// `true` if the connection was accepted; `false` if rejected.
        success: bool,
        /// The authoritative player ID assigned by the server.
        player_id: PlayerId,
    },

    /// Reports a player's world-space position.
    ///
    /// Packet ID: `0x10`
    Position {
        /// The player this position update belongs to.
        player_id: PlayerId,
        /// World-space X coordinate (f32 LE).
        x: f32,
        /// World-space Y coordinate, vertical axis (f32 LE).
        y: f32,
        /// World-space Z coordinate (f32 LE).
        z: f32,
    },

    /// Reports a player's look direction as compressed angles.
    ///
    /// Both angles are quantised to a single byte to minimise bandwidth.
    /// Use [`encode_yaw`]/[`decode_yaw`] and [`encode_pitch`]/[`decode_pitch`]
    /// to convert between degrees and wire values.
    ///
    /// Packet ID: `0x11`
    Rotation {
        /// The player this rotation update belongs to.
        player_id: PlayerId,
        /// Horizontal look angle. `0`–`255` maps linearly to `0°`–`360°`.
        yaw: u8,
        /// Vertical look angle. `0`–`255` maps linearly to `-90°`–`+90°`.
        pitch: u8,
    },

    /// Notifies clients that a single block in the world has changed.
    ///
    /// Packet ID: `0x20`
    BlockChange {
        /// Block X coordinate (i32 LE).
        x: i32,
        /// Block Y coordinate (i32 LE).
        y: i32,
        /// Block Z coordinate (i32 LE).
        z: i32,
        /// Numeric block type ID. `0` conventionally represents air.
        block_type: u8,
    },

    /// A chat message sent by a player.
    ///
    /// Packet ID: `0x30`
    Chat {
        /// The player who sent the message.
        player_id: PlayerId,
        /// UTF-8 message text, length-prefixed on the wire.
        message: String,
    },

    /// Signals that a player has left the session.
    ///
    /// Packet ID: `0x40`
    Disconnect {
        /// The player who disconnected.
        player_id: PlayerId,
    },

    /// Latency probe sent to the remote peer. Expects a matching [`Packet::Pong`].
    ///
    /// Packet ID: `0xFE`
    Ping {
        /// Sender's timestamp (e.g. milliseconds since epoch). Echoed back
        /// unchanged in the corresponding [`Packet::Pong`] so the sender can
        /// calculate round-trip time.
        timestamp: u64,
    },

    /// Reply to a [`Packet::Ping`]; echoes the original timestamp unchanged.
    ///
    /// Packet ID: `0xFF`
    Pong {
        /// The timestamp copied verbatim from the corresponding [`Packet::Ping`].
        timestamp: u64,
    },
}

impl Packet {
    /// Returns the single-byte packet ID used as a discriminant on the wire.
    fn packet_id(&self) -> u8 {
        match self {
            Packet::Connect { .. } => 0x01,
            Packet::ConnectAck { .. } => 0x02,
            Packet::Position { .. } => 0x10,
            Packet::Rotation { .. } => 0x11,
            Packet::BlockChange { .. } => 0x20,
            Packet::Chat { .. } => 0x30,
            Packet::Disconnect { .. } => 0x40,
            Packet::Ping { .. } => 0xFE,
            Packet::Pong { .. } => 0xFF,
        }
    }

    /// Serialises the packet into a length-prefixed byte buffer ready for sending.
    ///
    /// Wire layout:
    /// ```text
    /// [ u16 LE: total length including ID byte ] [ u8: packet ID ] [ payload ]
    /// ```
    ///
    /// # Returns
    ///
    /// A [`Vec<u8>`] containing the complete framed packet.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.push(self.packet_id());

        match self {
            Packet::Connect {
                player_id,
                username,
            } => {
                buf.extend_from_slice(&player_id.to_le_bytes());
                write_string(&mut buf, username);
            }
            Packet::ConnectAck { success, player_id } => {
                buf.push(if *success { 1 } else { 0 });
                buf.extend_from_slice(&player_id.to_le_bytes());
            }
            Packet::Position { player_id, x, y, z } => {
                buf.extend_from_slice(&player_id.to_le_bytes());
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf.extend_from_slice(&z.to_le_bytes());
            }
            Packet::Rotation {
                player_id,
                yaw,
                pitch,
            } => {
                buf.extend_from_slice(&player_id.to_le_bytes());
                buf.push(*yaw);
                buf.push(*pitch);
            }
            Packet::BlockChange {
                x,
                y,
                z,
                block_type,
            } => {
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf.extend_from_slice(&z.to_le_bytes());
                buf.push(*block_type);
            }
            Packet::Chat { player_id, message } => {
                buf.extend_from_slice(&player_id.to_le_bytes());
                write_string(&mut buf, message);
            }
            Packet::Disconnect { player_id } => {
                buf.extend_from_slice(&player_id.to_le_bytes());
            }
            Packet::Ping { timestamp } | Packet::Pong { timestamp } => {
                buf.extend_from_slice(&timestamp.to_le_bytes());
            }
        }

        // Prepend the 2-byte length header (covers ID byte + payload).
        let len = buf.len() as u16;
        let mut result = Vec::with_capacity(2 + buf.len());
        result.extend_from_slice(&len.to_le_bytes());
        result.extend(buf);
        result
    }

    /// Deserialises a packet from a raw byte slice produced by [`Packet::to_bytes`].
    ///
    /// Expects the full framed format:
    /// ```text
    /// [ u16 LE: length ] [ u8: packet ID ] [ payload ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidData`] if:
    /// - `data` is shorter than 3 bytes (2-byte length + 1-byte ID).
    /// - The packet ID is not recognised.
    /// - A string field contains invalid UTF-8.
    ///
    /// Returns any [`std::io::Error`] propagated from the underlying cursor reads
    /// (e.g. [`ErrorKind::UnexpectedEof`] if the payload is truncated).
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(Error::new(ErrorKind::InvalidData, "Packet too short"));
        }

        let mut cursor = Cursor::new(data);

        let mut len_bytes = [0u8; 2];
        cursor.read_exact(&mut len_bytes)?;
        let _len = u16::from_le_bytes(len_bytes);
        let mut id = [0u8; 1];
        cursor.read_exact(&mut id)?;

        match id[0] {
            0x01 => {
                let player_id = read_u32(&mut cursor)?;
                let username = read_string(&mut cursor)?;
                Ok(Packet::Connect {
                    player_id,
                    username,
                })
            }
            0x02 => {
                let mut b = [0u8; 1];
                cursor.read_exact(&mut b)?;
                let player_id = read_u32(&mut cursor)?;
                Ok(Packet::ConnectAck {
                    success: b[0] != 0,
                    player_id,
                })
            }
            0x10 => {
                let player_id = read_u32(&mut cursor)?;
                let x = read_f32(&mut cursor)?;
                let y = read_f32(&mut cursor)?;
                let z = read_f32(&mut cursor)?;
                Ok(Packet::Position { player_id, x, y, z })
            }
            0x11 => {
                let player_id = read_u32(&mut cursor)?;
                let mut angles = [0u8; 2];
                cursor.read_exact(&mut angles)?;
                Ok(Packet::Rotation {
                    player_id,
                    yaw: angles[0],
                    pitch: angles[1],
                })
            }
            0x20 => {
                let x = read_i32(&mut cursor)?;
                let y = read_i32(&mut cursor)?;
                let z = read_i32(&mut cursor)?;
                let mut bt = [0u8; 1];
                cursor.read_exact(&mut bt)?;
                Ok(Packet::BlockChange {
                    x,
                    y,
                    z,
                    block_type: bt[0],
                })
            }
            0x30 => {
                let player_id = read_u32(&mut cursor)?;
                let message = read_string(&mut cursor)?;
                Ok(Packet::Chat { player_id, message })
            }
            0x40 => {
                let player_id = read_u32(&mut cursor)?;
                Ok(Packet::Disconnect { player_id })
            }
            0xFE => {
                let timestamp = read_u64(&mut cursor)?;
                Ok(Packet::Ping { timestamp })
            }
            0xFF => {
                let timestamp = read_u64(&mut cursor)?;
                Ok(Packet::Pong { timestamp })
            }
            _ => Err(Error::new(ErrorKind::InvalidData, "Unknown packet ID")),
        }
    }
}

/// Writes a UTF-8 string into `buf` as a 2-byte little-endian length prefix
/// followed by the raw UTF-8 bytes.
fn write_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Reads a length-prefixed UTF-8 string from `cursor`.
///
/// Expects `[ u16 LE: byte length ][ N bytes: UTF-8 ]`.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidData`] if the bytes are not valid UTF-8, or
/// propagates any I/O error from the cursor.
fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let mut len_bytes = [0u8; 2];
    cursor.read_exact(&mut len_bytes)?;
    let len = u16::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;

    String::from_utf8(buf).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8"))
}

/// Reads a little-endian `u32` from `cursor`.
fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0u8; 4];
    cursor.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Reads a little-endian `u64` from `cursor`.
fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut bytes = [0u8; 8];
    cursor.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Encodes a yaw angle in degrees to the wire byte representation used in
/// [`Packet::Rotation`].
///
/// The full `0°`–`360°` range maps linearly to `0`–`255`. The input is first
/// normalised to `[0°, 360°)` so negative values and values above `360°` are
/// handled correctly.
///
/// # Example
/// ```
/// assert_eq!(encode_yaw(0.0),   0);
/// assert_eq!(encode_yaw(180.0), 128);
/// assert_eq!(encode_yaw(360.0), 0);   // wraps back to 0
/// ```
pub fn encode_yaw(degrees: f32) -> u8 {
    (((degrees % 360.0) + 360.0) % 360.0 / 360.0 * 256.0) as u8
}

/// Decodes a wire yaw byte back to degrees in the range `[0°, 360°)`.
///
/// Inverse of [`encode_yaw`].
pub fn decode_yaw(val: u8) -> f32 {
    val as f32 / 256.0 * 360.0
}

/// Encodes a pitch angle in degrees to the wire byte representation used in
/// [`Packet::Rotation`].
///
/// Input is clamped to `[-90°, +90°]` before encoding.
/// `-90°` maps to `0`; `+90°` maps to `255`.
///
/// # Example
/// ```
/// assert_eq!(encode_pitch(-90.0), 0);
/// assert_eq!(encode_pitch(0.0),   127);
/// assert_eq!(encode_pitch(90.0),  255);
/// ```
pub fn encode_pitch(degrees: f32) -> u8 {
    ((degrees.clamp(-90.0, 90.0) + 90.0) / 180.0 * 255.0) as u8
}

/// Decodes a wire pitch byte back to degrees in the range `[-90°, +90°]`.
///
/// Inverse of [`encode_pitch`].
pub fn decode_pitch(val: u8) -> f32 {
    val as f32 / 255.0 * 180.0 - 90.0
}

/// Reads a little-endian `f32` from `cursor`.
fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut bytes = [0u8; 4];
    cursor.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

/// Reads a little-endian `i32` from `cursor`.
fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut bytes = [0u8; 4];
    cursor.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}