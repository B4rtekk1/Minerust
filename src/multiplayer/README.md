# src/multiplayer/ - Networking & Multiplayer Module

## Overview

The `multiplayer/` module implements a client-server architecture for multiplayer gaming. It handles network communication using QUIC (primary) and TCP (fallback), player synchronization, chunk streaming to clients, and authoritative server validation.

## Module Structure

```
multiplayer/
├── mod.rs              ← Module declaration and public API
├── client.rs           ← Client-side networking logic
├── server.rs           ← Server-side game logic
├── network.rs          ← Shared networking utilities
├── protocol.rs         ← Message protocol definitions
├── transport.rs        ← Transport layer abstraction
├── quic.rs             ← QUIC protocol implementation
├── tcp.rs              ← TCP fallback implementation
└── player.rs           ← Remote player state management
```

## File Documentation

### `mod.rs` - Module Root
**Purpose:** Declares submodules and provides multiplayer API.

**Key Types:**
- `MultiplayerManager` - Main multiplayer coordinator
- `ConnectionMode` - Client or Server

**Key Functions:**
- `new_client(address) → Client` - Create client connection
- `new_server(port) → Server` - Start server
- `update() → ()` - Process network events

### `protocol.rs` - Message Protocol
**Purpose:** Defines all message types for client-server communication.

**Message Categories:**

#### **Handshake Messages**
```rust
pub enum HandshakeMessage {
    ClientHello {
        username: String,
        version: u32,
        client_token: u32,
    },
    ServerHello {
        server_id: u32,
        seed: u64,
        difficulty: u8,
    },
    ClientReady,
}
```

#### **World Data Messages**
```rust
pub enum WorldMessage {
    ChunkData {
        chunk_x: i32,
        chunk_z: i32,
        data: Vec<u8>,              // Compressed chunk
        is_full_update: bool,
    },
    BlockUpdate {
        x: i32,
        y: i32,
        z: i32,
        block_type: u32,
    },
    ChunkUnload {
        chunk_x: i32,
        chunk_z: i32,
    },
}
```

#### **Player Movement Messages**
```rust
pub enum MovementMessage {
    PlayerMove {
        player_id: u32,
        position: [f32; 3],
        rotation: [f32; 2],         // Yaw, Pitch
        velocity: [f32; 3],
    },
    PlayerJump {
        player_id: u32,
    },
    PlayerSprint {
        player_id: u32,
        is_sprinting: bool,
    },
}
```

#### **Chat & Entity Messages**
```rust
pub enum GameMessage {
    ChatMessage {
        sender_id: u32,
        text: String,
        timestamp: u64,
    },
    EntityUpdate {
        entity_id: u32,
        position: [f32; 3],
        data: Vec<u8>,
    },
}
```

**Serialization:**
- Uses `postcard` for efficient binary serialization
- Supports compression (gzip) for large messages
- Version negotiation for protocol compatibility

**Message Priorities:**
```
HIGH:    PlayerMove, PlayerJump (frequently sent)
MEDIUM:  ChunkData, BlockUpdate
LOW:     ChatMessage, EntityUpdate
```

### `client.rs` - Client Implementation
**Purpose:** Implements client-side networking and game state management.

**Client Connection Lifecycle:**

```
Client Start
    ↓
Cmotd=Welcome to Minerust!
    ├─ QUIC (preferred)
    └─ TCP (fallback)
    ↓
Send ClientHello
    ↓
Receive ServerHello
    ↓
Send ClientReady
    ↓
Connected (game loop)
    ├─ Receive chunk data
    ├─ Receive entity updates
    ├─ Send player movement
    └─ Send block updates
    ↓
Disconnect/Reconnect on error
```

**Key Types:**
```rust
pub struct GameClient {
    pub connection: Connection         // QUIC/TCP connection
    pub local_player: LocalPlayer
    pub remote_players: HashMap<u32, RemotePlayer>
    pub loaded_chunks: HashMap<(i32, i32), Chunk>
    pub pending_messages: VecDeque<ClientMessage>
}

pub enum ClientMessage {
    Move(MovementMessage),
    BlockPlace(BlockUpdate),
    BlockBreak(BlockUpdate),
    Chat(String),
}
```

**Client Logic:**
```
Input (Player presses W key)
    ↓
Local movement update
    ↓
Predict movement on client (no lag)
    ↓
Queue movement message
    ↓
Send to server (every ~10ms)
    ↓
Receive server validation
    ├─ If correct: keep prediction
    └─ If wrong: snap to server position
```

**Prediction System:**
```rust
// Client-side:
player.pos += velocity * delta_time;    // Immediate
send_message(PlayerMove { pos, velocity });

// Server:
validate_move(pos);
broadcast_to_clients(RemotePlayerMove { pos });

// Back on client:
if received_pos != predicted_pos {
    player.pos = received_pos;           // Snap correction
}
```

**Key Functions:**
- `new(server_address) → Client` - Connect to server
- `update() → ()` - Process network events
- `send_move(pos, rot) → ()` - Send player movement
- `place_block(x, y, z) → ()` - Send block placement
- `disconnect() → ()` - Close connection

### `server.rs` - Server Implementation
**Purpose:** Authoritative server that validates all gameplay and manages player state.

**Server Loop:**
```
Server Start (listen on port)
    ↓
Accept connections
    ├─ Verify protocol version
    ├─ Allocate player ID
    └─ Send server info + seed
    ↓
Game Loop (60 Hz)
    ├─ Process all player inputs
    ├─ Validate movements
    ├─ Update world state
    ├─ Broadcast changes to clients
    └─ Unload distant chunks
```

**Key Types:**
```rust
pub struct GameServer {
    pub listener: Listener              // Network listener
    pub players: HashMap<u32, ServerPlayer>
    pub loaded_chunks: HashMap<(i32, i32), Chunk>
    pub world: World
}

pub struct ServerPlayer {
    pub id: u32
    pub connection: Connection
    pub username: String
    pub position: [f32; 3]
    pub rotation: [f32; 2]
    pub loaded_chunks: HashSet<(i32, i32)>
    pub last_update: Instant
}
```

**Validation:**

```
Client claims: "Teleport to (999, 64, 999)"
    ↓
Server checks:
  ├─ Distance from last position (max 10 blocks/frame)
  ├─ Is position in valid area?
  ├─ Are blocks solid underneath?
  └─ Anti-cheat: is speed too high?
    ↓
If valid: Accept and broadcast
If invalid: Reject, snap back, warn
```

**Chunk Management:**
```
Player at chunk (10, 10)
    ↓
Server loads chunks in radius 12
    ↓
Unloads chunks beyond distance 16
    ↓
Sends ChunkData to client
    ↓
Client renders chunks
```

**Block Breaking/Placing Validation:**
```
Client: "Break block at (100, 64, 100)"
    ↓
Server verifies:
  ├─ Is block in loaded chunk?
  ├─ Is player close enough? (max 6 blocks)
  ├─ Is block in line of sight?
  ├─ Is tool appropriate?
  └─ Anti-grief checks?
    ↓
If valid: Break, broadcast, drop item
If invalid: Reject, resend block data
```

**Key Functions:**
- `new(port) → Server` - Start server on port
- `update() → ()` - Main server loop
- `accept_connection() → ()` - Handle new client
- `validate_movement(player, new_pos) → bool` - Anti-cheat
- `broadcast_chunk(chunk) → ()` - Send to all clients

### `transport.rs` - Transport Layer
**Purpose:** Abstract network transport (QUIC or TCP).

**Abstraction:**
```rust
pub trait Transport {
    fn send(&mut self, data: Vec<u8>) -> Result<()>;
    fn recv(&mut self) -> Result<Option<Vec<u8>>>;
    fn is_connected(&self) -> bool;
}

pub struct QuicTransport { ... }  // implements Transport
pub struct TcpTransport { ... }   // implements Transport
```

**Selection Logic:**
```
Client tries to connect
    ↓
Attempt QUIC (faster, lower latency)
    ├─ Succeeds → Use QUIC
    └─ Fails → Try TCP (fallback)
         ├─ Succeeds → Use TCP
         └─ Fails → Error, cannot connect
```

### `quic.rs` - QUIC Protocol
**Purpose:** QUIC implementation for low-latency, reliable connections.

**QUIC Advantages:**
- Built on UDP (lower latency than TCP)
- TLS 1.3 encryption
- Connection migration (resume on network change)
- Multiplexed streams
- Better for games than TCP

**Connection Setup:**
```
Client                           Server
  │                               │
  ├─ QUIC Initial ──────────────→ │
  │                               │
  │ ← QUIC Handshake + Hello ─────┤
  │                               │
  ├─ QUIC Handshake Complete ──→ │
  │                               │
  ├─ Encrypted Data ────────────→ │
  │                               │
  Connected & Encrypted ✓
```

**Stream Multiplexing:**
```
Stream 0: Player movement (high frequency)
Stream 1: Chat messages (low frequency)
Stream 2: Chunk data (bulk transfer)
Stream 3: Entity updates

All on same connection (no head-of-line blocking)
```

**Key Functions:**
- `connect(addr) → QuicConnection` - Establish QUIC connection
- `open_stream() → Stream` - Create new bidirectional stream
- `send(data) → ()` - Send data (automatically fragmented)
- `recv() → Vec<u8>` - Receive data (automatically reassembled)

### `tcp.rs` - TCP Fallback
**Purpose:** TCP implementation as fallback when QUIC unavailable.

**Used When:**
- QUIC not available (very old OS, restrictive network)
- QUIC connection fails
- Server doesn't support QUIC

**TCP Limitations vs QUIC:**
- Higher latency (TCP handshake + TLS setup)
- Head-of-line blocking (one lost packet delays all)
- Connection restart on network change

**Key Functions:**
Similar API to QUIC for compatibility.

### `network.rs` - Shared Utilities
**Purpose:** Common networking functions used by both client and server.

**Utilities:**
```rust
pub fn compress_chunk(data: Vec<u8>) → Vec<u8>
pub fn decompress_chunk(data: Vec<u8>) → Vec<u8>
pub fn calculate_hash(data: &[u8]) → u32
pub fn is_valid_message(data: &[u8]) → bool
pub fn measure_latency() → Duration
```

**Chunk Compression:**
```
Uncompressed chunk: 65KB (16×16×256 blocks × 1 byte)
    ↓
Compress (gzip)
    ↓
~4-8 KB (90-95% compression)
    ↓
Send over network
    ↓
Decompress on client
```

**Delta Compression (Optimization):**
```
Full chunk update: 8KB
    ↓
Change 1 block
    ↓
Delta update: 
  - Block position (12 bytes)
  - New block type (1 byte)
  - Total: 13 bytes instead of 8KB
```

### `player.rs` - Remote Player Management
**Purpose:** Manages state of other players seen by local player.

**Key Types:**
```rust
pub struct RemotePlayer {
    pub id: u32
    pub username: String
    pub position: [f32; 3]
    pub rotation: [f32; 2]
    pub velocity: [f32; 3]
    pub model: PlayerModel
    pub last_update: Instant
    pub interpolation: Interpolator
}
```

**Player Interpolation:**
```
Receive pos update from server
  T0: position = (10, 64, 10)
  T1: position = (11, 64, 11) (100ms later)
  
Interpolate between frames
  T0.5: position = (10.5, 64, 10.5)
  
Visual smoothness ✓ (hides network latency)
```

**Player Rendering:**
- Draw character model at interpolated position
- Smooth rotation (yaw/pitch interpolation)
- Animation blending between movement states

## Network Protocol Summary

### Message Flow (Typical)

```
CONNECT PHASE:
Client → ClientHello
Server → ServerHello
Client → ClientReady
Server → Ready (start sending chunks)

GAMEPLAY PHASE (60 Hz):
Client → PlayerMove (every frame)
Server → PlayerMove (broadcast to others)
Server → ChunkData (as needed)
Client → BlockUpdate (on block change)
Server → BlockUpdate (broadcast)
```

### Bandwidth Estimation

**Per Player Per Second:**
- Position updates: 60 frames × 20 bytes = 1.2 KB/s
- Chunk data: 4 chunks × 8 KB = 32 KB/s (initial)
- Chat: Variable, ~1 KB/msg
- **Total**: ~40 KB/s per player

**For 100 Players Server:**
- Incoming: 40 KB/s × 100 = 4 MB/s
- Outgoing: 4 MB/s per client connection = 400 MB/s (not practical)

**Optimizations:**
- Chunk streaming (send only visible chunks)
- Movement compression (delta encoding)
- Interest management (only update nearby players)
- Chunk LOD (distant chunks lower detail)

## Server Architecture

### Multi-threaded Design
```
Main Thread (game loop)
    ├─ Update world
    ├─ Validate inputs
    └─ Broadcast updates
    
Network Threads (4×)
    ├─ Accept connections
    ├─ Process incoming messages
    ├─ Send outgoing data
    └─ Handle timeouts
```

### Scalability
```
Small Server: 1-4 players
Medium Server: 4-16 players
Large Server: 16-64 players (with optimization)
Massive: 64+ (requires sharding/clustering)
```

## Integration with Other Modules

```
multiplayer/ ←→ app/       (Game loop coordination)
multiplayer/ ←→ world/     (Chunk loading for clients)
multiplayer/ ←→ player/    (Remote player position)
multiplayer/ ←→ core/      (Chunk data serialization)
```

## Anti-Cheat & Security

**Server-Side Validation:**
```
Every client action verified on server:
  ✓ Block breaking distance
  ✓ Movement speed
  ✓ Item validity
  ✓ Block placement legality
```

**Encryption:**
```
All client-server communication encrypted (QUIC TLS 1.3)
  - Prevents packet inspection
  - Prevents message injection
  - Prevents man-in-the-middle attacks
```

**Detection:**
```
Suspicious behavior:
  ├─ Moving too fast
  ├─ Breaking blocks through walls
  ├─ Placing blocks at impossible positions
  └─ Breaking blocks too fast
  
Action: Warn, kick, or ban
```

---

**Key Takeaway:** The `multiplayer/` module provides a robust, secure client-server architecture using modern networking (QUIC), with authoritative server validation to ensure fair gameplay and smooth multiplayer experiences.

