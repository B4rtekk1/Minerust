use std::sync::Arc;

use crate::logger::{LogLevel, log};
use crate::multiplayer::protocol::Packet;
use crate::multiplayer::tcp::TcpServer;

/// Runs a standalone dedicated multiplayer server that accepts TCP connections
/// and relays packets between all connected clients.
///
/// # Lifecycle
///
/// 1. Binds a [`TcpServer`] to `addr`.
/// 2. Enters an infinite accept loop on the calling task.
/// 3. For each accepted connection, spawns a dedicated Tokio task that owns
///    the receive loop for that client.
/// 4. The server runs until the process is killed; there is currently no
///    graceful shutdown signal.
///
/// # Packet handling
///
/// Every packet received from a client is mutated so its `player_id` field
/// reflects the server-assigned connection ID rather than whatever the client
/// sent.  This prevents clients from spoofing another player's identity.
///
/// | Packet variant   | Server action                                                    |
/// |------------------|------------------------------------------------------------------|
/// | `Connect`        | Overwrites `player_id`; sends a `ConnectAck` back to the sender.|
/// | `Position`       | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Rotation`       | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Chat`           | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Disconnect`     | Overwrites `player_id`; broadcast to all other clients.         |
/// | All other types  | Broadcast as-is (no mutation).                                  |
///
/// On a receive error the client is considered disconnected: a synthetic
/// `Disconnect` packet is broadcast to all remaining peers and the client is
/// removed from the server's connection table.
///
/// # Parameters
/// - `addr` – The `host:port` string to listen on (e.g. `"0.0.0.0:25565"`).
///
/// # Errors
/// Logs to `stderr` and returns early if the server cannot bind to `addr`.
/// Per-client receive/send errors are logged but do not terminate the server.
pub async fn run_dedicated_server(addr: &str) {
    match TcpServer::bind(addr).await {
        Ok(server_inst) => {
            // Wrap in Arc so the handle can be cheaply cloned into each
            // per-client task without requiring a global or thread-local.
            let server = Arc::new(server_inst);
            log(
                LogLevel::Info,
                &format!("Server successfully bound to {}", addr),
            );
            log(LogLevel::Info, "Waiting for connections...");
            // Flush stdout immediately so the operator sees the startup
            // message even if stdout is line-buffered (e.g., piped to a file).
            let _ = std::io::Write::flush(&mut std::io::stdout());

            let server_seed: u32 = rand::random();
            log(LogLevel::Info, &format!("Server world seed: {}", server_seed));

            // Runs on the calling task forever.  Each accepted connection is
            // handed off to a new Tokio task so `accept` is free to resume
            // waiting for the next client immediately.
            loop {
                match server.accept().await {
                    Ok((id, conn)) => {
                        log(
                            LogLevel::Info,
                            &format!(
                                "Accepted connection from {} with assigned ID {}",
                                conn.addr(),
                                id
                            ),
                        );
                        // Clone the Arc handle; the spawned task takes ownership
                        // of this clone so the borrow checker is satisfied.
                        let server_clone = server.clone();

                        // ── Per-client receive loop (spawned task) ──────── //
                        tokio::spawn(async move {
                            loop {
                                match conn.recv().await {
                                    Ok(mut packet) => {
                                        // ── Player-ID stamping ──────────── //
                                        // Overwrite the `player_id` field on
                                        // every packet variant that carries one.
                                        // This ensures that broadcasted packets
                                        // always carry the server-authoritative ID
                                        // rather than whatever the client supplied,
                                        // preventing identity spoofing.
                                        match packet {
                                            Packet::Connect {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                                // `ConnectAck` tells the client
                                                // which ID the server assigned to
                                                // it so it can stamp outgoing
                                                // packets correctly from here on.
                                                let ack = Packet::ConnectAck {
                                                    success: true,
                                                    player_id: id,
                                                    seed: server_seed,
                                                };
                                                let _ = conn.send(&ack).await;
                                            }
                                            Packet::Position {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Rotation {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Chat {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Disconnect {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            // Packet variants that carry no
                                            // player_id (e.g. server-only control
                                            // packets) are forwarded unchanged.
                                            _ => {}
                                        }

                                        // Relay the (possibly mutated) packet to
                                        // every client except the one that sent it.
                                        // Errors here are intentionally ignored:
                                        // a failed send to one peer should not
                                        // drop the packet for all others.
                                        let _ = server_clone.broadcast_except(&packet, id).await;
                                    }

                                    // ── Client disconnection ────────────── //
                                    // Any receive error is treated as a clean
                                    // disconnect (TCP RST, EOF, decode failure).
                                    Err(_) => {
                                        log(
                                            LogLevel::Info,
                                            &format!(
                                                "Connection error with client {}; treating as disconnect",
                                                id
                                            ),
                                        );
                                        // Synthesize a Disconnect packet so that
                                        // remaining clients can remove this player
                                        // from their local state (despawn model,
                                        // remove name tag, etc.).
                                        let disconnect_packet =
                                            Packet::Disconnect { player_id: id };
                                        let _ = server_clone
                                            .broadcast_except(&disconnect_packet, id)
                                            .await;

                                        // Remove the connection from the server's
                                        // internal table so it is no longer
                                        // included in future broadcasts.
                                        server_clone.remove_client(id).await;

                                        // Exit the receive loop; the task ends
                                        // naturally and the connection is dropped.
                                        break;
                                    }
                                }
                            }
                        });
                    }

                    Err(e) => {
                        // A single failed accept does not abort the server;
                        // log the error and continue waiting for the next client.
                        log(LogLevel::Error, &format!("Accept error: {}", e));
                    }
                }
            }
        }

        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Failed to bind server to {}: {}", addr, e),
            );
        }
    }
}
