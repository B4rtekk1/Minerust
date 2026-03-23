/// Represents a remote player in the game world, storing their position,
/// orientation, and display name.
///
/// Received from the server and used for rendering other players
/// and their nametags in the world.
#[derive(Debug, Clone)]
pub struct RemotePlayer {
    /// World-space X coordinate.
    pub x: f32,
    /// World-space Y coordinate (vertical axis).
    pub y: f32,
    /// World-space Z coordinate.
    pub z: f32,
    /// Horizontal rotation in radians. `0.0` faces +Z; increases counter-clockwise.
    pub yaw: f32,
    /// Vertical look angle in radians. Positive values look upward.
    pub pitch: f32,
    /// The player's display name, shown above their head as a nametag.
    pub username: String,
}

/// A resolved screen-space label for a remote player, ready to be passed
/// to the UI/text rendering layer.
///
/// Produced by [`queue_remote_players_labels`] after projecting each player's
/// world position into 2-D screen coordinates.
pub struct PlayerLabel {
    /// The player's display name to render.
    pub username: String,
    /// Horizontal screen position in pixels (origin at left edge).
    pub screen_x: f32,
    /// Vertical screen position in pixels (origin at top edge).
    pub screen_y: f32,
}

/// Projects all remote players' nametag positions into screen space and
/// returns a list of labels ready for rendering.
///
/// For each player a point `2.2` units above their feet is transformed by
/// `view_proj` into clip space. Players behind the camera (`w ≤ 0`) are
/// culled and produce no label. The surviving clip-space positions are
/// converted to pixel coordinates using the standard NDC-to-screen mapping:
///
/// ```text
/// screen_x = (x_ndc + 1) / 2 * width
/// screen_y = (1 - y_ndc) / 2 * height   // Y is flipped: NDC +Y is screen top
/// ```
///
/// # Arguments
///
/// * `remote_players` - Map of player ID → [`RemotePlayer`] received from the server.
/// * `view_proj`      - Combined view–projection matrix for the local camera.
/// * `width`          - Render target width in pixels.
/// * `height`         - Render target height in pixels.
///
/// # Returns
///
/// A [`Vec<PlayerLabel>`] containing one entry per visible remote player.
/// The order of entries is arbitrary (reflects `HashMap` iteration order).
pub fn queue_remote_players_labels(
    remote_players: &std::collections::HashMap<u32, RemotePlayer>,
    view_proj: &glam::Mat4,
    width: f32,
    height: f32,
) -> Vec<PlayerLabel> {
    let mut labels = Vec::new();

    for (_id, player) in remote_players {
        // Place the label origin slightly above the player's head.
        let pos = glam::Vec4::new(player.x, player.y + 2.2, player.z, 1.0);
        let clip_pos = *view_proj * pos;

        // Cull players behind the camera; w ≤ 0 means the point is at or
        // behind the near plane, making the perspective divide undefined.
        if clip_pos.w > 0.0 {
            let screen_x = (clip_pos.x / clip_pos.w + 1.0) / 2.0 * width;
            let screen_y = (1.0 - clip_pos.y / clip_pos.w) / 2.0 * height;

            labels.push(PlayerLabel {
                username: player.username.clone(),
                screen_x,
                screen_y,
            });
        }
    }
    labels
}