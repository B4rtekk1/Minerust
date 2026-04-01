/// Tracks all vital statistics and progression for the local player.
///
/// All stat fields use `f32` for smooth interpolation in the HUD.
/// Health and hunger follow a 0–20 scale (matching classic voxel-game conventions);
/// energy uses 0–100, and experience resets to 0 at each level-up.
pub struct PlayerInfo {
    /// Current health points. Range: `[0.0, 20.0]`.
    ///
    /// At `0.0` the player is dead. Damage and regeneration modify this value.
    pub health: f32,

    /// Current hunger points. Range: `[0.0, 20.0]`.
    ///
    /// Decreases over time and with physical activity. When hunger reaches `0.0`,
    /// health regeneration stops and the player may begin taking damage.
    pub hunger: f32,

    /// Hunger saturation buffer. Depleted before [`Self::hunger`] starts falling.
    ///
    /// Acts as a secondary hunger reserve. Food items typically restore both
    /// hunger and saturation. Initialised to `5.0`.
    pub saturation: f32,

    /// Current energy level. Range: `[0.0, 100.0]`.
    ///
    /// Used for actions such as sprinting or mining. Regenerates when the
    /// player is idle and hunger is sufficient.
    pub energy: f32,

    /// Accumulated experience points within the current level.
    ///
    /// Resets to `0.0` on level-up (excess XP carries over). Use
    /// [`Self::add_experience`] to modify this field so level-ups are handled
    /// automatically.
    pub experience: f32,

    /// Current player level. Starts at `0` and increments via [`Self::add_experience`].
    pub level: u32,
}

impl PlayerInfo {
    /// Creates a new `PlayerInfo` with full stats and no progression.
    ///
    /// Default values: health `20.0`, hunger `20.0`, saturation `5.0`,
    /// energy `100.0`, experience `0.0`, level `0`.
    pub fn new() -> Self {
        Self {
            health: 20.0,
            hunger: 20.0,
            saturation: 5.0,
            energy: 100.0,
            experience: 0.0,
            level: 0,
        }
    }

    /// Adds `amount` experience points and triggers level-ups as needed.
    ///
    /// If the accumulated experience meets or exceeds [`Self::experience_to_next_level`],
    /// the level increments and excess XP carries over. Multiple levels can be
    /// gained in a single call if `amount` is large enough.
    pub fn add_experience(&mut self, amount: f32) {
        self.experience += amount;
        while self.experience >= self.experience_to_next_level() {
            self.experience -= self.experience_to_next_level();
            self.level += 1;
        }
    }

    /// Returns the experience points required to advance from the current level to the next.
    ///
    /// Computed as:
    ///
    /// ![XP formula](https://raw.githubusercontent.com/B4rtekk1/Minerust/main/assets_docs/exp_equation.png)
    ///
    /// ```text
    /// xp = (level + 1)² × 10
    /// ```
    ///
    /// | Level | XP required |
    /// |-------|-------------|
    /// | 0     | 10          |
    /// | 1     | 40          |
    /// | 2     | 90          |
    /// | 3     | 160         |
    /// | 10    | 1210        |
    pub fn experience_to_next_level(&self) -> f32 {
        (self.level as f32 + 1.0).powf(2.0) * 10.0
    }
}
