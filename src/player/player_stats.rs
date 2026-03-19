pub struct PlayerInfo {
    pub health: f32,
    pub hunger: f32,
    pub saturation: f32,
    pub energy: f32,
    pub experience: f32,
    pub level: u32,
}

impl PlayerInfo {
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

    pub fn add_experience(&mut self, amount: f32) {
        self.experience += amount;
        while self.experience >= self.experience_to_next_level() {
            self.experience -= self.experience_to_next_level();
            self.level += 1;
        }
    }

    /// ![](https://raw.githubusercontent.com/B4rtekk1/Minerust/main/assets_docs/exp_equation.png)
    pub fn experience_to_next_level(&self) -> f32 {
        (self.level as f32 + 1.0).powf(2.0) * 10.0
        // 1 level: 40exp
        // 2 level: 90exp
        // 3 level: 160exp
        //10 le
    }
}