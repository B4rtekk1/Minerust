#[derive(Default)]
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump_held: bool,
    pub jump: bool,
    pub sprint: bool,
    pub sneak: bool,
    pub left_mouse: bool,
    pub right_mouse: bool,
}

#[derive(Default)]
pub struct DiggingState {
    pub target: Option<(i32, i32, i32)>,
    pub progress: f32,
    pub break_time: f32,
}
