//! Mini Minecraft 3D Game
//!
//! Main entry point that delegates to the app module.

mod app;
mod multiplayer;
mod ui;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("off")),
        )
        .init();

    app::run_game();
}
