mod api;
mod backend;
mod conductor;
mod state;

use state::AppState;
use tracing_subscriber;

#[tokio::main]
async fn main() {
    // Init logging
    tracing_subscriber::fmt()
        .with_target(true)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    // Load config from env
    let bind_addr = std::env::var("ORCH_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0".into());
    let bind_port = std::env::var("ORCH_BIND_PORT").unwrap_or_else(|_| "9999".into());
    let addr = format!("{}:{}", bind_addr, bind_port);

    tracing::info!("starting still-waiting orchestrator");
    tracing::info!("binding to {}", addr);
    tracing::info!("API base: http://{}/v1", addr);

    // Build app state
    let state = AppState::new();

    // Build router
    let app = api::routes(state);

    // Serve
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!("listening on {}", addr);
    axum::serve(listener, app).await.unwrap();
}
