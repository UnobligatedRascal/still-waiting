mod api;
mod backend;
mod conductor;
mod state;

use axum::routing::get;
use state::AppState;
use tower_http::services::ServeDir;
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

    let bind_addr = std::env::var("ORCH_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0".into());
    let bind_port = std::env::var("ORCH_BIND_PORT").unwrap_or_else(|_| "9999".into());
    let addr = format!("{}:{}", bind_addr, bind_port);

    let static_dir = std::env::var("ORCH_STATIC_DIR").unwrap_or_else(|_| "gui/dist".into());

    tracing::info!("starting still-waiting orchestrator");
    tracing::info!("binding to {}", addr);
    tracing::info!("GUI: http://{}", addr);
    tracing::info!("API: http://{}/v1", addr);
    tracing::info!("static dir: {}", static_dir);

    let state = AppState::new();

    // API routes under /v1
    let api_router = api::routes(state.clone());

    // Static file serving with SPA fallback
    let app = axum::Router::new()
        .nest("/v1", api_router)
        .route("/", get(spa_fallback))
        .fallback_service(
            ServeDir::new(&static_dir)
                .not_found_service(get(spa_fallback)),
        );

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!("listening on {}", addr);
    axum::serve(listener, app).await.unwrap();
}

async fn spa_fallback() -> axum::response::Redirect {
    axum::response::Redirect::to("/index.html")
}
