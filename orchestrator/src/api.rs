use crate::state::{AppState, JobStatus, TrainingJob, LogLevel, LogEntry};
use anyhow::Result;
use axum::{
    extract::{Path, State, Json},
    http::StatusCode,
    response::Json as JsonResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::conductor;

#[derive(Serialize)]
pub struct JobResponse {
    pub id: String,
    pub status: String,
    pub object: String,
}

#[derive(Deserialize)]
pub struct CreateJobRequest {
    pub model: String,
    pub format: Option<String>,
    pub config: Option<serde_json::Value>,
    pub target_steps: Option<u64>,
    pub context_length: Option<u64>,
}

#[derive(Deserialize)]
pub struct ConductorFeedback {
    pub instruction: String,
}

#[derive(Deserialize)]
pub struct CheckpointRequest {
    pub metrics: serde_json::Value,
    pub path: String,
    pub step: u64,
}

#[derive(Serialize)]
pub struct SystemStatus {
    pub orchestrator: OrchestratorInfo,
    pub jobs: JobCounts,
    pub gpu_info: Option<String>,
    pub restart_instructions: String,
}

#[derive(Serialize)]
pub struct OrchestratorInfo {
    pub version: String,
    pub uptime_seconds: u64,
    pub bind_addr: String,
}

#[derive(Serialize)]
pub struct JobCounts {
    pub total: usize,
    pub running: usize,
    pub queued: usize,
    pub paused: usize,
    pub completed: usize,
    pub failed: usize,
}

pub fn routes(state: AppState) -> Router {
    Router::new()
        // Training jobs
        .route("/v1/training/jobs", post(create_job))
        .route("/v1/training/jobs", get(list_jobs))
        .route("/v1/training/jobs/:id", get(get_job))
        .route("/v1/training/jobs/:id/pause", post(pause_job))
        .route("/v1/training/jobs/:id/resume", post(resume_job))
        .route("/v1/training/jobs/:id/checkpoint", post(report_checkpoint))
        .route("/v1/training/jobs/:id/conductor", post(conductor_feedback))
        .route("/v1/training/jobs/:id/logs", post(report_log))
        .route("/v1/training/jobs/:id/logs", get(get_logs))
        .route("/v1/training/jobs/:id/complete", post(complete_job))
        .route("/v1/training/jobs/:id/fail", post(fail_job))
        // Models list
        .route("/v1/models", get(list_models))
        // System
        .route("/v1/system/status", get(system_status))
        .with_state(state)
}

async fn system_status(
    State(state): State<AppState>,
) -> JsonResponse<SystemStatus> {
    let jobs = state.list_jobs().await;
    let counts = JobCounts {
        total: jobs.len(),
        running: jobs.iter().filter(|j| j.status == JobStatus::Running).count(),
        queued: jobs.iter().filter(|j| j.status == JobStatus::Queued).count(),
        paused: jobs.iter().filter(|j| j.status == JobStatus::Paused).count(),
        completed: jobs.iter().filter(|j| j.status == JobStatus::Completed).count(),
        failed: jobs.iter().filter(|j| j.status == JobStatus::Failed).count(),
    };

    let uptime = std::time::UNIX_EPOCH.elapsed().map_or(0, |d| d.as_secs());
    let bind_addr = std::env::var("ORCH_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0".into())
        + ":" + &std::env::var("ORCH_BIND_PORT").unwrap_or_else(|_| "9999".into());

    // Try to get GPU info (best effort)
    let gpu_info = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=index,name,memory.used,memory.total,temperature.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .filter(|s| !s.is_empty());

    JsonResponse(SystemStatus {
        orchestrator: OrchestratorInfo {
            version: env!("CARGO_PKG_VERSION").into(),
            uptime_seconds: uptime,
            bind_addr,
        },
        jobs: counts,
        gpu_info,
        restart_instructions: "Use startup script: sudo <PROJECT_ROOT>/deploy/start_still_waiting.sh restart\nOr systemd: sudo systemctl restart still-waiting-orchestrator".into(),
    })
}

async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> Result<JsonResponse<JobResponse>, StatusCode> {
    let id = Uuid::new_v4();
    let now = chrono::Utc::now();
    let job = TrainingJob {
        id,
        model_ref: req.model,
        format: req.format.unwrap_or_else(|| "transformers".into()),
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        current_step: 0,
        target_steps: req.target_steps,
        context_length: req.context_length,
        checkpoints: vec![],
        config: req.config.unwrap_or(serde_json::json!({})),
        conductor_notes: vec![],
        worker_pid: None,
        error_message: None,
    };

    state.jobs.write().await.insert(id, job.clone());

    // TODO: Signal worker pool to pick up this job
    tracing::info!(
        job_id = %id,
        model = %job.model_ref,
        target_steps = ?job.target_steps,
        "Training job created",
    );

    // Immediately mark as running (worker pool will pick it up)
    job.status; // suppress unused warning, TODO: update status

    Ok(JsonResponse(JobResponse {
        id: id.to_string(),
        status: "queued".into(),
        object: "training.job".into(),
    }))
}

async fn list_jobs(
    State(state): State<AppState>,
) -> JsonResponse<serde_json::Value> {
    let jobs = state.list_jobs().await;
    JsonResponse(serde_json::json!({
        "object": "list",
        "data": jobs
    }))
}

async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<JsonResponse<TrainingJob>, StatusCode> {
    let job = state.get_job(&id).await;
    match job {
        Some(job) => Ok(JsonResponse(job)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn pause_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<JsonResponse<JobResponse>, StatusCode> {
    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&id) {
        job.status = JobStatus::Paused;
        job.updated_at = chrono::Utc::now();
        // TODO: Signal worker to pause
        Ok(JsonResponse(JobResponse {
            id: id.to_string(),
            status: "paused".into(),
            object: "training.job".into(),
        }))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn resume_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<JsonResponse<JobResponse>, StatusCode> {
    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&id) {
        job.status = JobStatus::Running;
        job.updated_at = chrono::Utc::now();
        // TODO: Signal worker to resume
        Ok(JsonResponse(JobResponse {
            id: id.to_string(),
            status: "running".into(),
            object: "training.job".into(),
        }))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn report_checkpoint(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
    Json(req): Json<CheckpointRequest>,
) -> Result<StatusCode, StatusCode> {
    use crate::state::CheckpointMeta;

    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&job_id) {
        let path_clone = req.path.clone();
        let ckpt = CheckpointMeta {
            id: Uuid::new_v4(),
            step: req.step,
            path: req.path,
            created_at: chrono::Utc::now(),
            metrics: serde_json::from_value(req.metrics.clone()).unwrap_or_default(),
            notes: None,
        };
        job.checkpoints.push(ckpt.clone());
        job.current_step = req.step;
        job.updated_at = chrono::Utc::now();

        // Run conductor evaluation
        drop(jobs);
        if let Ok(Some(instruction)) = conductor::evaluate_and_maybe_edit(&state, job_id, &ckpt).await {
            conductor::apply_surgical_edit(&state, job_id, &instruction).await.ok();
        }

        tracing::info!(
            job_id = %job_id,
            step = req.step,
            ckpt_path = %path_clone,
            "Checkpoint reported",
        );
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn conductor_feedback(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(body): Json<ConductorFeedback>,
) -> Result<StatusCode, StatusCode> {
    conductor::apply_surgical_edit(&state, id, &body.instruction)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::OK)
}

async fn complete_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, StatusCode> {
    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&id) {
        job.status = JobStatus::Completed;
        job.updated_at = chrono::Utc::now();
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn fail_job(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(error): Json<serde_json::Value>,
) -> Result<StatusCode, StatusCode> {
    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&id) {
        job.status = JobStatus::Failed;
        job.error_message = Some(error.to_string());
        job.updated_at = chrono::Utc::now();
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn list_models(
    State(state): State<AppState>,
) -> JsonResponse<serde_json::Value> {
    let models = state.models.read().await.values().cloned().collect::<Vec<_>>();
    JsonResponse(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

#[derive(Deserialize)]
pub struct LogRequest {
    pub level: Option<String>,
    pub message: String,
    pub step: Option<u64>,
}

async fn report_log(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
    Json(req): Json<LogRequest>,
) -> Result<StatusCode, StatusCode> {
    let level = match req.level.as_deref() {
        Some("WARN") | Some("WARNING") => LogLevel::WARN,
        Some("ERROR") => LogLevel::ERROR,
        Some("DEBUG") => LogLevel::DEBUG,
        _ => LogLevel::INFO,
    };
    let entry = LogEntry {
        timestamp: chrono::Utc::now(),
        level,
        message: req.message,
        step: req.step,
    };
    state.add_log(&job_id, entry).await;
    Ok(StatusCode::OK)
}

#[derive(Deserialize)]
pub struct GetLogsQuery {
    #[serde(default)]
    pub from: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    200
}

async fn get_logs(
    State(state): State<AppState>,
    Path(job_id): Path<Uuid>,
    axum::extract::Query(query): axum::extract::Query<GetLogsQuery>,
) -> Result<JsonResponse<LogsResponse>, StatusCode> {
    let logs = state.get_logs(&job_id, query.from, query.limit).await;
    let total = state.log_count(&job_id).await;
    Ok(JsonResponse(LogsResponse {
        logs,
        total,
        from: query.from,
        limit: query.limit,
    }))
}

#[derive(Serialize)]
pub struct LogsResponse {
    pub logs: Vec<LogEntry>,
    pub total: usize,
    pub from: usize,
    pub limit: usize,
}
