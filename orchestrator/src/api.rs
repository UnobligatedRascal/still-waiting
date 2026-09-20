use crate::state::{AppState, JobStatus, TrainingJob};
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
        .route("/v1/training/jobs/:id/complete", post(complete_job))
        .route("/v1/training/jobs/:id/fail", post(fail_job))
        // Models list
        .route("/v1/models", get(list_models))
        .with_state(state)
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
