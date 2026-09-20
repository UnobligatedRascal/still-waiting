use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Checkpointed,
    Paused,
    Completed,
    Failed,
    SurgicallyEdited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub id: Uuid,
    pub step: u64,
    pub path: String,
    pub created_at: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub id: Uuid,
    pub model_ref: String,
    pub format: String,
    pub status: JobStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub current_step: u64,
    pub target_steps: Option<u64>,
    pub context_length: Option<u64>,
    pub checkpoints: Vec<CheckpointMeta>,
    pub config: serde_json::Value,
    pub conductor_notes: Vec<String>,
    // Worker tracking
    pub worker_pid: Option<u32>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: Uuid,
    pub name: String,
    pub format: String,
    pub path: String,
    pub source_job_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Default)]
pub struct AppState {
    pub jobs: Arc<RwLock<HashMap<Uuid, TrainingJob>>>,
    pub models: Arc<RwLock<HashMap<Uuid, ModelEntry>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_job(&self, id: &Uuid) -> Option<TrainingJob> {
        self.jobs.read().await.get(id).cloned()
    }

    pub async fn list_jobs(&self) -> Vec<TrainingJob> {
        self.jobs.read().await.values().cloned().collect()
    }
}
