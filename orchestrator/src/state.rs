use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Log level for training job logs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum LogLevel {
    INFO,
    WARN,
    ERROR,
    DEBUG,
}

/// A single log entry from a training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub step: Option<u64>,
}

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

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<RwLock<HashMap<Uuid, TrainingJob>>>,
    pub models: Arc<RwLock<HashMap<Uuid, ModelEntry>>>,
    pub logs: Arc<RwLock<HashMap<Uuid, Vec<LogEntry>>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            logs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn get_job(&self, id: &Uuid) -> Option<TrainingJob> {
        self.jobs.read().await.get(id).cloned()
    }

    pub async fn list_jobs(&self) -> Vec<TrainingJob> {
        self.jobs.read().await.values().cloned().collect()
    }

    /// Add a log entry to a job. Keeps max 5000 entries per job.
    pub async fn add_log(&self, job_id: &Uuid, entry: LogEntry) {
        let mut logs = self.logs.write().await;
        let entries = logs.entry(*job_id).or_insert_with(Vec::new);
        entries.push(entry);
        if entries.len() > 5000 {
            entries.drain(..entries.len() - 5000);
        }
    }

    /// Get logs for a job with optional pagination.
    pub async fn get_logs(&self, job_id: &Uuid, from: usize, limit: usize) -> Vec<LogEntry> {
        let logs = self.logs.read().await;
        let entries = match logs.get(job_id) {
            Some(e) => e,
            None => return vec![],
        };
        let from = from.min(entries.len());
        let to = (from + limit).min(entries.len());
        entries[from..to].to_vec()
    }

    /// Get log count for a job.
    pub async fn log_count(&self, job_id: &Uuid) -> usize {
        let logs = self.logs.read().await;
        logs.get(job_id).map_or(0, |e| e.len())
    }
}
