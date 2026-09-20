/// Backend dispatch trait for model training.
/// 
/// Concrete implementations run in the Python worker.
/// This trait defines the Rust-side contract for IPC with training backends.
///
/// For Kepler sm_37 specifics, see the Python backends module.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Prepare model for training (load weights, apply adapters, etc.)
    async fn prepare(&self, model_ref: &str, config: &Value) -> Result<()>;
    
    /// Execute one training step (or batch). Returns metrics including loss.
    async fn train_step(&self) -> Result<Value>;
    
    /// Save a checkpoint at the given step.
    async fn save_checkpoint(&self, step: u64, path: &str) -> Result<()>;
    
    /// Load a checkpoint from the given path.
    async fn load_checkpoint(&self, path: &str) -> Result<()>;
    
    /// Export the trained model in the requested format.
    /// Supported: "safetensors", "gguf"
    async fn export(&self, format: &str, path: &str) -> Result<()>;
}

/// Python worker IPC client.
/// 
/// The worker runs as a separate process (or torchrun-managed processes)
/// and communicates via HTTP or shared filesystem.
#[async_trait]
pub trait WorkerClient: Send + Sync {
    /// Launch a training job on the worker.
    async fn launch_job(&self, job_id: &str, config: &Value) -> Result<()>;
    
    /// Pause an active job.
    async fn pause_job(&self, job_id: &str) -> Result<()>;
    
    /// Resume a paused job.
    async fn resume_job(&self, job_id: &str) -> Result<()>;
    
    /// Terminate a job.
    async fn kill_job(&self, job_id: &str) -> Result<()>;
}

// TODO: Implement concrete WorkerClient that manages Python worker processes.
// Options:
// - HTTP client to a worker REST API
// - Direct subprocess management with std::process::Command
// - Shared filesystem with sentinel files
