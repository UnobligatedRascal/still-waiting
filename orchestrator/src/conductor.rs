use crate::state::{AppState, CheckpointMeta, JobStatus};
use anyhow::Result;
use uuid::Uuid;

/// Evaluate training progress and potentially trigger a surgical edit.
/// 
/// This is the hook for the "median progress arc" logic—comparing
/// actual training trajectory against expected, deciding whether to
/// snip back to an earlier checkpoint and re-train with modifications.
/// 
/// Future implementation:
/// - Load "golden arc" metrics (expected loss curve shape)
/// - Compare latest checkpoint metrics to arc
/// - If deviation > threshold → recommend snip_to:<earlier_ckpt_id>
/// - If stagnating → recommend hyperparameter change
/// - Human-in-the-loop: mark suspicious checkpoints for review
pub async fn evaluate_and_maybe_edit(
    state: &AppState,
    job_id: Uuid,
    latest_ckpt: &CheckpointMeta,
) -> Result<Option<String>> {
    // TODO: Implement median arc comparison logic
    tracing::info!(
        job_id = %job_id,
        step = latest_ckpt.step,
        ckpt_id = %latest_ckpt.id,
        loss = ?latest_ckpt.metrics.get("loss"),
        "Conductor evaluating checkpoint",
    );

    // For now: no automatic edits. Return Ok(None) = "keep training"
    Ok(None)
}

/// Apply a surgical edit instruction to a job.
/// 
/// Supported instruction formats (future):
/// - "snip_to:<ckpt_id>" — load earlier checkpoint, discard progress after it
/// - "inject_preference:<json>" — add RLHF-style preference data
/// - "adjust_lr:<value>" — change learning rate mid-training
/// - "merge_with:<job_id>" — blend weights from another job
pub async fn apply_surgical_edit(
    state: &AppState,
    job_id: Uuid,
    edit_instruction: &str,
) -> Result<()> {
    let mut jobs = state.jobs.write().await;
    if let Some(job) = jobs.get_mut(&job_id) {
        job.status = JobStatus::SurgicallyEdited;
        job.conductor_notes.push(edit_instruction.to_string());
        job.updated_at = chrono::Utc::now();
        
        tracing::info!(
            job_id = %job_id,
            instruction = edit_instruction,
            "Applied surgical edit",
        );
    } else {
        return Err(anyhow::anyhow!("Job not found: {}", job_id));
    }
    Ok(())
}
