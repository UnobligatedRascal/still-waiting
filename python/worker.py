"""
Training worker for agent-orchestrator.
Runs on NOUGHT. Manages Python training processes with Kepler-aware backends.

UnobligatedRascal — Making old hardware sing.
"""
import os
import json
import time
import sys
import re
import requests
import torch
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configuration
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "2048"))
ORCH_URL = os.getenv("ORCH_URL", "http://localhost:9999")
NUMA_NODE = int(os.getenv("NUMA_NODE", "0"))  # Worker's NUMA domain
GPUS_PER_NUMA = 4  # NUMA0=GPU0-3, NUMA1=GPU4-7

# Checkpoint directory (auto-resolves to writable path)
try:
    from checkpoints import get_checkpoint_dir
except ImportError:
    # Fallback if checkpoints module not available
    def get_checkpoint_dir():
        import os, pathlib
        path = pathlib.Path(os.getenv("CHECKPOINT_DIR", "/home/whistler/still-waiting/python/checkpoints"))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

CHECKPOINT_DIR = get_checkpoint_dir()

# Backend imports
from backends.transformers_backend import KeplerTransformersBackend

# Configuration
def get_local_rank():
    """Get local rank from torch.distributed or env."""
    return int(os.getenv("LOCAL_RANK", "0"))

def get_world_size():
    """Get world size from torch.distributed or env."""
    return int(os.getenv("WORLD_SIZE", "1"))

def send_log(job_id: str, level: str, message: str, step: Optional[int] = None):
    """POST a log entry to orchestrator. Non-blocking, silent on failure."""
    try:
        payload = {"level": level, "message": message}
        if step is not None:
            payload["step"] = step
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/logs",
            json=payload,
            timeout=2
        )
    except:
        pass

def notify_orchestrator(job_id: str, step: int, path: str, metrics: Dict[str, float]):
    """POST checkpoint to orchestrator."""
    try:
        resp = requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/checkpoint",
            json={
                "step": step,
                "path": path,
                "metrics": metrics
            },
            timeout=10
        )
        if resp.status_code != 200:
            msg = f"WARNING: Checkpoint report failed: {resp.status_code}"
            print(msg)
            send_log(job_id, "WARN", msg, step)
    except Exception as e:
        msg = f"ERROR: Failed to notify orchestrator: {e}"
        print(msg)
        send_log(job_id, "ERROR", msg)

def fail_job(job_id: str, error: str, step: Optional[int] = None):
    """Report job failure to orchestrator."""
    send_log(job_id, "ERROR", f"Job failed: {error}", step)
    try:
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/fail",
            json={"error": error},
            timeout=10
        )
    except:
        pass

def poll_conductor(job_id: str) -> Optional[str]:
    """Poll orchestrator for conductor instructions."""
    try:
        resp = requests.get(f"{ORCH_URL}/v1/training/jobs/{job_id}", timeout=5)
        if resp.status_code == 200:
            job = resp.json()
            if job.get("status") == "surgically_edited":
                # Get the latest instruction
                notes = job.get("conductor_notes", [])
                if notes:
                    return notes[-1]
            if job.get("status") == "paused":
                return "pause"
    except:
        pass
    return None

# Log line parser: extracts step number and level from common training output formats
LOG_LINE_RE = re.compile(r"(\d+)/\d+|step[_\s]*(\d+)", re.IGNORECASE)

def parse_log_line(line: str, current_step: int) -> tuple:
    """Parse a log line into (level, message, step_hint)."""
    level = "INFO"
    step_hint = current_step
    msg = line.strip()
    
    if any(w in msg.upper() for w in ["ERROR", "EXCEPTION", "FAILED", "FAILURE", "TRACEBACK"]):
        level = "ERROR"
    elif any(w in msg.upper() for w in ["WARNING", "WARN", "OOM", "OUT OF MEMORY", "DEPRECATED"]):
        level = "WARN"
    
    match = LOG_LINE_RE.search(msg)
    if match:
        step_hint = int(match.group(1) or match.group(2))
    
    return level, msg, step_hint

def run_job(job_id: str, job_config: Dict[str, Any]):
    """Run a training job. Called by worker pool or directly."""
    
    model_ref = job_config["model_ref"]
    target_steps = job_config.get("target_steps", 10000)
    config = job_config.get("config", {})
    current_step = job_config.get("current_step", 0)
    num_train_epochs = config.get("num_train_epochs", 1)
    
    msg = f"Starting job {job_id}: {model_ref} -> {target_steps} steps"
    print(msg)
    send_log(job_id, "INFO", msg)
    
    # Initialize backend
    try:
        backend = KeplerTransformersBackend(config)
        send_log(job_id, "INFO", f"Backend initialized: KeplerTransformersBackend")
    except Exception as e:
        fail_job(job_id, f"Backend init failed: {e}")
        raise
    
    # Prepare model
    try:
        backend.prepare(model_ref, config)
        send_log(job_id, "INFO", f"Model prepared: {model_ref}")
    except Exception as e:
        fail_job(job_id, f"Model prepare failed: {e}")
        raise
    
    # Notify orchestrator we're running
    try:
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/resume",
            timeout=10
        )
        send_log(job_id, "INFO", f"Job status changed to RUNNING")
    except Exception as e:
        send_log(job_id, "WARN", f"Failed to signal RUNNING status: {e}")
    
    # Step callback for periodic logging, conductor polling, and checkpointing
    def step_callback(step: int, loss: float, lr: float):
        nonlocal current_step
        current_step = step
        
        # Log periodic progress (every 16 steps)
        if step % 16 == 0:
            loss_val = f"{loss:.4f}"
            log_msg = f"Step {step}/{target_steps}, loss={loss_val}, lr={lr:.2e}"
            send_log(job_id, "INFO", log_msg, step)
            if get_local_rank() == 0:
                print(log_msg, flush=True)
        
        # Poll conductor every 64 steps
        if step % 64 == 0:
            instruction = poll_conductor(job_id)
            if instruction == "pause":
                msg = f"Job {job_id}: Paused by conductor"
                print(msg)
                send_log(job_id, "INFO", msg)
                raise SystemExit(0)  # Clean exit on pause
            elif instruction and instruction != "pause":
                msg = f"Job {job_id}: Conductor instruction: {instruction}"
                print(msg)
                send_log(job_id, "INFO", msg)
                apply_edit(backend, job_id, instruction)
        
        # Checkpoint every CHECKPOINT_EVERY steps
        if step % CHECKPOINT_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/{job_id}/step_{step}"
            os.makedirs(ckpt_path, exist_ok=True)
            metrics = {"loss": float(loss), "lr": float(lr), "step": step}
            backend.save_checkpoint(step, ckpt_path)
            notify_orchestrator(job_id, step, ckpt_path, metrics)
            send_log(job_id, "INFO", f"Checkpoint saved: step_{step} -> {ckpt_path}", step)
    
    # Training loop: iterate over epochs, each epoch processes full dataset
    for epoch in range(num_train_epochs):
        if current_step >= target_steps:
            break
        
        try:
            # Signal epoch start for DistributedSampler
            if hasattr(backend.data_loader.sampler, "set_epoch"):
                backend.data_loader.sampler.set_epoch(epoch)
            
            # Train one full epoch with step callback
            epoch_result = backend.train_epoch(step_callback=step_callback)
            current_step = backend.step_count
            
            # Log epoch summary
            loss_val = epoch_result.get("avg_loss", "N/A")
            if isinstance(loss_val, (int, float)):
                loss_val = f"{loss_val:.4f}"
            log_msg = f"Epoch {epoch+1}/{num_train_epochs} complete: {current_step}/{target_steps} steps, avg_loss={loss_val}"
            send_log(job_id, "INFO", log_msg, current_step)
            if get_local_rank() == 0:
                print(log_msg, flush=True)
            
        except SystemExit:
            # Clean exit from conductor pause
            raise
        except Exception as e:
            fail_job(job_id, f"Training error at step {current_step}: {e}", current_step)
            raise
    
    # Final checkpoint if we reached target steps but didn't align with CHECKPOINT_EVERY
    if current_step >= target_steps and current_step % CHECKPOINT_EVERY != 0:
        try:
            ckpt_path = f"{CHECKPOINT_DIR}/{job_id}/step_{current_step}"
            os.makedirs(ckpt_path, exist_ok=True)
            metrics = {"loss": 0.0, "lr": 0.0, "step": current_step}
            backend.save_checkpoint(current_step, ckpt_path)
            notify_orchestrator(job_id, current_step, ckpt_path, metrics)
            send_log(job_id, "INFO", f"Final checkpoint saved: step_{current_step} -> {ckpt_path}", current_step)
        except Exception as e:
            send_log(job_id, "WARN", f"Failed to save final checkpoint: {e}", current_step)
    
    # Complete
    try:
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/complete",
            timeout=10
        )
        msg = f"Job {job_id}: Completed at step {current_step}"
        print(msg)
        send_log(job_id, "INFO", msg)
    except Exception as e:
        msg = f"ERROR: Failed to mark job complete: {e}"
        print(msg)
        send_log(job_id, "ERROR", msg)

def apply_edit(backend, job_id: str, instruction: str):
    """Apply conductor instruction."""
    send_log(job_id, "INFO", f"Applying conductor edit: {instruction}")
    if instruction.startswith("snip_to:"):
        ckpt_id = instruction.split(":", 1)[1]
        msg = f"Job {job_id}: Snipping to checkpoint {ckpt_id}"
        print(msg)
        send_log(job_id, "INFO", msg)
        # TODO: Load checkpoint by ID
    elif instruction.startswith("inject_preference:"):
        data = json.loads(instruction.split(":", 1)[1])
        msg = f"Job {job_id}: Injecting preference data"
        print(msg)
        send_log(job_id, "INFO", msg)
        backend.inject_preference_data(data)
    elif instruction.startswith("adjust_lr:"):
        new_lr = float(instruction.split(":", 1)[1])
        msg = f"Job {job_id}: Adjusting LR to {new_lr}"
        print(msg)
        send_log(job_id, "INFO", msg)
        backend.set_learning_rate(new_lr)

def main():
    """Entry point. Called by torchrun or directly."""
    
    # Parse job info from args or env
    job_id = os.getenv("JOB_ID") or sys.argv[1]
    job_config_path = os.getenv("JOB_CONFIG") or sys.argv[2]
    
    with open(job_config_path) as f:
        job_config = json.load(f)
    
    # Initialize distributed if needed
    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
    
    local_rank = get_local_rank()
    send_log(job_id, "INFO", f"Worker started: LOCAL_RANK={local_rank}, WORLD_SIZE={get_world_size()}, NUMA={NUMA_NODE}")

    run_job(job_id, job_config)

if __name__ == "__main__":
    main()
