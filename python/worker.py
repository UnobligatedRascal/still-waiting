"""
Training worker for agent-orchestrator.
Runs on NOUGHT. Manages Python training processes with Kepler-aware backends.

UnobligatedRascal — Making old hardware sing.
"""
import os
import json
import time
import sys
import requests
import torch
from pathlib import Path
from typing import Any, Dict, Optional

# Backend imports
from backends.transformers_backend import KeplerTransformersBackend

# Configuration
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "2048"))
ORCH_URL = os.getenv("ORCH_URL", "http://localhost:9999")
NUMA_NODE = int(os.getenv("NUMA_NODE", "0"))  # Worker's NUMA domain
GPUS_PER_NUMA = 4  # NUMA0=GPU0-3, NUMA1=GPU4-7

def get_local_rank():
    """Get local rank from torch.distributed or env."""
    return int(os.getenv("LOCAL_RANK", "0"))

def get_world_size():
    """Get world size from torch.distributed or env."""
    return int(os.getenv("WORLD_SIZE", "1"))

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
            print(f"WARNING: Checkpoint report failed: {resp.status_code}")
    except Exception as e:
        print(f"ERROR: Failed to notify orchestrator: {e}")

def fail_job(job_id: str, error: str):
    """Report job failure to orchestrator."""
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

def run_job(job_id: str, job_config: Dict[str, Any]):
    """Run a training job. Called by worker pool or directly."""
    
    model_ref = job_config["model_ref"]
    target_steps = job_config.get("target_steps", 10000)
    config = job_config.get("config", {})
    current_step = job_config.get("current_step", 0)
    
    print(f"Starting job {job_id}: {model_ref} -> {target_steps} steps")
    
    # Initialize backend
    try:
        backend = KeplerTransformersBackend(config)
    except Exception as e:
        fail_job(job_id, f"Backend init failed: {e}")
        raise
    
    # Prepare model
    try:
        backend.prepare(model_ref, config)
    except Exception as e:
        fail_job(job_id, f"Model prepare failed: {e}")
        raise
    
    # Notify orchestrator we're running
    try:
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/resume",
            timeout=10
        )
    except:
        pass
    
    # Training loop
    while current_step < target_steps:
        try:
            # Train step
            metrics = backend.train_step()
            current_step += 1
            
            # Check conductor
            instruction = poll_conductor(job_id)
            if instruction == "pause":
                print(f"Job {job_id}: Paused by conductor")
                break
            elif instruction and instruction != "pause":
                print(f"Job {job_id}: Conductor instruction: {instruction}")
                apply_edit(backend, job_id, instruction)
            
            # Checkpoint
            if current_step % CHECKPOINT_EVERY == 0:
                ckpt_path = f"/data/checkpoints/{job_id}/step_{current_step}"
                os.makedirs(ckpt_path, exist_ok=True)
                backend.save_checkpoint(current_step, ckpt_path)
                notify_orchestrator(job_id, current_step, ckpt_path, metrics)
                
                # Log progress
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f"Job {job_id}: Step {current_step}/{target_steps}, loss={metrics.get('loss', 'N/A'):.4f}")
            
        except Exception as e:
            fail_job(job_id, f"Training error at step {current_step}: {e}")
            raise
    
    # Complete
    try:
        requests.post(
            f"{ORCH_URL}/v1/training/jobs/{job_id}/complete",
            timeout=10
        )
        print(f"Job {job_id}: Completed at step {current_step}")
    except Exception as e:
        print(f"ERROR: Failed to mark job complete: {e}")

def apply_edit(backend, job_id: str, instruction: str):
    """Apply conductor instruction."""
    if instruction.startswith("snip_to:"):
        ckpt_id = instruction.split(":", 1)[1]
        print(f"Job {job_id}: Snipping to checkpoint {ckpt_id}")
        # TODO: Load checkpoint by ID
    elif instruction.startswith("inject_preference:"):
        data = json.loads(instruction.split(":", 1)[1])
        print(f"Job {job_id}: Injecting preference data")
        backend.inject_preference_data(data)
    elif instruction.startswith("adjust_lr:"):
        new_lr = float(instruction.split(":", 1)[1])
        print(f"Job {job_id}: Adjusting LR to {new_lr}")
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
    
    run_job(job_id, job_config)

if __name__ == "__main__":
    main()
