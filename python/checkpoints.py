"""Checkpoint directory helper for still-waiting.

Creates and returns a writable checkpoint path.
Falls back through sensible locations if /data doesn't exist.

UnobligatedRascal — Ancient hardware, fresh ambition.
"""
import os
from pathlib import Path


def get_checkpoint_dir() -> str:
    """Get the checkpoint base directory, creating it if needed.

    Priority:
    1. CHECKPOINT_DIR env var (if set and writable)
    2. /data/checkpoints (if /data exists)
    3. /home/whistler/still-waiting/python/checkpoints

    Returns:
        Absolute path to writable checkpoint directory
    """
    # Check env var first
    if os.environ.get("CHECKPOINT_DIR"):
        path = Path(os.environ["CHECKPOINT_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # Try /data/checkpoints
    data_path = Path("/data/checkpoints")
    if data_path.parent.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return str(data_path)

    # Fallback: project directory
    fallback = Path("/home/whistler/still-waiting/python/checkpoints")
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)
