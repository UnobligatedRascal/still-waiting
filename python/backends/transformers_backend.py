"""
Kepler-optimized Transformers backend.

Uses PyTorch 1.14 + Transformers 4.30 + PEFT for LoRA fine-tuning
on sm_37 Kepler GPUs. No tensor cores, so we use F32 matmul via cuBLAS
legacy APIs (Sgemm, not SgemmEx).

Key considerations:
- NO gradient checkpointing via transformers' built-in (uses ops Kepler doesn't support)
- NO BF16 (Kepler is FP32/FP16 only, FP16 is slow)
- Use F32 for compute, F16 for storage if needed
- NUMA pinning required outside this module (worker.py handles via numactl)
- Distributed via torch.distributed (DDP)

UnobligatedRascal — Making old hardware sing.
"""
import os
import gc
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader, Dataset

# Transformers + PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Distributed
from torch.nn.parallel import DistributedDataParallel as DDP


class KeplerTransformersBackend:
    """
    LoRA fine-tuning backend optimized for Kepler sm_37.
    
    Configuration options (via job config):
    - max_seq_length: int (default 2048)
    - lora_r: int (default 16)
    - lora_alpha: int (default 32)
    - lora_dropout: float (default 0.05)
    - learning_rate: float (default 2e-4)
    - batch_size: int (default 4 per GPU)
    - gradient_accumulation_steps: int (default 8)
    - num_train_epochs: int (default 1)
    - target_modules: list[str] (default: ["q_proj", "v_proj"])
    - dataset_path: str (local path or HuggingFace id)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Check NVIDIA driver and CUDA installation.")
        
        # Report GPU info
        self._log_gpu_info()
    
    def _log_gpu_info(self):
        """Log GPU configuration for debugging."""
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        
        if local_rank == 0:
            print(f"GPU info: {torch.cuda.device_count()} GPUs available")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB, "
                      f"compute_cap={props.major}.{props.minor}")
            print(f"Distributed: rank={local_rank}, world_size={world_size}")
    
    def prepare(self, model_ref: str, config: Dict[str, Any] = None):
        """Load model and tokenizer, apply LoRA adapters."""
        
        cfg = self.config.copy()
        if config:
            cfg.update(config)
        
        max_seq_length = cfg.get("max_seq_length", 2048)
        lora_r = cfg.get("lora_r", 16)
        lora_alpha = cfg.get("lora_alpha", 32)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        target_modules = cfg.get("target_modules", ["q_proj", "v_proj"])
        
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        
        # Set device
        if torch.distributed.is_initialized():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
        
        # Load tokenizer
        print(f"[{local_rank}] Loading tokenizer: {model_ref}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model — F32 for Kepler (FP16 is slow without tensor cores)
        print(f"[{local_rank}] Loading model: {model_ref}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=torch.float32,  # F32 for Kepler compatibility
            trust_remote_code=True,
            # Low VRAM: use device_map="auto" if needed, but DDP handles distribution
        )
        
        # Apply LoRA
        print(f"[{local_rank}] Applying LoRA (r={lora_r}, alpha={lora_alpha})")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Move to device
        self.model.to(device)
        
        # Wrap with DDP if distributed
        if torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        
        self.model.train()
    
    def prepare_dataset(self, dataset_path: str, config: Dict[str, Any] = None):
        """Load and tokenize dataset."""
        cfg = self.config.copy()
        if config:
            cfg.update(config)
        
        max_seq_length = cfg.get("max_seq_length", 2048)
        
        # TODO: Implement dataset loading from local path, HF dataset, or custom format
        raise NotImplementedError("Dataset loading not yet implemented")
    
    def train_step(self) -> Dict[str, float]:
        """Execute one training step. Returns metrics."""
        
        # TODO: Implement manual training step or use Trainer
        # For now, return placeholder
        return {"loss": 0.0}
    
    def save_checkpoint(self, step: int, path: str):
        """Save model checkpoint."""
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(path)
        
        # Save tokenizer separately (it's shared)
        # tokenizer.save_pretrained(path)  # Only once, not every checkpoint
        
        # Force CUDA cache clear after checkpoint (Kepler has limited VRAM)
        gc.collect()
        torch.cuda.empty_cache()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # TODO: Load checkpoint and merge with LoRA adapters
        pass
    
    def set_learning_rate(self, lr: float):
        """Adjust learning rate mid-training."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def inject_preference_data(self, data: Dict[str, Any]):
        """Inject preference data for RLHF-style training."""
        # TODO: Implement
        pass
    
    def export(self, fmt: str, path: str):
        """Export trained model."""
        if fmt == "safetensors":
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_save.save_pretrained(path, safe_serialization=True)
        elif fmt == "gguf":
            # TODO: Convert to GGUF via llama.cpp/llama-cpp-python
            raise NotImplementedError("GGUF export not yet implemented")
        else:
            raise ValueError(f"Unknown format: {fmt}")
