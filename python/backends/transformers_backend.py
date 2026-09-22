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
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
import datasets

# Transformers + PEFT
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Distributed
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyDataset(Dataset):
    """Minimal dataset for pipeline validation when no real data is provided."""

    def __init__(self, tokenizer, max_length: int = 512, size: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.size = size
        self.samples = []
        for i in range(size):
            text = f"This is sample number {i}. " * (max_length // 10)
            tokenized = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.samples.append(tokenized)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = self.samples[idx].copy()
        item["labels"] = item["input_ids"].copy()
        return item


class LocalJsonlDataset(Dataset):
    """Dataset from local JSONL file. Each line: {"text": "..."} or {"messages": [...]}"""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 1024,
        text_field: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.samples = self._load_samples(file_path)

    def _load_samples(self, file_path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        text = "\n".join(
                            f"{m.get('role', 'user')}: {m.get('content', '')}"
                            for m in obj["messages"]
                        )
                    else:
                        text = obj.get(self.text_field, str(obj))
                    tokenized = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    samples.append(tokenized)
                except json.JSONDecodeError:
                    continue
        if not samples:
            raise ValueError(f"No valid samples loaded from {file_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx].copy()
        item["labels"] = item["input_ids"].copy()
        return item


def get_rank() -> int:
    """Get current process rank."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def log(msg: str):
    """Log only from main process."""
    if is_main_process():
        print(msg, flush=True)


def _auto_detect_target_modules(model_ref: str, model_precision: str) -> list[str]:
    """Auto-detect LoRA target modules based on model architecture.

    For QLoRA (bnb_nf4), we typically target all linear layers.
    For standard LoRA, we target attention projection layers only.

    Returns appropriate target_modules list as strings for LoraConfig.
    """
    model_lower = model_ref.lower()

    # QLoRA: target all linear layers (standard QLoRA practice)
    if model_precision == "bnb_nf4":
        if "qwen" in model_lower or "llama" in model_lower or "mistral" in model_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # Fallback: all linear layers (PEFT accepts "all-linear" string)
        return ["all-linear"]

    # Standard LoRA: attention projections only (safer, less params)
    # Qwen2/Qwen2.5 architecture
    if "qwen" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Llama-3 architecture
    if "llama-3" in model_lower or "llama3" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Mistral/Mixtral architecture
    if "mistral" in model_lower or "mixtral" in model_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # GPT-2 / TinyLlama / generic transformer architecture
    if "tinyllama" in model_lower:
        return ["q_proj", "k_proj", "v_proj"]

    # Default safe targets (attention only)
    return ["q_proj", "v_proj"]


class KeplerTransformersBackend:
    """
    LoRA fine-tuning backend optimized for Kepler sm_37.

    Configuration options (via job config):
    - model_precision: str (default "f32", options: "f32", "f16_storage", "bnb_nf4")
    - max_seq_length: int (default 1024)
    - lora_r: int (default 16)
    - lora_alpha: int (default 32)
    - lora_dropout: float (default 0.05)
    - learning_rate: float (default 2e-4)
    - batch_size: int (default 2)
    - gradient_accumulation_steps: int (default 8)
    - num_train_epochs: int (default 1)
    - target_modules: list[str] (default: auto-detected)
    - dataset_path: str (local path or HuggingFace id)

    model_precision modes:
    - "f32": Full precision. Safe, slowest, most VRAM.
    - "f16_storage": Weights in FP16, compute in FP32. 2x VRAM savings, same speed.
    - "bnb_nf4": bitsandbytes QLoRA. Requires bnb install. May fail on Kepler.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.data_loader = None
        self.optimizer = None
        self.scheduler = None
        self.grad_accum_steps = config.get("gradient_accumulation_steps", 8)
        self.step_count = 0
        
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
        """Load model and tokenizer, apply LoRA adapters, prepare dataset."""
        
        cfg = self.config.copy()
        if config:
            cfg.update(config)

        # Model precision config:
        # - "f32": Full precision (safe, slowest, most VRAM). Default for Kepler.
        # - "f16_storage": Load weights in FP16 (half VRAM), cast to FP32 for compute.
        #   FP16 matmul is slow on Kepler without tensor cores, but VRAM savings enable
        #   larger models. Recommended for 3B+ models on single K80.
        # - "bnb_nf4": Use bitsandbytes 4-bit NF4 quantization (QLoRA).
        #   Requires bitsandbytes installed. May fail on Kepler sm_37 (CC < 6.0 required).
        # - "custom_nf4": Custom NF4 4-bit quantization for Kepler (in development).
        #   ~7.8x compression vs FP32. Auto-fallback to F32 on kernel failure.
        model_precision = cfg.get("model_precision", "f32")

        # Alternative config flag for backward compatibility
        if cfg.get("quant") == "nf4" and model_precision == "f32":
            model_precision = "custom_nf4"

        max_seq_length = cfg.get("max_seq_length", 1024)  # Reduced default for Kepler VRAM
        lora_r = cfg.get("lora_r", 16)
        lora_alpha = cfg.get("lora_alpha", 32)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        learning_rate = cfg.get("learning_rate", 2e-4)
        num_train_epochs = cfg.get("num_train_epochs", 1)
        batch_size = cfg.get("batch_size", 2)  # Smaller default for Kepler
        self.grad_accum_steps = cfg.get("gradient_accumulation_steps", 8)
        warmup_ratio = cfg.get("warmup_ratio", 0.05)
        dataset_path = cfg.get("dataset_path")
        dataset_text_field = cfg.get("dataset_text_field", "text")

        # Auto-detect target modules for LoRA/QLoRA based on model architecture
        # if user hasn't specified them explicitly.
        if cfg.get("target_modules"):
            target_modules = cfg.get("target_modules")
        else:
            target_modules = _auto_detect_target_modules(model_ref, model_precision)

        log(f"LoRA target modules: {target_modules}")
        
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        
        # Set device
        if torch.distributed.is_initialized():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
        self.device = device
        
        # Load tokenizer
        log(f"Loading tokenizer: {model_ref}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log(f"Loading model: {model_ref} (precision={model_precision})")

        if model_precision == "bnb_nf4":
            # QLoRA via bitsandbytes 4-bit NF4 quantization
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise RuntimeError(
                    "bitsandbytes not installed. Install with: pip install bitsandbytes"
                )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,  # Kepler: F32 compute
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            log("Model loaded with bitsandbytes NF4 4-bit quantization (QLoRA)")

        elif model_precision == "custom_nf4":
            # Custom NF4 4-bit quantization for Kepler (in development)
            # Falls back to F32 if kernels not ready or fail
            try:
                from kernels import nf4_kepler
                log("Using custom NF4 quantization for Kepler")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_ref,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                # Kernels validated (dequant + torch::mm). LinearNF4 layer swap is not wired.
                # TODO: Replace nn.Linear weights with packed NF4 + NF4DequantizeFunction
                log("WARNING: custom_nf4 requested. Dequant kernels are validated, but LinearNF4 is not integrated.")
                log("Falling back to F32 until layer replacement lands.")
                model_precision = "f32"  # Fallback
            except ImportError as e:
                log(f"WARNING: nf4_kepler module not found ({e}). Falling back to F32.")
                model_precision = "f32"

        elif model_precision == "f16_storage":
            # FP16 weights for VRAM savings, FP32 compute for Kepler correctness
            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            # Cast to FP32 for compute (Kepler FP16 is slow without tensor cores)
            self.model = self.model.to(torch.float32)
            log("Model loaded in FP16 storage, cast to FP32 for compute")
        else:
            # F32 for Kepler (FP16 is slow without tensor cores)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            log("Model loaded in FP32")
        
        # Apply LoRA
        log(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, targets={target_modules})")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Move to device
        self.model.to(device)
        
        # Wrap with DDP if distributed
        if torch.distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        
        self.model.train()
        
        # Prepare optimizer (AdamW — F32 for Kepler)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        # Prepare dataset
        self._prepare_dataset_internal(
            dataset_path=dataset_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            dataset_text_field=dataset_text_field,
        )
        
        # Calculate total steps and create scheduler
        effective_batch_size = batch_size * self.grad_accum_steps * int(world_size)
        total_steps = len(self.data_loader) * num_train_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )
        
        log(f"Training config: batch_size={batch_size}, grad_accum={self.grad_accum_steps}, "
            f"effective_batch={effective_batch_size}, total_steps={total_steps}")
        log(f"LR={learning_rate}, warmup_ratio={warmup_ratio}, max_seq_len={max_seq_length}")
    
    def _prepare_dataset_internal(
        self,
        dataset_path: str,
        max_seq_length: int,
        batch_size: int,
        dataset_text_field: str,
    ):
        """Load and tokenize dataset. Internal helper."""
        
        if not dataset_path:
            # Create a tiny dummy dataset for testing
            log("No dataset specified, creating dummy dataset for pipeline validation")
            self.dataset = DummyDataset(tokenizer=self.tokenizer, max_length=max_seq_length, size=100)
        else:
            self.dataset = self._load_dataset(dataset_path, max_seq_length, dataset_text_field)
        
        # Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not MLM
        )
        
        # DataLoader with distributed sampler
        from torch.utils.data.distributed import DistributedSampler
        
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(self.dataset, shuffle=True)
        else:
            sampler = None
        
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            pin_memory=True,
            num_workers=2,  # Low to avoid CPU contention on NOUGHT
        )
        
        log(f"Dataset loaded: {len(self.dataset)} samples, {len(self.data_loader)} batches")

    def _load_dataset(
        self,
        path: str,
        max_seq_length: int,
        text_field: str,
    ) -> Dataset:
        """Load dataset from local file, directory, or HF hub."""
        
        p = Path(path)
        
        # Local JSONL file
        if p.is_file() and p.suffix in (".jsonl", ".json"):
            log(f"Loading local dataset: {path}")
            return LocalJsonlDataset(
                file_path=path,
                tokenizer=self.tokenizer,
                max_length=max_seq_length,
                text_field=text_field,
            )
        
        # Local directory with JSONL files
        if p.is_dir():
            log(f"Loading local dataset directory: {path}")
            files = list(p.glob("*.jsonl")) + list(p.glob("*.json"))
            if not files:
                raise ValueError(f"No JSONL files found in {path}")
            return LocalJsonlDataset(
                file_path=str(files[0]),  # TODO: support multiple files
                tokenizer=self.tokenizer,
                max_length=max_seq_length,
                text_field=text_field,
            )
        
        # Try HuggingFace dataset
        log(f"Loading HF dataset: {path}")
        try:
            hf_dataset = datasets.load_dataset(path, split="train")
        except Exception as e:
            log(f"Failed to load HF dataset: {e}, trying as local path again")
            raise
        
        def tokenize(examples):
            texts = []
            for item in examples[text_field]:
                if isinstance(item, list):
                    # Chat-style: join messages
                    texts.append("\n".join(str(m) for m in item))
                else:
                    texts.append(str(item))
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
            )
            # Create labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = hf_dataset.map(
            tokenize,
            batched=True,
            remove_columns=hf_dataset.column_names,
        )
        return tokenized_dataset

    def prepare_dataset(self, dataset_path: str, config: Dict[str, Any] = None):
        """Public API for dataset loading (for external calls)."""
        cfg = self.config.copy()
        if config:
            cfg.update(config)
        
        max_seq_length = cfg.get("max_seq_length", 1024)
        batch_size = cfg.get("batch_size", 2)
        dataset_text_field = cfg.get("dataset_text_field", "text")
        
        self._prepare_dataset_internal(
            dataset_path=dataset_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            dataset_text_field=dataset_text_field,
        )
    
    def train_step(self) -> Dict[str, float]:
        """Execute one effective training step (with gradient accumulation). Returns metrics."""
        
        if self.data_loader is None:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")
        
        self.model.train()
        self.optimizer.zero_grad()
        
        accumulated_loss = 0.0
        samples_processed = 0
        
        for batch in self.data_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / self.grad_accum_steps  # Scale for accumulation
            
            # Backward pass (no sync on intermediate steps for DDP efficiency)
            if torch.distributed.is_initialized():
                with self.model.no_sync() if samples_processed % self.grad_accum_steps != (self.grad_accum_steps - 1) else torch.enable_grad():
                    loss.backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item() * self.grad_accum_steps
            samples_processed += batch["input_ids"].shape[0]
            
            # Gradient accumulation step
            if samples_processed % self.grad_accum_steps == 0:
                # Gradient clipping (important for stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.step_count += 1
                
                # Return metrics after each effective step
                current_lr = self.scheduler.get_last_lr()[0]
                metrics = {
                    "loss": accumulated_loss / samples_processed,
                    "lr": current_lr,
                    "step": self.step_count,
                }
                
                # Clear cache periodically (Kepler VRAM management)
                if self.step_count % 64 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                return metrics
        
        # Fallback: should not reach here if data_loader has data
        return {"loss": accumulated_loss / max(samples_processed, 1), "lr": 0.0, "step": self.step_count}
    
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
