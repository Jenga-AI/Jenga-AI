"""
Training Utilities for JengaHub

This module provides utility functions for setting up training environments,
preparing datasets, and managing training workflows.
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import os
import random
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import yaml
import mlflow

from ..core.config import MultiModalConfig
from ..core.model import JengaHubMultiModalModel
from ..core.tracking import JengaHubMLflowLogger, setup_jengahub_tracking
from ..data.processor import UnifiedDataProcessor, ProcessedSample


def setup_training(
    config: MultiModalConfig,
    seed: int = None,
    output_dir: str = None,
    resume_from_checkpoint: str = None,
    local_rank: int = -1
) -> Dict[str, Any]:
    """
    Setup training environment with proper initialization.
    
    Args:
        config: Training configuration
        seed: Random seed for reproducibility
        output_dir: Training output directory
        resume_from_checkpoint: Path to checkpoint to resume from
        local_rank: Local rank for distributed training
        
    Returns:
        Dictionary with setup information
    """
    
    # Set random seeds for reproducibility
    if seed is None:
        seed = config.random_seed
    
    set_seed(seed)
    
    # Setup output directory
    if output_dir is None:
        output_dir = config.output_dir
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_path)
    logger = logging.getLogger(__name__)
    
    # Setup distributed training
    distributed_info = setup_distributed_training(local_rank)
    
    # Setup device
    device = setup_device(config.device, distributed_info['local_rank'])
    
    # Log setup information
    logger.info("=== Training Setup ===")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed: {distributed_info['is_distributed']}")
    if distributed_info['is_distributed']:
        logger.info(f"World size: {distributed_info['world_size']}")
        logger.info(f"Local rank: {distributed_info['local_rank']}")
    
    # Save configuration
    config_path = output_path / "config.yaml"
    config.to_yaml(str(config_path))
    logger.info(f"Configuration saved: {config_path}")
    
    setup_info = {
        'seed': seed,
        'output_dir': output_path,
        'device': device,
        'distributed_info': distributed_info,
        'resume_from_checkpoint': resume_from_checkpoint,
        'config_path': config_path
    }
    
    return setup_info


def set_seed(seed: int):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    
    log_file = output_dir / f"training_{int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def setup_distributed_training(local_rank: int = -1) -> Dict[str, Any]:
    """Setup distributed training environment."""
    
    distributed_info = {
        'is_distributed': False,
        'world_size': 1,
        'local_rank': 0,
        'rank': 0
    }
    
    # Check for distributed training setup
    if local_rank != -1 or 'WORLD_SIZE' in os.environ:
        # Initialize distributed training
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(local_rank)
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)
        
        try:
            dist.init_process_group(backend='nccl')
            distributed_info.update({
                'is_distributed': True,
                'world_size': dist.get_world_size(),
                'local_rank': dist.get_rank(),
                'rank': dist.get_rank()
            })
            
            logging.info(f"Distributed training initialized: rank {dist.get_rank()}/{dist.get_world_size()}")
            
        except Exception as e:
            logging.warning(f"Failed to initialize distributed training: {e}")
            logging.info("Falling back to single-process training")
    
    return distributed_info


def setup_device(device_config: str = "auto", local_rank: int = 0) -> torch.device:
    """Setup training device."""
    
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)
    
    # Set CUDA device for current process
    if device.type == "cuda":
        torch.cuda.set_device(device)
    
    return device


def prepare_datasets(
    config: MultiModalConfig,
    data_processor: UnifiedDataProcessor,
    distributed_info: Dict[str, Any] = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Prepare training, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        data_processor: Data processor instance
        distributed_info: Distributed training information
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    logger = logging.getLogger(__name__)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Process each data path in configuration
    for task_name, data_path in config.data_paths.items():
        if not Path(data_path).exists():
            logger.warning(f"Data path not found: {data_path}")
            continue
        
        try:
            # Load task-specific data
            task_samples = data_processor.load_dataset(data_path, task_name, split="train")
            
            # Split into train/val/test (80/10/10 by default)
            n_samples = len(task_samples)
            n_train = int(0.8 * n_samples)
            n_val = int(0.1 * n_samples)
            
            train_samples.extend(task_samples[:n_train])
            val_samples.extend(task_samples[n_train:n_train + n_val])
            test_samples.extend(task_samples[n_train + n_val:])
            
            logger.info(f"Loaded {len(task_samples)} samples for task '{task_name}'")
            
        except Exception as e:
            logger.error(f"Failed to load data for task '{task_name}': {e}")
    
    # Create balanced datasets
    if config.training.enable_round_robin:
        train_samples = data_processor.create_balanced_dataset(train_samples, balance_by="task")
        val_samples = data_processor.create_balanced_dataset(val_samples, balance_by="task")
    
    # Log dataset statistics
    logger.info(f"Dataset sizes - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Create data collator
    collate_fn = data_processor.create_batch_collator()
    
    # Setup samplers for distributed training
    if distributed_info and distributed_info['is_distributed']:
        train_sampler = DistributedSampler(train_samples, shuffle=True)
        val_sampler = DistributedSampler(val_samples, shuffle=False) if val_samples else None
        test_sampler = DistributedSampler(test_samples, shuffle=False) if test_samples else None
    else:
        train_sampler = RandomSampler(train_samples)
        val_sampler = SequentialSampler(val_samples) if val_samples else None
        test_sampler = SequentialSampler(test_samples) if test_samples else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_samples,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Important for distributed training
    )
    
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            val_samples,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=config.training.dataloader_num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    test_loader = None
    if test_samples:
        test_loader = DataLoader(
            test_samples,
            batch_size=config.training.batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=config.training.dataloader_num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    return train_loader, val_loader, test_loader


def log_training_info(
    model: JengaHubMultiModalModel,
    config: MultiModalConfig,
    train_loader: DataLoader,
    logger: Optional[JengaHubMLflowLogger] = None
):
    """Log comprehensive training information."""
    
    log = logging.getLogger(__name__)
    
    # Model information
    model_stats = model.get_model_statistics()
    
    log.info("=== Model Information ===")
    log.info(f"Total parameters: {model_stats['total_parameters']:,}")
    log.info(f"Trainable parameters: {model_stats['trainable_parameters']:,}")
    
    if 'lora_total_parameters' in model_stats:
        efficiency = model_stats['lora_total_parameters'] / model_stats['total_parameters'] * 100
        log.info(f"LoRA parameters: {model_stats['lora_total_parameters']:,} ({efficiency:.2f}%)")
    
    # Training configuration
    log.info("=== Training Configuration ===")
    log.info(f"Batch size: {config.training.batch_size}")
    log.info(f"Learning rate: {config.training.learning_rate:.2e}")
    log.info(f"Epochs: {config.training.num_epochs}")
    log.info(f"Optimizer: {config.training.optimizer}")
    log.info(f"Scheduler: {config.training.scheduler}")
    log.info(f"Mixed precision: {config.training.fp16}")
    
    # Dataset information
    log.info("=== Dataset Information ===")
    log.info(f"Training batches: {len(train_loader)}")
    log.info(f"Samples per epoch: {len(train_loader) * config.training.batch_size}")
    log.info(f"Total training steps: {len(train_loader) * config.training.num_epochs}")
    
    # MLflow logging
    if logger:
        logger.log_dataset_info({
            'training_batches': len(train_loader),
            'batch_size': config.training.batch_size,
            'total_samples': len(train_loader) * config.training.batch_size,
            'total_steps': len(train_loader) * config.training.num_epochs
        })


def save_training_summary(
    results: Dict[str, Any],
    output_dir: Path,
    config: MultiModalConfig
):
    """Save comprehensive training summary."""
    
    summary = {
        'training_results': results,
        'configuration': config.to_dict(),
        'timestamp': torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    }
    
    # Save as JSON
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save as YAML for human readability
    summary_yaml_path = output_dir / "training_summary.yaml"
    with open(summary_yaml_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    logging.info(f"Training summary saved: {summary_path}")


def create_training_run_name(config: MultiModalConfig) -> str:
    """Create a descriptive training run name."""
    
    import time
    timestamp = int(time.time())
    
    # Include key configuration details in the name
    run_name = f"jengahub_{config.project_name}"
    run_name += f"_lr{config.training.learning_rate:.0e}"
    run_name += f"_bs{config.training.batch_size}"
    run_name += f"_ep{config.training.num_epochs}"
    
    if config.nested_lora.n_levels > 0:
        run_name += f"_lora{config.nested_lora.n_levels}l"
    
    run_name += f"_{timestamp}"
    
    return run_name


def setup_mlflow_tracking(
    config: MultiModalConfig,
    output_dir: Path,
    experiment_name: str = None
) -> JengaHubMLflowLogger:
    """Setup MLflow tracking for the training run."""
    
    if experiment_name is None:
        experiment_name = f"{config.project_name}_training"
    
    # Setup MLflow tracking
    logger = setup_jengahub_tracking(config, experiment_name)
    
    # Create run name
    run_name = create_training_run_name(config)
    
    return logger


def validate_training_config(config: MultiModalConfig) -> List[str]:
    """Validate training configuration and return issues."""
    
    issues = []
    
    # Basic validation
    if config.training.batch_size <= 0:
        issues.append("Batch size must be positive")
    
    if config.training.learning_rate <= 0:
        issues.append("Learning rate must be positive")
    
    if config.training.num_epochs <= 0:
        issues.append("Number of epochs must be positive")
    
    # Data paths validation
    if not config.data_paths:
        issues.append("No data paths specified")
    else:
        for task, path in config.data_paths.items():
            if not Path(path).exists():
                issues.append(f"Data path for '{task}' does not exist: {path}")
    
    # LoRA configuration validation
    if config.nested_lora.n_levels > 0:
        if len(config.nested_lora.update_frequencies) != config.nested_lora.n_levels:
            issues.append("Number of update frequencies must match number of LoRA levels")
        
        if len(config.nested_lora.rank_scaling) != config.nested_lora.n_levels:
            issues.append("Number of rank scaling factors must match number of LoRA levels")
    
    # Memory configuration validation
    if hasattr(config, 'memory_system'):
        if config.memory_system.enable_memory_replay and config.memory_system.replay_buffer_size <= 0:
            issues.append("Replay buffer size must be positive when memory replay is enabled")
    
    return issues


class TrainingProfiler:
    """Profiler for training performance analysis."""
    
    def __init__(self, output_dir: Path, profile_memory: bool = True):
        self.output_dir = output_dir
        self.profile_memory = profile_memory
        self.profiler = None
    
    def start(self):
        """Start profiling."""
        if torch.cuda.is_available():
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        else:
            activities = [torch.profiler.ProfilerActivity.CPU]
        
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=True,
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir / "profiler"))
        )
        
        self.profiler.start()
    
    def step(self):
        """Step the profiler."""
        if self.profiler:
            self.profiler.step()
    
    def stop(self):
        """Stop profiling and save results."""
        if self.profiler:
            self.profiler.stop()
            
            # Save profiler summary
            summary_path = self.output_dir / "profiler_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            
            logging.info(f"Profiler results saved: {self.output_dir / 'profiler'}")


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    
    import platform
    import psutil
    
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3)
    }
    
    if torch.cuda.is_available():
        system_info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory_gb': [
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(torch.cuda.device_count())
            ]
        })
    
    return system_info