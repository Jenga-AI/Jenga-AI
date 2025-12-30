"""
JengaHub Training Module

This module provides comprehensive training infrastructure for JengaHub multimodal models,
including distributed training, advanced optimizations, and production-ready features.
"""

from .trainer import JengaHubTrainer
from .distributed import DistributedTrainer
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    MemoryMonitor,
    LoRAScheduler
)
from .optimizers import create_optimizer, create_scheduler
from .utils import (
    setup_training,
    prepare_datasets,
    log_training_info
)

__all__ = [
    "JengaHubTrainer",
    "DistributedTrainer", 
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler", 
    "MemoryMonitor",
    "LoRAScheduler",
    "create_optimizer",
    "create_scheduler",
    "setup_training",
    "prepare_datasets",
    "log_training_info"
]