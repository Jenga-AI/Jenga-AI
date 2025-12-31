"""
PEFT (Parameter-Efficient Fine-Tuning) utilities for Jenga-AI
Shared across all fine-tuning modules
"""

from .config import PEFTConfig, FreezingConfig
from .model_wrapper import apply_peft, freeze_layers, detect_model_type

__all__ = [
    'PEFTConfig',
    'FreezingConfig', 
    'apply_peft',
    'freeze_layers',
    'detect_model_type'
]
