"""
Universal PEFT configuration classes for all model types
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PEFTConfig:
    """
    Universal PEFT configuration for all model types (Seq2Seq, CausalLM, Encoder)
    
    Supports LoRA and other parameter-efficient fine-tuning methods.
    """
    enabled: bool = False
    method: str = "lora"  # Currently only "lora" is supported
    
    # LoRA specific parameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    
    # Common parameters
    bias: str = "none"  # "none", "all", or "lora_only"
    modules_to_save: Optional[List[str]] = None
    
    # Advanced options
    fan_in_fan_out: bool = False
    merge_weights: bool = False

@dataclass
class FreezingConfig:
    """
    Layer freezing configuration for preventing catastrophic forgetting
    """
    enabled: bool = False
    freeze_embeddings: bool = True
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    num_layers_to_freeze: Optional[int] = None  # Freeze first N layers from bottom
    layers_to_freeze: Optional[List[int]] = None  # Specific layer indices to freeze
