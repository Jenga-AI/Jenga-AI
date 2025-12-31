"""
Knowledge Distillation configuration
"""

from dataclasses import dataclass

@dataclass
class DistillationConfig:
    """
    Configuration for knowledge distillation (teacher-student learning)
    """
    enabled: bool = False
    teacher_model: str = None  # Path or HF model ID
    distillation_alpha: float = 0.5  # Balance between task loss and distillation loss
    temperature: float = 2.0  # Temperature for softening teacher outputs
    distillation_type: str = "soft"  # "soft" or "hard"
