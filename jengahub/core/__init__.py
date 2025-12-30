"""
JengaHub Core Components

This module contains the central architecture that unifies speech and text processing
through nested learning and continuum memory systems.
"""

from .config import (
    MultiModalConfig,
    NestedLoRAConfig,
    AudioConfig,
    TextConfig,
    TrainingConfig
)

from .model import (
    JengaHubModel,
    AudioTextBridge,
    HierarchicalFusion
)

from .memory import (
    ContinuumMemorySystem,
    NestedMemoryLevel,
    LanguageFamilyHub
)

__all__ = [
    "MultiModalConfig",
    "NestedLoRAConfig", 
    "AudioConfig",
    "TextConfig",
    "TrainingConfig",
    "JengaHubModel",
    "AudioTextBridge",
    "HierarchicalFusion",
    "ContinuumMemorySystem",
    "NestedMemoryLevel",
    "LanguageFamilyHub"
]