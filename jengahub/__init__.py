"""
JengaHub: Unified Multimodal AI Framework for African Languages

JengaHub combines the power of NestedWhisper and Jenga-AI to create a comprehensive
multimodal framework that excels at code-switching, hierarchical learning, and 
African language processing.

Key Components:
- Unified Configuration System (YAML-based)
- Continuum Memory System (hierarchical memory)
- Nested LoRA (efficient adaptation)
- Code-Switching Bridge (multimodal awareness)
- Unified Data Processing (audio + text)
- MLflow Tracking (comprehensive experimentation)
- Dynamic Model Serving (smart caching)

Author: JengaHub Development Team
License: MIT
"""

__version__ = "1.0.0"

from .core.config import (
    MultiModalConfig, 
    AudioConfig, 
    TextConfig,
    NestedLoRAConfig,
    LanguageBridgeSpec,
    TrainingConfig,
    DEFAULT_KIKUYU_CONFIG,
    DEFAULT_SWAHILI_CONFIG
)

from .core.model import (
    JengaHubMultiModalModel,
    JengaHubModelFactory,
    AudioTextBridge,
    HierarchicalFusion,
    load_pretrained_jengahub,
    save_jengahub_model
)

from .core.memory import (
    ContinuumMemorySystem,
    NestedMemoryLevel,
    LanguageFamilyHub
)

from .core.nested_lora import (
    NestedLoRALinear,
    NestedLoRAConverter,
    NestedLoRAScheduler
)

from .core.code_switching import (
    MultimodalCodeSwitchingBridge,
    CodeSwitchingDetector,
    LanguageIdentificationHead,
    SwitchType,
    SwitchPoint,
    detect_linguistic_triggers
)

from .core.tracking import (
    JengaHubMLflowLogger,
    MultiExperimentTracker,
    TrainingMetrics,
    setup_jengahub_tracking,
    create_performance_dashboard_data
)

from .core.serving import (
    JengaHubServingEngine,
    ModelManager,
    SmartCache,
    InferenceRequest,
    InferenceResult,
    ModelType,
    CacheLevel,
    create_serving_engine
)

from .data.processor import (
    UnifiedDataProcessor,
    AudioProcessor,
    TextProcessor,
    ProcessedSample
)

# Default configurations for quick setup
PRESETS = {
    'kikuyu': DEFAULT_KIKUYU_CONFIG,
    'swahili': DEFAULT_SWAHILI_CONFIG
}

def get_preset_config(language: str) -> MultiModalConfig:
    """Get a preset configuration for a specific language."""
    if language.lower() in PRESETS:
        return PRESETS[language.lower()]
    else:
        raise ValueError(f"No preset available for language: {language}")

# Quick start function
def create_jengahub_model(
    preset: str = "swahili",
    tasks: list = None,
    config: MultiModalConfig = None
) -> JengaHubMultiModalModel:
    """
    Quick start function to create a JengaHub model.
    
    Args:
        preset: Language preset ('swahili' or 'kikuyu')
        tasks: List of tasks to support
        config: Custom configuration (overrides preset)
        
    Returns:
        Configured JengaHub model
    """
    if config is None:
        config = get_preset_config(preset)
        
    if tasks is not None:
        # Update tasks in config
        config.text.tasks = tasks
    
    return JengaHubModelFactory.create_base_model(config)

__all__ = [
    # Core Configuration
    "MultiModalConfig", 
    "AudioConfig", 
    "TextConfig",
    "NestedLoRAConfig",
    "LanguageBridgeSpec",
    "TrainingConfig",
    "DEFAULT_KIKUYU_CONFIG",
    "DEFAULT_SWAHILI_CONFIG",
    
    # Model Components
    "JengaHubMultiModalModel",
    "JengaHubModelFactory",
    "AudioTextBridge",
    "HierarchicalFusion",
    "load_pretrained_jengahub",
    "save_jengahub_model",
    
    # Memory System
    "ContinuumMemorySystem",
    "NestedMemoryLevel",
    "LanguageFamilyHub",
    
    # LoRA System
    "NestedLoRALinear",
    "NestedLoRAConverter",
    "NestedLoRAScheduler",
    
    # Code-switching
    "MultimodalCodeSwitchingBridge",
    "CodeSwitchingDetector",
    "LanguageIdentificationHead",
    "SwitchType",
    "SwitchPoint",
    "detect_linguistic_triggers",
    
    # Tracking
    "JengaHubMLflowLogger",
    "MultiExperimentTracker",
    "TrainingMetrics",
    "setup_jengahub_tracking",
    "create_performance_dashboard_data",
    
    # Serving
    "JengaHubServingEngine",
    "ModelManager",
    "SmartCache",
    "InferenceRequest",
    "InferenceResult",
    "ModelType",
    "CacheLevel",
    "create_serving_engine",
    
    # Data Processing
    "UnifiedDataProcessor",
    "AudioProcessor",
    "TextProcessor",
    "ProcessedSample",
    
    # Utility Functions
    "PRESETS",
    "get_preset_config",
    "create_jengahub_model"
]