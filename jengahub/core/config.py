"""
Unified Configuration System for JengaHub

This module provides a comprehensive configuration system that merges
the YAML-based configurations from both Jenga-AI and NestedWhisper,
enabling seamless multi-modal AI training and deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml


@dataclass
class AudioConfig:
    """Configuration for audio processing and speech recognition."""
    
    # Model settings
    base_model: str = "openai/whisper-small"
    sampling_rate: int = 16000
    n_mels: int = 80
    hop_length: int = 160
    win_length: int = 400
    
    # NestedWhisper settings
    enable_nested_learning: bool = True
    nested_levels: int = 5
    update_frequencies: List[int] = field(default_factory=lambda: [1, 10, 100, 1000, 10000])
    memory_sizes: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    
    # Code-switching settings
    enable_frame_lid: bool = True
    lid_threshold: float = 0.7
    smoothing_window: int = 5
    
    # Languages supported
    primary_language: str = "swahili"
    secondary_languages: List[str] = field(default_factory=lambda: ["english", "kikuyu"])
    language_family: str = "bantu"


@dataclass 
class TextConfig:
    """Configuration for text processing and NLP tasks."""
    
    # Model settings
    base_model: str = "bert-base-multilingual-cased"
    max_seq_length: int = 512
    hidden_size: int = 768
    
    # Multi-task settings
    tasks: List[str] = field(default_factory=lambda: ["classification", "ner"])
    enable_attention_fusion: bool = True
    task_embeddings_dim: int = 768
    
    # LoRA settings
    enable_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])


@dataclass
class NestedLoRAConfig:
    """Advanced nested LoRA configuration for hierarchical adaptation."""
    
    # Nested levels configuration
    n_levels: int = 5
    base_rank: int = 16
    rank_scaling: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Update frequencies (how often each level updates)
    update_frequencies: List[int] = field(default_factory=lambda: [1, 10, 100, 1000, 10000])
    
    # Learning rates for each level
    learning_rates: Dict[str, float] = field(default_factory=lambda: {
        "level_0": 1e-4,  # Fast adaptation (tokens/frames)
        "level_1": 5e-5,  # Medium (phonemes/words)  
        "level_2": 1e-5,  # Language-specific
        "level_3": 5e-6,  # Language family
        "level_4": 1e-6   # Universal/base
    })
    
    # Alpha values for each level
    alpha_values: List[int] = field(default_factory=lambda: [32, 24, 16, 12, 8])
    
    # Dropout for each level
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.08, 0.06, 0.04, 0.02])


@dataclass
class LanguageBridgeSpec:
    """Linguistic Bridge Specification (LBS) configuration."""
    
    # Basic language info
    language: str
    iso_code: str
    family: str
    anchor_language: str = "swahili"
    
    # Phonetic mapping
    shared_phones: List[str] = field(default_factory=list)
    unique_phones: List[str] = field(default_factory=list)
    phonetic_similarity: float = 0.0
    
    # Morphology
    morphology_type: str = "agglutinative"  # agglutinative, fusional, isolating
    has_noun_classes: bool = False
    is_tonal: bool = False
    
    # Code-switching patterns
    common_switch_types: List[str] = field(default_factory=list)
    intra_word_switching: bool = False
    
    # Nested configuration
    active_levels: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    family_levels: List[int] = field(default_factory=lambda: [3])


@dataclass
class TrainingConfig:
    """Training configuration combining both frameworks."""
    
    # Basic training settings
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Multi-modal training
    audio_weight: float = 0.5
    text_weight: float = 0.5
    bridge_loss_weight: float = 0.1
    
    # Optimization settings
    optimizer: str = "adamw"
    scheduler: str = "linear"
    gradient_clipping: float = 1.0
    
    # Memory optimization
    enable_gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # QLoRA settings
    enable_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # Anti-catastrophic forgetting
    enable_ewc: bool = True
    ewc_lambda: float = 0.4
    enable_memory_replay: bool = True
    replay_buffer_size: int = 1000
    
    # Round-robin sampling for multi-task
    enable_round_robin: bool = True
    task_sampling_strategy: str = "equal"  # equal, weighted, adaptive
    
    # Evaluation
    eval_strategy: str = "epoch"  # steps, epoch
    eval_steps: int = 500
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 50
    logging_dir: str = "./logs"
    use_mlflow: bool = True
    mlflow_experiment_name: str = "jengahub_training"


@dataclass
class MultiModalConfig:
    """Top-level configuration for JengaHub multi-modal system."""
    
    # Project metadata
    project_name: str = "jengahub_project"
    description: str = "Multi-modal African language AI"
    version: str = "1.0.0"
    
    # Sub-configurations
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    nested_lora: NestedLoRAConfig = field(default_factory=NestedLoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Language bridge specifications
    language_bridges: Dict[str, LanguageBridgeSpec] = field(default_factory=dict)
    
    # Global settings
    device: str = "auto"  # auto, cuda, cpu
    random_seed: int = 42
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Data paths
    data_paths: Dict[str, str] = field(default_factory=dict)
    
    # Model hub settings
    hub_token: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "MultiModalConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MultiModalConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "audio" in config_dict:
            config_dict["audio"] = AudioConfig(**config_dict["audio"])
        
        if "text" in config_dict:
            config_dict["text"] = TextConfig(**config_dict["text"])
            
        if "nested_lora" in config_dict:
            config_dict["nested_lora"] = NestedLoRAConfig(**config_dict["nested_lora"])
            
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
            
        # Handle language bridge specifications
        if "language_bridges" in config_dict:
            bridges = {}
            for lang, bridge_config in config_dict["language_bridges"].items():
                bridges[lang] = LanguageBridgeSpec(**bridge_config)
            config_dict["language_bridges"] = bridges
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        # Basic fields
        for field_name, field_value in self.__dict__.items():
            if field_name in ["audio", "text", "nested_lora", "training"]:
                config_dict[field_name] = field_value.__dict__
            elif field_name == "language_bridges":
                config_dict[field_name] = {
                    lang: bridge.__dict__ for lang, bridge in field_value.items()
                }
            else:
                config_dict[field_name] = field_value
                
        return config_dict
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check nested levels consistency
        if len(self.nested_lora.update_frequencies) != self.nested_lora.n_levels:
            issues.append("Number of update frequencies must match number of nested levels")
            
        if len(self.audio.update_frequencies) != self.audio.nested_levels:
            issues.append("Audio nested levels and update frequencies mismatch")
        
        # Check language consistency
        if self.audio.primary_language not in self.language_bridges:
            issues.append(f"Primary language '{self.audio.primary_language}' not in language bridges")
            
        # Check data paths
        for task, path in self.data_paths.items():
            if not Path(path).exists():
                issues.append(f"Data path for '{task}' does not exist: {path}")
        
        return issues


# Default configurations for common use cases
DEFAULT_KIKUYU_CONFIG = MultiModalConfig(
    project_name="kikuyu_multilingual",
    audio=AudioConfig(
        primary_language="kikuyu",
        secondary_languages=["english", "swahili"],
        language_family="bantu"
    ),
    text=TextConfig(
        tasks=["sentiment_analysis", "ner", "classification"]
    ),
    language_bridges={
        "kikuyu": LanguageBridgeSpec(
            language="kikuyu",
            iso_code="ki",
            family="bantu", 
            anchor_language="swahili",
            morphology_type="agglutinative",
            has_noun_classes=True,
            is_tonal=False
        )
    }
)

DEFAULT_SWAHILI_CONFIG = MultiModalConfig(
    project_name="swahili_base",
    audio=AudioConfig(
        primary_language="swahili",
        secondary_languages=["english"],
        language_family="bantu"
    ),
    text=TextConfig(
        tasks=["classification", "qa", "summarization"]
    ),
    language_bridges={
        "swahili": LanguageBridgeSpec(
            language="swahili", 
            iso_code="sw",
            family="bantu",
            anchor_language="english",
            morphology_type="agglutinative",
            has_noun_classes=True,
            is_tonal=False
        )
    }
)