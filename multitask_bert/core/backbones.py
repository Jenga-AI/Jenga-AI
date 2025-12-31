import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all JengaAI backbones."""
    def __init__(self, model_name: str, config: Any, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Must return a dict containing 'last_hidden_state' and optionally 'pooler_output'."""
        pass

class TextBackbone(BaseBackbone):
    """Standard Transformer backbone for text tasks (BERT, RoBERTa, etc)."""
    def __init__(self, model_name: str, config: Any, **kwargs):
        super().__init__(model_name, config, **kwargs)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # Only pass token_type_ids if the model type supports it
        if token_type_ids is not None and self.encoder.config.model_type in ["bert", "xlnet", "albert"]:
            model_inputs["token_type_ids"] = token_type_ids
            
        outputs = self.encoder(**model_inputs, **kwargs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": getattr(outputs, "pooler_output", outputs.last_hidden_state[:, 0, :])
        }

class SequentialBackbone(BaseBackbone):
    """
    Backbone for Tabular and Time-Series data (AIOps/Security)
    Uses a configurable MLP/Feature Extractor.
    """
    def __init__(self, model_name: str, config=None, **kwargs):
        super().__init__(model_name, config)
        # In a real scenario, this would load a pre-defined MLP architecture
        # For now, we'll use a simple projection layer as a placeholder
        self.input_dim = kwargs.get("input_dim", 128) # Default for telemetry features
        self.hidden_dim = config.hidden_size if (config and hasattr(config, 'hidden_size')) else 768
        
        # Create a fake config object so model.py can access encoder.config.hidden_size
        from types import SimpleNamespace
        if not hasattr(self, 'config') or self.config is None:
            self.config = SimpleNamespace(hidden_size=self.hidden_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
        # Wrap the Sequential in a namespace so it has a .config attribute
        self.encoder.config = SimpleNamespace(hidden_size=self.hidden_dim)

        
    def forward(self, features, **kwargs):
        # features shape: [batch, seq_len, input_dim] or [batch, input_dim]
        if features.dim() == 2:
            features = features.unsqueeze(1) # Add dummy seq_len dimension
            
        hidden_states = self.encoder(features)
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": hidden_states[:, 0, :]
        }

class AudioBackbone(BaseBackbone):
    """Backbone for Audio models like Whisper."""
    def __init__(self, model_name: str, config: Any, **kwargs):
        super().__init__(model_name, config, **kwargs)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

    def forward(self, input_features, **kwargs):
        # Whisper uses 'input_features' instead of 'input_ids'
        outputs = self.encoder(input_features=input_features, **kwargs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.last_hidden_state.mean(dim=1) # Average pooling for audio
        }

class BackboneManager:
    """Registry to manage and instantiate different backbone types."""
    _REGISTRY = {
        "text": TextBackbone,
        "audio": AudioBackbone,
        "sequential": SequentialBackbone
    }

    @classmethod
    def create(cls, backbone_type: str, model_name: str, config: Any, **kwargs) -> BaseBackbone:
        if backbone_type not in cls._REGISTRY:
            raise ValueError(f"Unknown backbone type: {backbone_type}. Supported: {list(cls._REGISTRY.keys())}")
        
        backbone_class = cls._REGISTRY[backbone_type]
        return backbone_class(model_name, config, **kwargs)
