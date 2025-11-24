import torch # Added import
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..core.config import TaskConfig
import torch.nn as nn # Import torch.nn

class BaseTask(nn.Module, ABC): # Inherit from nn.Module
    """
    An abstract base class for defining a task (e.g., classification, NER).
    It is initialized from a TaskConfig and defines the task-specific head.
    """
    def __init__(self, config: TaskConfig, hidden_size: int): # Added hidden_size
        super().__init__() # Call nn.Module's constructor
        self.config = config # Store the config
        self.name = config.name
        self.type = config.type
        self.hidden_size = hidden_size # Store hidden_size
        self.heads = nn.ModuleDict() # Initialize heads as a ModuleDict

    @abstractmethod
    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[Any],
        encoder_outputs: Any, # This will contain last_hidden_state and pooled_output
        **kwargs
    ) -> Dict[str, Any]:
        """
        Defines the forward pass for the task-specific head(s).

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: Labels for the task.
            encoder_outputs: Outputs from the shared encoder (e.g., last_hidden_state, pooled_output).

        Returns:
            A dictionary containing logits and loss (if labels are provided).
        """
        raise NotImplementedError
