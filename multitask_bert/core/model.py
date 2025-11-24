import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from typing import List, Dict, Any, Optional
from ..tasks.base import BaseTask
from ..tasks import TASK_REGISTRY # Import the TASK_REGISTRY
from ..core.config import TaskConfig, ModelConfig # Import TaskConfig and ModelConfig
from .fusion import AttentionFusion

class MultiTaskModel(PreTrainedModel):
    """
    A generic multi-task model that uses a shared encoder and task-specific heads.
    This version's forward pass is designed to handle a batch from a single task at a time.
    """
    def __init__(self, config: AutoConfig, model_config: ModelConfig, task_configs: List[TaskConfig]): # Modified signature
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_config.base_model)
        
        # Instantiate tasks based on task_configs and TASK_REGISTRY
        self.tasks = nn.ModuleList()
        for task_config in task_configs:
            if task_config.type not in TASK_REGISTRY:
                raise ValueError(f"Unknown task type: {task_config.type}")
            
            # Instantiate the task, passing the task_config and the encoder's hidden_size
            task_class = TASK_REGISTRY[task_config.type]
            task_instance = task_class(config=task_config, hidden_size=self.encoder.config.hidden_size)
            self.tasks.append(task_instance)

        self.fusion = None
        if model_config.fusion:
            self.fusion = AttentionFusion(self.encoder.config, len(self.tasks))

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the input embeddings layer of the model.
        """
        return self.encoder.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        """
        Sets the input embeddings layer of the model.
        """
        self.encoder.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_id: int,
        labels: Any = None,
        token_type_ids: Optional[torch.Tensor] = None, # Make token_type_ids optional
        **kwargs
    ):
        """
        The forward pass for a single task's batch.

        Args:
            input_ids: Input token ids (batch_size, seq_len).
            attention_mask: Attention mask (batch_size, seq_len).
            task_id: The index of the task to run.
            labels: The labels for this task's batch.
            token_type_ids: Token type ids (segment ids).

        Returns:
            A dictionary containing logits and loss (if labels are provided) from the specified task.
        """
        # Pass token_type_ids only if the model supports it
        encoder_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if token_type_ids is not None and self.encoder.config.model_type in ["bert", "xlnet", "roberta"]: # Add other models that use token_type_ids
            encoder_inputs['token_type_ids'] = token_type_ids
        
        encoder_outputs = self.encoder(**encoder_inputs, **kwargs)
        
        # The task's get_forward_output expects input_ids, attention_mask, token_type_ids, and labels
        # It will then use the encoder outputs internally.
        task = self.tasks[task_id]
        
        task_output = task.get_forward_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            encoder_outputs=encoder_outputs # Pass encoder outputs directly
        )
        
        return task_output