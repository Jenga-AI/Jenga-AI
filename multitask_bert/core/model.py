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
        from .backbones import BackboneManager
        
        # Determine input_dim for sequential/tabular models
        input_dim = 128 # Default
        for tc in task_configs:
            if hasattr(tc, 'input_dim') and tc.input_dim is not None:
                input_dim = tc.input_dim
                break
        
        # Instantiate the modular backbone
        self.backbone = BackboneManager.create(
            backbone_type=model_config.backbone_type,
            model_name=model_config.base_model,
            config=config,
            input_dim=input_dim
        )
        
        # Instantiate tasks based on task_configs and TASK_REGISTRY
        self.tasks = nn.ModuleList()
        # Use the backbone's internal encoder config for hidden size compatibility
        hidden_size = self.backbone.encoder.config.hidden_size
        
        for task_config in task_configs:
            if task_config.type not in TASK_REGISTRY:
                raise ValueError(f"Unknown task type: {task_config.type}")
            
            task_class = TASK_REGISTRY[task_config.type]
            task_instance = task_class(config=task_config, hidden_size=hidden_size)
            self.tasks.append(task_instance)

        self.fusion = None
        if model_config.fusion:
            self.fusion = AttentionFusion(self.encoder.config, len(self.tasks))

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the input embeddings layer of the model.
        """
        # Delegate to the backbone's encoder if it exists (standard for HF models)
        return self.backbone.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        """
        Sets the input embeddings layer of the model.
        """
        self.backbone.encoder.set_input_embeddings(value)

    def forward(
        self,
        task_id: int,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Any = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        The forward pass for a single task's batch.
        """
        # Collect all potential inputs
        inputs = {}
        if input_ids is not None: inputs['input_ids'] = input_ids
        if attention_mask is not None: inputs['attention_mask'] = attention_mask
        if token_type_ids is not None: inputs['token_type_ids'] = token_type_ids
        inputs.update(kwargs)
        
        # The backbone manager handles translating these inputs to its specific encoder
        encoder_outputs_dict = self.backbone(**inputs)
        
        # Wrap the dict output into an object that task heads expect (with .last_hidden_state etc)
        from types import SimpleNamespace
        encoder_outputs = SimpleNamespace(**encoder_outputs_dict)
        
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