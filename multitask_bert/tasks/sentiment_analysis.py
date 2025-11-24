from typing import Dict, Any, Optional, List

import torch
from torch import nn

from multitask_bert.tasks.base import BaseTask
from multitask_bert.core.config import TaskConfig


class SentimentAnalysisTask(BaseTask):
    """
    Task for Sentiment Analysis (single-label classification).
    """

    def __init__(self, config: TaskConfig, hidden_size: int):
        super().__init__(config, hidden_size)
        self.num_labels = len(config.labels) if config.labels else 2 # Default to 2 for binary sentiment
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor], # labels will be a single integer
        encoder_outputs: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs the forward pass for the Sentiment Analysis task.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: Single integer label for classification.
            encoder_outputs: Outputs from the shared encoder.

        Returns:
            A dictionary containing logits and loss (if labels are provided).
        """
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :] # Use [CLS] token output for classification

        logits = self.classifier(pooled_output)

        output = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits, labels.long())
            output["loss"] = loss
        
        return output

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculates metrics for the Sentiment Analysis task.
        """
        # For simplicity, we'll just return accuracy for now.
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return {"accuracy": accuracy}

    def get_predictions(self, logits: torch.Tensor) -> List[int]:
        """
        Converts logits to class predictions.
        """
        return torch.argmax(logits, dim=-1).tolist()
