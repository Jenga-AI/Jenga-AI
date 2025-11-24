from typing import Dict, Any, Optional, List

import torch
from torch import nn

from multitask_bert.tasks.base import BaseTask
from multitask_bert.core.config import TaskConfig # Import TaskConfig


class QATask(BaseTask):
    """
    Task for Question Answering (QA) which is treated as a multi-label classification problem.
    """

    def __init__(self, config: TaskConfig, hidden_size: int): # Added hidden_size
        super().__init__(config, hidden_size) # Pass hidden_size to super
        self.num_labels = len(self.config.labels) # Use self.config.labels
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.classifier = nn.Linear(self.hidden_size, self.num_labels) # Use self.hidden_size 

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        encoder_outputs: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs the forward pass for the QA task.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: Multi-hot encoded labels for multi-label classification.
            encoder_outputs: Outputs from the shared encoder.

        Returns:
            A dictionary containing logits and loss (if labels are provided).
        """
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token output

        output = {"logits": logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            output["loss"] = loss
        return output

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculates metrics for the QA task.
        For multi-label classification, we can use F1-score, precision, recall, and accuracy.
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Convert probabilities to predictions (0 or 1)
        predictions = (probs > 0.5).long()

        # Flatten labels and predictions for metric calculation
        labels_flat = labels.cpu().numpy().flatten()
        predictions_flat = predictions.cpu().numpy().flatten()

        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

        f1 = f1_score(labels_flat, predictions_flat, average="micro")
        precision = precision_score(labels_flat, predictions_flat, average="micro")
        recall = recall_score(labels_flat, predictions_flat, average="micro")
        accuracy = accuracy_score(labels_flat, predictions_flat)

        return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}

    def get_predictions(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Converts logits to multi-label predictions.
        """
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long().tolist()
        return predictions