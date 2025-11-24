from typing import Dict, Any, Optional, List

import torch
from torch import nn

from multitask_bert.tasks.base import BaseTask
from multitask_bert.core.config import TaskConfig


class RegressionTask(BaseTask):
    """
    Task for Regression.
    """

    def __init__(self, config: TaskConfig, hidden_size: int):
        super().__init__(config, hidden_size)
        self.loss_fn = nn.MSELoss() # Mean Squared Error Loss for regression
        self.regressor = nn.Linear(hidden_size, 1) # Output a single continuous value

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor], # labels will be a single continuous value
        encoder_outputs: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs the forward pass for the Regression task.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: Continuous labels for regression.
            encoder_outputs: Outputs from the shared encoder.

        Returns:
            A dictionary containing predictions and loss (if labels are provided).
        """
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :] # Use [CLS] token output for regression

        predictions = self.regressor(pooled_output)

        output = {"predictions": predictions}

        if labels is not None:
            loss = self.loss_fn(predictions.squeeze(), labels.float())
            output["loss"] = loss
        
        return output

    def get_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculates metrics for the Regression task.
        """
        # For simplicity, we'll just return a dummy metric (e.g., RMSE) for now.
        rmse = torch.sqrt(nn.MSELoss()(predictions.squeeze(), labels.float())).item()
        return {"rmse": rmse}

    def get_predictions(self, predictions: torch.Tensor) -> List[float]:
        """
        Converts raw predictions to a list of floats.
        """
        return predictions.squeeze().tolist()
