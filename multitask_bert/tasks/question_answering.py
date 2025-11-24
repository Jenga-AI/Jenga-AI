from typing import Dict, Any, Optional, List

import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering

from multitask_bert.tasks.base import BaseTask
from multitask_bert.core.config import TaskConfig


class QuestionAnsweringTask(BaseTask):
    """
    Task for Question Answering (QA).
    """

    def __init__(self, config: TaskConfig, hidden_size: int):
        super().__init__(config, hidden_size)
        # For Question Answering, we typically use a model head that predicts start and end logits
        # The number of labels is 2 (start_logit, end_logit)
        self.num_labels = 2 
        self.qa_outputs = nn.Linear(hidden_size, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def get_forward_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        labels: Optional[Any], # labels will be a tuple (start_positions, end_positions)
        encoder_outputs: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Performs the forward pass for the Question Answering task.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            labels: A tuple containing (start_positions, end_positions).
            encoder_outputs: Outputs from the shared encoder.

        Returns:
            A dictionary containing start_logits, end_logits, and loss (if labels are provided).
        """
        sequence_output = encoder_outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        output = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        if labels is not None:
            start_positions, end_positions = labels
            # If we are on multi-GPU, let's remove the dimension added by DataParallel
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = self.loss_fct.ignore_index
            start_positions.clamp_(0, logits.size(1))
            end_positions.clamp_(0, logits.size(1))

            loss = self.loss_fct(start_logits, start_positions) + self.loss_fct(end_logits, end_positions)
            output["loss"] = loss
        
        return output

    def get_metrics(self, start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor, end_positions: torch.Tensor) -> Dict[str, float]:
        """
        Calculates metrics for the Question Answering task.
        This is a placeholder and would typically involve more complex QA metrics like F1 and Exact Match.
        """
        # For simplicity, we'll just return a dummy metric for now.
        # In a real scenario, you'd use libraries like evaluate or implement SQuAD metrics.
        return {"qa_dummy_metric": 0.0}

    def get_predictions(self, start_logits: torch.Tensor, end_logits: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Converts logits to QA predictions.
        This is a placeholder and would involve post-processing to extract answer spans.
        """
        # For simplicity, return dummy predictions
        return [{"start": -1, "end": -1, "answer": "dummy"}]
