"""
Universal Teacher-Student wrapper for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class TeacherStudentWrapper(nn.Module):
    """
    Universal wrapper for knowledge distillation across all model types
    
    Works with Seq2Seq, CausalLM, and Encoder models from HuggingFace.
    """
    
    def __init__(self, student_model, teacher_model, distillation_alpha=0.5, temperature=2.0):
        """
        Initialize the TeacherStudentWrapper
        
        Args:
            student_model: The model being trained (student)
            teacher_model: The pre-trained model providing guidance (teacher)
            distillation_alpha: Weight for distillation loss (0-1)
            temperature: Temperature for softening outputs
        """
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.distillation_alpha = distillation_alpha
        self.temperature = temperature
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        logger.info(f"👨‍🏫 Initialized distillation with alpha={distillation_alpha}, temp={temperature}")
    
    def forward(self, *args, **kwargs):
        """
        Forward pass with distillation loss
        
        Compatible with all HuggingFace model forward signatures.
        """
        # Student forward pass
        student_output = self.student_model(*args, **kwargs)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_model(*args, **kwargs)
        
        # Calculate distillation loss
        if hasattr(student_output, 'logits') and hasattr(teacher_output, 'logits'):
            # Use logits for distillation
            distillation_loss = self._compute_distillation_loss(
                student_output.logits,
                teacher_output.logits
            )
            
            # Combine losses
            if hasattr(student_output, 'loss') and student_output.loss is not None:
                original_loss = student_output.loss
                combined_loss = (
                    (1 - self.distillation_alpha) * original_loss +
                    self.distillation_alpha * distillation_loss
                )
                student_output.loss = combined_loss
        
        return student_output
    
    def _compute_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute KL divergence distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            
        Returns:
            Distillation loss (scaled by temperature squared)
        """
        # Soften distributions with temperature
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )
        
        # Scale by temperature squared (standard practice)
        return kl_loss * (self.temperature ** 2)
