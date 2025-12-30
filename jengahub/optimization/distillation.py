"""
Knowledge Distillation for JengaHub

This module provides comprehensive knowledge distillation techniques including
response-based, feature-based, and attention-based distillation for model compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import logging
import copy
from pathlib import Path
import time
from tqdm import tqdm

from ..core.model import JengaHubMultiModalModel
from ..core.config import MultiModalConfig
from ..training.trainer import JengaHubTrainer


class KnowledgeDistiller:
    """Main knowledge distillation framework for JengaHub models."""
    
    def __init__(
        self,
        teacher_model: JengaHubMultiModalModel,
        student_model: JengaHubMultiModalModel,
        temperature: float = 4.0,
        alpha: float = 0.7,
        distillation_type: str = "response",
        feature_layers: Optional[List[str]] = None
    ):
        """
        Initialize knowledge distiller.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            temperature: Temperature for softmax distillation
            alpha: Weight for distillation loss vs task loss
            distillation_type: Type of distillation (response, feature, attention)
            feature_layers: Layers for feature distillation
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
        self.feature_layers = feature_layers or []
        
        self.logger = logging.getLogger(__name__)
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Initialize distillation losses
        self.distillation_losses = self._setup_distillation_losses()
    
    def distill(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[MultiModalConfig] = None,
        num_epochs: int = 10,
        output_dir: str = "./distillation_output"
    ) -> JengaHubMultiModalModel:
        """
        Perform knowledge distillation training.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            config: Training configuration
            num_epochs: Number of training epochs
            output_dir: Output directory
            
        Returns:
            Trained student model
        """
        self.logger.info(f"Starting knowledge distillation for {num_epochs} epochs")
        
        # Create custom trainer with distillation
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            student_model=self.student_model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            distiller=self,
            output_dir=output_dir
        )
        
        # Train with distillation
        results = trainer.train()
        
        self.logger.info("Knowledge distillation completed")
        return trainer.student_model
    
    def compute_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        task_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            task_loss: Original task loss
            
        Returns:
            Combined loss and loss components
        """
        distillation_loss = 0.0
        loss_components = {}
        
        if self.distillation_type == "response":
            # Response-based distillation (logit matching)
            distill_loss = self._compute_response_distillation_loss(
                student_outputs, teacher_outputs
            )
            distillation_loss += distill_loss
            loss_components['response_distillation'] = distill_loss.item()
        
        elif self.distillation_type == "feature":
            # Feature-based distillation
            distill_loss = self._compute_feature_distillation_loss(
                student_outputs, teacher_outputs
            )
            distillation_loss += distill_loss
            loss_components['feature_distillation'] = distill_loss.item()
        
        elif self.distillation_type == "attention":
            # Attention-based distillation
            distill_loss = self._compute_attention_distillation_loss(
                student_outputs, teacher_outputs
            )
            distillation_loss += distill_loss
            loss_components['attention_distillation'] = distill_loss.item()
        
        # Combine task loss and distillation loss
        combined_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        loss_components['task_loss'] = task_loss.item()
        loss_components['combined_loss'] = combined_loss.item()
        
        return combined_loss, loss_components
    
    def _setup_distillation_losses(self) -> Dict[str, nn.Module]:
        """Setup loss functions for distillation."""
        losses = {
            'kl_div': nn.KLDivLoss(reduction='batchmean'),
            'mse': nn.MSELoss(),
            'cosine_similarity': nn.CosineEmbeddingLoss()
        }
        return losses
    
    def _compute_response_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute response-based distillation loss."""
        student_logits = student_outputs.get('logits')
        teacher_logits = teacher_outputs.get('logits')
        
        if student_logits is None or teacher_logits is None:
            return torch.tensor(0.0, device=student_outputs.get('loss', torch.tensor(0.0)).device)
        
        # Apply temperature scaling
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = self.distillation_losses['kl_div'](student_probs, teacher_probs)
        
        # Scale by temperature squared
        distillation_loss *= (self.temperature ** 2)
        
        return distillation_loss
    
    def _compute_feature_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute feature-based distillation loss."""
        total_loss = 0.0
        
        student_features = student_outputs.get('hidden_states', {})
        teacher_features = teacher_outputs.get('hidden_states', {})
        
        for layer_name in self.feature_layers:
            if layer_name in student_features and layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # Align feature dimensions if necessary
                if student_feat.shape != teacher_feat.shape:
                    student_feat = self._align_features(student_feat, teacher_feat.shape)
                
                # MSE loss between features
                feat_loss = self.distillation_losses['mse'](student_feat, teacher_feat)
                total_loss += feat_loss
        
        return total_loss
    
    def _compute_attention_distillation_loss(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute attention-based distillation loss."""
        total_loss = 0.0
        
        student_attentions = student_outputs.get('attentions', [])
        teacher_attentions = teacher_outputs.get('attentions', [])
        
        min_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(min_layers):
            student_att = student_attentions[i]
            teacher_att = teacher_attentions[i]
            
            if student_att.shape != teacher_att.shape:
                # Handle different attention head counts
                student_att = self._align_attention_maps(student_att, teacher_att.shape)
            
            # MSE loss between attention maps
            att_loss = self.distillation_losses['mse'](student_att, teacher_att)
            total_loss += att_loss
        
        return total_loss
    
    def _align_features(self, student_feat: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Align student features to teacher feature dimensions."""
        if len(student_feat.shape) == 3 and len(target_shape) == 3:
            # Sequence features: (batch, seq_len, hidden_dim)
            if student_feat.size(-1) != target_shape[-1]:
                # Project to target dimension
                projection = nn.Linear(
                    student_feat.size(-1), 
                    target_shape[-1]
                ).to(student_feat.device)
                student_feat = projection(student_feat)
        
        return student_feat
    
    def _align_attention_maps(
        self, 
        student_att: torch.Tensor, 
        target_shape: torch.Size
    ) -> torch.Tensor:
        """Align student attention maps to teacher dimensions."""
        # Handle different number of attention heads
        if student_att.size(1) != target_shape[1]:  # num_heads dimension
            # Average over excess heads or repeat heads
            if student_att.size(1) > target_shape[1]:
                # Average excess heads
                ratio = student_att.size(1) // target_shape[1]
                student_att = student_att.view(
                    student_att.size(0), target_shape[1], ratio, 
                    student_att.size(2), student_att.size(3)
                ).mean(dim=2)
            else:
                # Repeat heads
                repeat_factor = target_shape[1] // student_att.size(1)
                student_att = student_att.repeat_interleave(repeat_factor, dim=1)
        
        return student_att


class DistillationTrainer(JengaHubTrainer):
    """Custom trainer for knowledge distillation."""
    
    def __init__(
        self,
        teacher_model: JengaHubMultiModalModel,
        student_model: JengaHubMultiModalModel,
        distiller: KnowledgeDistiller,
        **kwargs
    ):
        """Initialize distillation trainer."""
        # Initialize parent with student model
        super().__init__(model=student_model, **kwargs)
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distiller = distiller
        
        # Move teacher to same device as student
        self.teacher_model.to(self.device)
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train epoch with knowledge distillation."""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        loss_components = {}
        
        # Setup progress bar
        if self.rank == 0:
            pbar = tqdm(
                self.train_dataloader, 
                desc=f"Distillation Epoch {self.state.epoch + 1}",
                leave=False
            )
        else:
            pbar = self.train_dataloader
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**batch, return_dict=True)
            
            # Student forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    student_outputs = self.student_model(**batch, return_dict=True)
                    task_loss = student_outputs['loss']
                    
                    # Compute distillation loss
                    combined_loss, batch_loss_components = self.distiller.compute_distillation_loss(
                        student_outputs, teacher_outputs, task_loss
                    )
            else:
                student_outputs = self.student_model(**batch, return_dict=True)
                task_loss = student_outputs['loss']
                
                # Compute distillation loss
                combined_loss, batch_loss_components = self.distiller.compute_distillation_loss(
                    student_outputs, teacher_outputs, task_loss
                )
            
            # Scale loss for gradient accumulation
            combined_loss = combined_loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(combined_loss).backward()
            else:
                combined_loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.training_config.gradient_clipping > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), 
                        self.training_config.gradient_clipping
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Accumulate metrics
            batch_size = self._get_batch_size(batch)
            total_loss += combined_loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size
            
            # Accumulate loss components
            for key, value in batch_loss_components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value * batch_size
            
            # Update progress bar
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{combined_loss.item() * self.gradient_accumulation_steps:.4f}",
                    'task_loss': f"{batch_loss_components.get('task_loss', 0):.4f}",
                    'distill_loss': f"{batch_loss_components.get('response_distillation', 0):.4f}",
                    'lr': f"{current_lr:.2e}"
                })
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_loss_components = {
            key: value / total_samples 
            for key, value in loss_components.items()
        }
        
        metrics = {'loss': avg_loss, **avg_loss_components}
        
        # Store learning rate and loss history
        self.state.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.state.train_loss_history.append(avg_loss)
        
        return metrics


class AttentionDistiller:
    """Specialized distiller for attention mechanisms."""
    
    def __init__(
        self,
        attention_transfer_type: str = "mean",
        layer_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Initialize attention distiller.
        
        Args:
            attention_transfer_type: How to transfer attention (mean, sum, weighted)
            layer_mapping: Mapping from student layers to teacher layers
        """
        self.attention_transfer_type = attention_transfer_type
        self.layer_mapping = layer_mapping
        self.logger = logging.getLogger(__name__)
    
    def compute_attention_loss(
        self,
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention transfer loss."""
        total_loss = 0.0
        
        # Use layer mapping if provided
        if self.layer_mapping:
            for student_layer, teacher_layer in self.layer_mapping.items():
                if (student_layer < len(student_attentions) and 
                    teacher_layer < len(teacher_attentions)):
                    
                    student_att = student_attentions[student_layer]
                    teacher_att = teacher_attentions[teacher_layer]
                    
                    loss = self._compute_attention_distance(student_att, teacher_att)
                    total_loss += loss
        else:
            # Map layers sequentially
            min_layers = min(len(student_attentions), len(teacher_attentions))
            for i in range(min_layers):
                student_att = student_attentions[i]
                teacher_att = teacher_attentions[i]
                
                loss = self._compute_attention_distance(student_att, teacher_att)
                total_loss += loss
        
        return total_loss
    
    def _compute_attention_distance(
        self,
        student_att: torch.Tensor,
        teacher_att: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between attention maps."""
        # Handle different attention head counts
        if student_att.size(1) != teacher_att.size(1):
            if self.attention_transfer_type == "mean":
                # Average attention heads
                student_att = student_att.mean(dim=1, keepdim=True)
                teacher_att = teacher_att.mean(dim=1, keepdim=True)
            elif self.attention_transfer_type == "sum":
                # Sum attention heads
                student_att = student_att.sum(dim=1, keepdim=True)
                teacher_att = teacher_att.sum(dim=1, keepdim=True)
        
        # MSE loss between attention maps
        return F.mse_loss(student_att, teacher_att)


class FeatureDistiller:
    """Specialized distiller for intermediate features."""
    
    def __init__(
        self,
        feature_adaptation: str = "linear",
        feature_loss: str = "mse",
        layer_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize feature distiller.
        
        Args:
            feature_adaptation: How to adapt features (linear, conv, identity)
            feature_loss: Loss function for features (mse, cosine, kl)
            layer_mapping: Mapping from student to teacher layers
        """
        self.feature_adaptation = feature_adaptation
        self.feature_loss = feature_loss
        self.layer_mapping = layer_mapping or {}
        self.adaptation_layers = {}
        
        self.logger = logging.getLogger(__name__)
    
    def compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature distillation loss."""
        total_loss = 0.0
        
        for student_layer, teacher_layer in self.layer_mapping.items():
            if (student_layer in student_features and 
                teacher_layer in teacher_features):
                
                student_feat = student_features[student_layer]
                teacher_feat = teacher_features[teacher_layer]
                
                # Adapt student features to match teacher dimensions
                adapted_student_feat = self._adapt_features(
                    student_feat, teacher_feat, f"{student_layer}_to_{teacher_layer}"
                )
                
                # Compute feature loss
                loss = self._compute_feature_distance(adapted_student_feat, teacher_feat)
                total_loss += loss
        
        return total_loss
    
    def _adapt_features(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        layer_key: str
    ) -> torch.Tensor:
        """Adapt student features to match teacher feature dimensions."""
        if student_feat.shape == teacher_feat.shape:
            return student_feat
        
        # Create adaptation layer if it doesn't exist
        if layer_key not in self.adaptation_layers:
            if self.feature_adaptation == "linear":
                adaptation = nn.Linear(
                    student_feat.size(-1),
                    teacher_feat.size(-1)
                ).to(student_feat.device)
            elif self.feature_adaptation == "conv":
                adaptation = nn.Conv1d(
                    student_feat.size(-1),
                    teacher_feat.size(-1),
                    1
                ).to(student_feat.device)
            else:  # identity
                adaptation = nn.Identity()
            
            self.adaptation_layers[layer_key] = adaptation
        
        # Apply adaptation
        adapted_feat = self.adaptation_layers[layer_key](student_feat)
        
        return adapted_feat
    
    def _compute_feature_distance(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance between feature representations."""
        if self.feature_loss == "mse":
            return F.mse_loss(student_feat, teacher_feat)
        elif self.feature_loss == "cosine":
            return 1 - F.cosine_similarity(
                student_feat.view(-1), teacher_feat.view(-1), dim=0
            )
        elif self.feature_loss == "kl":
            # Normalize features to probabilities
            student_probs = F.log_softmax(student_feat, dim=-1)
            teacher_probs = F.softmax(teacher_feat, dim=-1)
            return F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        else:
            raise ValueError(f"Unsupported feature loss: {self.feature_loss}")


# Convenience functions
def distill_model(
    teacher_model: JengaHubMultiModalModel,
    student_model: JengaHubMultiModalModel,
    train_dataloader: torch.utils.data.DataLoader,
    config: Optional[MultiModalConfig] = None,
    temperature: float = 4.0,
    alpha: float = 0.7,
    num_epochs: int = 10,
    **kwargs
) -> JengaHubMultiModalModel:
    """
    Convenience function for knowledge distillation.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        train_dataloader: Training data
        config: Training configuration
        temperature: Distillation temperature
        alpha: Distillation weight
        num_epochs: Training epochs
        **kwargs: Additional arguments
        
    Returns:
        Distilled student model
    """
    distiller = KnowledgeDistiller(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=temperature,
        alpha=alpha,
        **kwargs
    )
    
    return distiller.distill(
        train_dataloader=train_dataloader,
        config=config,
        num_epochs=num_epochs
    )


def create_student_model(
    teacher_model: JengaHubMultiModalModel,
    compression_ratio: float = 0.5,
    architecture_type: str = "smaller"
) -> JengaHubMultiModalModel:
    """
    Create a smaller student model based on teacher architecture.
    
    Args:
        teacher_model: Teacher model
        compression_ratio: Compression ratio for model size
        architecture_type: Type of compression (smaller, pruned, quantized)
        
    Returns:
        Student model
    """
    if architecture_type == "smaller":
        # Create a smaller version of the teacher model
        # This would involve reducing hidden dimensions, layers, etc.
        # For now, return a copy of the teacher model
        student_model = copy.deepcopy(teacher_model)
        
        # Reduce hidden dimensions in transformer layers
        for module in student_model.modules():
            if hasattr(module, 'd_model'):
                module.d_model = int(module.d_model * compression_ratio)
            if hasattr(module, 'hidden_size'):
                module.hidden_size = int(module.hidden_size * compression_ratio)
        
        return student_model
    
    else:
        # Return a copy for other compression types
        return copy.deepcopy(teacher_model)


def compare_teacher_student(
    teacher_model: JengaHubMultiModalModel,
    student_model: JengaHubMultiModalModel,
    test_dataloader: torch.utils.data.DataLoader,
    num_batches: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Compare teacher and student model performance.
    
    Args:
        teacher_model: Teacher model
        student_model: Student model
        test_dataloader: Test data
        num_batches: Number of test batches
        
    Returns:
        Comparison results
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    # Evaluate teacher model
    logger.info("Evaluating teacher model...")
    teacher_metrics = _evaluate_distillation_model(teacher_model, test_dataloader, num_batches)
    results['teacher'] = teacher_metrics
    
    # Evaluate student model
    logger.info("Evaluating student model...")
    student_metrics = _evaluate_distillation_model(student_model, test_dataloader, num_batches)
    results['student'] = student_metrics
    
    # Calculate relative metrics
    student_metrics['performance_retention'] = student_metrics['accuracy'] / teacher_metrics['accuracy']
    student_metrics['speedup'] = teacher_metrics['inference_time'] / student_metrics['inference_time']
    student_metrics['compression_ratio'] = teacher_metrics['model_size'] / student_metrics['model_size']
    
    logger.info(f"Performance retention: {student_metrics['performance_retention']:.2%}")
    logger.info(f"Speedup: {student_metrics['speedup']:.2f}x")
    logger.info(f"Compression ratio: {student_metrics['compression_ratio']:.2f}x")
    
    return results


def _evaluate_distillation_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    num_batches: int
) -> Dict[str, float]:
    """Evaluate model for distillation comparison."""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_samples = 0
    inference_times = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(**batch, return_dict=True)
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
            
            # Accumulate metrics
            loss = outputs.get('loss', torch.tensor(0.0))
            batch_size = next(iter(batch.values())).size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': 1.0 / (1.0 + total_loss / total_samples),  # Simple accuracy proxy
        'inference_time': np.mean(inference_times),
        'model_size': model_size
    }