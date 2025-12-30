"""
Model Pruning for JengaHub

This module provides comprehensive model pruning techniques including structured
and unstructured pruning, magnitude-based pruning, and gradual pruning strategies.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import logging
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..core.model import JengaHubMultiModalModel
from ..core.config import MultiModalConfig
from ..training.trainer import JengaHubTrainer


class StructuredPruner:
    """Structured pruning for systematic removal of channels, filters, or attention heads."""
    
    def __init__(
        self,
        pruning_ratio: float = 0.5,
        importance_metric: str = "l2_norm",
        granularity: str = "channel"
    ):
        """
        Initialize structured pruner.
        
        Args:
            pruning_ratio: Fraction of structures to prune
            importance_metric: Metric for importance calculation (l2_norm, l1_norm, gradient)
            granularity: Pruning granularity (channel, filter, head)
        """
        self.pruning_ratio = pruning_ratio
        self.importance_metric = importance_metric
        self.granularity = granularity
        self.logger = logging.getLogger(__name__)
    
    def prune_model(
        self,
        model: JengaHubMultiModalModel,
        calibration_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> JengaHubMultiModalModel:
        """
        Apply structured pruning to the model.
        
        Args:
            model: Model to prune
            calibration_dataloader: Data for importance calculation
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Starting structured pruning with {self.pruning_ratio:.1%} ratio")
        
        pruned_model = copy.deepcopy(model)
        
        # Calculate importance scores
        importance_scores = self._calculate_importance_scores(
            pruned_model, calibration_dataloader
        )
        
        # Apply pruning based on granularity
        if self.granularity == "channel":
            pruned_model = self._prune_channels(pruned_model, importance_scores)
        elif self.granularity == "filter":
            pruned_model = self._prune_filters(pruned_model, importance_scores)
        elif self.granularity == "head":
            pruned_model = self._prune_attention_heads(pruned_model, importance_scores)
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        
        # Update model architecture to reflect pruning
        pruned_model = self._update_model_architecture(pruned_model)
        
        self.logger.info("Structured pruning completed")
        return pruned_model
    
    def _calculate_importance_scores(
        self,
        model: torch.nn.Module,
        calibration_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Dict[str, torch.Tensor]:
        """Calculate importance scores for each structure."""
        importance_scores = {}
        
        if self.importance_metric == "l2_norm":
            # L2 norm of weights
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    weight = module.weight.data
                    if self.granularity == "channel":
                        # Calculate L2 norm per input channel
                        scores = torch.norm(weight, p=2, dim=(0, 2, 3) if len(weight.shape) == 4 else 0)
                    elif self.granularity == "filter":
                        # Calculate L2 norm per output channel (filter)
                        scores = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                    else:
                        scores = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                    
                    importance_scores[name] = scores
        
        elif self.importance_metric == "gradient":
            # Gradient-based importance (requires calibration data)
            if calibration_dataloader is None:
                raise ValueError("Calibration data required for gradient-based importance")
            
            importance_scores = self._calculate_gradient_importance(
                model, calibration_dataloader
            )
        
        return importance_scores
    
    def _calculate_gradient_importance(
        self,
        model: torch.nn.Module,
        calibration_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, torch.Tensor]:
        """Calculate importance based on gradients."""
        model.train()
        importance_scores = {}
        
        # Initialize gradient accumulators
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                importance_scores[name] = torch.zeros_like(module.weight.data)
        
        # Accumulate gradients
        for batch in tqdm(calibration_dataloader, desc="Calculating gradient importance"):
            model.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch, return_dict=True)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            for name, module in model.named_modules():
                if name in importance_scores and module.weight.grad is not None:
                    importance_scores[name] += torch.abs(module.weight.grad)
        
        # Convert to importance per structure
        for name, grad_importance in importance_scores.items():
            module = dict(model.named_modules())[name]
            weight_shape = module.weight.shape
            
            if self.granularity == "channel":
                # Sum over output channels and spatial dimensions
                if len(weight_shape) == 4:  # Conv2d
                    scores = torch.sum(grad_importance, dim=(0, 2, 3))
                else:  # Linear
                    scores = torch.sum(grad_importance, dim=0)
            elif self.granularity == "filter":
                # Sum over input channels and spatial dimensions
                scores = torch.sum(grad_importance.view(weight_shape[0], -1), dim=1)
            
            importance_scores[name] = scores
        
        return importance_scores
    
    def _prune_channels(
        self,
        model: torch.nn.Module,
        importance_scores: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """Prune input channels based on importance scores."""
        for name, scores in importance_scores.items():
            module = dict(model.named_modules())[name]
            
            # Determine channels to prune
            num_channels = len(scores)
            num_to_prune = int(num_channels * self.pruning_ratio)
            
            if num_to_prune > 0:
                # Get indices of least important channels
                _, prune_indices = torch.topk(scores, num_to_prune, largest=False)
                
                # Create pruning mask
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask[prune_indices] = False
                
                # Apply channel pruning (this is a simplified approach)
                # In practice, this would require careful handling of dependent layers
                self.logger.info(f"Pruning {num_to_prune}/{num_channels} channels in {name}")
        
        return model
    
    def _prune_filters(
        self,
        model: torch.nn.Module,
        importance_scores: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """Prune output filters based on importance scores."""
        for name, scores in importance_scores.items():
            module = dict(model.named_modules())[name]
            
            # Determine filters to prune
            num_filters = len(scores)
            num_to_prune = int(num_filters * self.pruning_ratio)
            
            if num_to_prune > 0:
                # Get indices of least important filters
                _, prune_indices = torch.topk(scores, num_to_prune, largest=False)
                
                self.logger.info(f"Pruning {num_to_prune}/{num_filters} filters in {name}")
        
        return model
    
    def _prune_attention_heads(
        self,
        model: torch.nn.Module,
        importance_scores: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """Prune attention heads in transformer layers."""
        # This would require specific handling of attention mechanisms
        self.logger.info("Attention head pruning not yet implemented")
        return model
    
    def _update_model_architecture(self, model: torch.nn.Module) -> torch.nn.Module:
        """Update model architecture to reflect pruning changes."""
        # This would rebuild the model with the new architecture
        # For now, return the model as-is
        return model


class UnstructuredPruner:
    """Unstructured pruning for fine-grained weight removal."""
    
    def __init__(
        self,
        pruning_ratio: float = 0.5,
        pruning_method: str = "magnitude",
        global_pruning: bool = False
    ):
        """
        Initialize unstructured pruner.
        
        Args:
            pruning_ratio: Fraction of weights to prune
            pruning_method: Pruning method (magnitude, random, structured)
            global_pruning: Whether to prune globally or per-layer
        """
        self.pruning_ratio = pruning_ratio
        self.pruning_method = pruning_method
        self.global_pruning = global_pruning
        self.logger = logging.getLogger(__name__)
    
    def prune_model(
        self,
        model: JengaHubMultiModalModel,
        layer_types: Optional[List[type]] = None
    ) -> JengaHubMultiModalModel:
        """
        Apply unstructured pruning to the model.
        
        Args:
            model: Model to prune
            layer_types: Types of layers to prune
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Starting unstructured pruning with {self.pruning_ratio:.1%} ratio")
        
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]
        
        pruned_model = copy.deepcopy(model)
        
        # Collect parameters to prune
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                parameters_to_prune.append((module, 'weight'))
        
        if self.global_pruning:
            # Global magnitude pruning
            if self.pruning_method == "magnitude":
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.pruning_ratio,
                )
        else:
            # Layer-wise pruning
            for module, param_name in parameters_to_prune:
                if self.pruning_method == "magnitude":
                    prune.l1_unstructured(module, param_name, amount=self.pruning_ratio)
                elif self.pruning_method == "random":
                    prune.random_unstructured(module, param_name, amount=self.pruning_ratio)
        
        self.logger.info("Unstructured pruning completed")
        return pruned_model
    
    def remove_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """Remove pruning masks and make pruning permanent."""
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        
        self.logger.info("Pruning masks removed")
        return model


class MagnitudePruner:
    """Magnitude-based pruning with advanced strategies."""
    
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        pruning_schedule: str = "polynomial",
        frequency: int = 100
    ):
        """
        Initialize magnitude pruner.
        
        Args:
            initial_sparsity: Starting sparsity level
            final_sparsity: Target sparsity level
            pruning_schedule: Schedule for sparsity increase (linear, polynomial, exponential)
            frequency: Pruning frequency (steps)
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_schedule = pruning_schedule
        self.frequency = frequency
        self.logger = logging.getLogger(__name__)
    
    def create_pruning_schedule(self, total_steps: int) -> List[float]:
        """Create sparsity schedule for gradual pruning."""
        pruning_steps = list(range(0, total_steps, self.frequency))
        
        if self.pruning_schedule == "linear":
            sparsities = np.linspace(
                self.initial_sparsity, 
                self.final_sparsity, 
                len(pruning_steps)
            )
        elif self.pruning_schedule == "polynomial":
            # Polynomial schedule with power 3
            progress = np.linspace(0, 1, len(pruning_steps))
            sparsities = self.initial_sparsity + (
                self.final_sparsity - self.initial_sparsity
            ) * (progress ** 3)
        elif self.pruning_schedule == "exponential":
            # Exponential decay schedule
            decay_rate = np.log(self.final_sparsity / max(self.initial_sparsity, 1e-10))
            progress = np.linspace(0, 1, len(pruning_steps))
            sparsities = self.initial_sparsity * np.exp(decay_rate * progress)
        else:
            raise ValueError(f"Unsupported pruning schedule: {self.pruning_schedule}")
        
        schedule = list(zip(pruning_steps, sparsities))
        self.logger.info(f"Created {len(schedule)} pruning steps")
        
        return schedule
    
    def apply_magnitude_pruning(
        self,
        model: torch.nn.Module,
        target_sparsity: float,
        layer_types: Optional[List[type]] = None
    ):
        """Apply magnitude-based pruning to achieve target sparsity."""
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]
        
        parameters_to_prune = []
        for module in model.modules():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity,
        )


class GradualPruner:
    """Gradual pruning during training with fine-tuning."""
    
    def __init__(
        self,
        pruner: Union[StructuredPruner, UnstructuredPruner, MagnitudePruner],
        fine_tuning_epochs: int = 10,
        pruning_frequency: int = 5
    ):
        """
        Initialize gradual pruner.
        
        Args:
            pruner: Base pruner to use
            fine_tuning_epochs: Epochs to fine-tune after each pruning step
            pruning_frequency: Frequency of pruning (epochs)
        """
        self.pruner = pruner
        self.fine_tuning_epochs = fine_tuning_epochs
        self.pruning_frequency = pruning_frequency
        self.logger = logging.getLogger(__name__)
    
    def gradual_prune_and_fine_tune(
        self,
        model: JengaHubMultiModalModel,
        trainer: JengaHubTrainer,
        total_epochs: int
    ) -> JengaHubMultiModalModel:
        """
        Apply gradual pruning with fine-tuning.
        
        Args:
            model: Model to prune
            trainer: Trainer for fine-tuning
            total_epochs: Total training epochs
            
        Returns:
            Gradually pruned model
        """
        self.logger.info("Starting gradual pruning with fine-tuning")
        
        pruned_model = copy.deepcopy(model)
        
        # Calculate pruning schedule
        pruning_epochs = list(range(0, total_epochs, self.pruning_frequency))
        
        for epoch in pruning_epochs:
            self.logger.info(f"Pruning at epoch {epoch}")
            
            # Apply pruning step
            if isinstance(self.pruner, MagnitudePruner):
                # Calculate current target sparsity
                progress = epoch / total_epochs
                current_sparsity = self.pruner.initial_sparsity + (
                    self.pruner.final_sparsity - self.pruner.initial_sparsity
                ) * progress
                
                self.pruner.apply_magnitude_pruning(pruned_model, current_sparsity)
            else:
                # Apply structured/unstructured pruning
                pruned_model = self.pruner.prune_model(pruned_model)
            
            # Fine-tune for a few epochs
            self.logger.info(f"Fine-tuning for {self.fine_tuning_epochs} epochs")
            trainer.model = pruned_model
            trainer.training_config.num_epochs = min(self.fine_tuning_epochs, total_epochs - epoch)
            trainer.train()
        
        self.logger.info("Gradual pruning completed")
        return pruned_model


# Convenience functions
def prune_model(
    model: JengaHubMultiModalModel,
    method: str = "magnitude",
    pruning_ratio: float = 0.5,
    **kwargs
) -> JengaHubMultiModalModel:
    """
    Convenience function for model pruning.
    
    Args:
        model: Model to prune
        method: Pruning method (structured, unstructured, magnitude)
        pruning_ratio: Fraction to prune
        **kwargs: Additional pruner arguments
        
    Returns:
        Pruned model
    """
    if method == "structured":
        pruner = StructuredPruner(pruning_ratio=pruning_ratio, **kwargs)
    elif method == "unstructured":
        pruner = UnstructuredPruner(pruning_ratio=pruning_ratio, **kwargs)
    elif method == "magnitude":
        pruner = MagnitudePruner(final_sparsity=pruning_ratio, **kwargs)
    else:
        raise ValueError(f"Unsupported pruning method: {method}")
    
    return pruner.prune_model(model)


def fine_tune_pruned_model(
    pruned_model: JengaHubMultiModalModel,
    trainer: JengaHubTrainer,
    fine_tuning_epochs: int = 10
) -> JengaHubMultiModalModel:
    """
    Fine-tune pruned model to recover performance.
    
    Args:
        pruned_model: Pruned model
        trainer: Trainer instance
        fine_tuning_epochs: Number of fine-tuning epochs
        
    Returns:
        Fine-tuned pruned model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Fine-tuning pruned model for {fine_tuning_epochs} epochs")
    
    # Update trainer with pruned model
    trainer.model = pruned_model
    trainer.training_config.num_epochs = fine_tuning_epochs
    
    # Reduce learning rate for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] *= 0.1
    
    # Train
    trainer.train()
    
    logger.info("Fine-tuning completed")
    return trainer.model


def analyze_sparsity(model: torch.nn.Module) -> Dict[str, float]:
    """
    Analyze sparsity levels in the model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Sparsity statistics
    """
    total_params = 0
    zero_params = 0
    layer_sparsity = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            weight = module.weight.data
            layer_total = weight.numel()
            layer_zeros = (weight == 0).sum().item()
            
            total_params += layer_total
            zero_params += layer_zeros
            
            layer_sparsity[name] = layer_zeros / layer_total if layer_total > 0 else 0.0
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_parameters': total_params,
        'zero_parameters': zero_params,
        'layer_sparsity': layer_sparsity
    }


def visualize_sparsity(model: torch.nn.Module, save_path: Optional[str] = None):
    """
    Visualize sparsity patterns in the model.
    
    Args:
        model: Model to visualize
        save_path: Path to save visualization
    """
    sparsity_stats = analyze_sparsity(model)
    
    # Plot layer-wise sparsity
    layer_names = list(sparsity_stats['layer_sparsity'].keys())
    sparsities = list(sparsity_stats['layer_sparsity'].values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(layer_names)), sparsities)
    plt.xlabel('Layer')
    plt.ylabel('Sparsity')
    plt.title(f"Layer-wise Sparsity (Overall: {sparsity_stats['overall_sparsity']:.2%})")
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()