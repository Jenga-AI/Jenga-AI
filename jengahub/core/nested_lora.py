"""
Nested LoRA Implementation for JengaHub

This module implements hierarchical LoRA (Low-Rank Adaptation) that works
across multiple abstraction levels, supporting both text and audio tasks
with dynamic rank allocation and cross-level knowledge sharing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
from dataclasses import dataclass

from .config import NestedLoRAConfig


@dataclass
class LoRALayer:
    """Individual LoRA adaptation layer."""
    
    level_id: int
    rank: int
    alpha: int
    dropout: float
    A: nn.Parameter  # Down-projection
    B: nn.Parameter  # Up-projection
    scaling: float
    update_frequency: int
    step_count: int = 0
    active: bool = True


class NestedLoRALinear(nn.Module):
    """
    Linear layer with hierarchical LoRA adaptations.
    Supports multiple LoRA levels with different update frequencies.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: NestedLoRAConfig,
        target_module: str = "query",
        task_specific: bool = False
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.config = config
        self.target_module = target_module
        self.task_specific = task_specific
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Create nested LoRA levels
        self.lora_levels = nn.ModuleDict()
        self._create_nested_levels()
        
        # Cross-level interaction
        self.level_interaction = nn.MultiheadAttention(
            embed_dim=min(config.base_rank * 4, 256),
            num_heads=4,
            batch_first=True
        ) if config.n_levels > 1 else None
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleDict() if task_specific else None
        
        # Learnable level weights
        self.level_weights = nn.Parameter(
            torch.ones(config.n_levels) / config.n_levels
        )
        
        # Dynamic rank adjustment
        self.rank_controller = nn.Linear(self.in_features, config.n_levels)
        
        # Freeze base layer
        self.base_layer.requires_grad_(False)
    
    def _create_nested_levels(self):
        """Create hierarchical LoRA levels with decreasing ranks."""
        config = self.config
        
        for level in range(config.n_levels):
            # Calculate rank for this level
            rank_scale = config.rank_scaling[level] if level < len(config.rank_scaling) else 0.2
            level_rank = max(int(config.base_rank * rank_scale), 1)
            
            # Create LoRA matrices
            lora_A = nn.Parameter(
                torch.randn(level_rank, self.in_features) / math.sqrt(level_rank)
            )
            lora_B = nn.Parameter(
                torch.zeros(self.out_features, level_rank)
            )
            
            # Calculate scaling
            alpha = config.alpha_values[level] if level < len(config.alpha_values) else config.base_rank
            scaling = alpha / level_rank
            
            # Create dropout
            dropout_rate = config.dropout_rates[level] if level < len(config.dropout_rates) else 0.1
            dropout = nn.Dropout(dropout_rate)
            
            # Store level information
            level_dict = nn.ModuleDict({
                'lora_A': nn.Parameter(lora_A),
                'lora_B': nn.Parameter(lora_B), 
                'dropout': dropout
            })
            
            self.lora_levels[f'level_{level}'] = level_dict
            
            # Store metadata
            setattr(self, f'level_{level}_rank', level_rank)
            setattr(self, f'level_{level}_scaling', scaling)
            setattr(self, f'level_{level}_frequency', config.update_frequencies[level])
            setattr(self, f'level_{level}_steps', 0)
    
    def add_task_adapter(self, task_id: int, rank: int = None):
        """Add task-specific LoRA adapter."""
        if self.task_adapters is None:
            self.task_adapters = nn.ModuleDict()
        
        rank = rank or self.config.base_rank // 2
        
        task_A = nn.Parameter(
            torch.randn(rank, self.in_features) / math.sqrt(rank)
        )
        task_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        self.task_adapters[str(task_id)] = nn.ModuleDict({
            'lora_A': task_A,
            'lora_B': task_B,
            'scaling': self.config.base_rank / rank,
            'dropout': nn.Dropout(0.1)
        })
    
    def _compute_level_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Dynamically compute importance of each LoRA level."""
        # Global average pooling to get representation
        pooled = x.mean(dim=-2) if x.dim() > 2 else x
        
        # Compute level importance scores
        importance = self.rank_controller(pooled)  # [batch_size, n_levels]
        importance = torch.softmax(importance, dim=-1)
        
        return importance
    
    def _apply_cross_level_interaction(
        self, 
        level_outputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply cross-level attention to share information between levels."""
        if self.level_interaction is None or len(level_outputs) < 2:
            return level_outputs
        
        # Stack level outputs for attention
        stacked = torch.stack(level_outputs, dim=1)  # [batch, n_levels, features]
        
        # Apply self-attention across levels
        enhanced, _ = self.level_interaction(stacked, stacked, stacked)
        
        # Split back to individual levels
        enhanced_outputs = [enhanced[:, i] for i in range(enhanced.size(1))]
        
        return enhanced_outputs
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_id: Optional[int] = None,
        active_levels: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass with nested LoRA adaptation.
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            task_id: Task identifier for task-specific adaptation
            active_levels: Which LoRA levels to activate (None = all)
            
        Returns:
            Adapted output tensor
        """
        # Base layer forward pass
        base_output = self.base_layer(x)
        
        # Compute dynamic level importance
        level_importance = self._compute_level_importance(x)
        
        # Apply LoRA adaptations at each level
        level_outputs = []
        active_levels = active_levels or list(range(self.config.n_levels))
        
        for level in active_levels:
            level_key = f'level_{level}'
            if level_key not in self.lora_levels:
                continue
            
            level_dict = self.lora_levels[level_key]
            
            # Check if this level should update (based on frequency)
            level_steps = getattr(self, f'level_{level}_steps')
            update_freq = getattr(self, f'level_{level}_frequency')
            
            if self.training:
                setattr(self, f'level_{level}_steps', level_steps + 1)
            
            # Apply LoRA transformation: x @ A^T @ B^T
            lora_A = level_dict['lora_A']
            lora_B = level_dict['lora_B']
            dropout = level_dict['dropout']
            scaling = getattr(self, f'level_{level}_scaling')
            
            # LoRA computation
            x_dropped = dropout(x) if self.training else x
            down_proj = torch.matmul(x_dropped, lora_A.T)  # [batch, seq, rank]
            up_proj = torch.matmul(down_proj, lora_B.T)     # [batch, seq, out_features]
            
            level_output = up_proj * scaling
            level_outputs.append(level_output)
        
        # Apply cross-level interaction
        if len(level_outputs) > 1:
            level_outputs = self._apply_cross_level_interaction(level_outputs)
        
        # Combine level outputs with dynamic weighting
        if level_outputs:
            # Weight by learned weights and dynamic importance
            combined_weights = F.softmax(self.level_weights[:len(level_outputs)], dim=0)
            
            # Apply importance weighting per sample
            weighted_outputs = []
            for i, output in enumerate(level_outputs):
                if level_importance.dim() > 1:
                    # Per-sample importance
                    sample_weight = level_importance[:, i:i+1].unsqueeze(-1)
                    weighted_output = output * sample_weight * combined_weights[i]
                else:
                    # Global importance
                    weighted_output = output * combined_weights[i]
                weighted_outputs.append(weighted_output)
            
            lora_output = sum(weighted_outputs)
        else:
            lora_output = torch.zeros_like(base_output)
        
        # Apply task-specific adaptation
        if task_id is not None and self.task_adapters and str(task_id) in self.task_adapters:
            task_adapter = self.task_adapters[str(task_id)]
            
            task_A = task_adapter['lora_A']
            task_B = task_adapter['lora_B']
            task_scaling = task_adapter['scaling']
            task_dropout = task_adapter['dropout']
            
            x_task = task_dropout(x) if self.training else x
            task_down = torch.matmul(x_task, task_A.T)
            task_up = torch.matmul(task_down, task_B.T)
            task_output = task_up * task_scaling
            
            lora_output = lora_output + task_output
        
        return base_output + lora_output
    
    def get_level_statistics(self) -> Dict[str, any]:
        """Get statistics about LoRA levels."""
        stats = {
            'total_parameters': sum(
                p.numel() for level_dict in self.lora_levels.values() 
                for p in level_dict.parameters()
            ),
            'level_ranks': [
                getattr(self, f'level_{i}_rank') 
                for i in range(self.config.n_levels)
            ],
            'level_weights': self.level_weights.detach().cpu().tolist(),
            'update_frequencies': [
                getattr(self, f'level_{i}_frequency')
                for i in range(self.config.n_levels)
            ]
        }
        
        if self.task_adapters:
            stats['task_adapters'] = list(self.task_adapters.keys())
            stats['task_adapter_parameters'] = sum(
                p.numel() for adapter in self.task_adapters.values()
                for p in adapter.parameters()
            )
        
        return stats


class NestedLoRAConverter:
    """Utility to convert standard layers to NestedLoRA layers."""
    
    @staticmethod
    def convert_model(
        model: nn.Module,
        config: NestedLoRAConfig,
        target_modules: List[str] = ["query", "value", "key"],
        task_specific_modules: List[str] = ["classifier"]
    ) -> nn.Module:
        """
        Convert a model to use NestedLoRA layers.
        
        Args:
            model: Model to convert
            config: NestedLoRA configuration
            target_modules: Names of modules to replace with LoRA
            task_specific_modules: Modules that need task-specific adaptation
            
        Returns:
            Model with NestedLoRA layers
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module should be replaced
                should_replace = any(target in name.lower() for target in target_modules)
                is_task_specific = any(target in name.lower() for target in task_specific_modules)
                
                if should_replace:
                    # Create NestedLoRA replacement
                    lora_layer = NestedLoRALinear(
                        base_layer=module,
                        config=config,
                        target_module=name.split('.')[-1],
                        task_specific=is_task_specific
                    )
                    
                    # Replace in model
                    parent = model
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    
                    setattr(parent, name.split('.')[-1], lora_layer)
        
        return model
    
    @staticmethod
    def add_task_adapters(
        model: nn.Module,
        task_configs: Dict[int, Dict[str, any]]
    ):
        """Add task-specific adapters to NestedLoRA layers."""
        for name, module in model.named_modules():
            if isinstance(module, NestedLoRALinear) and module.task_specific:
                for task_id, task_config in task_configs.items():
                    rank = task_config.get('rank', module.config.base_rank // 2)
                    module.add_task_adapter(task_id, rank)
    
    @staticmethod
    def get_model_statistics(model: nn.Module) -> Dict[str, any]:
        """Get comprehensive statistics about LoRA adaptation in model."""
        stats = {
            'total_lora_layers': 0,
            'total_lora_parameters': 0,
            'layer_statistics': {}
        }
        
        for name, module in model.named_modules():
            if isinstance(module, NestedLoRALinear):
                layer_stats = module.get_level_statistics()
                stats['layer_statistics'][name] = layer_stats
                stats['total_lora_layers'] += 1
                stats['total_lora_parameters'] += layer_stats['total_parameters']
        
        # Calculate parameter efficiency
        total_model_params = sum(p.numel() for p in model.parameters())
        stats['parameter_efficiency'] = {
            'total_model_parameters': total_model_params,
            'lora_parameters': stats['total_lora_parameters'],
            'efficiency_ratio': stats['total_lora_parameters'] / total_model_params
        }
        
        return stats


class NestedLoRAScheduler:
    """Scheduler for dynamically adjusting LoRA levels during training."""
    
    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 1000,
        max_steps: int = 10000
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step_count = 0
    
    def step(self):
        """Update LoRA scheduling for one training step."""
        self.step_count += 1
        
        # Determine active levels based on training progress
        progress = min(self.step_count / self.max_steps, 1.0)
        
        for name, module in self.model.named_modules():
            if isinstance(module, NestedLoRALinear):
                # Early training: focus on fast adaptation levels (0, 1)
                # Later training: activate all levels
                if self.step_count < self.warmup_steps:
                    active_levels = [0, 1]
                elif progress < 0.5:
                    active_levels = [0, 1, 2]
                else:
                    active_levels = list(range(module.config.n_levels))
                
                # Store active levels for forward pass
                module.active_levels = active_levels
    
    def get_current_schedule(self) -> Dict[str, any]:
        """Get current scheduling state."""
        progress = min(self.step_count / self.max_steps, 1.0)
        
        return {
            'step_count': self.step_count,
            'progress': progress,
            'phase': 'warmup' if self.step_count < self.warmup_steps else 'main_training'
        }