"""
Optimizers and Schedulers for JengaHub Training

This module provides optimized optimizers and learning rate schedulers
for efficient training of JengaHub multimodal models.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, ReduceLROnPlateau, 
    StepLR, ExponentialLR, MultiStepLR
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Dict, Any, Optional, List, Union
import logging
import math


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with model-specific parameter grouping.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer (adamw, adam, sgd, etc.)
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        **kwargs: Additional optimizer parameters
    
    Returns:
        Configured optimizer
    """
    
    # Create parameter groups with different learning rates and weight decay
    param_groups = _create_parameter_groups(
        model, learning_rate, weight_decay
    )
    
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "sgd":
        momentum = kwargs.get('momentum', 0.9)
        optimizer = optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "rmsprop":
        alpha = kwargs.get('alpha', 0.99)
        optimizer = optim.RMSprop(
            param_groups,
            lr=learning_rate,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(
            param_groups,
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    logging.info(f"Created {optimizer_name} optimizer with {len(param_groups)} parameter groups")
    
    return optimizer


def _create_parameter_groups(
    model: torch.nn.Module,
    base_lr: float,
    base_weight_decay: float
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with specialized learning rates and weight decay.
    
    Different components get different learning rates:
    - LoRA parameters: Higher learning rate
    - Memory system: Medium learning rate  
    - Base encoders: Lower learning rate
    - Bias and LayerNorm: No weight decay
    """
    
    # Define parameter group configurations
    group_configs = {
        'lora': {'lr_scale': 2.0, 'weight_decay_scale': 0.5},
        'memory': {'lr_scale': 1.5, 'weight_decay_scale': 0.8},
        'encoder': {'lr_scale': 0.5, 'weight_decay_scale': 1.0},
        'head': {'lr_scale': 1.0, 'weight_decay_scale': 1.0},
        'bias_norm': {'lr_scale': 1.0, 'weight_decay_scale': 0.0}
    }
    
    # Categorize parameters
    param_groups = {name: [] for name in group_configs.keys()}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Categorize parameter based on name
        if 'lora' in name.lower():
            param_groups['lora'].append(param)
        elif 'memory' in name.lower() or 'continuum' in name.lower():
            param_groups['memory'].append(param)
        elif any(encoder in name.lower() for encoder in ['encoder', 'backbone']):
            param_groups['encoder'].append(param)
        elif 'bias' in name.lower() or 'norm' in name.lower() or 'ln' in name.lower():
            param_groups['bias_norm'].append(param)
        else:
            param_groups['head'].append(param)
    
    # Create optimizer parameter groups
    optimizer_groups = []
    
    for group_name, params in param_groups.items():
        if not params:  # Skip empty groups
            continue
        
        config = group_configs[group_name]
        group_lr = base_lr * config['lr_scale']
        group_wd = base_weight_decay * config['weight_decay_scale']
        
        optimizer_groups.append({
            'params': params,
            'lr': group_lr,
            'weight_decay': group_wd,
            'name': group_name
        })
        
        logging.info(
            f"Parameter group '{group_name}': {len(params)} params, "
            f"lr={group_lr:.2e}, wd={group_wd:.4f}"
        )
    
    return optimizer_groups


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "linear_warmup",
    num_training_steps: int = 1000,
    warmup_steps: int = 100,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        num_training_steps: Total training steps
        warmup_steps: Warmup steps
        **kwargs: Additional scheduler parameters
    
    Returns:
        Configured scheduler or None
    """
    
    if scheduler_name is None or scheduler_name.lower() == "none":
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    elif scheduler_name == "cosine":
        T_max = kwargs.get('T_max', num_training_steps)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
    
    elif scheduler_name == "step":
        step_size = kwargs.get('step_size', num_training_steps // 3)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_name == "multistep":
        milestones = kwargs.get('milestones', [num_training_steps // 3, 2 * num_training_steps // 3])
        gamma = kwargs.get('gamma', 0.1)
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif scheduler_name == "exponential":
        gamma = kwargs.get('gamma', 0.95)
        scheduler = ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    elif scheduler_name == "plateau":
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 10)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience
        )
    
    elif scheduler_name == "polynomial":
        scheduler = PolynomialLRScheduler(
            optimizer,
            total_steps=num_training_steps,
            warmup_steps=warmup_steps,
            power=kwargs.get('power', 1.0),
            end_lr=kwargs.get('end_lr', 1e-7)
        )
    
    elif scheduler_name == "one_cycle":
        max_lr = kwargs.get('max_lr', 0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=warmup_steps / num_training_steps
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    logging.info(f"Created {scheduler_name} scheduler")
    
    return scheduler


class PolynomialLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        power: float = 1.0,
        end_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.power = power
        self.end_lr = end_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay phase
            decay_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            
            decay_factor = (1 - current_step / decay_steps) ** self.power
            
            return [
                (base_lr - self.end_lr) * decay_factor + self.end_lr
                for base_lr in self.base_lrs
            ]


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_step / cosine_steps))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler that adjusts based on training dynamics."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-8,
        threshold: float = 1e-4,
        cooldown: int = 0
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.cooldown = cooldown
        
        self.best_metric = float('inf')
        self.wait_count = 0
        self.cooldown_count = 0
        
    def step(self, metric: float):
        """Step the scheduler with current metric value."""
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return
        
        if metric < self.best_metric - self.threshold:
            self.best_metric = metric
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        if self.wait_count >= self.patience:
            self._reduce_lr()
            self.wait_count = 0
            self.cooldown_count = self.cooldown
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            logging.info(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")


def get_optimizer_info(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """Get comprehensive optimizer information."""
    
    info = {
        'type': optimizer.__class__.__name__,
        'param_groups': len(optimizer.param_groups),
        'total_params': sum(len(group['params']) for group in optimizer.param_groups),
        'learning_rates': [group['lr'] for group in optimizer.param_groups],
        'weight_decays': [group.get('weight_decay', 0) for group in optimizer.param_groups]
    }
    
    # Add optimizer-specific parameters
    if hasattr(optimizer, 'defaults'):
        info['defaults'] = optimizer.defaults
    
    return info


def get_scheduler_info(scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> Dict[str, Any]:
    """Get comprehensive scheduler information."""
    
    if scheduler is None:
        return {'type': 'None'}
    
    info = {
        'type': scheduler.__class__.__name__,
        'last_epoch': scheduler.last_epoch,
        'current_lrs': scheduler.get_last_lr() if hasattr(scheduler, 'get_last_lr') else None
    }
    
    # Add scheduler-specific parameters
    if hasattr(scheduler, 'total_steps'):
        info['total_steps'] = scheduler.total_steps
    if hasattr(scheduler, 'warmup_steps'):
        info['warmup_steps'] = scheduler.warmup_steps
    if hasattr(scheduler, 'T_max'):
        info['T_max'] = scheduler.T_max
    
    return info


# Utility function for creating common optimizer-scheduler combinations
def create_training_components(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_training_steps: int
) -> tuple:
    """
    Create optimizer and scheduler from configuration.
    
    Args:
        model: Model to train
        config: Training configuration
        num_training_steps: Total training steps
    
    Returns:
        Tuple of (optimizer, scheduler)
    """
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name=config.get('optimizer', 'adamw'),
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config.get('scheduler', 'linear_warmup'),
        num_training_steps=num_training_steps,
        warmup_steps=config.get('warmup_steps', num_training_steps // 10)
    )
    
    return optimizer, scheduler