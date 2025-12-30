"""
Training Callbacks for JengaHub

This module provides a comprehensive callback system for training hooks,
monitoring, and advanced training features.
"""

import torch
import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ..core.nested_lora import NestedLoRAScheduler


class TrainingCallback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch. Return True to stop training."""
        pass


class EarlyStopping(TrainingCallback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = 'val_loss',
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
        self.stopped_epoch = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, trainer):
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.wait_count = 0
        self.stopped_epoch = 0
    
    def on_train_end(self, trainer):
        if self.stopped_epoch > 0:
            self.logger.info(f"Early stopping triggered at epoch {self.stopped_epoch}")
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        if self.monitor not in metrics:
            self.logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return False
        
        current_value = metrics[self.monitor]
        
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.wait_count = 0
            else:
                self.wait_count += 1
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.wait_count = 0
            else:
                self.wait_count += 1
        
        if self.wait_count >= self.patience:
            self.stopped_epoch = epoch + 1
            self.logger.info(f"Early stopping triggered. Best {self.monitor}: {self.best_value}")
            return True
        
        return False


class ModelCheckpoint(TrainingCallback):
    """Model checkpointing callback with configurable saving strategy."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 1,
        max_checkpoints: int = 5
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoint_paths = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure directory exists
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
    
    def on_train_begin(self, trainer):
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.checkpoint_paths = []
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        if (epoch + 1) % self.save_frequency != 0:
            return False
        
        should_save = not self.save_best_only
        
        if self.monitor in metrics:
            current_value = metrics[self.monitor]
            
            if self.mode == 'min':
                if current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
            else:  # mode == 'max'
                if current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
        
        if should_save:
            checkpoint_path = self._get_checkpoint_path(epoch, metrics.get(self.monitor))
            self._save_checkpoint(trainer, checkpoint_path)
            self._cleanup_old_checkpoints()
        
        return False
    
    def _get_checkpoint_path(self, epoch: int, metric_value: Optional[float]) -> Path:
        """Generate checkpoint path with epoch and metric information."""
        suffix = self.filepath.suffix
        stem = self.filepath.stem
        
        if metric_value is not None:
            filename = f"{stem}_epoch{epoch:03d}_{self.monitor}{metric_value:.4f}{suffix}"
        else:
            filename = f"{stem}_epoch{epoch:03d}{suffix}"
        
        return self.filepath.parent / filename
    
    def _save_checkpoint(self, trainer, checkpoint_path: Path):
        """Save model checkpoint."""
        try:
            # Get model state dict (handle distributed training)
            model_state_dict = (
                trainer.model.module.state_dict() if trainer.is_distributed 
                else trainer.model.state_dict()
            )
            
            checkpoint = {
                'epoch': trainer.state.epoch,
                'global_step': trainer.state.global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
                'config': trainer.config.to_dict(),
                'best_metric': self.best_value
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.checkpoint_paths.append(checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        while len(self.checkpoint_paths) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_paths.pop(0)
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduling callback with visualization."""
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        log_lr: bool = True,
        plot_lr: bool = True
    ):
        self.scheduler = scheduler
        self.log_lr = log_lr
        self.plot_lr = plot_lr
        
        self.lr_history = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, trainer):
        self.lr_history = []
    
    def on_train_end(self, trainer):
        if self.plot_lr and self.lr_history:
            self._plot_learning_rate(trainer.output_dir)
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        current_lr = trainer.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        
        if self.log_lr:
            self.logger.info(f"Epoch {epoch}: Learning Rate = {current_lr:.2e}")
        
        # Step scheduler if not handled by trainer
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor_value = metrics.get('val_loss', metrics.get('loss', 0))
            self.scheduler.step(monitor_value)
        else:
            self.scheduler.step()
        
        return False
    
    def _plot_learning_rate(self, output_dir: Path):
        """Plot learning rate evolution."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.lr_history)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True)
            
            lr_plot_path = output_dir / 'learning_rate_schedule.png'
            plt.savefig(lr_plot_path)
            plt.close()
            
            self.logger.info(f"Learning rate plot saved: {lr_plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to plot learning rate: {e}")


class MemoryMonitor(TrainingCallback):
    """Monitor system and GPU memory usage during training."""
    
    def __init__(
        self,
        log_frequency: int = 10,
        plot_memory: bool = True
    ):
        self.log_frequency = log_frequency
        self.plot_memory = plot_memory
        
        self.cpu_memory_history = []
        self.gpu_memory_history = []
        self.epochs = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, trainer):
        self.cpu_memory_history = []
        self.gpu_memory_history = []
        self.epochs = []
        
        self.logger.info("Memory monitoring started")
        self._log_system_info()
    
    def on_train_end(self, trainer):
        if self.plot_memory:
            self._plot_memory_usage(trainer.output_dir)
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        if (epoch + 1) % self.log_frequency == 0:
            cpu_memory = psutil.virtual_memory().percent
            
            gpu_memory = 0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            
            self.cpu_memory_history.append(cpu_memory)
            self.gpu_memory_history.append(gpu_memory)
            self.epochs.append(epoch)
            
            self.logger.info(
                f"Epoch {epoch}: CPU Memory: {cpu_memory:.1f}%, "
                f"GPU Memory: {gpu_memory:.1f}%"
            )
        
        return False
    
    def _log_system_info(self):
        """Log system information."""
        cpu_count = psutil.cpu_count()
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        self.logger.info(f"System: {cpu_count} CPUs, {total_memory:.1f}GB RAM")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self.logger.info(f"GPU: {gpu_count} devices, {gpu_memory:.1f}GB memory")
    
    def _plot_memory_usage(self, output_dir: Path):
        """Plot memory usage over time."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # CPU memory plot
            ax1.plot(self.epochs, self.cpu_memory_history, 'b-', label='CPU Memory')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('CPU Memory Usage (%)')
            ax1.set_title('CPU Memory Usage')
            ax1.grid(True)
            ax1.legend()
            
            # GPU memory plot
            if torch.cuda.is_available() and self.gpu_memory_history:
                ax2.plot(self.epochs, self.gpu_memory_history, 'r-', label='GPU Memory')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('GPU Memory Usage (%)')
                ax2.set_title('GPU Memory Usage')
                ax2.grid(True)
                ax2.legend()
            
            plt.tight_layout()
            
            memory_plot_path = output_dir / 'memory_usage.png'
            plt.savefig(memory_plot_path)
            plt.close()
            
            self.logger.info(f"Memory usage plot saved: {memory_plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to plot memory usage: {e}")


class LoRAScheduler(TrainingCallback):
    """Callback for managing LoRA level scheduling during training."""
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        full_activation_epoch: int = 10
    ):
        self.warmup_epochs = warmup_epochs
        self.full_activation_epoch = full_activation_epoch
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, trainer):
        if hasattr(trainer, 'lora_scheduler') and trainer.lora_scheduler:
            self.logger.info("LoRA scheduling enabled")
        else:
            self.logger.warning("LoRA scheduler not found in trainer")
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, trainer, epoch: int):
        if hasattr(trainer, 'lora_scheduler') and trainer.lora_scheduler:
            # Update LoRA scheduler based on epoch
            current_schedule = trainer.lora_scheduler.get_current_schedule()
            
            if epoch < self.warmup_epochs:
                # Gradual activation during warmup
                active_levels = list(range(min(epoch + 1, trainer.model.config.nested_lora.n_levels)))
            elif epoch >= self.full_activation_epoch:
                # All levels active
                active_levels = list(range(trainer.model.config.nested_lora.n_levels))
            else:
                # Progressive activation
                progress = (epoch - self.warmup_epochs) / (self.full_activation_epoch - self.warmup_epochs)
                n_active = int(progress * trainer.model.config.nested_lora.n_levels) + self.warmup_epochs
                active_levels = list(range(min(n_active, trainer.model.config.nested_lora.n_levels)))
            
            self.logger.info(f"Epoch {epoch}: LoRA active levels: {active_levels}")
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        return False


class MetricsLogger(TrainingCallback):
    """Enhanced metrics logging with visualization."""
    
    def __init__(
        self,
        log_frequency: int = 1,
        plot_metrics: bool = True,
        save_metrics: bool = True
    ):
        self.log_frequency = log_frequency
        self.plot_metrics = plot_metrics
        self.save_metrics = save_metrics
        
        self.metrics_history = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def on_train_begin(self, trainer):
        self.metrics_history = []
        self.logger.info("Enhanced metrics logging started")
    
    def on_train_end(self, trainer):
        if self.plot_metrics:
            self._plot_training_curves(trainer.output_dir)
        
        if self.save_metrics:
            self._save_metrics_history(trainer.output_dir)
    
    def on_epoch_begin(self, trainer, epoch: int):
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> bool:
        if (epoch + 1) % self.log_frequency == 0:
            # Store metrics with additional information
            extended_metrics = {
                'epoch': epoch,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                'global_step': trainer.state.global_step,
                **metrics
            }
            
            self.metrics_history.append(extended_metrics)
            
            # Log key metrics
            log_msg = f"Epoch {epoch}: "
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_msg += f"{key}: {value:.4f}, "
            
            self.logger.info(log_msg.rstrip(', '))
        
        return False
    
    def _plot_training_curves(self, output_dir: Path):
        """Plot comprehensive training curves."""
        if not self.metrics_history:
            return
        
        try:
            # Extract metrics for plotting
            epochs = [m['epoch'] for m in self.metrics_history]
            train_losses = [m.get('loss', 0) for m in self.metrics_history]
            val_losses = [m.get('val_loss', 0) for m in self.metrics_history if 'val_loss' in m]
            learning_rates = [m.get('learning_rate', 0) for m in self.metrics_history]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
            if val_losses and len(val_losses) == len(epochs):
                axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Learning rate
            axes[0, 1].plot(epochs, learning_rates, 'g-')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
            
            # Task-specific losses (if available)
            task_metrics = {}
            for metrics in self.metrics_history:
                for key, value in metrics.items():
                    if key not in ['epoch', 'loss', 'val_loss', 'learning_rate', 'global_step']:
                        if key not in task_metrics:
                            task_metrics[key] = []
                        task_metrics[key].append(value)
            
            if task_metrics:
                ax = axes[1, 0]
                for task_name, values in task_metrics.items():
                    if len(values) == len(epochs):
                        ax.plot(epochs, values, label=task_name)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Task Loss')
                ax.set_title('Task-Specific Losses')
                ax.legend()
                ax.grid(True)
            
            # Memory usage (if available)
            # This would be filled by MemoryMonitor callback
            
            plt.tight_layout()
            
            curves_plot_path = output_dir / 'training_curves.png'
            plt.savefig(curves_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training curves saved: {curves_plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to plot training curves: {e}")
    
    def _save_metrics_history(self, output_dir: Path):
        """Save metrics history to JSON file."""
        try:
            metrics_path = output_dir / 'training_metrics.json'
            
            import json
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            self.logger.info(f"Metrics history saved: {metrics_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save metrics history: {e}")


# Utility function to create common callback combinations
def create_default_callbacks(
    output_dir: str,
    patience: int = 10,
    save_frequency: int = 5,
    monitor: str = 'val_loss'
) -> List[TrainingCallback]:
    """Create a default set of callbacks for training."""
    
    callbacks = [
        EarlyStopping(
            patience=patience,
            monitor=monitor,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=f"{output_dir}/checkpoint_{{epoch:03d}}.pth",
            monitor=monitor,
            save_best_only=False,
            save_frequency=save_frequency
        ),
        MemoryMonitor(
            log_frequency=10,
            plot_memory=True
        ),
        LoRAScheduler(
            warmup_epochs=5,
            full_activation_epoch=15
        ),
        MetricsLogger(
            log_frequency=1,
            plot_metrics=True,
            save_metrics=True
        )
    ]
    
    return callbacks