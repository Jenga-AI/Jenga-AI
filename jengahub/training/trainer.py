"""
JengaHub Trainer

Comprehensive training infrastructure for JengaHub multimodal models with advanced features
including mixed precision, gradient accumulation, model checkpointing, and MLflow integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import time
import logging
import os
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np

from ..core.config import MultiModalConfig
from ..core.model import JengaHubMultiModalModel, save_jengahub_model
from ..core.tracking import JengaHubMLflowLogger, TrainingMetrics
from ..core.nested_lora import NestedLoRAScheduler
from ..data.processor import UnifiedDataProcessor, ProcessedSample
from .callbacks import TrainingCallback
from .optimizers import create_optimizer, create_scheduler


@dataclass
class TrainingState:
    """Container for training state information."""
    
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    should_stop: bool = False
    train_loss_history: List[float] = None
    val_loss_history: List[float] = None
    learning_rates: List[float] = None
    
    def __post_init__(self):
        if self.train_loss_history is None:
            self.train_loss_history = []
        if self.val_loss_history is None:
            self.val_loss_history = []
        if self.learning_rates is None:
            self.learning_rates = []


class JengaHubTrainer:
    """
    Comprehensive trainer for JengaHub multimodal models with production-ready features.
    
    Key Features:
    - Mixed precision training
    - Gradient accumulation
    - Model checkpointing
    - MLflow integration
    - Memory optimization
    - LoRA scheduling
    - Multi-task learning support
    - Code-switching aware training
    """
    
    def __init__(
        self,
        model: JengaHubMultiModalModel,
        config: MultiModalConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        callbacks: List[TrainingCallback] = None,
        logger: Optional[JengaHubMLflowLogger] = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.callbacks = callbacks or []
        self.logger = logger
        self.output_dir = Path(output_dir) if output_dir else Path("./training_output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Training configuration
        self.training_config = config.training
        self.device = torch.device(config.device if config.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup training state
        self.state = TrainingState()
        
        # Setup optimizers and schedulers
        self.optimizer = create_optimizer(
            self.model, 
            self.training_config.optimizer, 
            self.training_config.learning_rate,
            self.training_config.weight_decay
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            self.training_config.scheduler,
            num_training_steps=len(train_dataloader) * self.training_config.num_epochs,
            warmup_steps=self.training_config.warmup_steps
        )
        
        # Mixed precision setup
        self.use_amp = self.training_config.fp16 and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = getattr(
            self.training_config, 'gradient_accumulation_steps', 1
        )
        
        # LoRA scheduler setup
        self.lora_scheduler = None
        if hasattr(self.model, 'text_encoder') or hasattr(self.model, 'audio_encoder'):
            self.lora_scheduler = NestedLoRAScheduler(
                self.model,
                warmup_steps=self.training_config.warmup_steps,
                max_steps=len(train_dataloader) * self.training_config.num_epochs
            )
        
        # Model checkpointing
        self.best_model_path = self.output_dir / "best_model.pth"
        self.last_checkpoint_path = self.output_dir / "last_checkpoint.pth"
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup for distributed training if available
        self.is_distributed = dist.is_available() and dist.is_initialized()
        if self.is_distributed:
            self.model = DistributedDataParallel(self.model)
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"training_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.training_logger = logging.getLogger("JengaHubTrainer")
        self.training_logger.info(f"Training initialized with device: {self.device}")
        self.training_logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.training_logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results and metrics
        """
        self.training_logger.info("Starting training...")
        
        # Initialize callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.state.epoch, self.training_config.num_epochs):
                self.state.epoch = epoch
                
                # Epoch callbacks
                for callback in self.callbacks:
                    callback.on_epoch_begin(self, epoch)
                
                # Training epoch
                train_metrics = self._train_epoch()
                
                # Validation epoch
                val_metrics = None
                if self.val_dataloader is not None:
                    val_metrics = self._validate_epoch()
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        val_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Update LoRA scheduler
                if self.lora_scheduler:
                    self.lora_scheduler.step()
                
                # Log metrics
                combined_metrics = {**train_metrics}
                if val_metrics:
                    combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                self._log_epoch_metrics(epoch, combined_metrics)
                
                # Model checkpointing
                current_metric = val_metrics['loss'] if val_metrics else train_metrics['loss']
                if current_metric < self.state.best_metric:
                    self.state.best_metric = current_metric
                    self.state.best_epoch = epoch
                    self.state.patience_counter = 0
                    self._save_best_model()
                else:
                    self.state.patience_counter += 1
                
                # Save regular checkpoint
                self._save_checkpoint()
                
                # Epoch callbacks
                for callback in self.callbacks:
                    if callback.on_epoch_end(self, epoch, combined_metrics):
                        self.state.should_stop = True
                        break
                
                # Early stopping check
                if self.state.should_stop:
                    self.training_logger.info(f"Training stopped early at epoch {epoch}")
                    break
                
                # Progress logging
                self.training_logger.info(
                    f"Epoch {epoch + 1}/{self.training_config.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}"
                    + (f" - Val Loss: {val_metrics['loss']:.4f}" if val_metrics else "")
                )
        
        except KeyboardInterrupt:
            self.training_logger.info("Training interrupted by user")
        
        except Exception as e:
            self.training_logger.error(f"Training failed with error: {str(e)}")
            raise
        
        finally:
            # Training complete callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
        
        total_time = time.time() - start_time
        
        # Final evaluation
        results = {
            'training_time': total_time,
            'best_epoch': self.state.best_epoch,
            'best_metric': self.state.best_metric,
            'total_steps': self.state.global_step,
            'final_learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Test evaluation if test set available
        if self.test_dataloader is not None:
            self.training_logger.info("Running final test evaluation...")
            test_metrics = self._test_model()
            results['test_metrics'] = test_metrics
        
        self.training_logger.info(f"Training completed in {total_time:.2f} seconds")
        return results
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        task_losses = {}
        
        # Setup progress bar
        if self.rank == 0:  # Only show progress on main process
            pbar = tqdm(
                self.train_dataloader, 
                desc=f"Training Epoch {self.state.epoch + 1}",
                leave=False
            )
        else:
            pbar = self.train_dataloader
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with mixed precision if enabled
            if self.use_amp:
                with autocast():
                    outputs = self.model(**batch, return_dict=True)
                    loss = outputs['loss']
            else:
                outputs = self.model(**batch, return_dict=True)
                loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.training_config.gradient_clipping > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
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
            total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size
            
            # Track task-specific losses
            if 'losses' in outputs:
                for task_name, task_loss in outputs['losses'].items():
                    if task_name not in task_losses:
                        task_losses[task_name] = 0.0
                    task_losses[task_name] += task_loss.item() * batch_size
            
            # Update progress bar
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_task_losses = {
            task: task_loss / total_samples 
            for task, task_loss in task_losses.items()
        }
        
        metrics = {'loss': avg_loss, **avg_task_losses}
        
        # Store learning rate
        self.state.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.state.train_loss_history.append(avg_loss)
        
        return metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        task_losses = {}
        
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(
                    self.val_dataloader, 
                    desc=f"Validation Epoch {self.state.epoch + 1}",
                    leave=False
                )
            else:
                pbar = self.val_dataloader
            
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(**batch, return_dict=True)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch, return_dict=True)
                    loss = outputs['loss']
                
                # Accumulate metrics
                batch_size = self._get_batch_size(batch)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Track task-specific losses
                if 'losses' in outputs:
                    for task_name, task_loss in outputs['losses'].items():
                        if task_name not in task_losses:
                            task_losses[task_name] = 0.0
                        task_losses[task_name] += task_loss.item() * batch_size
        
        # Calculate average losses
        avg_loss = total_loss / total_samples
        avg_task_losses = {
            task: task_loss / total_samples 
            for task, task_loss in task_losses.items()
        }
        
        metrics = {'loss': avg_loss, **avg_task_losses}
        self.state.val_loss_history.append(avg_loss)
        
        return metrics
    
    def _test_model(self) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        
        # Load best model for testing
        if self.best_model_path.exists():
            self._load_best_model()
        
        total_loss = 0.0
        total_samples = 0
        task_losses = {}
        predictions = []
        
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(self.test_dataloader, desc="Testing", leave=False)
            else:
                pbar = self.test_dataloader
            
            for batch in pbar:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch, return_dict=True)
                loss = outputs['loss']
                
                # Store predictions for analysis
                if 'logits' in outputs:
                    predictions.extend(outputs['logits'].cpu().numpy())
                
                # Accumulate metrics
                batch_size = self._get_batch_size(batch)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Track task-specific losses
                if 'losses' in outputs:
                    for task_name, task_loss in outputs['losses'].items():
                        if task_name not in task_losses:
                            task_losses[task_name] = 0.0
                        task_losses[task_name] += task_loss.item() * batch_size
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        avg_task_losses = {
            task: task_loss / total_samples 
            for task, task_loss in task_losses.items()
        }
        
        # Save predictions
        predictions_path = self.output_dir / "test_predictions.pkl"
        with open(predictions_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        return {'loss': avg_loss, **avg_task_losses}
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to training device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Get batch size from batch dictionary."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 1
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch metrics to various loggers."""
        
        # MLflow logging
        if self.logger and self.rank == 0:
            training_metrics = TrainingMetrics(
                train_loss=metrics.get('loss', 0.0),
                val_loss=metrics.get('val_loss'),
                task_metrics={k: v for k, v in metrics.items() if k not in ['loss', 'val_loss']},
                epoch=epoch,
                step=self.state.global_step
            )
            self.logger.log_metrics(training_metrics, step=self.state.global_step)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        if self.rank != 0:  # Only save on main process
            return
        
        # Get model state dict (handle distributed training)
        model_state_dict = (
            self.model.module.state_dict() if self.is_distributed 
            else self.model.state_dict()
        )
        
        checkpoint = {
            'epoch': self.state.epoch,
            'global_step': self.state.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_state': asdict(self.state),
            'config': self.config.to_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        torch.save(checkpoint, self.last_checkpoint_path)
        self.training_logger.info(f"Checkpoint saved: {self.last_checkpoint_path}")
    
    def _save_best_model(self):
        """Save the best model."""
        if self.rank != 0:
            return
        
        model_to_save = self.model.module if self.is_distributed else self.model
        save_jengahub_model(model_to_save, str(self.best_model_path))
        self.training_logger.info(f"Best model saved: {self.best_model_path}")
    
    def _load_best_model(self):
        """Load the best model."""
        if not self.best_model_path.exists():
            self.training_logger.warning("Best model not found, using current model")
            return
        
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if self.is_distributed:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        self.training_logger.info("Best model loaded")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        if self.is_distributed:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        training_state = checkpoint.get('training_state', {})
        self.state.epoch = training_state.get('epoch', 0)
        self.state.global_step = training_state.get('global_step', 0)
        self.state.best_metric = training_state.get('best_metric', float('inf'))
        self.state.best_epoch = training_state.get('best_epoch', 0)
        self.state.patience_counter = training_state.get('patience_counter', 0)
        
        # Restore RNG state
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        
        self.training_logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.training_logger.info(f"Resuming from epoch {self.state.epoch}, step {self.state.global_step}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model and training statistics."""
        model = self.model.module if self.is_distributed else self.model
        
        stats = {
            'model_statistics': model.get_model_statistics(),
            'training_state': asdict(self.state),
            'optimizer_state': {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0),
            },
            'device_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'distributed': self.is_distributed,
                'world_size': self.world_size,
                'rank': self.rank
            }
        }
        
        if torch.cuda.is_available():
            stats['device_info']['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        
        return stats