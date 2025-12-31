"""
Security-specific trainer for tabular data models.
Handles features instead of input_ids/attention_mask.
"""
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import os

from ..core.config import ExperimentConfig
from .callbacks import CallbackHandler, BaseCallback


class SecurityTrainer:
    """
    Trainer specifically designed for security/tabular data models.
    Handles 'features' input instead of 'input_ids'.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: torch.nn.Module,
        train_datasets: Dict[str, Any],
        eval_datasets: Dict[str, Any],
        callbacks: Optional[List[BaseCallback]] = None
    ):
        self.config = config
        self.model = model
        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.training_args = config.training
        
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
        
        # Ensure NestedLearningCallback is present (core feature)
        from .callbacks import NestedLearningCallback
        if not any(isinstance(c, NestedLearningCallback) for c in callbacks):
            callbacks.append(NestedLearningCallback())
            
        self.callback_handler = CallbackHandler(callbacks)
        
        # Create task mapping
        self.task_map = {task.name: idx for idx, task in enumerate(config.tasks)}
        
        # Move model to device
        self.model.to(self.training_args.device)
        
        # Initialize logger
        self._init_logger()
    
    def _init_logger(self):
        """Initialize MLflow logger."""
        import mlflow
        mlflow.set_experiment(self.config.project_name)
        self.mlflow_run = mlflow.start_run()
        print(f"MLflow logger initialized. Experiment: '{self.config.project_name}'")
    
    def _create_dataloaders(self):
        """Create dataloaders for all tasks."""
        train_loaders = {}
        eval_loaders = {}
        
        for task_name in self.train_datasets.keys():
            train_loaders[task_name] = DataLoader(
                self.train_datasets[task_name],
                batch_size=self.training_args.batch_size,
                shuffle=True,
                collate_fn=self._security_collate_fn
            )
            
            eval_loaders[task_name] = DataLoader(
                self.eval_datasets[task_name],
                batch_size=self.training_args.batch_size,
                shuffle=False,
                collate_fn=self._security_collate_fn
            )
        
        return train_loaders, eval_loaders
    
    def _security_collate_fn(self, batch):
        """
        Collate function for security/tabular data.
        Expects 'features' and 'labels_*' keys.
        """
        # Stack features
        features = torch.stack([item['features'] for item in batch])
        
        # Collect labels (handle multi-head)
        labels = {}
        for key in batch[0].keys():
            if key.startswith('labels_'):
                labels[key] = torch.stack([item[key] for item in batch])
        
        return {
            'features': features,
            'labels': labels
        }
    
    def train(self):
        """Main training loop for security models."""
        print("Starting security model training...")
        
        # Create dataloaders
        train_loaders, eval_loaders = self._create_dataloaders()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay
        )
        
        # Training loop
        self.callback_handler.call('on_train_begin', self)
        
        for epoch in range(self.training_args.num_epochs):
            self.callback_handler.call('on_epoch_begin', self, epoch=epoch)
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            # Create iterators for all tasks
            train_iterators = {
                task_name: iter(loader) 
                for task_name, loader in train_loaders.items()
            }
            
            # Calculate total steps
            total_steps = sum(len(loader) for loader in train_loaders.values())
            pbar = tqdm(total=total_steps, desc=f"Training")
            
            # Round-robin training
            while train_iterators:
                for task_name in list(train_iterators.keys()):
                    try:
                        batch = next(train_iterators[task_name])
                    except StopIteration:
                        del train_iterators[task_name]
                        continue
                    
                    optimizer.zero_grad()
                    
                    task_id = self.task_map[task_name]
                    
                    # Move to device
                    features = batch['features'].to(self.training_args.device)
                    labels = batch['labels']
                    labels = {k: v.to(self.training_args.device) for k, v in labels.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        features=features,
                        task_id=task_id,
                        labels=labels
                    )
                    loss = outputs["loss"]
                    
                    # SECRECY SENTINEL: Trigger batch end hook with outputs
                    self.callback_handler.call('on_batch_end', self, task_id=task_id, loss=loss.item(), outputs=outputs)
                    
                    # Backward pass
                    loss.backward()
                    self.callback_handler.call('on_backward_end', self, task_id=task_id)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.update(1)
                    pbar.set_postfix({'loss': f'{loss.item():.2f}'})
            
            pbar.close()
            
            # Evaluation phase
            if self.training_args.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                self.callback_handler.call('on_epoch_end', self, epoch=epoch, metrics=eval_metrics)
                
                # Log to MLflow
                import mlflow
                mlflow.log_metrics(eval_metrics, step=epoch)
                mlflow.log_metric("train_loss", epoch_loss / num_batches, step=epoch)
                
                print(f"Epoch {epoch} Eval Metrics: {eval_metrics}")
            
            # Save checkpoint
            if self.training_args.save_strategy == "epoch":
                self._save_checkpoint(epoch, eval_metrics)
        
        self.callback_handler.call('on_train_end', self)
        print("Training complete.")
    
    def evaluate(self):
        """Evaluate the model on all tasks."""
        self.model.eval()
        _, eval_loaders = self._create_dataloaders()
        
        all_metrics = {}
        
        with torch.no_grad():
            for task_name, eval_loader in eval_loaders.items():
                task_id = self.task_map[task_name]
                total_loss = 0
                num_batches = 0
                
                pbar = tqdm(eval_loader, desc=f"Evaluating {task_name}")
                
                for batch in pbar:
                    features = batch['features'].to(self.training_args.device)
                    labels = batch['labels']
                    labels = {k: v.to(self.training_args.device) for k, v in labels.items()}
                    
                    outputs = self.model(
                        features=features,
                        task_id=task_id,
                        labels=labels
                    )
                    
                    total_loss += outputs["loss"].item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                all_metrics[f"eval_{task_name}_loss"] = avg_loss
        
        return all_metrics
    
    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        output_dir = os.path.join(self.training_args.output_dir, f"checkpoint-{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save best model if this is the best so far
        metric_name = self.training_args.metric_for_best_model
        if metric_name in metrics:
            if not hasattr(self, 'best_metric') or metrics[metric_name] < self.best_metric:
                self.best_metric = metrics[metric_name]
                best_dir = os.path.join(self.training_args.output_dir, "best_model")
                os.makedirs(best_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                print(f"✅ New best model saved! {metric_name}={self.best_metric:.4f}")
    
    def close(self):
        """Close the MLflow logger."""
        import mlflow
        mlflow.end_run()
        print("mlflow logger closed.")
