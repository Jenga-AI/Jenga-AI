
import os
import json
import logging
import mlflow
from datetime import datetime
from transformers import TrainerCallback
from seq2seq_models.utils.mlflow_manager import MLflowManager

logger = logging.getLogger(__name__)

class TranslationMLflowCallback(TrainerCallback):
    """MLflow callback for automatic metric logging during training"""
    
    def __init__(self, mlflow_manager: MLflowManager):
        self.mlflow_manager = mlflow_manager
        self.training_started = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log training start"""
        if not self.mlflow_manager.enabled:
            return
        
        try:
            if not self.training_started and mlflow.active_run():
                mlflow.log_metric("training_status_code", 1.0)
                mlflow.log_param("total_epochs", int(args.num_train_epochs))
                logger.info("✅ MLflow: Training started - status_code=1.0 (TRAINING)")
                self.training_started = True
        except Exception as e:
            logger.warning(f"⚠️ Error in on_train_begin: {e}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics"""
        if not self.mlflow_manager.enabled or logs is None:
            return
        
        try:
            numeric_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if not (isinstance(value, float) and (value != value)):
                        numeric_logs[key] = float(value)
            
            if numeric_logs:
                self.mlflow_manager.log_metrics(numeric_logs, step=state.global_step)
        except Exception as e:
            logger.warning(f"⚠️ Error in on_log: {e}")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log epoch progress"""
        if not self.mlflow_manager.enabled or state.epoch is None:
            return
        
        try:
            if state.global_step % 100 == 0 and mlflow.active_run():
                epoch_progress = (state.epoch / args.num_train_epochs) * 100
                self.mlflow_manager.log_metrics({
                    'epoch_progress_percent': epoch_progress,
                    'current_epoch': state.epoch
                }, step=state.global_step)
        except Exception as e:
            logger.warning(f"⚠️ Error in on_step_end: {e}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics"""
        if not self.mlflow_manager.enabled or metrics is None:
            return
        
        try:
            valid_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if not (isinstance(value, float) and (value != value)):
                        valid_metrics[key] = float(value)
            
            if valid_metrics and mlflow.active_run():
                self.mlflow_manager.log_metrics(valid_metrics, step=state.global_step)
        except Exception as e:
            logger.warning(f"⚠️ Error in on_evaluate: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Log training end"""
        if not self.mlflow_manager.enabled:
            return
        
        try:
            if mlflow.active_run():
                mlflow.log_metric("training_status_code", 2.0)
                if state.epoch is not None:
                    mlflow.log_metric("final_epoch", state.epoch)
                mlflow.log_metric("final_step", state.global_step)
                logger.info("✅ MLflow: Training finished - status_code=2.0 (FINISHED)")
        except Exception as e:
            logger.warning(f"⚠️ Error in on_train_end: {e}")


class EarlyFailureCallback(TrainerCallback):
    """Stops training early if BLEU stays at 0.0"""
    
    def __init__(self, patience=3):
        self.patience = patience
        self.eval_count = 0
        self.bleu_scores = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            bleu = metrics.get('eval_bleu', 0)
            self.bleu_scores.append(bleu)
            self.eval_count += 1
            
            if self.eval_count >= self.patience:
                recent_scores = self.bleu_scores[-self.patience:]
                if all(score < 0.01 for score in recent_scores):
                    logger.error("\n" + "="*70)
                    logger.error("❌ TRAINING FAILURE DETECTED!")
                    logger.error("="*70)
                    logger.error(f"BLEU has been at ~0.0 for {self.patience} consecutive evaluations.")
                    logger.error("Stopping training to save time.")
                    logger.error("="*70 + "\n")
                    
                    control.should_training_stop = True
        
        return control


class HuggingFaceHubCallback(TrainerCallback):
    """Callback to push model and metrics to HuggingFace Hub after each checkpoint"""

    def __init__(self, repo_id: str, token: str, output_dir: str, config: dict = None):
        self.repo_id = repo_id
        self.token = token
        self.output_dir = output_dir
        self.config = config or {}
        self.push_count = 0

    def on_save(self, args, state, control, **kwargs):
        """Push checkpoint and metrics to hub after each checkpoint save"""
        try:
            from huggingface_hub import HfApi

            self.push_count += 1
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

            logger.info(f"\n📤 Pushing checkpoint {state.global_step} to HuggingFace Hub...")
            
            api = HfApi(token=self.token)

            # Save current training metrics snapshot
            metrics_snapshot = {
                'checkpoint': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
                'training_history': state.log_history if hasattr(state, 'log_history') else [],
                'best_metric': state.best_metric,
                'total_steps': state.max_steps,
            }

            metrics_path = os.path.join(checkpoint_dir, 'training_metrics_snapshot.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_snapshot, f, indent=2, default=str)

            # Push the checkpoint with metrics
            api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Checkpoint {state.global_step} - Epoch {state.epoch:.2f}",
                path_in_repo=f"checkpoints/checkpoint-{state.global_step}"
            )

            # Also push a latest metrics file to root for easy access
            api.upload_file(
                path_or_fileobj=metrics_path,
                path_in_repo="latest_training_metrics.json",
                repo_id=self.repo_id,
                repo_type="model",
                commit_message=f"Update metrics at checkpoint {state.global_step}"
            )

            logger.info(f"✅ Checkpoint {state.global_step} pushed successfully!")

        except Exception as e:
            logger.warning(f"⚠️ Failed to push checkpoint to hub: {e}")
            # Don't fail training if push fails

        return control
