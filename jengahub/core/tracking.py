"""
Unified MLflow Tracking System for JengaHub

This module provides comprehensive experiment tracking and model management
for multimodal AI training, combining metrics from both audio and text
processing with African language-specific evaluations.
"""

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from contextlib import contextmanager
import tempfile

from .config import MultiModalConfig
from .memory import ContinuumMemorySystem
from .nested_lora import NestedLoRALinear
from .code_switching import MultimodalCodeSwitchingBridge


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    # Basic metrics
    train_loss: float
    val_loss: Optional[float] = None
    
    # Task-specific metrics
    task_metrics: Dict[str, float] = None
    
    # Language-specific metrics
    language_metrics: Dict[str, float] = None
    
    # Code-switching metrics
    switch_detection_f1: Optional[float] = None
    switch_type_accuracy: Optional[float] = None
    
    # Memory system metrics
    memory_utilization: Optional[float] = None
    attention_entropy: Optional[float] = None
    
    # LoRA metrics
    lora_efficiency: Optional[float] = None
    active_lora_levels: Optional[List[int]] = None
    
    # System metrics
    epoch: int = 0
    step: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.task_metrics is None:
            self.task_metrics = {}
        if self.language_metrics is None:
            self.language_metrics = {}


class JengaHubMLflowLogger:
    """
    Comprehensive MLflow logger for JengaHub multimodal training.
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: MultiModalConfig,
        tracking_uri: Optional[str] = None,
        model_registry_uri: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.config = config
        
        # Set MLflow URIs
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if model_registry_uri:
            mlflow.set_registry_uri(model_registry_uri)
        
        # Create or set experiment
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.client = MlflowClient()
        
        # Active run tracking
        self.active_run = None
        self.run_id = None
        
        # Metrics history
        self.metrics_history = []
        self.best_metrics = {}
        
        # Model artifacts tracking
        self.model_artifacts = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start MLflow run with automatic cleanup."""
        run_name = run_name or f"jengahub_run_{int(time.time())}"
        
        # Default tags
        default_tags = {
            "framework": "JengaHub",
            "version": self.config.version,
            "project": self.config.project_name,
            "multimodal": "true",
            "african_languages": "true"
        }
        
        if tags:
            default_tags.update(tags)
        
        try:
            self.active_run = mlflow.start_run(
                run_name=run_name,
                nested=nested,
                tags=default_tags
            )
            self.run_id = self.active_run.info.run_id
            
            # Log configuration
            self.log_config()
            
            yield self
            
        finally:
            if self.active_run:
                mlflow.end_run()
                self.active_run = None
                self.run_id = None
    
    def log_config(self):
        """Log complete configuration to MLflow."""
        config_dict = self.config.to_dict()
        
        # Flatten nested configuration for MLflow params
        flat_config = self._flatten_dict(config_dict)
        
        # Log parameters (MLflow has limits, so chunk if needed)
        param_chunks = self._chunk_dict(flat_config, max_size=100)
        
        for i, chunk in enumerate(param_chunks):
            mlflow.log_params(chunk)
        
        # Save full config as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            self.config.to_yaml(f.name)
            mlflow.log_artifact(f.name, "config")
    
    def log_metrics(
        self,
        metrics: Union[TrainingMetrics, Dict[str, Any]],
        step: Optional[int] = None
    ):
        """Log comprehensive training metrics."""
        if isinstance(metrics, TrainingMetrics):
            metrics_dict = asdict(metrics)
        else:
            metrics_dict = metrics
        
        # Extract step if not provided
        if step is None:
            step = metrics_dict.get('step', len(self.metrics_history))
        
        # Flatten metrics for MLflow
        flat_metrics = self._flatten_dict(metrics_dict)
        
        # Remove non-numeric values
        numeric_metrics = {
            k: v for k, v in flat_metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        
        # Log to MLflow
        for name, value in numeric_metrics.items():
            mlflow.log_metric(name, value, step=step)
        
        # Store in history
        self.metrics_history.append({
            'step': step,
            'metrics': metrics_dict,
            'timestamp': time.time()
        })
        
        # Update best metrics
        self._update_best_metrics(numeric_metrics)
        
        self.logger.info(f"Logged {len(numeric_metrics)} metrics at step {step}")
    
    def log_model_artifacts(
        self,
        model: nn.Module,
        model_name: str = "jengahub_model",
        save_state_dict: bool = True,
        save_full_model: bool = False
    ):
        """Log model artifacts with comprehensive metadata."""
        
        # Create model info
        model_info = {
            "model_name": model_name,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        # Add LoRA-specific info
        lora_info = self._extract_lora_info(model)
        if lora_info:
            model_info.update(lora_info)
        
        # Add memory system info
        memory_info = self._extract_memory_info(model)
        if memory_info:
            model_info.update(memory_info)
        
        # Save state dict
        if save_state_dict:
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                mlflow.log_artifact(f.name, f"models/{model_name}")
        
        # Save full model
        if save_full_model:
            mlflow.pytorch.log_model(
                model, 
                f"models/{model_name}_full",
                extra_files={
                    "model_info.json": json.dumps(model_info, indent=2),
                    "config.yaml": self.config.to_dict()
                }
            )
        
        # Log model info as metrics
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"model_{key}", value)
        
        # Save model info as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info, f, indent=2)
            mlflow.log_artifact(f.name, f"models/{model_name}")
        
        self.model_artifacts[model_name] = model_info
    
    def log_language_analysis(
        self,
        analysis: Dict[str, Any],
        step: Optional[int] = None
    ):
        """Log language-specific analysis results."""
        
        # Log language distribution
        if 'language_distribution' in analysis:
            lang_dist = analysis['language_distribution']
            for lang, prob in lang_dist.items():
                mlflow.log_metric(f"language_dist_{lang}", prob, step=step)
        
        # Log code-switching metrics
        if 'code_switching' in analysis:
            cs_metrics = analysis['code_switching']
            for metric, value in cs_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"code_switching_{metric}", value, step=step)
        
        # Save full analysis as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(analysis, f, indent=2)
            mlflow.log_artifact(f.name, "language_analysis")
    
    def log_performance_metrics(
        self,
        performance_data: Dict[str, Any],
        step: Optional[int] = None
    ):
        """Log system performance metrics."""
        
        perf_metrics = {}
        
        # Memory usage
        if 'memory_usage' in performance_data:
            mem_usage = performance_data['memory_usage']
            if isinstance(mem_usage, dict):
                for key, value in mem_usage.items():
                    if isinstance(value, (int, float)):
                        perf_metrics[f"memory_{key}"] = value
            else:
                perf_metrics['memory_usage'] = mem_usage
        
        # Training speed
        if 'training_speed' in performance_data:
            perf_metrics['training_speed'] = performance_data['training_speed']
        
        # GPU utilization
        if 'gpu_utilization' in performance_data:
            perf_metrics['gpu_utilization'] = performance_data['gpu_utilization']
        
        # Log metrics
        for metric, value in perf_metrics.items():
            mlflow.log_metric(metric, value, step=step)
    
    def log_dataset_info(
        self,
        dataset_stats: Dict[str, Any]
    ):
        """Log dataset statistics and information."""
        
        # Log basic stats
        basic_stats = [
            'total_samples', 'audio_samples', 'text_samples', 
            'multimodal_samples', 'code_switching_samples'
        ]
        
        for stat in basic_stats:
            if stat in dataset_stats:
                mlflow.log_param(f"dataset_{stat}", dataset_stats[stat])
        
        # Log language distribution
        if 'language_distribution' in dataset_stats:
            lang_dist = dataset_stats['language_distribution']
            for lang, count in lang_dist.items():
                mlflow.log_param(f"dataset_lang_{lang}", count)
        
        # Save full stats as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset_stats, f, indent=2)
            mlflow.log_artifact(f.name, "dataset")
    
    def register_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        stage: str = "None",
        description: str = None
    ) -> str:
        """Register model in MLflow Model Registry."""
        
        if not self.run_id:
            raise ValueError("No active run. Start a run first.")
        
        # Create model URI
        model_uri = f"runs:/{self.run_id}/models/{model_name}_full"
        
        # Register model
        model_version_obj = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "framework": "JengaHub",
                "version": self.config.version,
                "multimodal": "true",
                "african_languages": "true"
            }
        )
        
        registered_version = model_version_obj.version
        
        # Update model description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=registered_version,
                description=description
            )
        
        # Transition to specified stage
        if stage != "None":
            self.client.transition_model_version_stage(
                name=model_name,
                version=registered_version,
                stage=stage
            )
        
        self.logger.info(f"Registered model {model_name} version {registered_version}")
        return registered_version
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = False
    ) -> Optional[mlflow.entities.Run]:
        """Get best run based on a specific metric."""
        
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
        )
        
        return runs[0] if runs else None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs across specified metrics."""
        
        comparison = {}
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'name': run.info.run_name,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'metrics': {
                    metric: run.data.metrics.get(metric, None)
                    for metric in metrics
                }
            }
        
        return comparison
    
    def _extract_lora_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract LoRA-specific information from model."""
        lora_info = {}
        
        lora_layers = [m for m in model.modules() if isinstance(m, NestedLoRALinear)]
        
        if lora_layers:
            total_lora_params = sum(
                layer.get_level_statistics()['total_parameters']
                for layer in lora_layers
            )
            
            lora_info.update({
                "lora_layers_count": len(lora_layers),
                "lora_parameters": total_lora_params,
                "lora_efficiency": total_lora_params / sum(p.numel() for p in model.parameters())
            })
            
            # Get level statistics from first layer (representative)
            if lora_layers:
                first_layer_stats = lora_layers[0].get_level_statistics()
                lora_info.update({
                    f"lora_{k}": v for k, v in first_layer_stats.items()
                    if isinstance(v, (int, float, list))
                })
        
        return lora_info
    
    def _extract_memory_info(self, model: nn.Module) -> Dict[str, Any]:
        """Extract memory system information from model."""
        memory_info = {}
        
        memory_systems = [m for m in model.modules() if isinstance(m, ContinuumMemorySystem)]
        
        if memory_systems:
            memory_system = memory_systems[0]  # Take first one
            stats = memory_system.get_memory_statistics()
            
            memory_info.update({
                f"memory_{k}": v for k, v in stats.items()
                if isinstance(v, (int, float, list))
            })
        
        return memory_info
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) < 10:  # Avoid huge lists
                for i, item in enumerate(v):
                    if isinstance(item, (int, float, str, bool)):
                        items.append((f"{new_key}_{i}", item))
            elif isinstance(v, (str, int, float, bool)) and v is not None:
                items.append((new_key, v))
        
        return dict(items)
    
    def _chunk_dict(self, d: Dict[str, Any], max_size: int = 100) -> List[Dict[str, Any]]:
        """Chunk dictionary to avoid MLflow parameter limits."""
        items = list(d.items())
        chunks = []
        
        for i in range(0, len(items), max_size):
            chunk = dict(items[i:i + max_size])
            chunks.append(chunk)
        
        return chunks
    
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics tracking."""
        for name, value in metrics.items():
            if 'loss' in name.lower() or 'error' in name.lower():
                # Lower is better
                if name not in self.best_metrics or value < self.best_metrics[name]:
                    self.best_metrics[name] = value
            else:
                # Higher is better
                if name not in self.best_metrics or value > self.best_metrics[name]:
                    self.best_metrics[name] = value
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {
            'total_steps': len(self.metrics_history),
            'best_metrics': self.best_metrics,
            'latest_metrics': self.metrics_history[-1]['metrics'] if self.metrics_history else {},
            'training_duration': (
                self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp']
                if len(self.metrics_history) > 1 else 0
            )
        }
        
        return summary


class MultiExperimentTracker:
    """
    Track and compare multiple experiments for hyperparameter optimization
    and model selection.
    """
    
    def __init__(
        self,
        base_experiment_name: str = "JengaHub_Experiments",
        tracking_uri: Optional[str] = None
    ):
        self.base_experiment_name = base_experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.experiments = {}
    
    def create_experiment_series(
        self,
        series_name: str,
        base_config: MultiModalConfig,
        variations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Create a series of experiments with configuration variations.
        
        Args:
            series_name: Name for the experiment series
            base_config: Base configuration
            variations: List of configuration modifications
            
        Returns:
            List of experiment IDs
        """
        experiment_ids = []
        
        for i, variation in enumerate(variations):
            # Modify base config
            modified_config = base_config.__class__.from_dict({
                **base_config.to_dict(),
                **variation
            })
            
            # Create experiment
            exp_name = f"{self.base_experiment_name}_{series_name}_{i:03d}"
            experiment = mlflow.create_experiment(
                exp_name,
                tags={
                    "series": series_name,
                    "variation_index": str(i),
                    "base_experiment": self.base_experiment_name
                }
            )
            
            experiment_ids.append(experiment)
            self.experiments[exp_name] = {
                'id': experiment,
                'config': modified_config,
                'variation': variation
            }
        
        return experiment_ids
    
    def get_series_results(
        self,
        series_name: str,
        metric: str = "val_loss"
    ) -> Dict[str, Any]:
        """Get results summary for an experiment series."""
        
        series_experiments = [
            exp for name, exp in self.experiments.items()
            if series_name in name
        ]
        
        results = []
        
        for exp in series_experiments:
            runs = self.client.search_runs([exp['id']])
            
            if runs:
                best_run = min(runs, key=lambda r: r.data.metrics.get(metric, float('inf')))
                
                results.append({
                    'experiment_id': exp['id'],
                    'config_variation': exp['variation'],
                    'best_metric': best_run.data.metrics.get(metric),
                    'run_id': best_run.info.run_id
                })
        
        # Sort by metric performance
        results.sort(key=lambda x: x['best_metric'] or float('inf'))
        
        return {
            'series_name': series_name,
            'total_experiments': len(results),
            'best_result': results[0] if results else None,
            'all_results': results
        }


# Utility functions for experiment management
def setup_jengahub_tracking(
    config: MultiModalConfig,
    experiment_name: Optional[str] = None
) -> JengaHubMLflowLogger:
    """Setup JengaHub tracking with default configuration."""
    
    exp_name = experiment_name or config.training.mlflow_experiment_name
    
    return JengaHubMLflowLogger(
        experiment_name=exp_name,
        config=config
    )


def create_performance_dashboard_data(
    logger: JengaHubMLflowLogger,
    include_model_comparison: bool = True
) -> Dict[str, Any]:
    """Create data for performance dashboard visualization."""
    
    dashboard_data = {
        'experiment_summary': logger.get_metrics_summary(),
        'training_history': logger.metrics_history,
        'best_metrics': logger.best_metrics,
        'model_artifacts': logger.model_artifacts
    }
    
    if include_model_comparison and len(logger.metrics_history) > 1:
        # Add trend analysis
        recent_metrics = logger.metrics_history[-10:]  # Last 10 steps
        dashboard_data['recent_trends'] = {
            metric: [step['metrics'].get(metric) for step in recent_metrics]
            for metric in ['train_loss', 'val_loss', 'memory_utilization']
            if any(step['metrics'].get(metric) is not None for step in recent_metrics)
        }
    
    return dashboard_data