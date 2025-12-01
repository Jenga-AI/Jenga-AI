"""
MLflow Utilities for Jenga-AI Project

This module provides utility functions for MLflow experiment tracking,
model logging, and management.
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path


def load_mlflow_config(config_path: str = "mlflow_config.yaml") -> Dict[str, Any]:
    """
    Load MLflow configuration from YAML file.
    
    Args:
        config_path: Path to MLflow configuration file
        
    Returns:
        Dictionary containing MLflow configuration
    """
    if not os.path.exists(config_path):
        # Use default configuration if file doesn't exist
        return {
            "tracking": {
                "uri": "./mlruns",
                "default_experiment": "jenga-ai-experiments",
                "artifact_location": "./mlruns"
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_mlflow(
    config_path: str = "mlflow_config.yaml",
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> str:
    """
    Initialize MLflow with configuration.
    
    Args:
        config_path: Path to MLflow configuration file
        tracking_uri: Override tracking URI from config
        experiment_name: Override experiment name from config
        
    Returns:
        Experiment ID
    """
    # Load configuration
    config = load_mlflow_config(config_path)
    
    # Set tracking URI
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or config["tracking"]["uri"]
    mlflow.set_tracking_uri(uri)
    
    # Get or create experiment
    exp_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or config["tracking"]["default_experiment"]
    experiment_id = get_or_create_experiment(exp_name, config["tracking"].get("artifact_location"))
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name=exp_name)
    
    # Set up auto-logging if enabled
    if config.get("logging", {}).get("auto_log", {}).get("pytorch", False):
        try:
            mlflow.pytorch.autolog()
        except Exception as e:
            print(f"Warning: Could not enable PyTorch autolog: {e}")
    
    if config.get("logging", {}).get("auto_log", {}).get("transformers", False):
        try:
            mlflow.transformers.autolog()
        except Exception as e:
            print(f"Warning: Could not enable Transformers autolog: {e}")
    
    print(f"✓ MLflow initialized")
    print(f"  Tracking URI: {uri}")
    print(f"  Experiment: {exp_name} (ID: {experiment_id})")
    
    return experiment_id


def get_or_create_experiment(
    experiment_name: str,
    artifact_location: Optional[str] = None
) -> str:
    """
    Get existing experiment or create new one.
    
    Args:
        experiment_name: Name of the experiment
        artifact_location: Location to store artifacts
        
    Returns:
        Experiment ID
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is not None:
        return experiment.experiment_id
    
    return mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_location
    )


def log_experiment_config(config: Dict[str, Any], prefix: str = "") -> None:
    """
    Log experiment configuration parameters to MLflow.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names
    """
    def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_config = flatten_dict(config, prefix)
    mlflow.log_params(flat_config)


def log_metrics_dict(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Training step/epoch number
        prefix: Prefix for metric names
    """
    metrics_with_prefix = {
        f"{prefix}{k}" if prefix else k: v 
        for k, v in metrics.items()
    }
    mlflow.log_metrics(metrics_with_prefix, step=step)


def log_model_with_signature(
    model: Any,
    artifact_path: str,
    signature: Optional[Any] = None,
    input_example: Optional[Any] = None,
    registered_model_name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log model with input/output signature.
    
    Args:
        model: Model to log
        artifact_path: Path within the run's artifact directory
        signature: MLflow model signature
        input_example: Example input for the model
        registered_model_name: Name for model registry
        **kwargs: Additional arguments for mlflow.pytorch.log_model
    """
    mlflow.pytorch.log_model(
        model,
        artifact_path,
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        **kwargs
    )


def get_best_run(
    experiment_name: str,
    metric: str = "val_loss",
    ascending: bool = True
) -> Optional[mlflow.entities.Run]:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize
        ascending: True if lower is better, False if higher is better
        
    Returns:
        Best run object or None if no runs found
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    return runs[0] if runs else None


def load_model_from_run(run_id: str, artifact_path: str = "model") -> Any:
    """
    Load a model from a specific run.
    
    Args:
        run_id: MLflow run ID
        artifact_path: Path to model artifact within the run
        
    Returns:
        Loaded model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.pytorch.load_model(model_uri)


def log_artifact_local(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file as an artifact.
    
    Args:
        local_path: Path to local file
        artifact_path: Destination path within artifact directory
    """
    mlflow.log_artifact(local_path, artifact_path)


def set_tags(tags: Dict[str, str]) -> None:
    """
    Set tags for the current run.
    
    Args:
        tags: Dictionary of tag names and values
    """
    mlflow.set_tags(tags)


def log_dataset_info(
    dataset_name: str,
    train_size: int,
    val_size: int,
    test_size: Optional[int] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log dataset information as parameters.
    
    Args:
        dataset_name: Name of the dataset
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples (optional)
        additional_info: Additional dataset information
    """
    params = {
        "dataset_name": dataset_name,
        "train_size": train_size,
        "val_size": val_size,
    }
    
    if test_size is not None:
        params["test_size"] = test_size
    
    if additional_info:
        params.update(additional_info)
    
    mlflow.log_params(params)


def create_nested_run(run_name: str, parent_run_id: Optional[str] = None):
    """
    Create a nested run (useful for hyperparameter tuning).
    
    Args:
        run_name: Name of the nested run
        parent_run_id: ID of the parent run
        
    Returns:
        Context manager for the nested run
    """
    return mlflow.start_run(run_name=run_name, nested=True)


def end_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.
    
    Args:
        status: Status of the run (FINISHED, FAILED, KILLED)
    """
    mlflow.end_run(status=status)
