"""
Model Quantization for JengaHub

This module provides comprehensive quantization techniques including dynamic quantization,
static quantization, and quantization-aware training (QAT) for efficient model deployment.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import default_qconfig, QConfig
from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver
from torch.quantization.fake_quantize import default_fake_quant, default_weight_fake_quant
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
import time
import copy
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..core.model import JengaHubMultiModalModel
from ..core.config import MultiModalConfig
from ..training.trainer import JengaHubTrainer


class DynamicQuantizer:
    """Dynamic quantization for JengaHub models."""
    
    def __init__(self, dtype: torch.dtype = torch.qint8):
        """
        Initialize dynamic quantizer.
        
        Args:
            dtype: Quantization data type (qint8, qfloat16)
        """
        self.dtype = dtype
        self.logger = logging.getLogger(__name__)
    
    def quantize(
        self,
        model: JengaHubMultiModalModel,
        qconfig_spec: Optional[Dict] = None
    ) -> torch.nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: Model to quantize
            qconfig_spec: Custom quantization configuration
            
        Returns:
            Dynamically quantized model
        """
        self.logger.info("Applying dynamic quantization...")
        
        # Prepare model for quantization
        model.eval()
        model_copy = copy.deepcopy(model)
        
        # Default quantization spec for linear and LSTM layers
        if qconfig_spec is None:
            qconfig_spec = {
                torch.nn.Linear: default_qconfig,
                torch.nn.LSTM: default_qconfig,
                torch.nn.MultiheadAttention: default_qconfig
            }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            qconfig_spec,
            dtype=self.dtype
        )
        
        self.logger.info(f"Dynamic quantization completed with dtype: {self.dtype}")
        return quantized_model
    
    def quantize_modules(
        self,
        model: JengaHubMultiModalModel,
        module_types: List[type] = None
    ) -> torch.nn.Module:
        """
        Quantize specific module types.
        
        Args:
            model: Model to quantize
            module_types: List of module types to quantize
            
        Returns:
            Quantized model
        """
        if module_types is None:
            module_types = [torch.nn.Linear, torch.nn.LSTM]
        
        qconfig_spec = {mod_type: default_qconfig for mod_type in module_types}
        
        return self.quantize(model, qconfig_spec)


class StaticQuantizer:
    """Static quantization with calibration for JengaHub models."""
    
    def __init__(
        self,
        backend: str = "fbgemm",
        observer: str = "minmax"
    ):
        """
        Initialize static quantizer.
        
        Args:
            backend: Quantization backend (fbgemm, qnnpack)
            observer: Observer type (minmax, histogram)
        """
        self.backend = backend
        self.observer = observer
        self.logger = logging.getLogger(__name__)
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
    
    def prepare_model(
        self,
        model: JengaHubMultiModalModel,
        example_inputs: Dict[str, torch.Tensor]
    ) -> torch.nn.Module:
        """
        Prepare model for static quantization.
        
        Args:
            model: Model to prepare
            example_inputs: Example inputs for calibration
            
        Returns:
            Prepared model with observers
        """
        self.logger.info("Preparing model for static quantization...")
        
        model.eval()
        model_prepared = copy.deepcopy(model)
        
        # Configure quantization
        qconfig = self._create_qconfig()
        model_prepared.qconfig = qconfig
        
        # Prepare model
        model_prepared = torch.quantization.prepare(model_prepared)
        
        self.logger.info("Model prepared for calibration")
        return model_prepared
    
    def calibrate(
        self,
        model: torch.nn.Module,
        calibration_dataloader: torch.utils.data.DataLoader,
        num_batches: int = 100
    ) -> torch.nn.Module:
        """
        Calibrate model with representative data.
        
        Args:
            model: Prepared model with observers
            calibration_dataloader: Calibration data
            num_batches: Number of batches for calibration
            
        Returns:
            Calibrated model
        """
        self.logger.info(f"Calibrating model with {num_batches} batches...")
        
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_dataloader, desc="Calibrating")):
                if i >= num_batches:
                    break
                
                # Move batch to CPU for quantization
                batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass for observer collection
                model(**batch)
        
        self.logger.info("Calibration completed")
        return model
    
    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Convert calibrated model to quantized model.
        
        Args:
            model: Calibrated model
            
        Returns:
            Quantized model
        """
        self.logger.info("Converting to quantized model...")
        
        quantized_model = torch.quantization.convert(model)
        
        self.logger.info("Static quantization completed")
        return quantized_model
    
    def quantize(
        self,
        model: JengaHubMultiModalModel,
        calibration_dataloader: torch.utils.data.DataLoader,
        example_inputs: Dict[str, torch.Tensor],
        num_calibration_batches: int = 100
    ) -> torch.nn.Module:
        """
        Full static quantization pipeline.
        
        Args:
            model: Model to quantize
            calibration_dataloader: Calibration data
            example_inputs: Example inputs
            num_calibration_batches: Number of calibration batches
            
        Returns:
            Statically quantized model
        """
        # Prepare model
        prepared_model = self.prepare_model(model, example_inputs)
        
        # Calibrate
        calibrated_model = self.calibrate(
            prepared_model, 
            calibration_dataloader, 
            num_calibration_batches
        )
        
        # Convert to quantized
        quantized_model = self.convert(calibrated_model)
        
        return quantized_model
    
    def _create_qconfig(self) -> QConfig:
        """Create quantization configuration."""
        
        if self.observer == "minmax":
            activation_observer = MovingAverageMinMaxObserver
            weight_observer = MovingAverageMinMaxObserver
        elif self.observer == "histogram":
            activation_observer = HistogramObserver
            weight_observer = HistogramObserver
        else:
            raise ValueError(f"Unsupported observer: {self.observer}")
        
        qconfig = QConfig(
            activation=activation_observer.with_args(dtype=torch.quint8),
            weight=weight_observer.with_args(dtype=torch.qint8)
        )
        
        return qconfig


class QATTrainer:
    """Quantization-Aware Training for JengaHub models."""
    
    def __init__(
        self,
        model: JengaHubMultiModalModel,
        config: MultiModalConfig,
        backend: str = "fbgemm"
    ):
        """
        Initialize QAT trainer.
        
        Args:
            model: Model to train with QAT
            config: Training configuration
            backend: Quantization backend
        """
        self.model = model
        self.config = config
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        
        # Prepare model for QAT
        self.model_qat = self._prepare_qat_model()
    
    def _prepare_qat_model(self) -> torch.nn.Module:
        """Prepare model for quantization-aware training."""
        self.logger.info("Preparing model for QAT...")
        
        # Create QAT model
        model_qat = copy.deepcopy(self.model)
        model_qat.train()
        
        # Set QAT config
        qat_qconfig = torch.quantization.get_default_qat_qconfig(self.backend)
        model_qat.qconfig = qat_qconfig
        
        # Prepare for QAT
        model_qat = torch.quantization.prepare_qat(model_qat)
        
        self.logger.info("Model prepared for QAT")
        return model_qat
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 5,
        freeze_bn_epochs: int = 2
    ) -> torch.nn.Module:
        """
        Train model with quantization-aware training.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of training epochs
            freeze_bn_epochs: Epochs to freeze batch norm
            
        Returns:
            Trained QAT model
        """
        self.logger.info(f"Starting QAT for {num_epochs} epochs...")
        
        # Create trainer for QAT
        trainer = JengaHubTrainer(
            model=self.model_qat,
            config=self.config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir="./qat_output"
        )
        
        # Train for initial epochs with normal BN
        if freeze_bn_epochs < num_epochs:
            self.logger.info(f"Training with normal BN for {freeze_bn_epochs} epochs")
            trainer.training_config.num_epochs = freeze_bn_epochs
            trainer.train()
        
        # Freeze batch norm and continue training
        if freeze_bn_epochs < num_epochs:
            self.logger.info("Freezing BatchNorm and continuing training...")
            self._freeze_bn(self.model_qat)
            
            trainer.state.epoch = freeze_bn_epochs
            trainer.training_config.num_epochs = num_epochs
            trainer.train()
        
        self.logger.info("QAT training completed")
        return self.model_qat
    
    def convert_to_quantized(self, qat_model: torch.nn.Module) -> torch.nn.Module:
        """
        Convert QAT model to quantized model for inference.
        
        Args:
            qat_model: Trained QAT model
            
        Returns:
            Quantized inference model
        """
        self.logger.info("Converting QAT model to quantized model...")
        
        qat_model.eval()
        quantized_model = torch.quantization.convert(qat_model)
        
        self.logger.info("QAT model conversion completed")
        return quantized_model
    
    def _freeze_bn(self, model: torch.nn.Module):
        """Freeze batch normalization layers."""
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def quantize_model(
    model: JengaHubMultiModalModel,
    method: str = "dynamic",
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    **kwargs
) -> torch.nn.Module:
    """
    Convenience function for model quantization.
    
    Args:
        model: Model to quantize
        method: Quantization method (dynamic, static, qat)
        calibration_data: Calibration data for static quantization
        **kwargs: Additional arguments for specific quantizers
        
    Returns:
        Quantized model
    """
    if method == "dynamic":
        quantizer = DynamicQuantizer(**kwargs)
        return quantizer.quantize(model)
    
    elif method == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        
        quantizer = StaticQuantizer(**kwargs)
        example_inputs = next(iter(calibration_data))
        return quantizer.quantize(model, calibration_data, example_inputs)
    
    elif method == "qat":
        raise NotImplementedError("Use QATTrainer class for quantization-aware training")
    
    else:
        raise ValueError(f"Unsupported quantization method: {method}")


def calibrate_model(
    model: JengaHubMultiModalModel,
    calibration_dataloader: torch.utils.data.DataLoader,
    num_batches: int = 100,
    backend: str = "fbgemm"
) -> torch.nn.Module:
    """
    Calibrate model for static quantization.
    
    Args:
        model: Model to calibrate
        calibration_dataloader: Calibration data
        num_batches: Number of calibration batches
        backend: Quantization backend
        
    Returns:
        Calibrated model ready for conversion
    """
    quantizer = StaticQuantizer(backend=backend)
    example_inputs = next(iter(calibration_dataloader))
    
    prepared_model = quantizer.prepare_model(model, example_inputs)
    calibrated_model = quantizer.calibrate(prepared_model, calibration_dataloader, num_batches)
    
    return calibrated_model


def compare_quantized_models(
    original_model: JengaHubMultiModalModel,
    quantized_models: Dict[str, torch.nn.Module],
    test_dataloader: torch.utils.data.DataLoader,
    num_batches: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Compare original and quantized models on accuracy and performance.
    
    Args:
        original_model: Original full-precision model
        quantized_models: Dictionary of quantized models
        test_dataloader: Test data
        num_batches: Number of test batches
        
    Returns:
        Comparison results
    """
    logger = logging.getLogger(__name__)
    results = {}
    
    # Evaluate original model
    logger.info("Evaluating original model...")
    original_metrics = _evaluate_model(original_model, test_dataloader, num_batches)
    results['original'] = original_metrics
    
    # Evaluate quantized models
    for name, q_model in quantized_models.items():
        logger.info(f"Evaluating {name} model...")
        q_metrics = _evaluate_model(q_model, test_dataloader, num_batches)
        results[name] = q_metrics
    
    # Calculate relative metrics
    for name, metrics in results.items():
        if name != 'original':
            metrics['accuracy_retention'] = metrics['accuracy'] / original_metrics['accuracy']
            metrics['speedup'] = original_metrics['inference_time'] / metrics['inference_time']
            metrics['size_reduction'] = original_metrics['model_size'] / metrics['model_size']
    
    return results


def _evaluate_model(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    num_batches: int
) -> Dict[str, float]:
    """Evaluate model performance and accuracy."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    inference_times = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            if i >= num_batches:
                break
            
            # Move to CPU for quantized models
            batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(**batch, return_dict=True)
            end_time = time.time()
            
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