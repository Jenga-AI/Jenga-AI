"""
Optimization Utilities for JengaHub

This module provides comprehensive utilities for model optimization including
benchmarking, analysis, configuration management, and optimization pipelines.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import hashlib

from ..core.model import JengaHubMultiModalModel
from ..core.config import MultiModalConfig


@dataclass
class OptimizationConfig:
    """Configuration for model optimization pipeline."""
    
    # Quantization settings
    enable_quantization: bool = True
    quantization_method: str = "dynamic"  # dynamic, static, qat
    quantization_dtype: str = "qint8"
    calibration_batches: int = 100
    
    # ONNX export settings
    enable_onnx_export: bool = True
    onnx_opset_version: int = 11
    onnx_optimization_level: str = "all"
    enable_dynamic_axes: bool = True
    
    # TensorRT settings
    enable_tensorrt: bool = False
    tensorrt_precision: str = "fp16"
    tensorrt_max_batch_size: int = 32
    tensorrt_workspace_size: int = 1 << 30  # 1GB
    
    # Pruning settings
    enable_pruning: bool = False
    pruning_method: str = "magnitude"
    pruning_ratio: float = 0.5
    pruning_schedule: str = "gradual"
    
    # Distillation settings
    enable_distillation: bool = False
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    student_compression_ratio: float = 0.5
    
    # General settings
    benchmark_iterations: int = 100
    enable_profiling: bool = True
    output_dir: str = "./optimization_output"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OptimizationConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ModelOptimizer:
    """Comprehensive model optimization pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize model optimizer with configuration."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.optimization_results = {}
        
        # Setup logging
        self._setup_logging()
    
    def optimize_model(
        self,
        model: JengaHubMultiModalModel,
        example_inputs: Dict[str, torch.Tensor],
        calibration_dataloader: Optional[torch.utils.data.DataLoader] = None,
        validation_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive optimization pipeline.
        
        Args:
            model: Model to optimize
            example_inputs: Example inputs for tracing/export
            calibration_dataloader: Data for calibration
            validation_dataloader: Data for validation
            
        Returns:
            Optimization results and optimized models
        """
        self.logger.info("Starting comprehensive model optimization pipeline")
        
        # Create model hash for caching
        model_hash = self._compute_model_hash(model)
        self.optimization_results['model_hash'] = model_hash
        
        # Baseline evaluation
        baseline_metrics = self._evaluate_baseline(model, example_inputs, validation_dataloader)
        self.optimization_results['baseline'] = baseline_metrics
        
        optimized_models = {'original': model}
        
        # 1. Quantization
        if self.config.enable_quantization:
            self.logger.info("Running quantization optimization...")
            quantized_model = self._apply_quantization(model, calibration_dataloader)
            if quantized_model:
                optimized_models['quantized'] = quantized_model
                quant_metrics = self._evaluate_model(quantized_model, example_inputs, validation_dataloader)
                self.optimization_results['quantized'] = quant_metrics
        
        # 2. ONNX Export
        if self.config.enable_onnx_export:
            self.logger.info("Running ONNX export optimization...")
            onnx_path = self._export_to_onnx(model, example_inputs)
            if onnx_path:
                self.optimization_results['onnx_path'] = str(onnx_path)
                # Benchmark ONNX model
                onnx_metrics = self._benchmark_onnx_model(onnx_path, example_inputs)
                self.optimization_results['onnx'] = onnx_metrics
        
        # 3. TensorRT (if enabled and available)
        if self.config.enable_tensorrt:
            try:
                self.logger.info("Running TensorRT optimization...")
                tensorrt_path = self._convert_to_tensorrt(onnx_path, example_inputs)
                if tensorrt_path:
                    self.optimization_results['tensorrt_path'] = str(tensorrt_path)
                    trt_metrics = self._benchmark_tensorrt_model(tensorrt_path, example_inputs)
                    self.optimization_results['tensorrt'] = trt_metrics
            except Exception as e:
                self.logger.warning(f"TensorRT optimization failed: {e}")
        
        # 4. Pruning
        if self.config.enable_pruning:
            self.logger.info("Running pruning optimization...")
            pruned_model = self._apply_pruning(model, calibration_dataloader)
            if pruned_model:
                optimized_models['pruned'] = pruned_model
                pruned_metrics = self._evaluate_model(pruned_model, example_inputs, validation_dataloader)
                self.optimization_results['pruned'] = pruned_metrics
        
        # 5. Knowledge Distillation
        if self.config.enable_distillation:
            self.logger.info("Running knowledge distillation...")
            student_model = self._apply_distillation(model, calibration_dataloader)
            if student_model:
                optimized_models['distilled'] = student_model
                distilled_metrics = self._evaluate_model(student_model, example_inputs, validation_dataloader)
                self.optimization_results['distilled'] = distilled_metrics
        
        # Generate optimization report
        self._generate_optimization_report()
        
        # Save results
        self._save_optimization_results()
        
        self.logger.info("Model optimization pipeline completed")
        
        return {
            'optimized_models': optimized_models,
            'results': self.optimization_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _evaluate_baseline(
        self,
        model: torch.nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        validation_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Dict[str, Any]:
        """Evaluate baseline model performance."""
        self.logger.info("Evaluating baseline model...")
        return self._evaluate_model(model, example_inputs, validation_dataloader)
    
    def _evaluate_model(
        self,
        model: torch.nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        validation_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        device = next(model.parameters()).device
        model.eval()
        
        metrics = {}
        
        # Model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        metrics['model_size_mb'] = model_size_mb
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['total_parameters'] = total_params
        metrics['trainable_parameters'] = trainable_params
        
        # Inference benchmarking
        input_tensors = {k: v.to(device) for k, v in example_inputs.items() if isinstance(v, torch.Tensor)}
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**input_tensors)
        
        # Benchmark
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(self.config.benchmark_iterations):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                outputs = model(**input_tensors)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB
        
        # Performance metrics
        metrics['mean_inference_time'] = np.mean(inference_times)
        metrics['std_inference_time'] = np.std(inference_times)
        metrics['throughput'] = 1.0 / np.mean(inference_times)
        metrics['p95_latency'] = np.percentile(inference_times, 95)
        
        if memory_usage:
            metrics['mean_memory_usage_mb'] = np.mean(memory_usage)
            metrics['peak_memory_usage_mb'] = np.max(memory_usage)
        
        # Accuracy evaluation (if validation data provided)
        if validation_dataloader:
            accuracy_metrics = self._evaluate_accuracy(model, validation_dataloader)
            metrics.update(accuracy_metrics)
        
        return metrics
    
    def _evaluate_accuracy(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 50
    ) -> Dict[str, float]:
        """Evaluate model accuracy."""
        device = next(model.parameters()).device
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch, return_dict=True)
                loss = outputs.get('loss', torch.tensor(0.0))
                
                batch_size = next(iter(batch.values())).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return {
            'validation_loss': total_loss / total_samples,
            'validation_accuracy': 1.0 / (1.0 + total_loss / total_samples)  # Simple proxy
        }
    
    def _apply_quantization(
        self,
        model: torch.nn.Module,
        calibration_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Optional[torch.nn.Module]:
        """Apply model quantization."""
        try:
            from .quantization import quantize_model
            
            return quantize_model(
                model,
                method=self.config.quantization_method,
                calibration_data=calibration_dataloader
            )
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return None
    
    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        example_inputs: Dict[str, torch.Tensor]
    ) -> Optional[Path]:
        """Export model to ONNX."""
        try:
            from .onnx_export import export_to_onnx
            
            onnx_path = self.output_dir / "model.onnx"
            
            export_to_onnx(
                model,
                example_inputs,
                str(onnx_path),
                optimize=True,
                opset_version=self.config.onnx_opset_version
            )
            
            return onnx_path
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            return None
    
    def _benchmark_onnx_model(
        self,
        onnx_path: Path,
        example_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Benchmark ONNX model."""
        try:
            from .onnx_export import benchmark_onnx_model, torch_to_numpy_inputs
            
            numpy_inputs = torch_to_numpy_inputs(example_inputs)
            return benchmark_onnx_model(
                str(onnx_path),
                numpy_inputs,
                num_runs=self.config.benchmark_iterations
            )
        except Exception as e:
            self.logger.error(f"ONNX benchmarking failed: {e}")
            return {}
    
    def _convert_to_tensorrt(
        self,
        onnx_path: Path,
        example_inputs: Dict[str, torch.Tensor]
    ) -> Optional[Path]:
        """Convert ONNX to TensorRT."""
        try:
            from .tensorrt import convert_to_tensorrt
            
            trt_path = self.output_dir / "model.trt"
            
            convert_to_tensorrt(
                str(onnx_path),
                str(trt_path),
                precision=self.config.tensorrt_precision,
                max_batch_size=self.config.tensorrt_max_batch_size
            )
            
            return trt_path
        except Exception as e:
            self.logger.error(f"TensorRT conversion failed: {e}")
            return None
    
    def _benchmark_tensorrt_model(
        self,
        trt_path: Path,
        example_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Benchmark TensorRT model."""
        try:
            from .tensorrt import benchmark_tensorrt_model
            from .onnx_export import torch_to_numpy_inputs
            
            numpy_inputs = torch_to_numpy_inputs(example_inputs)
            return benchmark_tensorrt_model(
                str(trt_path),
                numpy_inputs,
                num_runs=self.config.benchmark_iterations
            )
        except Exception as e:
            self.logger.error(f"TensorRT benchmarking failed: {e}")
            return {}
    
    def _apply_pruning(
        self,
        model: torch.nn.Module,
        calibration_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Optional[torch.nn.Module]:
        """Apply model pruning."""
        try:
            from .pruning import prune_model
            
            return prune_model(
                model,
                method=self.config.pruning_method,
                pruning_ratio=self.config.pruning_ratio
            )
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return None
    
    def _apply_distillation(
        self,
        teacher_model: torch.nn.Module,
        calibration_dataloader: Optional[torch.utils.data.DataLoader]
    ) -> Optional[torch.nn.Module]:
        """Apply knowledge distillation."""
        try:
            from .distillation import create_student_model, distill_model
            
            if calibration_dataloader is None:
                self.logger.warning("No calibration data for distillation")
                return None
            
            # Create student model
            student_model = create_student_model(
                teacher_model,
                compression_ratio=self.config.student_compression_ratio
            )
            
            # Perform distillation
            distilled_model = distill_model(
                teacher_model,
                student_model,
                calibration_dataloader,
                temperature=self.config.distillation_temperature,
                alpha=self.config.distillation_alpha,
                num_epochs=5  # Quick distillation
            )
            
            return distilled_model
        except Exception as e:
            self.logger.error(f"Distillation failed: {e}")
            return None
    
    def _compute_model_hash(self, model: torch.nn.Module) -> str:
        """Compute hash of model architecture and parameters."""
        model_str = str(model)
        param_str = ""
        
        for name, param in model.named_parameters():
            param_str += f"{name}:{param.shape}"
        
        combined_str = model_str + param_str
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def _generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        report = {
            'summary': self._generate_summary(),
            'detailed_metrics': self.optimization_results,
            'recommendations': self._generate_recommendations(),
            'system_info': self._get_system_info()
        }
        
        # Save report
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Optimization report saved: {report_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate optimization summary."""
        baseline = self.optimization_results.get('baseline', {})
        baseline_time = baseline.get('mean_inference_time', 1.0)
        baseline_size = baseline.get('model_size_mb', 1.0)
        
        summary = {
            'optimizations_applied': [],
            'best_latency_improvement': 0.0,
            'best_size_reduction': 0.0,
            'accuracy_retention': 1.0
        }
        
        for opt_name, opt_results in self.optimization_results.items():
            if opt_name in ['baseline', 'model_hash']:
                continue
            
            if isinstance(opt_results, dict):
                summary['optimizations_applied'].append(opt_name)
                
                # Calculate improvements
                opt_time = opt_results.get('mean_inference_time', baseline_time)
                opt_size = opt_results.get('model_size_mb', baseline_size)
                
                latency_improvement = (baseline_time - opt_time) / baseline_time
                size_reduction = (baseline_size - opt_size) / baseline_size
                
                summary['best_latency_improvement'] = max(
                    summary['best_latency_improvement'], latency_improvement
                )
                summary['best_size_reduction'] = max(
                    summary['best_size_reduction'], size_reduction
                )
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        baseline = self.optimization_results.get('baseline', {})
        
        # Analyze results and generate recommendations
        if 'quantized' in self.optimization_results:
            quant_results = self.optimization_results['quantized']
            baseline_time = baseline.get('mean_inference_time', 1.0)
            quant_time = quant_results.get('mean_inference_time', baseline_time)
            
            if quant_time < baseline_time * 0.8:
                recommendations.append("Quantization provides significant speedup - recommended for deployment")
            else:
                recommendations.append("Quantization speedup is minimal - consider other optimizations")
        
        if 'onnx' in self.optimization_results:
            recommendations.append("ONNX export successful - enables cross-platform deployment")
        
        if 'tensorrt' in self.optimization_results:
            recommendations.append("TensorRT optimization available for NVIDIA GPU deployment")
        
        if not recommendations:
            recommendations.append("Consider enabling more optimization techniques for better performance")
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_memory_gb'] = [gpu.memoryTotal / 1024 for gpu in gpus]
        except:
            pass
        
        return info
    
    def _save_optimization_results(self):
        """Save optimization results to file."""
        results_path = self.output_dir / "optimization_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.optimization_results, f)
        
        self.logger.info(f"Optimization results saved: {results_path}")
    
    def _setup_logging(self):
        """Setup logging for optimization."""
        log_file = self.output_dir / "optimization.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )


class BenchmarkSuite:
    """Comprehensive benchmarking suite for model comparison."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def benchmark_models(
        self,
        models: Dict[str, torch.nn.Module],
        example_inputs: Dict[str, torch.Tensor],
        validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
        num_runs: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple models comprehensively.
        
        Args:
            models: Dictionary of model name to model
            example_inputs: Example inputs for benchmarking
            validation_dataloader: Validation data
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results for each model
        """
        self.logger.info(f"Benchmarking {len(models)} models...")
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Benchmarking {name}...")
            
            model_metrics = self._benchmark_single_model(
                model, example_inputs, validation_dataloader, num_runs
            )
            
            results[name] = model_metrics
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _benchmark_single_model(
        self,
        model: torch.nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        validation_dataloader: Optional[torch.utils.data.DataLoader],
        num_runs: int
    ) -> Dict[str, Any]:
        """Benchmark a single model."""
        optimizer = ModelOptimizer(OptimizationConfig(benchmark_iterations=num_runs))
        return optimizer._evaluate_model(model, example_inputs, validation_dataloader)
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]):
        """Generate comparison report and visualizations."""
        # Create comparison plots
        self._plot_performance_comparison(results)
        self._plot_size_comparison(results)
        
        # Save detailed results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved: {results_path}")
    
    def _plot_performance_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot performance comparison."""
        model_names = list(results.keys())
        latencies = [results[name].get('mean_inference_time', 0) * 1000 for name in model_names]  # ms
        throughputs = [results[name].get('throughput', 0) for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency comparison
        ax1.bar(model_names, latencies)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Model Latency Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax2.bar(model_names, throughputs)
        ax2.set_ylabel('Throughput (inferences/sec)')
        ax2.set_title('Model Throughput Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_size_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Plot model size comparison."""
        model_names = list(results.keys())
        sizes = [results[name].get('model_size_mb', 0) for name in model_names]
        param_counts = [results[name].get('total_parameters', 0) / 1e6 for name in model_names]  # millions
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Size comparison
        ax1.bar(model_names, sizes)
        ax1.set_ylabel('Model Size (MB)')
        ax1.set_title('Model Size Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        ax2.bar(model_names, param_counts)
        ax2.set_ylabel('Parameters (Millions)')
        ax2.set_title('Parameter Count Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "size_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()


# Convenience functions
def compare_models(
    models: Dict[str, torch.nn.Module],
    example_inputs: Dict[str, torch.Tensor],
    validation_dataloader: Optional[torch.utils.data.DataLoader] = None,
    output_dir: str = "./model_comparison"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models comprehensively.
    
    Args:
        models: Dictionary of models to compare
        example_inputs: Example inputs
        validation_dataloader: Validation data
        output_dir: Output directory for results
        
    Returns:
        Comparison results
    """
    benchmark_suite = BenchmarkSuite(output_dir)
    return benchmark_suite.benchmark_models(models, example_inputs, validation_dataloader)


def analyze_model_efficiency(
    model: torch.nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    target_latency_ms: Optional[float] = None,
    target_size_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze model efficiency against targets.
    
    Args:
        model: Model to analyze
        example_inputs: Example inputs
        target_latency_ms: Target latency in milliseconds
        target_size_mb: Target model size in MB
        
    Returns:
        Efficiency analysis results
    """
    optimizer = ModelOptimizer(OptimizationConfig())
    metrics = optimizer._evaluate_model(model, example_inputs, None)
    
    analysis = {
        'current_metrics': metrics,
        'efficiency_score': 0.0,
        'meets_targets': True,
        'recommendations': []
    }
    
    # Check against targets
    current_latency_ms = metrics.get('mean_inference_time', 0) * 1000
    current_size_mb = metrics.get('model_size_mb', 0)
    
    if target_latency_ms and current_latency_ms > target_latency_ms:
        analysis['meets_targets'] = False
        analysis['recommendations'].append(
            f"Latency ({current_latency_ms:.1f}ms) exceeds target ({target_latency_ms}ms)"
        )
    
    if target_size_mb and current_size_mb > target_size_mb:
        analysis['meets_targets'] = False
        analysis['recommendations'].append(
            f"Size ({current_size_mb:.1f}MB) exceeds target ({target_size_mb}MB)"
        )
    
    # Calculate efficiency score (higher is better)
    throughput = metrics.get('throughput', 1.0)
    size_efficiency = 1.0 / max(current_size_mb, 1.0)
    analysis['efficiency_score'] = throughput * size_efficiency
    
    return analysis


def create_optimization_config(
    target_platform: str = "cpu",
    performance_priority: str = "balanced",
    size_constraints: Optional[float] = None
) -> OptimizationConfig:
    """
    Create optimization configuration for specific deployment scenario.
    
    Args:
        target_platform: Target platform (cpu, gpu, mobile, edge)
        performance_priority: Priority (speed, size, accuracy, balanced)
        size_constraints: Maximum model size in MB
        
    Returns:
        Optimization configuration
    """
    config = OptimizationConfig()
    
    # Platform-specific settings
    if target_platform == "cpu":
        config.enable_quantization = True
        config.quantization_method = "dynamic"
        config.enable_onnx_export = True
        config.enable_tensorrt = False
    
    elif target_platform == "gpu":
        config.enable_quantization = True
        config.quantization_method = "static"
        config.enable_onnx_export = True
        config.enable_tensorrt = True
        config.tensorrt_precision = "fp16"
    
    elif target_platform in ["mobile", "edge"]:
        config.enable_quantization = True
        config.quantization_method = "qat"
        config.quantization_dtype = "qint8"
        config.enable_pruning = True
        config.pruning_ratio = 0.7
        config.enable_distillation = True
        config.student_compression_ratio = 0.3
    
    # Performance priority settings
    if performance_priority == "speed":
        config.enable_quantization = True
        config.enable_onnx_export = True
        config.enable_tensorrt = True
    
    elif performance_priority == "size":
        config.enable_pruning = True
        config.pruning_ratio = 0.8
        config.enable_distillation = True
        config.student_compression_ratio = 0.25
        config.enable_quantization = True
    
    elif performance_priority == "accuracy":
        config.pruning_ratio = 0.3
        config.quantization_method = "qat"
        config.enable_distillation = False
    
    return config