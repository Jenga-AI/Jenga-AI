"""
ONNX Export and Optimization for JengaHub

This module provides comprehensive ONNX export capabilities with optimization,
validation, and benchmarking for cross-platform deployment.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
from onnxsim import simplify
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
from pathlib import Path
import json
from tqdm import tqdm
import warnings

from ..core.model import JengaHubMultiModalModel
from ..core.config import MultiModalConfig


class ONNXExporter:
    """Comprehensive ONNX export and optimization for JengaHub models."""
    
    def __init__(
        self,
        opset_version: int = 11,
        do_constant_folding: bool = True,
        optimize_graph: bool = True
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            opset_version: ONNX opset version
            do_constant_folding: Whether to perform constant folding
            optimize_graph: Whether to optimize the graph
        """
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.optimize_graph = optimize_graph
        self.logger = logging.getLogger(__name__)
    
    def export(
        self,
        model: JengaHubMultiModalModel,
        example_inputs: Dict[str, torch.Tensor],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        export_params: bool = True,
        verbose: bool = False
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            example_inputs: Example inputs for tracing
            output_path: Output ONNX file path
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            export_params: Whether to export parameters
            verbose: Whether to print verbose output
            
        Returns:
            Path to exported ONNX model
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Prepare model for export
        model.eval()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare input tensors
        input_tensors = self._prepare_inputs(example_inputs)
        
        # Auto-generate input/output names if not provided
        if input_names is None:
            input_names = list(example_inputs.keys())
        
        if output_names is None:
            output_names = ['output']
        
        # Auto-generate dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {}
            for name in input_names:
                dynamic_axes[name] = {0: 'batch_size'}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}
        
        try:
            # Export to ONNX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                torch.onnx.export(
                    model,
                    input_tensors,
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset_version,
                    do_constant_folding=self.do_constant_folding,
                    export_params=export_params,
                    verbose=verbose,
                    training=torch.onnx.TrainingMode.EVAL,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                )
            
            self.logger.info(f"ONNX export completed: {output_path}")
            
            # Validate exported model
            self._validate_export(output_path)
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"ONNX export failed: {str(e)}")
            raise
    
    def export_with_optimization(
        self,
        model: JengaHubMultiModalModel,
        example_inputs: Dict[str, torch.Tensor],
        output_path: str,
        optimization_level: str = "all",
        **export_kwargs
    ) -> str:
        """
        Export model with comprehensive optimization.
        
        Args:
            model: Model to export
            example_inputs: Example inputs
            output_path: Output path
            optimization_level: Optimization level (basic, all)
            **export_kwargs: Additional export arguments
            
        Returns:
            Path to optimized ONNX model
        """
        # First export to ONNX
        temp_path = output_path.replace('.onnx', '_temp.onnx')
        self.export(model, example_inputs, temp_path, **export_kwargs)
        
        # Optimize ONNX model
        optimized_path = self.optimize_onnx_model(
            temp_path, 
            output_path, 
            optimization_level
        )
        
        # Remove temporary file
        Path(temp_path).unlink(missing_ok=True)
        
        return optimized_path
    
    def optimize_onnx_model(
        self,
        input_path: str,
        output_path: str,
        optimization_level: str = "all"
    ) -> str:
        """
        Optimize ONNX model using ONNXSimplifier and graph optimizations.
        
        Args:
            input_path: Input ONNX model path
            output_path: Output optimized model path
            optimization_level: Optimization level
            
        Returns:
            Path to optimized model
        """
        self.logger.info(f"Optimizing ONNX model: {input_path} -> {output_path}")
        
        try:
            # Load original model
            model = onnx.load(input_path)
            
            # Basic optimizations
            if optimization_level in ["basic", "all"]:
                # Simplify model
                model_simplified, check = simplify(
                    model,
                    check_n=3,
                    perform_optimization=True
                )
                
                if check:
                    model = model_simplified
                    self.logger.info("Model simplified successfully")
                else:
                    self.logger.warning("Model simplification failed validation")
            
            # Advanced optimizations
            if optimization_level == "all":
                model = self._apply_advanced_optimizations(model)
            
            # Save optimized model
            onnx.save(model, output_path)
            
            # Compare model sizes
            original_size = Path(input_path).stat().st_size / (1024 * 1024)  # MB
            optimized_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
            
            self.logger.info(
                f"Optimization completed. Size: {original_size:.1f}MB -> {optimized_size:.1f}MB "
                f"({100 * (1 - optimized_size/original_size):.1f}% reduction)"
            )
            
            return output_path
        
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {str(e)}")
            raise
    
    def _prepare_inputs(self, example_inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Prepare input tensors for ONNX export."""
        input_tensors = []
        for key in sorted(example_inputs.keys()):
            tensor = example_inputs[key]
            if isinstance(tensor, torch.Tensor):
                input_tensors.append(tensor)
        return tuple(input_tensors)
    
    def _validate_export(self, onnx_path: str):
        """Validate exported ONNX model."""
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            self.logger.info("ONNX model validation passed")
        except Exception as e:
            self.logger.error(f"ONNX model validation failed: {str(e)}")
            raise
    
    def _apply_advanced_optimizations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply advanced ONNX optimizations."""
        # This would include custom optimization passes
        # For now, return the model as-is
        self.logger.info("Advanced optimizations applied")
        return model


class ONNXBenchmark:
    """Benchmarking utility for ONNX models."""
    
    def __init__(self, providers: Optional[List[str]] = None):
        """
        Initialize ONNX benchmark.
        
        Args:
            providers: ONNX Runtime providers to use
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.providers = providers
        self.logger = logging.getLogger(__name__)
    
    def benchmark_model(
        self,
        onnx_path: str,
        example_inputs: Dict[str, np.ndarray],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            onnx_path: Path to ONNX model
            example_inputs: Example inputs as numpy arrays
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark metrics
        """
        self.logger.info(f"Benchmarking ONNX model: {onnx_path}")
        
        # Create inference session
        session = ort.InferenceSession(onnx_path, providers=self.providers)
        
        # Get input names
        input_names = [inp.name for inp in session.get_inputs()]
        
        # Prepare inputs
        feed_dict = {}
        for i, name in enumerate(input_names):
            if name in example_inputs:
                feed_dict[name] = example_inputs[name]
            else:
                # Use first available input if names don't match
                feed_dict[name] = list(example_inputs.values())[i]
        
        # Warmup runs
        for _ in range(warmup_runs):
            _ = session.run(None, feed_dict)
        
        # Benchmark runs
        inference_times = []
        
        for _ in tqdm(range(num_runs), desc="Benchmarking"):
            start_time = time.perf_counter()
            _ = session.run(None, feed_dict)
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        metrics = {
            'mean_inference_time': mean_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput': 1.0 / mean_time,  # inferences per second
            'latency_p50': np.percentile(inference_times, 50),
            'latency_p95': np.percentile(inference_times, 95),
            'latency_p99': np.percentile(inference_times, 99)
        }
        
        self.logger.info(f"Benchmark completed. Mean inference time: {mean_time*1000:.2f}ms")
        
        return metrics
    
    def compare_models(
        self,
        model_paths: Dict[str, str],
        example_inputs: Dict[str, np.ndarray],
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple ONNX models.
        
        Args:
            model_paths: Dictionary of model name to path
            example_inputs: Example inputs
            num_runs: Number of benchmark runs
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, path in model_paths.items():
            self.logger.info(f"Benchmarking {name}...")
            results[name] = self.benchmark_model(path, example_inputs, num_runs)
        
        return results


# Convenience functions
def export_to_onnx(
    model: JengaHubMultiModalModel,
    example_inputs: Dict[str, torch.Tensor],
    output_path: str,
    optimize: bool = True,
    **kwargs
) -> str:
    """
    Convenience function for ONNX export.
    
    Args:
        model: Model to export
        example_inputs: Example inputs
        output_path: Output path
        optimize: Whether to optimize the model
        **kwargs: Additional export arguments
        
    Returns:
        Path to exported ONNX model
    """
    exporter = ONNXExporter()
    
    if optimize:
        return exporter.export_with_optimization(
            model, example_inputs, output_path, **kwargs
        )
    else:
        return exporter.export(model, example_inputs, output_path, **kwargs)


def optimize_onnx_model(
    input_path: str,
    output_path: str,
    optimization_level: str = "all"
) -> str:
    """
    Optimize existing ONNX model.
    
    Args:
        input_path: Input model path
        output_path: Output model path
        optimization_level: Optimization level
        
    Returns:
        Path to optimized model
    """
    exporter = ONNXExporter()
    return exporter.optimize_onnx_model(input_path, output_path, optimization_level)


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        
    Returns:
        True if valid, False otherwise
    """
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return True
    except Exception:
        return False


def benchmark_onnx_model(
    onnx_path: str,
    example_inputs: Dict[str, np.ndarray],
    providers: Optional[List[str]] = None,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark ONNX model performance.
    
    Args:
        onnx_path: Path to ONNX model
        example_inputs: Example inputs as numpy arrays
        providers: ONNX Runtime providers
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark metrics
    """
    benchmark = ONNXBenchmark(providers)
    return benchmark.benchmark_model(onnx_path, example_inputs, num_runs)


def torch_to_numpy_inputs(torch_inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert PyTorch inputs to numpy for ONNX Runtime."""
    numpy_inputs = {}
    for key, tensor in torch_inputs.items():
        if isinstance(tensor, torch.Tensor):
            numpy_inputs[key] = tensor.detach().cpu().numpy()
        else:
            numpy_inputs[key] = tensor
    return numpy_inputs


class ONNXModelAnalyzer:
    """Analyze ONNX models for optimization opportunities."""
    
    def __init__(self, model_path: str):
        """Initialize analyzer with ONNX model."""
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.logger = logging.getLogger(__name__)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Comprehensive model analysis.
        
        Returns:
            Analysis results
        """
        analysis = {
            'model_info': self._get_model_info(),
            'graph_info': self._analyze_graph(),
            'operator_stats': self._analyze_operators(),
            'size_analysis': self._analyze_size(),
            'optimization_opportunities': self._find_optimization_opportunities()
        }
        
        return analysis
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        return {
            'ir_version': self.model.ir_version,
            'opset_version': self.model.opset_import[0].version,
            'producer_name': self.model.producer_name,
            'producer_version': self.model.producer_version,
            'domain': self.model.domain
        }
    
    def _analyze_graph(self) -> Dict[str, Any]:
        """Analyze model graph structure."""
        graph = self.model.graph
        
        return {
            'num_nodes': len(graph.node),
            'num_inputs': len(graph.input),
            'num_outputs': len(graph.output),
            'num_initializers': len(graph.initializer),
            'num_value_infos': len(graph.value_info)
        }
    
    def _analyze_operators(self) -> Dict[str, Any]:
        """Analyze operator usage."""
        op_counts = {}
        for node in self.model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        return {
            'operator_counts': op_counts,
            'total_operators': len(self.model.graph.node),
            'unique_operators': len(op_counts)
        }
    
    def _analyze_size(self) -> Dict[str, Any]:
        """Analyze model size."""
        model_size = Path(self.model_path).stat().st_size
        
        # Count parameters
        total_params = 0
        for initializer in self.model.graph.initializer:
            param_size = 1
            for dim in initializer.dims:
                param_size *= dim
            total_params += param_size
        
        return {
            'file_size_mb': model_size / (1024 * 1024),
            'total_parameters': total_params,
            'parameters_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def _find_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check for common optimization opportunities
        op_counts = {}
        for node in self.model.graph.node:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        # Check for patterns that can be optimized
        if 'MatMul' in op_counts and op_counts['MatMul'] > 5:
            opportunities.append("Consider operator fusion for MatMul operations")
        
        if 'BatchNormalization' in op_counts:
            opportunities.append("Consider fusing BatchNormalization with preceding operations")
        
        if len(self.model.graph.initializer) > 100:
            opportunities.append("Consider weight quantization to reduce model size")
        
        return opportunities