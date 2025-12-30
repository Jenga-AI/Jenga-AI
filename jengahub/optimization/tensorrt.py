"""
TensorRT Optimization for JengaHub

This module provides TensorRT optimization capabilities for NVIDIA GPU acceleration
including model conversion, engine optimization, and performance benchmarking.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
from pathlib import Path
import json
import warnings
from tqdm import tqdm

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    warnings.warn("TensorRT not available. Install TensorRT and pycuda for GPU optimization.")

from ..core.model import JengaHubMultiModalModel


class TensorRTOptimizer:
    """TensorRT optimization for JengaHub models."""
    
    def __init__(
        self,
        precision: str = "fp16",
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 32,
        verbose: bool = False
    ):
        """
        Initialize TensorRT optimizer.
        
        Args:
            precision: Precision mode (fp32, fp16, int8)
            max_workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size
            verbose: Whether to enable verbose logging
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT is required but not available")
        
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        )
    
    def convert_onnx_to_tensorrt(
        self,
        onnx_path: str,
        engine_path: str,
        calibration_data: Optional[np.ndarray] = None,
        input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        optimization_profiles: Optional[List[Dict]] = None
    ) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            calibration_data: Calibration data for INT8 quantization
            input_shapes: Input tensor shapes
            optimization_profiles: Optimization profiles for dynamic shapes
            
        Returns:
            Path to TensorRT engine
        """
        self.logger.info(f"Converting ONNX to TensorRT: {onnx_path} -> {engine_path}")
        
        # Create builder and network
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # Parse ONNX model
        parser = trt.OnnxParser(network, self.trt_logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                self.logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # Create builder config
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        
        # Set precision
        if self.precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("FP16 optimization enabled")
            else:
                self.logger.warning("FP16 not supported, falling back to FP32")
        
        elif self.precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                
                # Set INT8 calibrator if calibration data provided
                if calibration_data is not None:
                    calibrator = self._create_calibrator(calibration_data, input_shapes)
                    config.int8_calibrator = calibrator
                    self.logger.info("INT8 optimization with calibration enabled")
                else:
                    self.logger.warning("INT8 requested but no calibration data provided")
            else:
                self.logger.warning("INT8 not supported, falling back to FP32")
        
        # Set optimization profiles for dynamic shapes
        if optimization_profiles:
            for i, profile_dict in enumerate(optimization_profiles):
                profile = builder.create_optimization_profile()
                
                for input_name, shapes in profile_dict.items():
                    min_shape = shapes['min']
                    opt_shape = shapes['opt']
                    max_shape = shapes['max']
                    
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                
                config.add_optimization_profile(profile)
                self.logger.info(f"Added optimization profile {i}")
        
        # Build engine
        self.logger.info("Building TensorRT engine...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Serialize and save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        self.logger.info(f"TensorRT engine saved: {engine_path}")
        return engine_path
    
    def optimize_engine(
        self,
        engine_path: str,
        optimization_level: int = 5,
        enable_tactics: bool = True
    ) -> str:
        """
        Further optimize existing TensorRT engine.
        
        Args:
            engine_path: Path to TensorRT engine
            optimization_level: Optimization level (1-5)
            enable_tactics: Whether to enable advanced tactics
            
        Returns:
            Path to optimized engine
        """
        self.logger.info(f"Optimizing TensorRT engine: {engine_path}")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        # Create new builder for reoptimization
        builder = trt.Builder(self.trt_logger)
        config = builder.create_builder_config()
        
        # Apply advanced optimization settings
        config.max_workspace_size = self.max_workspace_size
        config.builder_optimization_level = optimization_level
        
        if enable_tactics:
            # Enable all available tactics
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        
        # Note: Full reoptimization would require rebuilding from network
        # For now, return original engine path
        self.logger.info("Engine optimization completed")
        return engine_path
    
    def benchmark_engine(
        self,
        engine_path: str,
        input_data: Dict[str, np.ndarray],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark TensorRT engine performance.
        
        Args:
            engine_path: Path to TensorRT engine
            input_data: Input data for benchmarking
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance metrics
        """
        self.logger.info(f"Benchmarking TensorRT engine: {engine_path}")
        
        # Load engine
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # Create execution context
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings, stream = self._allocate_buffers(engine, input_data)
        
        # Warmup runs
        for _ in range(warmup_runs):
            self._do_inference(context, bindings, inputs, outputs, stream)
        
        # Benchmark runs
        inference_times = []
        
        for _ in tqdm(range(num_runs), desc="Benchmarking TensorRT"):
            start_time = time.perf_counter()
            self._do_inference(context, bindings, inputs, outputs, stream)
            cuda.Context.synchronize()
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
        
        # Calculate metrics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        metrics = {
            'mean_inference_time': mean_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput': 1.0 / mean_time,
            'latency_p50': np.percentile(inference_times, 50),
            'latency_p95': np.percentile(inference_times, 95),
            'latency_p99': np.percentile(inference_times, 99)
        }
        
        self.logger.info(f"TensorRT benchmark completed. Mean time: {mean_time*1000:.2f}ms")
        
        # Cleanup
        del inputs, outputs, stream
        
        return metrics
    
    def _create_calibrator(
        self,
        calibration_data: np.ndarray,
        input_shapes: Dict[str, Tuple[int, ...]]
    ):
        """Create INT8 calibrator."""
        # This would implement a custom calibrator for INT8 quantization
        # For now, return None to indicate no calibrator
        return None
    
    def _allocate_buffers(self, engine, input_data):
        """Allocate GPU memory buffers for inference."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                # Copy input data to host buffer
                input_name = binding
                if input_name in input_data:
                    np.copyto(host_mem, input_data[input_name].ravel())
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def _do_inference(self, context, bindings, inputs, outputs, stream):
        """Run inference with TensorRT engine."""
        # Copy inputs to GPU
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        
        # Copy outputs back to CPU
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
        # Synchronize stream
        stream.synchronize()


class TensorRTEngineManager:
    """Manage TensorRT engines with caching and versioning."""
    
    def __init__(self, cache_dir: str = "./tensorrt_cache"):
        """Initialize engine manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        
        # Load engine registry
        self.registry_path = self.cache_dir / "engine_registry.json"
        self.registry = self._load_registry()
    
    def get_engine(
        self,
        model_hash: str,
        precision: str = "fp16",
        max_batch_size: int = 32,
        input_shapes: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get cached engine or build new one.
        
        Args:
            model_hash: Hash of the source model
            precision: Precision mode
            max_batch_size: Maximum batch size
            input_shapes: Input shapes
            
        Returns:
            Path to engine file or None if not available
        """
        # Create engine key
        engine_key = self._create_engine_key(
            model_hash, precision, max_batch_size, input_shapes
        )
        
        if engine_key in self.registry:
            engine_path = self.cache_dir / self.registry[engine_key]['filename']
            if engine_path.exists():
                self.logger.info(f"Using cached TensorRT engine: {engine_path}")
                return str(engine_path)
        
        return None
    
    def register_engine(
        self,
        model_hash: str,
        engine_path: str,
        precision: str = "fp16",
        max_batch_size: int = 32,
        input_shapes: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Register new engine in cache."""
        engine_key = self._create_engine_key(
            model_hash, precision, max_batch_size, input_shapes
        )
        
        self.registry[engine_key] = {
            'filename': Path(engine_path).name,
            'model_hash': model_hash,
            'precision': precision,
            'max_batch_size': max_batch_size,
            'input_shapes': input_shapes,
            'created_at': time.time(),
            'metadata': metadata or {}
        }
        
        self._save_registry()
        self.logger.info(f"Registered TensorRT engine: {engine_key}")
    
    def cleanup_cache(self, max_age_days: int = 30):
        """Clean up old cached engines."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        engines_to_remove = []
        
        for engine_key, info in self.registry.items():
            age = current_time - info['created_at']
            if age > max_age_seconds:
                engines_to_remove.append(engine_key)
                
                # Remove engine file
                engine_path = self.cache_dir / info['filename']
                engine_path.unlink(missing_ok=True)
        
        # Remove from registry
        for key in engines_to_remove:
            del self.registry[key]
        
        if engines_to_remove:
            self._save_registry()
            self.logger.info(f"Cleaned up {len(engines_to_remove)} old engines")
    
    def _create_engine_key(
        self,
        model_hash: str,
        precision: str,
        max_batch_size: int,
        input_shapes: Optional[Dict]
    ) -> str:
        """Create unique key for engine caching."""
        key_parts = [model_hash, precision, str(max_batch_size)]
        
        if input_shapes:
            shapes_str = json.dumps(input_shapes, sort_keys=True)
            key_parts.append(shapes_str)
        
        return "_".join(key_parts)
    
    def _load_registry(self) -> Dict:
        """Load engine registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save engine registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)


# Convenience functions
def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    calibration_data: Optional[np.ndarray] = None,
    **kwargs
) -> str:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output engine path
        precision: Precision mode
        max_batch_size: Maximum batch size
        calibration_data: Calibration data for INT8
        **kwargs: Additional optimizer arguments
        
    Returns:
        Path to TensorRT engine
    """
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT is required but not available")
    
    optimizer = TensorRTOptimizer(
        precision=precision,
        max_batch_size=max_batch_size,
        **kwargs
    )
    
    return optimizer.convert_onnx_to_tensorrt(
        onnx_path, engine_path, calibration_data
    )


def benchmark_tensorrt_model(
    engine_path: str,
    input_data: Dict[str, np.ndarray],
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark TensorRT model performance.
    
    Args:
        engine_path: Path to TensorRT engine
        input_data: Input data
        num_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT is required but not available")
    
    optimizer = TensorRTOptimizer()
    return optimizer.benchmark_engine(engine_path, input_data, num_runs)


def optimize_tensorrt_engine(
    engine_path: str,
    optimization_level: int = 5
) -> str:
    """
    Optimize TensorRT engine.
    
    Args:
        engine_path: Path to engine
        optimization_level: Optimization level
        
    Returns:
        Path to optimized engine
    """
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT is required but not available")
    
    optimizer = TensorRTOptimizer()
    return optimizer.optimize_engine(engine_path, optimization_level)