"""
JengaHub Model Optimization Module

This module provides comprehensive model optimization capabilities for efficient
deployment and inference, including quantization, ONNX export, TensorRT optimization,
pruning, and knowledge distillation.
"""

from .quantization import (
    DynamicQuantizer,
    StaticQuantizer, 
    QATTrainer,
    quantize_model,
    calibrate_model,
    compare_quantized_models
)

from .onnx_export import (
    ONNXExporter,
    export_to_onnx,
    optimize_onnx_model,
    validate_onnx_model,
    benchmark_onnx_model
)

from .tensorrt import (
    TensorRTOptimizer,
    convert_to_tensorrt,
    benchmark_tensorrt_model,
    optimize_tensorrt_engine
)

from .pruning import (
    StructuredPruner,
    UnstructuredPruner,
    MagnitudePruner,
    GradualPruner,
    prune_model,
    fine_tune_pruned_model
)

from .distillation import (
    KnowledgeDistiller,
    AttentionDistiller,
    FeatureDistiller,
    distill_model,
    create_student_model
)

from .utils import (
    ModelOptimizer,
    BenchmarkSuite,
    OptimizationConfig,
    compare_models,
    analyze_model_efficiency
)

__all__ = [
    # Quantization
    'DynamicQuantizer',
    'StaticQuantizer',
    'QATTrainer',
    'quantize_model',
    'calibrate_model',
    'compare_quantized_models',
    
    # ONNX Export
    'ONNXExporter',
    'export_to_onnx',
    'optimize_onnx_model',
    'validate_onnx_model',
    'benchmark_onnx_model',
    
    # TensorRT
    'TensorRTOptimizer',
    'convert_to_tensorrt',
    'benchmark_tensorrt_model',
    'optimize_tensorrt_engine',
    
    # Pruning
    'StructuredPruner',
    'UnstructuredPruner',
    'MagnitudePruner',
    'GradualPruner',
    'prune_model',
    'fine_tune_pruned_model',
    
    # Distillation
    'KnowledgeDistiller',
    'AttentionDistiller',
    'FeatureDistiller',
    'distill_model',
    'create_student_model',
    
    # Utils
    'ModelOptimizer',
    'BenchmarkSuite',
    'OptimizationConfig',
    'compare_models',
    'analyze_model_efficiency'
]