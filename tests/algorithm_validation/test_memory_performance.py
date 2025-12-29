#!/usr/bin/env python3
"""
Memory Efficiency and Performance Analysis
Tests for memory usage, gradient accumulation, and CPU optimization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading

from multitask_bert.core.model import MultiTaskModel
from multitask_bert.core.config import ModelConfig
from multitask_bert.tasks.classification import SingleLabelClassificationTask
from multitask_bert.tasks.ner import NERTask
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Memory usage profile"""
    phase: str
    rss_mb: float
    vms_mb: float
    percent: float
    peak_mb: float
    timestamp: float


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    forward_time: float
    backward_time: float
    memory_peak: float
    memory_allocated: float
    batch_size: int
    sequence_length: int
    throughput_samples_per_sec: float


class MemoryMonitor:
    """Real-time memory monitoring"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.profiles = []
        self.monitoring = False
        self.peak_memory = 0.0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                profile = MemoryProfile(
                    phase="monitoring",
                    rss_mb=memory_info.rss / 1024 / 1024,
                    vms_mb=memory_info.vms / 1024 / 1024,
                    percent=memory_percent,
                    peak_mb=self.peak_memory,
                    timestamp=time.time()
                )
                
                self.profiles.append(profile)
                self.peak_memory = max(self.peak_memory, profile.rss_mb)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                break
    
    @contextmanager
    def profile_phase(self, phase_name: str):
        """Context manager for profiling a specific phase"""
        process = psutil.Process()
        
        # Before
        gc.collect()
        memory_before = process.memory_info()
        
        profile_before = MemoryProfile(
            phase=f"{phase_name}_start",
            rss_mb=memory_before.rss / 1024 / 1024,
            vms_mb=memory_before.vms / 1024 / 1024,
            percent=process.memory_percent(),
            peak_mb=self.peak_memory,
            timestamp=time.time()
        )
        self.profiles.append(profile_before)
        
        try:
            yield
        finally:
            # After
            memory_after = process.memory_info()
            profile_after = MemoryProfile(
                phase=f"{phase_name}_end",
                rss_mb=memory_after.rss / 1024 / 1024,
                vms_mb=memory_after.vms / 1024 / 1024,
                percent=process.memory_percent(),
                peak_mb=self.peak_memory,
                timestamp=time.time()
            )
            self.profiles.append(profile_after)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory usage summary"""
        if not self.profiles:
            return {}
        
        rss_values = [p.rss_mb for p in self.profiles]
        
        return {
            'peak_memory_mb': self.peak_memory,
            'min_memory_mb': min(rss_values),
            'max_memory_mb': max(rss_values),
            'avg_memory_mb': np.mean(rss_values),
            'memory_range_mb': max(rss_values) - min(rss_values),
            'total_profiles': len(self.profiles)
        }


class TestMemoryPerformance:
    """Test suite for memory efficiency and performance"""
    
    def setUp(self):
        """Setup for each test"""
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def create_test_model(self, model_name: str = "prajjwal1/bert-tiny", 
                         num_tasks: int = 2, use_fusion: bool = True) -> MultiTaskModel:
        """Create a test model"""
        config = AutoConfig.from_pretrained(model_name)
        model_config = ModelConfig(base_model=model_name, fusion=use_fusion)
        
        tasks = []
        for i in range(num_tasks):
            if i % 2 == 0:
                task = SingleLabelClassificationTask(
                    name=f"classification_{i}",
                    label_map={0: "negative", 1: "positive"}
                )
            else:
                task = NERTask(
                    name=f"ner_{i}",
                    label_map={0: "O", 1: "B-PER", 2: "I-PER"}
                )
            tasks.append(task)
        
        model = MultiTaskModel(config, model_config, tasks)
        return model
    
    def create_test_batch(self, batch_size: int, seq_len: int, vocab_size: int = 1000) -> Dict[str, torch.Tensor]:
        """Create a test batch"""
        return {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': torch.randint(0, 2, (batch_size,))
        }
    
    def test_memory_usage_scaling(self):
        """Test how memory usage scales with batch size and sequence length"""
        logger.info("Testing memory usage scaling...")
        
        model = self.create_test_model()
        monitor = MemoryMonitor()
        
        # Test different configurations
        configs = [
            (2, 32),   # Small: 2 samples, 32 tokens
            (4, 32),   # Medium: 4 samples, 32 tokens
            (8, 32),   # Large: 8 samples, 32 tokens
            (4, 64),   # Long: 4 samples, 64 tokens
            (4, 128),  # Very long: 4 samples, 128 tokens
        ]
        
        results = {}
        
        for batch_size, seq_len in configs:
            config_name = f"bs{batch_size}_sl{seq_len}"
            logger.info(f"Testing configuration: {config_name}")
            
            with monitor.profile_phase(config_name):
                batch = self.create_test_batch(batch_size, seq_len)
                
                # Forward pass
                start_time = time.time()
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_id=0,
                    labels=batch['labels']
                )
                forward_time = time.time() - start_time
                
                # Backward pass
                start_time = time.time()
                loss = outputs['loss']
                loss.backward()
                backward_time = time.time() - start_time
                
                # Calculate throughput
                total_samples = batch_size
                total_time = forward_time + backward_time
                throughput = total_samples / total_time if total_time > 0 else 0
                
                results[config_name] = PerformanceMetrics(
                    forward_time=forward_time,
                    backward_time=backward_time,
                    memory_peak=monitor.peak_memory,
                    memory_allocated=0,  # Will be filled by monitoring
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    throughput_samples_per_sec=throughput
                )
                
                logger.info(f"  Forward: {forward_time:.4f}s, Backward: {backward_time:.4f}s")
                logger.info(f"  Throughput: {throughput:.2f} samples/sec")
                
                # Clear gradients and cache
                model.zero_grad()
                del outputs, loss
                gc.collect()
        
        # Analyze scaling patterns
        self._analyze_scaling_patterns(results)
        
        return results
    
    def _analyze_scaling_patterns(self, results: Dict[str, PerformanceMetrics]):
        """Analyze memory and time scaling patterns"""
        logger.info("\n📊 SCALING ANALYSIS:")
        
        # Group by sequence length
        seq_len_groups = {}
        for config_name, metrics in results.items():
            seq_len = metrics.sequence_length
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append((metrics.batch_size, metrics))
        
        for seq_len, configs in seq_len_groups.items():
            configs.sort(key=lambda x: x[0])  # Sort by batch size
            logger.info(f"\nSequence Length {seq_len}:")
            
            prev_batch_size, prev_metrics = None, None
            for batch_size, metrics in configs:
                logger.info(f"  Batch {batch_size}: {metrics.throughput_samples_per_sec:.2f} samples/sec")
                
                if prev_metrics:
                    # Calculate scaling efficiency
                    batch_ratio = batch_size / prev_batch_size
                    throughput_ratio = metrics.throughput_samples_per_sec / prev_metrics.throughput_samples_per_sec
                    efficiency = throughput_ratio / batch_ratio
                    
                    logger.info(f"    Scaling efficiency: {efficiency:.2f} (1.0 = perfect linear)")
                    
                    if efficiency < 0.7:
                        logger.warning(f"    ⚠️  Poor scaling efficiency: {efficiency:.2f}")
                
                prev_batch_size, prev_metrics = batch_size, metrics
    
    def test_gradient_accumulation_efficiency(self):
        """Test efficiency of gradient accumulation vs larger batches"""
        logger.info("Testing gradient accumulation efficiency...")
        
        model = self.create_test_model()
        monitor = MemoryMonitor()
        
        # Test configurations
        configs = [
            ("direct_large", 8, 1),      # Direct large batch
            ("accumulated_2x4", 4, 2),  # 2 steps of 4 samples
            ("accumulated_4x2", 2, 4),  # 4 steps of 2 samples
            ("accumulated_8x1", 1, 8),  # 8 steps of 1 sample
        ]
        
        results = {}
        
        for config_name, batch_size, accumulation_steps in configs:
            logger.info(f"Testing: {config_name}")
            
            with monitor.profile_phase(config_name):
                total_loss = 0.0
                start_time = time.time()
                
                model.zero_grad()
                
                for step in range(accumulation_steps):
                    batch = self.create_test_batch(batch_size, 32)
                    
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task_id=0,
                        labels=batch['labels']
                    )
                    
                    loss = outputs['loss'] / accumulation_steps  # Scale loss
                    loss.backward()
                    total_loss += loss.item()
                
                # Simulate optimizer step
                total_time = time.time() - start_time
                
                # Calculate effective batch size and throughput
                effective_batch_size = batch_size * accumulation_steps
                throughput = effective_batch_size / total_time
                
                results[config_name] = {
                    'effective_batch_size': effective_batch_size,
                    'total_time': total_time,
                    'throughput': throughput,
                    'memory_peak': monitor.peak_memory,
                    'total_loss': total_loss
                }
                
                logger.info(f"  Time: {total_time:.4f}s, Throughput: {throughput:.2f} samples/sec")
                logger.info(f"  Memory peak: {monitor.peak_memory:.2f} MB")
                
                model.zero_grad()
                gc.collect()
        
        # Compare efficiency
        baseline = results["direct_large"]
        logger.info("\n📊 GRADIENT ACCUMULATION COMPARISON:")
        logger.info(f"Baseline (direct large): {baseline['throughput']:.2f} samples/sec")
        
        for config_name, result in results.items():
            if config_name != "direct_large":
                efficiency = result['throughput'] / baseline['throughput']
                memory_ratio = result['memory_peak'] / baseline['memory_peak']
                
                logger.info(f"{config_name}:")
                logger.info(f"  Throughput efficiency: {efficiency:.2f}")
                logger.info(f"  Memory ratio: {memory_ratio:.2f}")
                
                if efficiency > 0.8 and memory_ratio < 0.7:
                    logger.info("  ✅ Good tradeoff: similar speed, less memory")
                elif memory_ratio < 0.5:
                    logger.info("  💾 Memory efficient: significant memory savings")
                elif efficiency < 0.5:
                    logger.warning("  ⚠️  Slow: significant speed penalty")
        
        return results
    
    def test_model_size_vs_performance(self):
        """Test performance across different model sizes"""
        logger.info("Testing model size vs performance...")
        
        model_configs = [
            ("prajjwal1/bert-tiny", "BERT-Tiny (4.4M params)"),
            ("prajjwal1/bert-mini", "BERT-Mini (11M params)"),
            ("distilbert-base-uncased", "DistilBERT (66M params)"),
        ]
        
        results = {}
        
        for model_name, description in model_configs:
            logger.info(f"Testing {description}...")
            
            try:
                monitor = MemoryMonitor()
                
                with monitor.profile_phase(f"model_load_{model_name}"):
                    model = self.create_test_model(model_name, num_tasks=2)
                    
                    # Count parameters
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Test performance
                with monitor.profile_phase(f"inference_{model_name}"):
                    batch = self.create_test_batch(4, 32)
                    
                    # Warmup
                    for _ in range(3):
                        _ = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            task_id=0
                        )
                    
                    # Timed inference
                    start_time = time.time()
                    for _ in range(10):
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            task_id=0,
                            labels=batch['labels']
                        )
                    inference_time = (time.time() - start_time) / 10
                
                results[model_name] = {
                    'description': description,
                    'total_params': total_params,
                    'trainable_params': trainable_params,
                    'inference_time': inference_time,
                    'memory_peak': monitor.peak_memory,
                    'samples_per_sec': 4 / inference_time,  # batch_size / time
                    'memory_per_param': monitor.peak_memory / (total_params / 1e6)  # MB per million params
                }
                
                logger.info(f"  Params: {total_params/1e6:.1f}M, Time: {inference_time:.4f}s")
                logger.info(f"  Memory: {monitor.peak_memory:.2f}MB, Throughput: {4/inference_time:.2f} samples/sec")
                
                del model
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to test {model_name}: {e}")
                continue
        
        # Analyze size vs performance tradeoffs
        self._analyze_model_tradeoffs(results)
        
        return results
    
    def _analyze_model_tradeoffs(self, results: Dict[str, Any]):
        """Analyze model size vs performance tradeoffs"""
        logger.info("\n📊 MODEL SIZE vs PERFORMANCE ANALYSIS:")
        
        # Sort by parameter count
        sorted_models = sorted(results.items(), key=lambda x: x[1]['total_params'])
        
        for model_name, metrics in sorted_models:
            efficiency_score = (
                (1 / metrics['inference_time']) *  # Speed factor
                (1 / (metrics['memory_peak'] / 1000))  # Memory efficiency factor
            )
            
            logger.info(f"\n{metrics['description']}:")
            logger.info(f"  Parameters: {metrics['total_params']/1e6:.1f}M")
            logger.info(f"  Speed: {metrics['samples_per_sec']:.2f} samples/sec")
            logger.info(f"  Memory: {metrics['memory_peak']:.2f} MB")
            logger.info(f"  Memory/Param: {metrics['memory_per_param']:.2f} MB/Mparam")
            logger.info(f"  Efficiency Score: {efficiency_score:.4f}")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during training"""
        logger.info("Testing for memory leaks...")
        
        model = self.create_test_model()
        monitor = MemoryMonitor()
        monitor.start_monitoring()
        
        memory_snapshots = []
        
        try:
            # Simulate training loop
            for epoch in range(5):
                logger.info(f"Epoch {epoch + 1}/5")
                
                # Record memory at start of epoch
                process = psutil.Process()
                memory_snapshots.append(process.memory_info().rss / 1024 / 1024)
                
                for batch_idx in range(10):
                    batch = self.create_test_batch(4, 32)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        task_id=batch_idx % 2,
                        labels=batch['labels']
                    )
                    
                    # Backward pass
                    loss = outputs['loss']
                    loss.backward()
                    
                    # Simulate optimizer step
                    model.zero_grad()
                    
                    # Clean up
                    del outputs, loss
                    
                    if batch_idx % 5 == 0:
                        gc.collect()
                
                # Explicit cleanup at end of epoch
                gc.collect()
                
        finally:
            monitor.stop_monitoring()
        
        # Analyze memory trend
        logger.info("\n📊 MEMORY LEAK ANALYSIS:")
        for i, memory_mb in enumerate(memory_snapshots):
            logger.info(f"Epoch {i + 1}: {memory_mb:.2f} MB")
        
        if len(memory_snapshots) >= 3:
            # Calculate memory growth trend
            memory_growth = memory_snapshots[-1] - memory_snapshots[0]
            avg_growth_per_epoch = memory_growth / (len(memory_snapshots) - 1)
            
            logger.info(f"Total memory growth: {memory_growth:.2f} MB")
            logger.info(f"Average growth per epoch: {avg_growth_per_epoch:.2f} MB")
            
            if avg_growth_per_epoch > 50:  # More than 50MB per epoch
                logger.warning("⚠️  Potential memory leak detected!")
                return False
            elif avg_growth_per_epoch > 20:
                logger.warning("⚠️  Moderate memory growth detected")
                return True
            else:
                logger.info("✅ No significant memory leak detected")
                return True
        
        return True
    
    def test_cpu_optimization_techniques(self):
        """Test various CPU optimization techniques"""
        logger.info("Testing CPU optimization techniques...")
        
        optimizations = [
            ("baseline", {}),
            ("mixed_precision", {"use_amp": True}),
            ("reduced_precision", {"dtype": torch.float16}),
            ("cpu_threads", {"num_threads": 4}),
        ]
        
        results = {}
        
        for opt_name, opt_config in optimizations:
            logger.info(f"Testing optimization: {opt_name}")
            
            try:
                # Apply optimizations
                if "num_threads" in opt_config:
                    torch.set_num_threads(opt_config["num_threads"])
                
                model = self.create_test_model()
                
                if "dtype" in opt_config:
                    model = model.to(dtype=opt_config["dtype"])
                
                # Test performance
                batch = self.create_test_batch(4, 32)
                
                if "dtype" in opt_config:
                    for key in batch:
                        if batch[key].dtype == torch.float32:
                            batch[key] = batch[key].to(dtype=opt_config["dtype"])
                
                # Warmup
                for _ in range(3):
                    try:
                        _ = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            task_id=0
                        )
                    except Exception as e:
                        logger.warning(f"Warmup failed for {opt_name}: {e}")
                        break
                else:
                    # Timed run
                    start_time = time.time()
                    for _ in range(10):
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            task_id=0,
                            labels=batch['labels']
                        )
                    elapsed_time = time.time() - start_time
                    
                    results[opt_name] = {
                        'avg_time': elapsed_time / 10,
                        'throughput': 40 / elapsed_time,  # 4 samples * 10 iterations / time
                        'config': opt_config
                    }
                    
                    logger.info(f"  Average time: {results[opt_name]['avg_time']:.4f}s")
                    logger.info(f"  Throughput: {results[opt_name]['throughput']:.2f} samples/sec")
                
                del model
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Optimization {opt_name} failed: {e}")
                continue
        
        # Compare optimizations
        if "baseline" in results:
            baseline_throughput = results["baseline"]["throughput"]
            logger.info("\n📊 CPU OPTIMIZATION COMPARISON:")
            logger.info(f"Baseline: {baseline_throughput:.2f} samples/sec")
            
            for opt_name, result in results.items():
                if opt_name != "baseline":
                    speedup = result["throughput"] / baseline_throughput
                    logger.info(f"{opt_name}: {speedup:.2f}x speedup")
        
        return results


def run_memory_performance_validation():
    """Run the complete memory and performance validation"""
    logger.info("=" * 80)
    logger.info("RUNNING MEMORY EFFICIENCY & PERFORMANCE VALIDATION")
    logger.info("=" * 80)
    
    test_suite = TestMemoryPerformance()
    
    try:
        test_suite.setUp()
        
        # Run all tests
        scaling_results = test_suite.test_memory_usage_scaling()
        logger.info("✅ Memory usage scaling test COMPLETED")
        
        accumulation_results = test_suite.test_gradient_accumulation_efficiency()
        logger.info("✅ Gradient accumulation efficiency test COMPLETED")
        
        model_size_results = test_suite.test_model_size_vs_performance()
        logger.info("✅ Model size vs performance test COMPLETED")
        
        leak_detection = test_suite.test_memory_leak_detection()
        logger.info(f"✅ Memory leak detection: {'PASSED' if leak_detection else 'FAILED'}")
        
        cpu_optimization_results = test_suite.test_cpu_optimization_techniques()
        logger.info("✅ CPU optimization techniques test COMPLETED")
        
        logger.info("=" * 80)
        logger.info("ALL MEMORY & PERFORMANCE TESTS COMPLETED! ✅")
        logger.info("=" * 80)
        
        return {
            'scaling': scaling_results,
            'accumulation': accumulation_results,
            'model_sizes': model_size_results,
            'leak_detection': leak_detection,
            'cpu_optimization': cpu_optimization_results,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Memory and performance validation failed: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    run_memory_performance_validation()