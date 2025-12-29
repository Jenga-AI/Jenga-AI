#!/usr/bin/env python3
"""
Multi-Task vs Single-Task Learning Benchmark
Tests for negative transfer detection and performance comparison
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import json

from multitask_bert.core.model import MultiTaskModel
from multitask_bert.core.config import ModelConfig, TrainingConfig
from multitask_bert.tasks.classification import SingleLabelClassificationTask
from multitask_bert.tasks.ner import NERTask
from multitask_bert.training.trainer import Trainer
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark experiment"""
    task_name: str
    model_type: str  # 'single_task' or 'multi_task'
    final_accuracy: float
    final_loss: float
    training_time: float
    convergence_epochs: int
    memory_peak: int


class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, size: int = 100, seq_len: int = 32, vocab_size: int = 1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate deterministic data for reproducibility
        torch.manual_seed(42)
        self.input_ids = torch.randint(0, vocab_size, (size, seq_len))
        self.attention_mask = torch.ones(size, seq_len)
        
        # Generate labels based on task
        self.classification_labels = torch.randint(0, 2, (size,))  # Binary classification
        self.ner_labels = torch.randint(0, 3, (size, seq_len))     # 3-class NER
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'classification_labels': self.classification_labels[idx],
            'ner_labels': self.ner_labels[idx]
        }


class MultiTaskBenchmark:
    """Comprehensive multi-task learning benchmark"""
    
    def __init__(self):
        self.results = []
        self.model_name = "prajjwal1/bert-tiny"  # Lightweight model for testing
    
    def create_tasks(self) -> List:
        """Create test tasks"""
        classification_task = SingleLabelClassificationTask(
            name="sentiment_classification",
            label_map={0: "negative", 1: "positive"}
        )
        
        ner_task = NERTask(
            name="entity_recognition",
            label_map={0: "O", 1: "B-PER", 2: "I-PER"}
        )
        
        return [classification_task, ner_task]
    
    def create_single_task_model(self, task) -> nn.Module:
        """Create a single-task model"""
        config = AutoConfig.from_pretrained(self.model_name)
        model_config = ModelConfig(base_model=self.model_name, fusion=False)
        
        model = MultiTaskModel(config, model_config, [task])
        return model
    
    def create_multi_task_model(self, tasks: List, use_fusion: bool = True) -> nn.Module:
        """Create a multi-task model"""
        config = AutoConfig.from_pretrained(self.model_name)
        model_config = ModelConfig(base_model=self.model_name, fusion=use_fusion)
        
        model = MultiTaskModel(config, model_config, tasks)
        return model
    
    def train_model(self, model: nn.Module, dataset: MockDataset, task_name: str, 
                   task_id: int = 0, max_epochs: int = 10) -> BenchmarkResult:
        """Train a model and return benchmark results"""
        logger.info(f"Training model for task: {task_name}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Track metrics
        losses = []
        accuracies = []
        start_time = time.time()
        
        # Simple training loop
        model.train()
        for epoch in range(max_epochs):
            epoch_losses = []
            correct_predictions = 0
            total_predictions = 0
            
            # Create simple dataloader
            indices = torch.randperm(len(dataset))
            batch_size = 8
            
            for batch_start in range(0, len(dataset), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                
                # Prepare batch
                batch_input_ids = dataset.input_ids[batch_indices]
                batch_attention_mask = dataset.attention_mask[batch_indices]
                
                if 'classification' in task_name:
                    batch_labels = dataset.classification_labels[batch_indices]
                else:
                    batch_labels = dataset.ner_labels[batch_indices]
                
                # Forward pass
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    task_id=task_id,
                    labels=batch_labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track metrics
                epoch_losses.append(loss.item())
                
                # Calculate accuracy
                if 'classification' in task_name:
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                    total_predictions += batch_labels.size(0)
                else:
                    # For NER, calculate token-level accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    mask = batch_attention_mask.bool()
                    correct_predictions += ((predictions == batch_labels) & mask).sum().item()
                    total_predictions += mask.sum().item()
            
            # Calculate epoch metrics
            epoch_loss = np.mean(epoch_losses)
            epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch}: Loss {epoch_loss:.4f}, Accuracy {epoch_accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        # Determine convergence epoch (when loss stabilizes)
        convergence_epoch = self._find_convergence_epoch(losses)
        
        return BenchmarkResult(
            task_name=task_name,
            model_type="single_task" if len(model.tasks) == 1 else "multi_task",
            final_accuracy=accuracies[-1],
            final_loss=losses[-1],
            training_time=training_time,
            convergence_epochs=convergence_epoch,
            memory_peak=self._get_memory_usage()
        )
    
    def _find_convergence_epoch(self, losses: List[float], window_size: int = 3) -> int:
        """Find the epoch where training converged"""
        if len(losses) < window_size * 2:
            return len(losses)
        
        for i in range(window_size, len(losses) - window_size):
            window1 = losses[i-window_size:i]
            window2 = losses[i:i+window_size]
            
            if abs(np.mean(window1) - np.mean(window2)) < 0.01:
                return i
        
        return len(losses)
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    
    def benchmark_single_vs_multi_task(self, use_fusion: bool = True) -> Dict[str, Any]:
        """Run comprehensive single vs multi-task benchmark"""
        logger.info("=" * 80)
        logger.info("RUNNING SINGLE-TASK vs MULTI-TASK BENCHMARK")
        logger.info("=" * 80)
        
        results = {
            'single_task': [],
            'multi_task': [],
            'comparison': {},
            'negative_transfer_detected': False
        }
        
        # Create tasks and dataset
        tasks = self.create_tasks()
        dataset = MockDataset(size=200)  # Larger dataset for better benchmarking
        
        # 1. Train single-task models
        logger.info("Training single-task models...")
        for i, task in enumerate(tasks):
            model = self.create_single_task_model(task)
            result = self.train_model(model, dataset, task.name, task_id=0)
            results['single_task'].append(result)
            logger.info(f"Single-task {task.name}: Acc {result.final_accuracy:.4f}, Loss {result.final_loss:.4f}")
        
        # 2. Train multi-task model
        logger.info("Training multi-task model...")
        multi_model = self.create_multi_task_model(tasks, use_fusion=use_fusion)
        
        for i, task in enumerate(tasks):
            result = self.train_model(multi_model, dataset, task.name, task_id=i)
            results['multi_task'].append(result)
            logger.info(f"Multi-task {task.name}: Acc {result.final_accuracy:.4f}, Loss {result.final_loss:.4f}")
        
        # 3. Compare results and detect negative transfer
        for i, task in enumerate(tasks):
            single_result = results['single_task'][i]
            multi_result = results['multi_task'][i]
            
            accuracy_diff = multi_result.final_accuracy - single_result.final_accuracy
            loss_diff = multi_result.final_loss - single_result.final_loss
            
            results['comparison'][task.name] = {
                'accuracy_improvement': accuracy_diff,
                'loss_change': loss_diff,
                'time_ratio': multi_result.training_time / single_result.training_time,
                'memory_ratio': multi_result.memory_peak / single_result.memory_peak
            }
            
            # Detect negative transfer (multi-task performs worse)
            if accuracy_diff < -0.05 or loss_diff > 0.1:
                results['negative_transfer_detected'] = True
                logger.warning(f"⚠️  Negative transfer detected for {task.name}")
                logger.warning(f"   Accuracy drop: {accuracy_diff:.4f}")
                logger.warning(f"   Loss increase: {loss_diff:.4f}")
        
        # 4. Generate summary report
        self._generate_benchmark_report(results)
        
        return results
    
    def benchmark_fusion_impact(self) -> Dict[str, Any]:
        """Benchmark the impact of attention fusion"""
        logger.info("=" * 80)
        logger.info("BENCHMARKING ATTENTION FUSION IMPACT")
        logger.info("=" * 80)
        
        results = {
            'with_fusion': [],
            'without_fusion': [],
            'fusion_improvement': {}
        }
        
        tasks = self.create_tasks()
        dataset = MockDataset(size=200)
        
        # Test with fusion
        logger.info("Training multi-task model WITH fusion...")
        model_with_fusion = self.create_multi_task_model(tasks, use_fusion=True)
        for i, task in enumerate(tasks):
            result = self.train_model(model_with_fusion, dataset, task.name, task_id=i)
            results['with_fusion'].append(result)
        
        # Test without fusion
        logger.info("Training multi-task model WITHOUT fusion...")
        model_without_fusion = self.create_multi_task_model(tasks, use_fusion=False)
        for i, task in enumerate(tasks):
            result = self.train_model(model_without_fusion, dataset, task.name, task_id=i)
            results['without_fusion'].append(result)
        
        # Compare fusion impact
        for i, task in enumerate(tasks):
            with_fusion = results['with_fusion'][i]
            without_fusion = results['without_fusion'][i]
            
            results['fusion_improvement'][task.name] = {
                'accuracy_gain': with_fusion.final_accuracy - without_fusion.final_accuracy,
                'loss_reduction': without_fusion.final_loss - with_fusion.final_loss,
                'convergence_speedup': without_fusion.convergence_epochs - with_fusion.convergence_epochs
            }
            
            logger.info(f"Fusion impact on {task.name}:")
            logger.info(f"  Accuracy: {results['fusion_improvement'][task.name]['accuracy_gain']:+.4f}")
            logger.info(f"  Loss: {results['fusion_improvement'][task.name]['loss_reduction']:+.4f}")
        
        return results
    
    def _generate_benchmark_report(self, results: Dict[str, Any]):
        """Generate a detailed benchmark report"""
        logger.info("=" * 80)
        logger.info("BENCHMARK SUMMARY REPORT")
        logger.info("=" * 80)
        
        logger.info("\n📊 PERFORMANCE COMPARISON:")
        for task_name, comparison in results['comparison'].items():
            logger.info(f"\n{task_name.upper()}:")
            logger.info(f"  Accuracy change: {comparison['accuracy_improvement']:+.4f}")
            logger.info(f"  Loss change: {comparison['loss_change']:+.4f}")
            logger.info(f"  Time ratio: {comparison['time_ratio']:.2f}x")
            logger.info(f"  Memory ratio: {comparison['memory_ratio']:.2f}x")
        
        logger.info(f"\n🔍 NEGATIVE TRANSFER: {'DETECTED' if results['negative_transfer_detected'] else 'NOT DETECTED'}")
        
        # Overall verdict
        avg_accuracy_improvement = np.mean([
            comp['accuracy_improvement'] for comp in results['comparison'].values()
        ])
        
        if avg_accuracy_improvement > 0.01:
            logger.info("✅ VERDICT: Multi-task learning shows POSITIVE transfer!")
        elif avg_accuracy_improvement > -0.01:
            logger.info("⚪ VERDICT: Multi-task learning shows NEUTRAL transfer")
        else:
            logger.info("❌ VERDICT: Multi-task learning shows NEGATIVE transfer")
        
        logger.info("=" * 80)


def run_multitask_benchmark():
    """Run the complete multi-task benchmark suite"""
    benchmark = MultiTaskBenchmark()
    
    try:
        # Run main benchmark
        results = benchmark.benchmark_single_vs_multi_task(use_fusion=True)
        
        # Run fusion benchmark
        fusion_results = benchmark.benchmark_fusion_impact()
        
        logger.info("🎉 ALL MULTI-TASK BENCHMARKS COMPLETED SUCCESSFULLY!")
        
        return {
            'main_benchmark': results,
            'fusion_benchmark': fusion_results,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Multi-task benchmark failed: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    run_multitask_benchmark()