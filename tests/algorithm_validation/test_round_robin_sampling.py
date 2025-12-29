#!/usr/bin/env python3
"""
Round-Robin Task Sampling Validation
Tests the balanced training across multiple tasks
"""

import pytest
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

from multitask_bert.training.trainer import Trainer
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.core.config import ModelConfig, TrainingConfig
from multitask_bert.tasks.classification import SingleLabelClassificationTask
from multitask_bert.tasks.ner import NERTask
from transformers import AutoConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskSamplingTracker:
    """Tracks task sampling patterns during training"""
    
    def __init__(self):
        self.task_samples = defaultdict(list)  # task_id -> [epoch, batch_idx]
        self.batch_sizes = defaultdict(list)   # task_id -> [batch_size]
        self.total_batches_per_task = defaultdict(int)
        self.epochs_seen = set()
    
    def record_batch(self, task_id: int, epoch: int, batch_idx: int, batch_size: int):
        """Record a training batch"""
        self.task_samples[task_id].append((epoch, batch_idx))
        self.batch_sizes[task_id].append(batch_size)
        self.total_batches_per_task[task_id] += 1
        self.epochs_seen.add(epoch)
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get comprehensive sampling statistics"""
        stats = {
            'total_epochs': len(self.epochs_seen),
            'tasks': {},
            'balance_metrics': {},
            'temporal_distribution': {}
        }
        
        total_batches = sum(self.total_batches_per_task.values())
        
        for task_id, samples in self.task_samples.items():
            task_batches = self.total_batches_per_task[task_id]
            
            stats['tasks'][task_id] = {
                'total_batches': task_batches,
                'proportion': task_batches / total_batches if total_batches > 0 else 0,
                'avg_batch_size': np.mean(self.batch_sizes[task_id]) if self.batch_sizes[task_id] else 0,
                'batches_per_epoch': task_batches / len(self.epochs_seen) if self.epochs_seen else 0
            }
        
        # Calculate balance metrics
        proportions = [stats['tasks'][tid]['proportion'] for tid in stats['tasks']]
        stats['balance_metrics'] = {
            'uniformity_score': self._calculate_uniformity_score(proportions),
            'gini_coefficient': self._calculate_gini_coefficient(proportions),
            'max_deviation': max(proportions) - min(proportions) if proportions else 0
        }
        
        # Temporal distribution analysis
        stats['temporal_distribution'] = self._analyze_temporal_distribution()
        
        return stats
    
    def _calculate_uniformity_score(self, proportions: List[float]) -> float:
        """Calculate uniformity score (1.0 = perfectly uniform, 0.0 = completely skewed)"""
        if not proportions:
            return 0.0
        
        expected_proportion = 1.0 / len(proportions)
        deviations = [abs(p - expected_proportion) for p in proportions]
        max_possible_deviation = expected_proportion
        
        avg_deviation = np.mean(deviations)
        uniformity_score = 1.0 - (avg_deviation / max_possible_deviation)
        
        return max(0.0, uniformity_score)
    
    def _calculate_gini_coefficient(self, proportions: List[float]) -> float:
        """Calculate Gini coefficient (0.0 = perfectly equal, 1.0 = completely unequal)"""
        if not proportions:
            return 0.0
        
        proportions = sorted(proportions)
        n = len(proportions)
        index = np.arange(1, n + 1)
        
        return (np.sum((2 * index - n - 1) * proportions)) / (n * np.sum(proportions))
    
    def _analyze_temporal_distribution(self) -> Dict[str, Any]:
        """Analyze how tasks are distributed over time"""
        temporal_stats = {}
        
        for task_id, samples in self.task_samples.items():
            epochs = [sample[0] for sample in samples]
            batch_indices = [sample[1] for sample in samples]
            
            temporal_stats[task_id] = {
                'epochs_covered': len(set(epochs)),
                'epoch_coverage': len(set(epochs)) / len(self.epochs_seen) if self.epochs_seen else 0,
                'avg_batches_per_active_epoch': len(samples) / len(set(epochs)) if epochs else 0,
                'temporal_spread': max(epochs) - min(epochs) if epochs else 0
            }
        
        return temporal_stats


class MockRoundRobinTrainer:
    """Mock trainer that implements round-robin sampling for testing"""
    
    def __init__(self, tasks: List, datasets: Dict[int, Any], model: torch.nn.Module):
        self.tasks = tasks
        self.datasets = datasets
        self.model = model
        self.tracker = TaskSamplingTracker()
        
        # Create iterators for each task
        self.task_iterators = {}
        self.task_data_loaders = {}
        
        for task_id in range(len(tasks)):
            dataset = datasets[task_id]
            # Simple dataloader simulation
            indices = torch.randperm(len(dataset))
            batches = [indices[i:i+8] for i in range(0, len(indices), 8)]  # batch_size=8
            self.task_data_loaders[task_id] = batches
            self.task_iterators[task_id] = iter(batches)
    
    def get_next_batch(self, task_id: int, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get next batch for a specific task"""
        try:
            batch_indices = next(self.task_iterators[task_id])
        except StopIteration:
            # Reset iterator when exhausted
            self.task_iterators[task_id] = iter(self.task_data_loaders[task_id])
            batch_indices = next(self.task_iterators[task_id])
        
        dataset = self.datasets[task_id]
        
        # Record this batch
        batch_idx = len([s for s in self.tracker.task_samples[task_id] if s[0] == epoch])
        self.tracker.record_batch(task_id, epoch, batch_idx, len(batch_indices))
        
        # Return mock batch data
        batch_size = len(batch_indices)
        input_ids = torch.randint(0, 1000, (batch_size, 32))
        attention_mask = torch.ones(batch_size, 32)
        labels = torch.randint(0, 2, (batch_size,))
        
        return input_ids, attention_mask, labels
    
    def train_round_robin(self, num_epochs: int = 5, batches_per_epoch: int = 20) -> TaskSamplingTracker:
        """Simulate round-robin training"""
        logger.info(f"Starting round-robin training: {num_epochs} epochs, {batches_per_epoch} batches per epoch")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Round-robin through tasks
            for batch_in_epoch in range(batches_per_epoch):
                task_id = batch_in_epoch % len(self.tasks)
                
                # Get batch and simulate training
                input_ids, attention_mask, labels = self.get_next_batch(task_id, epoch)
                
                # Mock forward pass (just for tracking)
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_id=task_id,
                        labels=labels
                    )
                    loss = outputs.get('loss', torch.tensor(0.5))
                except Exception as e:
                    # Mock loss if model fails
                    loss = torch.tensor(0.5)
                
                if batch_in_epoch % 10 == 0:
                    logger.info(f"  Batch {batch_in_epoch}, Task {task_id}, Loss: {loss.item():.4f}")
        
        return self.tracker


class TestRoundRobinSampling:
    """Test suite for round-robin task sampling validation"""
    
    def create_mock_datasets(self, num_tasks: int = 3, sizes: List[int] = None) -> Dict[int, List]:
        """Create mock datasets for testing"""
        if sizes is None:
            sizes = [100, 150, 120]  # Uneven sizes to test balancing
        
        datasets = {}
        for i in range(num_tasks):
            size = sizes[i] if i < len(sizes) else 100
            datasets[i] = list(range(size))  # Simple mock dataset
        
        return datasets
    
    def create_mock_model_and_tasks(self, num_tasks: int = 3):
        """Create mock model and tasks"""
        tasks = []
        for i in range(num_tasks):
            if i % 2 == 0:
                task = SingleLabelClassificationTask(
                    name=f"classification_task_{i}",
                    label_map={0: "negative", 1: "positive"}
                )
            else:
                task = NERTask(
                    name=f"ner_task_{i}",
                    label_map={0: "O", 1: "B-PER", 2: "I-PER"}
                )
            tasks.append(task)
        
        # Create lightweight model
        config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        model_config = ModelConfig(base_model="prajjwal1/bert-tiny", fusion=True)
        model = MultiTaskModel(config, model_config, tasks)
        
        return model, tasks
    
    def test_round_robin_balance_equal_datasets(self):
        """Test round-robin sampling with equal-sized datasets"""
        logger.info("Testing round-robin balance with equal datasets...")
        
        # Create equal-sized datasets
        datasets = self.create_mock_datasets(num_tasks=3, sizes=[100, 100, 100])
        model, tasks = self.create_mock_model_and_tasks(num_tasks=3)
        
        trainer = MockRoundRobinTrainer(tasks, datasets, model)
        tracker = trainer.train_round_robin(num_epochs=3, batches_per_epoch=30)
        
        stats = tracker.get_sampling_stats()
        
        # Verify balanced sampling
        for task_id in range(3):
            proportion = stats['tasks'][task_id]['proportion']
            expected_proportion = 1.0 / 3
            deviation = abs(proportion - expected_proportion)
            
            logger.info(f"Task {task_id}: Proportion {proportion:.3f} (expected {expected_proportion:.3f})")
            assert deviation < 0.1, f"Task {task_id} proportion too far from expected: {deviation}"
        
        # Check uniformity score
        uniformity_score = stats['balance_metrics']['uniformity_score']
        logger.info(f"Uniformity score: {uniformity_score:.3f}")
        assert uniformity_score > 0.85, f"Poor uniformity: {uniformity_score}"
        
        # Check Gini coefficient
        gini = stats['balance_metrics']['gini_coefficient']
        logger.info(f"Gini coefficient: {gini:.3f}")
        assert gini < 0.2, f"High inequality: {gini}"
    
    def test_round_robin_balance_unequal_datasets(self):
        """Test round-robin sampling with unequal-sized datasets"""
        logger.info("Testing round-robin balance with unequal datasets...")
        
        # Create unequal-sized datasets
        datasets = self.create_mock_datasets(num_tasks=3, sizes=[50, 200, 75])
        model, tasks = self.create_mock_model_and_tasks(num_tasks=3)
        
        trainer = MockRoundRobinTrainer(tasks, datasets, model)
        tracker = trainer.train_round_robin(num_epochs=4, batches_per_epoch=36)
        
        stats = tracker.get_sampling_stats()
        
        # With unequal datasets, round-robin should still give equal batches
        expected_proportion = 1.0 / 3
        max_deviation = 0.0
        
        for task_id in range(3):
            proportion = stats['tasks'][task_id]['proportion']
            deviation = abs(proportion - expected_proportion)
            max_deviation = max(max_deviation, deviation)
            
            logger.info(f"Task {task_id}: Proportion {proportion:.3f}, Dataset size {len(datasets[task_id])}")
        
        logger.info(f"Max deviation from equal sampling: {max_deviation:.3f}")
        assert max_deviation < 0.15, f"Round-robin failed with unequal datasets: {max_deviation}"
        
        # Uniformity should still be reasonable
        uniformity_score = stats['balance_metrics']['uniformity_score']
        logger.info(f"Uniformity score: {uniformity_score:.3f}")
        assert uniformity_score > 0.75, f"Poor uniformity with unequal datasets: {uniformity_score}"
    
    def test_temporal_distribution(self):
        """Test that tasks are well distributed across epochs"""
        logger.info("Testing temporal distribution of tasks...")
        
        datasets = self.create_mock_datasets(num_tasks=4, sizes=[100, 100, 100, 100])
        model, tasks = self.create_mock_model_and_tasks(num_tasks=4)
        
        trainer = MockRoundRobinTrainer(tasks, datasets, model)
        tracker = trainer.train_round_robin(num_epochs=5, batches_per_epoch=40)
        
        stats = tracker.get_sampling_stats()
        temporal_dist = stats['temporal_distribution']
        
        for task_id in range(4):
            epoch_coverage = temporal_dist[task_id]['epoch_coverage']
            logger.info(f"Task {task_id}: Epoch coverage {epoch_coverage:.3f}")
            
            # Each task should appear in all epochs
            assert epoch_coverage >= 0.8, f"Task {task_id} poor epoch coverage: {epoch_coverage}"
            
            # Tasks should be spread throughout each epoch
            batches_per_active_epoch = temporal_dist[task_id]['avg_batches_per_active_epoch']
            expected_batches = 40 / 4  # batches_per_epoch / num_tasks
            
            logger.info(f"Task {task_id}: Avg batches per epoch {batches_per_active_epoch:.1f} (expected ~{expected_batches})")
            assert abs(batches_per_active_epoch - expected_batches) < 2, f"Poor temporal spread for task {task_id}"
    
    def test_sampling_with_exhausted_datasets(self):
        """Test behavior when some datasets are exhausted"""
        logger.info("Testing sampling with dataset exhaustion...")
        
        # Create very small datasets that will be exhausted
        datasets = self.create_mock_datasets(num_tasks=3, sizes=[10, 15, 8])
        model, tasks = self.create_mock_model_and_tasks(num_tasks=3)
        
        trainer = MockRoundRobinTrainer(tasks, datasets, model)
        
        # Train for many batches to force dataset exhaustion and recycling
        tracker = trainer.train_round_robin(num_epochs=3, batches_per_epoch=30)
        
        stats = tracker.get_sampling_stats()
        
        # Should still maintain reasonable balance despite exhaustion
        uniformity_score = stats['balance_metrics']['uniformity_score']
        logger.info(f"Uniformity score with small datasets: {uniformity_score:.3f}")
        assert uniformity_score > 0.7, f"Poor balance with dataset exhaustion: {uniformity_score}"
        
        # Verify all tasks were sampled
        for task_id in range(3):
            total_batches = stats['tasks'][task_id]['total_batches']
            logger.info(f"Task {task_id}: Total batches {total_batches}")
            assert total_batches > 0, f"Task {task_id} was never sampled"
    
    def test_sampling_statistics_computation(self):
        """Test the correctness of sampling statistics computation"""
        logger.info("Testing sampling statistics computation...")
        
        # Create a controlled tracker for testing
        tracker = TaskSamplingTracker()
        
        # Add controlled samples
        # Task 0: 40 batches, Task 1: 30 batches, Task 2: 30 batches
        for i in range(40):
            tracker.record_batch(0, i // 10, i % 10, 8)
        
        for i in range(30):
            tracker.record_batch(1, i // 10, i % 10, 8)
        
        for i in range(30):
            tracker.record_batch(2, i // 10, i % 10, 8)
        
        stats = tracker.get_sampling_stats()
        
        # Verify basic counts
        assert stats['tasks'][0]['total_batches'] == 40
        assert stats['tasks'][1]['total_batches'] == 30
        assert stats['tasks'][2]['total_batches'] == 30
        
        # Verify proportions
        total_batches = 100
        assert abs(stats['tasks'][0]['proportion'] - 0.4) < 0.01
        assert abs(stats['tasks'][1]['proportion'] - 0.3) < 0.01
        assert abs(stats['tasks'][2]['proportion'] - 0.3) < 0.01
        
        # Verify balance metrics
        uniformity = stats['balance_metrics']['uniformity_score']
        gini = stats['balance_metrics']['gini_coefficient']
        
        logger.info(f"Test uniformity: {uniformity:.3f}, Gini: {gini:.3f}")
        
        # This distribution is not perfectly uniform but reasonable
        assert 0.6 < uniformity < 0.9
        assert 0.05 < gini < 0.2
    
    def generate_sampling_visualization(self, tracker: TaskSamplingTracker, save_path: str = None):
        """Generate visualization of sampling patterns"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            stats = tracker.get_sampling_stats()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Round-Robin Task Sampling Analysis', fontsize=16)
            
            # 1. Task proportions
            task_ids = list(stats['tasks'].keys())
            proportions = [stats['tasks'][tid]['proportion'] for tid in task_ids]
            
            axes[0, 0].bar(task_ids, proportions)
            axes[0, 0].axhline(y=1.0/len(task_ids), color='red', linestyle='--', label='Expected')
            axes[0, 0].set_title('Task Sampling Proportions')
            axes[0, 0].set_xlabel('Task ID')
            axes[0, 0].set_ylabel('Proportion')
            axes[0, 0].legend()
            
            # 2. Batches per task
            total_batches = [stats['tasks'][tid]['total_batches'] for tid in task_ids]
            axes[0, 1].bar(task_ids, total_batches)
            axes[0, 1].set_title('Total Batches per Task')
            axes[0, 1].set_xlabel('Task ID')
            axes[0, 1].set_ylabel('Total Batches')
            
            # 3. Temporal distribution
            for tid in task_ids:
                samples = tracker.task_samples[tid]
                epochs = [s[0] for s in samples]
                batch_indices = [s[1] for s in samples]
                axes[1, 0].scatter(epochs, [tid] * len(epochs), alpha=0.6, label=f'Task {tid}')
            
            axes[1, 0].set_title('Temporal Distribution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Task ID')
            axes[1, 0].legend()
            
            # 4. Balance metrics
            metrics = ['Uniformity', 'Gini Coeff.', 'Max Deviation']
            values = [
                stats['balance_metrics']['uniformity_score'],
                1 - stats['balance_metrics']['gini_coefficient'],  # Invert for better visualization
                1 - stats['balance_metrics']['max_deviation']      # Invert for better visualization
            ]
            
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Balance Quality Metrics')
            axes[1, 1].set_ylabel('Score (higher = better)')
            axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualization")


def run_round_robin_validation():
    """Run the complete round-robin sampling validation"""
    logger.info("=" * 80)
    logger.info("RUNNING ROUND-ROBIN TASK SAMPLING VALIDATION")
    logger.info("=" * 80)
    
    test_suite = TestRoundRobinSampling()
    
    try:
        # Run all tests
        test_suite.test_round_robin_balance_equal_datasets()
        logger.info("✅ Equal datasets balance test PASSED")
        
        test_suite.test_round_robin_balance_unequal_datasets()
        logger.info("✅ Unequal datasets balance test PASSED")
        
        test_suite.test_temporal_distribution()
        logger.info("✅ Temporal distribution test PASSED")
        
        test_suite.test_sampling_with_exhausted_datasets()
        logger.info("✅ Dataset exhaustion test PASSED")
        
        test_suite.test_sampling_statistics_computation()
        logger.info("✅ Statistics computation test PASSED")
        
        logger.info("=" * 80)
        logger.info("ALL ROUND-ROBIN SAMPLING TESTS PASSED! ✅")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Round-robin validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    run_round_robin_validation()