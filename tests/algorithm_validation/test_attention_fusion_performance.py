#!/usr/bin/env python3
"""
Comprehensive Attention Fusion Performance Validation
Tests the actual performance gains of attention fusion mechanism
"""

import pytest
import torch
import torch.nn as nn
import time
import logging
from typing import Dict, List, Tuple
from unittest.mock import MagicMock
import numpy as np

from multitask_bert.core.fusion import AttentionFusion
from multitask_bert.core.model import MultiTaskModel
from multitask_bert.tasks.classification import SingleLabelClassificationTask
from multitask_bert.tasks.ner import NERTask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self, hidden_size=128, num_attention_heads=8):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = 1000
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-12


class MockModelConfig:
    """Mock model configuration"""
    def __init__(self, base_model="prajjwal1/bert-tiny", fusion=True):
        self.base_model = base_model
        self.fusion = fusion
        self.dropout = 0.1


def create_mock_data(batch_size: int = 8, seq_len: int = 32, hidden_size: int = 128) -> torch.Tensor:
    """Create mock training data"""
    return torch.randn(batch_size, seq_len, hidden_size)


def create_mock_tasks() -> List:
    """Create mock tasks for testing"""
    sentiment_task = SingleLabelClassificationTask(
        name="sentiment",
        label_map={0: "negative", 1: "positive"}
    )
    ner_task = NERTask(
        name="ner",
        label_map={0: "O", 1: "B-PER", 2: "I-PER"}
    )
    return [sentiment_task, ner_task]


class TestAttentionFusionPerformance:
    """Test suite for attention fusion performance validation"""
    
    @pytest.fixture
    def mock_config(self):
        return MockConfig()
    
    @pytest.fixture
    def mock_model_config(self):
        return MockModelConfig()
    
    @pytest.fixture
    def fusion_layer(self, mock_config):
        return AttentionFusion(mock_config, num_tasks=2)
    
    def test_fusion_vs_no_fusion_performance(self, mock_config, mock_model_config):
        """Test performance comparison between fusion and no-fusion"""
        logger.info("Testing attention fusion vs no-fusion performance...")
        
        # Create test data
        batch_size, seq_len, hidden_size = 8, 32, mock_config.hidden_size
        shared_representation = create_mock_data(batch_size, seq_len, hidden_size)
        
        # Test with fusion
        fusion_layer = AttentionFusion(mock_config, num_tasks=2)
        
        # Warm up
        for _ in range(5):
            _ = fusion_layer(shared_representation, task_id=0)
        
        # Time with fusion
        start_time = time.time()
        for _ in range(100):
            output_with_fusion = fusion_layer(shared_representation, task_id=0)
        fusion_time = time.time() - start_time
        
        # Time without fusion (direct pass-through)
        start_time = time.time()
        for _ in range(100):
            output_without_fusion = shared_representation
        no_fusion_time = time.time() - start_time
        
        logger.info(f"Fusion time: {fusion_time:.4f}s")
        logger.info(f"No-fusion time: {no_fusion_time:.4f}s")
        logger.info(f"Fusion overhead: {((fusion_time - no_fusion_time) / no_fusion_time) * 100:.2f}%")
        
        # Assertions
        assert output_with_fusion.shape == shared_representation.shape
        assert fusion_time > no_fusion_time  # Fusion should add some overhead
        assert fusion_time < no_fusion_time * 3  # But not more than 3x overhead
        
        # Test that fusion actually changes the representation
        assert not torch.allclose(output_with_fusion, shared_representation, atol=1e-6)
    
    def test_task_specific_attention_weights(self, mock_config):
        """Test that different tasks produce different attention weights"""
        logger.info("Testing task-specific attention differentiation...")
        
        fusion_layer = AttentionFusion(mock_config, num_tasks=3)
        batch_size, seq_len, hidden_size = 4, 16, mock_config.hidden_size
        shared_representation = create_mock_data(batch_size, seq_len, hidden_size)
        
        # Get outputs for different tasks
        outputs = {}
        attention_scores = {}
        
        for task_id in range(3):
            with torch.no_grad():
                # Forward pass
                output = fusion_layer(shared_representation, task_id)
                outputs[task_id] = output
                
                # Extract attention scores for analysis
                task_embedding = fusion_layer.task_embeddings(torch.tensor([task_id]))
                task_embedding_expanded = task_embedding.unsqueeze(0).expand(
                    shared_representation.size(0), shared_representation.size(1), -1
                )
                combined = torch.cat([shared_representation, task_embedding_expanded], dim=2)
                scores = fusion_layer.attention_layer(combined)
                attention_scores[task_id] = scores
        
        # Verify that different tasks produce different outputs
        for i in range(3):
            for j in range(i + 1, 3):
                cosine_sim = torch.nn.functional.cosine_similarity(
                    outputs[i].flatten(), outputs[j].flatten(), dim=0
                )
                logger.info(f"Task {i} vs Task {j} output similarity: {cosine_sim:.4f}")
                
                # Attention scores should be different for different tasks
                attention_sim = torch.nn.functional.cosine_similarity(
                    attention_scores[i].flatten(), attention_scores[j].flatten(), dim=0
                )
                logger.info(f"Task {i} vs Task {j} attention similarity: {attention_sim:.4f}")
                
                # Outputs should not be identical (some differentiation)
                assert cosine_sim < 0.99, f"Tasks {i} and {j} produce too similar outputs"
                assert attention_sim < 0.95, f"Tasks {i} and {j} have too similar attention"
    
    def test_attention_fusion_memory_efficiency(self, mock_config):
        """Test memory efficiency of attention fusion"""
        logger.info("Testing attention fusion memory efficiency...")
        
        fusion_layer = AttentionFusion(mock_config, num_tasks=2)
        
        # Test with increasing batch sizes
        memory_usage = []
        batch_sizes = [2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            seq_len, hidden_size = 64, mock_config.hidden_size
            shared_representation = create_mock_data(batch_size, seq_len, hidden_size)
            
            # Measure memory before
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Forward pass
            output = fusion_layer(shared_representation, task_id=0)
            
            # Measure memory after
            memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            memory_diff = memory_after - memory_before
            
            memory_usage.append(memory_diff)
            logger.info(f"Batch size {batch_size}: Memory usage {memory_diff} bytes")
            
            # Cleanup
            del output, shared_representation
        
        # Memory usage should scale roughly linearly with batch size
        if len(memory_usage) > 1:
            ratios = [memory_usage[i] / memory_usage[i-1] for i in range(1, len(memory_usage))]
            avg_ratio = np.mean(ratios)
            logger.info(f"Average memory scaling ratio: {avg_ratio:.2f}")
            
            # Should scale roughly with batch size (2x batch ≈ 2x memory)
            assert 1.5 < avg_ratio < 3.0, f"Memory scaling seems inefficient: {avg_ratio}"
    
    def test_attention_fusion_convergence(self, mock_config):
        """Test that attention fusion helps with convergence"""
        logger.info("Testing attention fusion convergence properties...")
        
        fusion_layer = AttentionFusion(mock_config, num_tasks=2)
        
        # Create optimizer
        optimizer = torch.optim.Adam(fusion_layer.parameters(), lr=0.001)
        
        # Create dummy target
        batch_size, seq_len, hidden_size = 8, 32, mock_config.hidden_size
        target_output = torch.randn(batch_size, seq_len, hidden_size)
        
        losses = []
        
        # Training loop
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            shared_representation = create_mock_data(batch_size, seq_len, hidden_size)
            output = fusion_layer(shared_representation, task_id=0)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(output, target_output)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss {loss.item():.6f}")
        
        # Check convergence
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        improvement = (initial_loss - final_loss) / initial_loss
        
        logger.info(f"Initial loss: {initial_loss:.6f}")
        logger.info(f"Final loss: {final_loss:.6f}")
        logger.info(f"Improvement: {improvement:.2%}")
        
        # Should show some convergence
        assert improvement > 0.1, f"Fusion layer didn't converge well: {improvement:.2%}"
        assert final_loss < initial_loss * 0.8, "Loss didn't decrease sufficiently"
    
    def test_attention_fusion_gradient_flow(self, mock_config):
        """Test that gradients flow properly through attention fusion"""
        logger.info("Testing attention fusion gradient flow...")
        
        fusion_layer = AttentionFusion(mock_config, num_tasks=2)
        
        # Create input that requires gradients
        batch_size, seq_len, hidden_size = 4, 16, mock_config.hidden_size
        shared_representation = create_mock_data(batch_size, seq_len, hidden_size)
        shared_representation.requires_grad_(True)
        
        # Forward pass
        output = fusion_layer(shared_representation, task_id=0)
        
        # Compute dummy loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert shared_representation.grad is not None
        assert not torch.allclose(shared_representation.grad, torch.zeros_like(shared_representation.grad))
        
        # Check that all fusion layer parameters have gradients
        for name, param in fusion_layer.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Zero gradient for {name}"
        
        logger.info("Gradient flow test passed - all parameters receiving gradients")


def run_performance_validation():
    """Run the full performance validation suite"""
    logger.info("=" * 80)
    logger.info("RUNNING ATTENTION FUSION PERFORMANCE VALIDATION")
    logger.info("=" * 80)
    
    # Initialize test class
    test_class = TestAttentionFusionPerformance()
    mock_config = MockConfig()
    mock_model_config = MockModelConfig()
    
    try:
        # Run tests
        test_class.test_fusion_vs_no_fusion_performance(mock_config, mock_model_config)
        logger.info("✅ Fusion vs no-fusion performance test PASSED")
        
        test_class.test_task_specific_attention_weights(mock_config)
        logger.info("✅ Task-specific attention test PASSED")
        
        test_class.test_attention_fusion_memory_efficiency(mock_config)
        logger.info("✅ Memory efficiency test PASSED")
        
        test_class.test_attention_fusion_convergence(mock_config)
        logger.info("✅ Convergence test PASSED")
        
        test_class.test_attention_fusion_gradient_flow(mock_config)
        logger.info("✅ Gradient flow test PASSED")
        
        logger.info("=" * 80)
        logger.info("ALL ATTENTION FUSION PERFORMANCE TESTS PASSED! ✅")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance validation failed: {str(e)}")
        logger.error("=" * 80)
        return False


if __name__ == "__main__":
    run_performance_validation()