#!/usr/bin/env python3
"""
Attention Fusion Unit Tests
==========================
Tests the attention fusion mechanism functionality.

Usage:
    python -m pytest tests/unit/test_attention_fusion.py -v
    OR
    python tests/unit/test_attention_fusion.py
"""

import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.core.fusion import AttentionFusion
from transformers import AutoConfig


class TestAttentionFusion(unittest.TestCase):
    """Test attention fusion mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        self.hidden_size = self.config.hidden_size  # 128 for bert-tiny
        self.num_tasks = 3
        self.batch_size = 2
        self.seq_length = 10
        
        # Create sample inputs
        self.shared_representation = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size
        )
    
    def test_attention_fusion_initialization(self):
        """Test AttentionFusion layer initialization."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Verify attributes
        self.assertEqual(fusion.num_tasks, self.num_tasks)
        self.assertEqual(fusion.config, self.config)
        
        # Verify components
        self.assertIsInstance(fusion.task_embeddings, nn.Embedding)
        self.assertEqual(fusion.task_embeddings.num_embeddings, self.num_tasks)
        self.assertEqual(fusion.task_embeddings.embedding_dim, self.hidden_size)
        
        # Verify attention layer structure
        self.assertIsInstance(fusion.attention_layer, nn.Sequential)
        self.assertEqual(len(fusion.attention_layer), 3)  # Linear -> Tanh -> Linear
        
        # Check first linear layer input size (hidden_size * 2 for concat)
        first_linear = fusion.attention_layer[0]
        self.assertIsInstance(first_linear, nn.Linear)
        self.assertEqual(first_linear.in_features, self.hidden_size * 2)
        self.assertEqual(first_linear.out_features, self.hidden_size)
        
        # Check activation
        activation = fusion.attention_layer[1]
        self.assertIsInstance(activation, nn.Tanh)
        
        # Check final linear layer
        final_linear = fusion.attention_layer[2]
        self.assertIsInstance(final_linear, nn.Linear)
        self.assertEqual(final_linear.in_features, self.hidden_size)
        self.assertEqual(final_linear.out_features, 1)
    
    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Test with each task ID
        for task_id in range(self.num_tasks):
            with self.subTest(task_id=task_id):
                output = fusion.forward(self.shared_representation, task_id)
                
                # Verify output shape
                expected_shape = self.shared_representation.shape
                self.assertEqual(output.shape, expected_shape)
                
                # Verify output is not identical to input (fusion should modify it)
                self.assertFalse(torch.equal(output, self.shared_representation))
                
                # Verify output is finite
                self.assertTrue(torch.isfinite(output).all())
    
    def test_task_embedding_retrieval(self):
        """Test task embedding retrieval for different task IDs."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Get embeddings for different tasks
        embeddings = []
        for task_id in range(self.num_tasks):
            task_tensor = torch.tensor([task_id])
            embedding = fusion.task_embeddings(task_tensor)
            embeddings.append(embedding)
            
            # Verify embedding shape
            self.assertEqual(embedding.shape, (1, self.hidden_size))
        
        # Verify different tasks have different embeddings (with high probability)
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Embeddings should be different (random initialization)
                self.assertFalse(torch.equal(embeddings[i], embeddings[j]))
    
    def test_attention_computation(self):
        """Test attention mechanism computation."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        task_id = 0
        
        # Get task embedding
        device = self.shared_representation.device
        task_embedding = fusion.task_embeddings(torch.tensor([task_id], device=device))
        
        # Manually compute expected intermediate steps
        task_embedding_expanded = task_embedding.unsqueeze(0).expand(
            self.shared_representation.size(0), 
            self.shared_representation.size(1), 
            -1
        )
        
        # Verify expanded task embedding shape
        expected_expanded_shape = (self.batch_size, self.seq_length, self.hidden_size)
        self.assertEqual(task_embedding_expanded.shape, expected_expanded_shape)
        
        # Test concatenation
        combined_representation = torch.cat(
            [self.shared_representation, task_embedding_expanded], dim=2
        )
        expected_combined_shape = (self.batch_size, self.seq_length, self.hidden_size * 2)
        self.assertEqual(combined_representation.shape, expected_combined_shape)
    
    def test_attention_weights_properties(self):
        """Test properties of attention weights."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Create a modified forward pass to extract attention weights
        task_id = 0
        device = self.shared_representation.device
        
        # Get task embedding and expand
        task_embedding = fusion.task_embeddings(torch.tensor([task_id], device=device))
        task_embedding_expanded = task_embedding.unsqueeze(0).expand(
            self.shared_representation.size(0), 
            self.shared_representation.size(1), 
            -1
        )
        
        # Combine representations
        combined_representation = torch.cat(
            [self.shared_representation, task_embedding_expanded], dim=2
        )
        
        # Compute attention scores
        attention_scores = fusion.attention_layer(combined_representation)
        
        # Apply softmax
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        
        # Test attention weights properties
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_length, 1))
        
        # Attention weights should be non-negative
        self.assertTrue((attention_weights >= 0).all())
        
        # Attention weights should sum to 1 across sequence length for each batch
        for batch_idx in range(self.batch_size):
            attention_sum = attention_weights[batch_idx, :, 0].sum()
            self.assertAlmostEqual(attention_sum.item(), 1.0, places=5)
    
    def test_different_sequence_lengths(self):
        """Test fusion with different sequence lengths."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Test with different sequence lengths
        sequence_lengths = [5, 15, 32]
        
        for seq_len in sequence_lengths:
            with self.subTest(seq_len=seq_len):
                shared_rep = torch.randn(self.batch_size, seq_len, self.hidden_size)
                output = fusion.forward(shared_rep, 0)
                
                # Verify output shape matches input
                self.assertEqual(output.shape, shared_rep.shape)
                self.assertTrue(torch.isfinite(output).all())
    
    def test_different_batch_sizes(self):
        """Test fusion with different batch sizes."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                shared_rep = torch.randn(batch_size, self.seq_length, self.hidden_size)
                output = fusion.forward(shared_rep, 0)
                
                # Verify output shape matches input
                self.assertEqual(output.shape, shared_rep.shape)
                self.assertTrue(torch.isfinite(output).all())
    
    def test_invalid_task_id(self):
        """Test handling of invalid task IDs."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Test with task ID >= num_tasks
        with self.assertRaises(IndexError):
            fusion.forward(self.shared_representation, self.num_tasks)
        
        # Test with negative task ID
        with self.assertRaises(IndexError):
            fusion.forward(self.shared_representation, -1)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the fusion layer."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Create input with gradient tracking
        shared_rep = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size, 
            requires_grad=True
        )
        
        # Forward pass
        output = fusion.forward(shared_rep, 0)
        
        # Create dummy loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist
        self.assertIsNotNone(shared_rep.grad)
        self.assertTrue(torch.isfinite(shared_rep.grad).all())
        
        # Verify fusion parameters have gradients
        for param in fusion.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
    
    def test_fusion_output_difference(self):
        """Test that fusion produces different outputs for different tasks."""
        fusion = AttentionFusion(self.config, self.num_tasks)
        
        # Get outputs for different tasks
        outputs = []
        for task_id in range(min(self.num_tasks, 3)):  # Test first 3 tasks
            output = fusion.forward(self.shared_representation, task_id)
            outputs.append(output)
        
        # Verify outputs are different across tasks
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                # Outputs should be different for different tasks
                self.assertFalse(torch.equal(outputs[i], outputs[j]))
    
    def test_device_compatibility(self):
        """Test fusion works on different devices."""
        # Test CPU
        fusion_cpu = AttentionFusion(self.config, self.num_tasks)
        shared_rep_cpu = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        
        output_cpu = fusion_cpu.forward(shared_rep_cpu, 0)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            fusion_gpu = fusion_cpu.cuda()
            shared_rep_gpu = shared_rep_cpu.cuda()
            
            output_gpu = fusion_gpu.forward(shared_rep_gpu, 0)
            self.assertEqual(output_gpu.device.type, 'cuda')
    
    def test_deterministic_output(self):
        """Test that fusion produces deterministic outputs."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        fusion1 = AttentionFusion(self.config, self.num_tasks)
        
        torch.manual_seed(42)
        fusion2 = AttentionFusion(self.config, self.num_tasks)
        
        # Same input
        shared_rep = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        
        # Should produce same outputs with same seed
        output1 = fusion1.forward(shared_rep.clone(), 0)
        output2 = fusion2.forward(shared_rep.clone(), 0)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


def run_tests():
    """Run all attention fusion tests and generate report."""
    print("=" * 70)
    print("  ATTENTION FUSION UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAttentionFusion)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())