#!/usr/bin/env python3
"""
MultiTask Model Unit Tests
=========================
Tests the core multitask model functionality including shared encoder and task-specific heads.

Usage:
    python -m pytest tests/unit/test_multitask_model.py -v
    OR
    python tests/unit/test_multitask_model.py
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

from multitask_bert.core.model import MultiTaskModel
from multitask_bert.core.fusion import AttentionFusion
from multitask_bert.core.config import ModelConfig, FusionConfig, TaskConfig, HeadConfig
from multitask_bert.tasks.classification import ClassificationTask
from multitask_bert.tasks.ner import NERTask
from multitask_bert.tasks.base import TaskOutput
from transformers import AutoConfig


class TestMultiTaskModel(unittest.TestCase):
    """Test MultiTaskModel core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use tiny model for faster testing
        self.model_name = "prajjwal1/bert-tiny"
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = self.config.hidden_size  # 128 for bert-tiny
        
        # Create test tasks
        self.sentiment_task_config = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path="/dummy/path",
            heads=[HeadConfig(name="sentiment_head", num_labels=3)]
        )
        
        self.ner_task_config = TaskConfig(
            name="ner", 
            type="ner",
            data_path="/dummy/path",
            heads=[HeadConfig(name="ner_head", num_labels=5)]
        )
        
        # Create task instances
        self.tasks = [
            ClassificationTask(self.sentiment_task_config),
            NERTask(self.ner_task_config)
        ]
        
        # Sample inputs
        self.batch_size = 2
        self.seq_length = 10
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones(self.batch_size, self.seq_length)
        
        # Labels for different tasks
        self.sentiment_labels = torch.tensor([1, 2], dtype=torch.long)
        self.ner_labels = [
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0],  # Token-level labels
            [-100, 0, 3, 4, 0, 0, -100, -100, -100, -100]
        ]
    
    def test_model_initialization_without_fusion(self):
        """Test MultiTaskModel initialization without fusion."""
        model_config = ModelConfig(
            base_model=self.model_name,
            dropout=0.1,
            fusion=None
        )
        
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Verify basic attributes
        self.assertIsNotNone(model.encoder)
        self.assertEqual(len(model.tasks), 2)
        self.assertIsNone(model.fusion)
        
        # Verify tasks are ModuleList
        self.assertIsInstance(model.tasks, nn.ModuleList)
        
        # Verify encoder is the correct type
        self.assertEqual(model.encoder.config.model_type, self.config.model_type)
    
    def test_model_initialization_with_fusion(self):
        """Test MultiTaskModel initialization with fusion."""
        fusion_config = FusionConfig(type="attention", hidden_size=self.hidden_size)
        model_config = ModelConfig(
            base_model=self.model_name,
            dropout=0.1,
            fusion=fusion_config
        )
        
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Verify fusion is created
        self.assertIsNotNone(model.fusion)
        self.assertIsInstance(model.fusion, AttentionFusion)
        self.assertEqual(model.fusion.num_tasks, len(self.tasks))
    
    def test_input_embeddings_access(self):
        """Test getting and setting input embeddings."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test getting input embeddings
        embeddings = model.get_input_embeddings()
        self.assertIsInstance(embeddings, nn.Module)
        self.assertEqual(embeddings.num_embeddings, self.config.vocab_size)
        self.assertEqual(embeddings.embedding_dim, self.hidden_size)
        
        # Test setting input embeddings
        new_embeddings = nn.Embedding(self.config.vocab_size, self.hidden_size)
        model.set_input_embeddings(new_embeddings)
        retrieved_embeddings = model.get_input_embeddings()
        self.assertEqual(retrieved_embeddings, new_embeddings)
    
    def test_forward_pass_classification_task(self):
        """Test forward pass for classification task."""
        model_config = ModelConfig(base_model=self.model_name, fusion=None)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Forward pass for sentiment task (task_id=0)
        output = model.forward(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            task_id=0,
            labels=self.sentiment_labels
        )
        
        # Verify output structure
        self.assertIsInstance(output, TaskOutput)
        self.assertIsInstance(output.loss, (float, torch.Tensor))
        self.assertIn("sentiment_head", output.logits)
        
        # Verify logits shape
        logits = output.logits["sentiment_head"]
        expected_shape = (self.batch_size, 3)  # 3 classes for sentiment
        self.assertEqual(logits.shape, expected_shape)
        
        # Verify loss is finite
        if isinstance(output.loss, torch.Tensor):
            self.assertTrue(torch.isfinite(output.loss))
        else:
            self.assertTrue(output.loss != float('inf'))
    
    def test_forward_pass_ner_task(self):
        """Test forward pass for NER task."""
        model_config = ModelConfig(base_model=self.model_name, fusion=None)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Forward pass for NER task (task_id=1)
        output = model.forward(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            task_id=1,
            labels=self.ner_labels
        )
        
        # Verify output structure
        self.assertIsInstance(output, TaskOutput)
        self.assertIsInstance(output.loss, (float, torch.Tensor))
        self.assertIn("ner_head", output.logits)
        
        # Verify logits shape (sequence-level for NER)
        logits = output.logits["ner_head"]
        expected_shape = (self.batch_size, self.seq_length, 5)  # 5 classes for NER
        self.assertEqual(logits.shape, expected_shape)
        
        # Verify loss is finite
        if isinstance(output.loss, torch.Tensor):
            self.assertTrue(torch.isfinite(output.loss))
        else:
            self.assertTrue(output.loss != float('inf'))
    
    def test_forward_pass_with_fusion(self):
        """Test forward pass with attention fusion."""
        fusion_config = FusionConfig(type="attention", hidden_size=self.hidden_size)
        model_config = ModelConfig(
            base_model=self.model_name,
            fusion=fusion_config
        )
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test both tasks with fusion
        for task_id, labels in enumerate([self.sentiment_labels, self.ner_labels]):
            with self.subTest(task_id=task_id):
                output = model.forward(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    task_id=task_id,
                    labels=labels
                )
                
                # Verify output structure
                self.assertIsInstance(output, TaskOutput)
                self.assertIsInstance(output.loss, (float, torch.Tensor))
                self.assertIsInstance(output.logits, dict)
                
                # Verify fusion was applied (no direct way to test, but check it doesn't crash)
                self.assertTrue(len(output.logits) > 0)
    
    def test_forward_pass_without_labels(self):
        """Test forward pass without labels (inference mode)."""
        model_config = ModelConfig(base_model=self.model_name, fusion=None)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Forward pass without labels
        output = model.forward(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            task_id=0,
            labels=None
        )
        
        # Verify output structure
        self.assertIsInstance(output, TaskOutput)
        # Loss might be None or 0 in inference mode
        self.assertTrue(output.loss is None or output.loss == 0 or torch.isfinite(torch.tensor(output.loss)))
        self.assertIn("sentiment_head", output.logits)
    
    def test_invalid_task_id(self):
        """Test handling of invalid task IDs."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test with task_id >= num_tasks
        with self.assertRaises(IndexError):
            model.forward(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                task_id=len(self.tasks),
                labels=None
            )
        
        # Test with negative task_id
        with self.assertRaises(IndexError):
            model.forward(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                task_id=-1,
                labels=None
            )
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test with different sequence lengths
        sequence_lengths = [5, 15, 32]
        
        for seq_len in sequence_lengths:
            with self.subTest(seq_len=seq_len):
                input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, seq_len))
                attention_mask = torch.ones(self.batch_size, seq_len)
                labels = torch.tensor([1, 0], dtype=torch.long)
                
                output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=0,
                    labels=labels
                )
                
                # Verify output is valid
                self.assertIsInstance(output, TaskOutput)
                logits = output.logits["sentiment_head"]
                self.assertEqual(logits.shape[0], self.batch_size)
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, self.seq_length))
                attention_mask = torch.ones(batch_size, self.seq_length)
                labels = torch.randint(0, 3, (batch_size,), dtype=torch.long)
                
                output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=0,
                    labels=labels
                )
                
                # Verify output is valid
                self.assertIsInstance(output, TaskOutput)
                logits = output.logits["sentiment_head"]
                self.assertEqual(logits.shape[0], batch_size)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Create inputs with gradient tracking
        input_ids = self.input_ids.requires_grad_(False)  # Input IDs don't need gradients
        
        # Forward pass
        output = model.forward(
            input_ids=input_ids,
            attention_mask=self.attention_mask,
            task_id=0,
            labels=self.sentiment_labels
        )
        
        # Backward pass
        if isinstance(output.loss, torch.Tensor):
            loss = output.loss
        else:
            loss = sum(output.logits.values()).sum()  # Create dummy loss if needed
        
        loss.backward()
        
        # Check gradients exist for model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")
    
    def test_token_type_ids_handling(self):
        """Test handling of token_type_ids for different model types."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Create token_type_ids
        token_type_ids = torch.zeros(self.batch_size, self.seq_length, dtype=torch.long)
        
        # Test with token_type_ids
        try:
            output = model.forward(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                task_id=0,
                labels=self.sentiment_labels,
                token_type_ids=token_type_ids
            )
            
            # Should work without errors
            self.assertIsInstance(output, TaskOutput)
            
        except Exception as e:
            # Some models don't support token_type_ids, should handle gracefully
            self.assertIn("token_type_ids", str(e).lower())
    
    def test_model_device_compatibility(self):
        """Test model works on different devices."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Test CPU
        model_cpu = model.cpu()
        input_ids_cpu = self.input_ids.cpu()
        attention_mask_cpu = self.attention_mask.cpu()
        labels_cpu = self.sentiment_labels.cpu()
        
        output_cpu = model_cpu.forward(
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu,
            task_id=0,
            labels=labels_cpu
        )
        
        self.assertIsInstance(output_cpu, TaskOutput)
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_ids_gpu = self.input_ids.cuda()
            attention_mask_gpu = self.attention_mask.cuda()
            labels_gpu = self.sentiment_labels.cuda()
            
            output_gpu = model_gpu.forward(
                input_ids=input_ids_gpu,
                attention_mask=attention_mask_gpu,
                task_id=0,
                labels=labels_gpu
            )
            
            self.assertIsInstance(output_gpu, TaskOutput)
    
    def test_model_parameters_count(self):
        """Test model has reasonable number of parameters."""
        model_config = ModelConfig(base_model=self.model_name)
        model = MultiTaskModel(self.config, model_config, self.tasks)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Verify reasonable parameter count for tiny model
        self.assertTrue(total_params > 0)
        self.assertTrue(trainable_params > 0)
        self.assertEqual(total_params, trainable_params)  # All should be trainable
        
        # For bert-tiny, should be around 4-5M parameters
        self.assertTrue(1_000_000 < total_params < 10_000_000)  # Reasonable range


def run_tests():
    """Run all multitask model tests and generate report."""
    print("=" * 70)
    print("  MULTITASK MODEL UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiTaskModel)
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