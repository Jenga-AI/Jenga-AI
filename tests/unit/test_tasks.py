#!/usr/bin/env python3
"""
Task Definition Unit Tests
=========================
Tests task creation, validation, and serialization functionality.

Usage:
    python -m pytest tests/unit/test_tasks.py -v
    OR
    python tests/unit/test_tasks.py
"""

import sys
import unittest
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.core.config import TaskConfig, HeadConfig
from multitask_bert.tasks.base import BaseTask, TaskOutput
from multitask_bert.tasks.classification import ClassificationTask
from multitask_bert.tasks.ner import NERTask
from multitask_bert.tasks.sentiment_analysis import SentimentAnalysisTask


class TestTaskConfigurations(unittest.TestCase):
    """Test task configuration creation and validation."""
    
    def test_head_config_creation(self):
        """Test HeadConfig creation and default values."""
        # Test with minimal parameters
        head = HeadConfig(name="test_head", num_labels=3)
        self.assertEqual(head.name, "test_head")
        self.assertEqual(head.num_labels, 3)
        self.assertEqual(head.weight, 1.0)  # Default weight
        
        # Test with custom weight
        head_weighted = HeadConfig(name="weighted_head", num_labels=5, weight=0.8)
        self.assertEqual(head_weighted.weight, 0.8)
    
    def test_task_config_creation(self):
        """Test TaskConfig creation and validation."""
        heads = [HeadConfig(name="classification_head", num_labels=2)]
        
        task = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path="/path/to/data.json",
            heads=heads
        )
        
        self.assertEqual(task.name, "sentiment")
        self.assertEqual(task.type, "single_label_classification")
        self.assertEqual(task.data_path, "/path/to/data.json")
        self.assertEqual(len(task.heads), 1)
        self.assertEqual(task.heads[0].name, "classification_head")
        self.assertIsNone(task.label_maps)  # Default
    
    def test_task_config_with_label_maps(self):
        """Test TaskConfig with label mappings."""
        heads = [HeadConfig(name="ner_head", num_labels=5)]
        label_maps = {
            "ner_head": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
        }
        
        task = TaskConfig(
            name="ner",
            type="ner",
            data_path="/path/to/ner.jsonl",
            heads=heads,
            label_maps=label_maps
        )
        
        self.assertEqual(task.name, "ner")
        self.assertEqual(task.type, "ner")
        self.assertIsNotNone(task.label_maps)
        self.assertIn("ner_head", task.label_maps)
        self.assertEqual(task.label_maps["ner_head"][0], "O")
    
    def test_multi_head_task_config(self):
        """Test TaskConfig with multiple heads."""
        heads = [
            HeadConfig(name="sentiment_head", num_labels=3, weight=1.0),
            HeadConfig(name="toxicity_head", num_labels=2, weight=0.5)
        ]
        
        task = TaskConfig(
            name="multi_classification",
            type="multi_label_classification",
            data_path="/path/to/multi_data.json",
            heads=heads
        )
        
        self.assertEqual(len(task.heads), 2)
        self.assertEqual(task.heads[0].name, "sentiment_head")
        self.assertEqual(task.heads[1].name, "toxicity_head")
        self.assertEqual(task.heads[0].weight, 1.0)
        self.assertEqual(task.heads[1].weight, 0.5)


class TestBaseTask(unittest.TestCase):
    """Test BaseTask abstract class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.task_config = TaskConfig(
            name="test_task",
            type="test_type", 
            data_path="/dummy/path",
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
    
    def test_base_task_initialization(self):
        """Test BaseTask initialization."""
        # BaseTask is abstract, so we need a concrete implementation
        class TestTask(BaseTask):
            def get_forward_output(self, feature, pooled_output, sequence_output, **kwargs):
                return TaskOutput(loss=0.0, logits={"test_head": torch.randn(1, 2)})
        
        task = TestTask(self.task_config)
        
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.type, "test_type")
        self.assertEqual(task.config, self.task_config)
        self.assertIsInstance(task.heads, torch.nn.ModuleDict)
    
    def test_task_output_structure(self):
        """Test TaskOutput dataclass."""
        logits = {"head1": torch.randn(2, 3), "head2": torch.randn(2, 2)}
        output = TaskOutput(loss=0.5, logits=logits)
        
        self.assertEqual(output.loss, 0.5)
        self.assertIn("head1", output.logits)
        self.assertIn("head2", output.logits)
        self.assertEqual(output.logits["head1"].shape, (2, 3))
        self.assertEqual(output.logits["head2"].shape, (2, 2))


class TestClassificationTask(unittest.TestCase):
    """Test ClassificationTask implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.single_label_config = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path="/dummy/path",
            heads=[HeadConfig(name="sentiment_head", num_labels=3)]
        )
        
        self.multi_label_config = TaskConfig(
            name="multi_sentiment",
            type="multi_label_classification", 
            data_path="/dummy/path",
            heads=[
                HeadConfig(name="sentiment_head", num_labels=3),
                HeadConfig(name="emotion_head", num_labels=4)
            ]
        )
    
    def test_single_label_classification_task_creation(self):
        """Test single-label classification task creation."""
        task = ClassificationTask(self.single_label_config)
        
        self.assertEqual(task.name, "sentiment")
        self.assertEqual(task.type, "single_label_classification")
        self.assertIn("sentiment_head", task.heads)
        
        # Verify the head is a Linear layer with correct dimensions
        head = task.heads["sentiment_head"]
        self.assertIsInstance(head, torch.nn.Linear)
        self.assertEqual(head.out_features, 3)
    
    def test_multi_label_classification_task_creation(self):
        """Test multi-label classification task creation."""
        task = ClassificationTask(self.multi_label_config)
        
        self.assertEqual(task.name, "multi_sentiment")
        self.assertEqual(task.type, "multi_label_classification")
        self.assertIn("sentiment_head", task.heads)
        self.assertIn("emotion_head", task.heads)
        
        # Verify both heads are Linear layers with correct dimensions
        sentiment_head = task.heads["sentiment_head"]
        emotion_head = task.heads["emotion_head"]
        
        self.assertIsInstance(sentiment_head, torch.nn.Linear)
        self.assertIsInstance(emotion_head, torch.nn.Linear)
        self.assertEqual(sentiment_head.out_features, 3)
        self.assertEqual(emotion_head.out_features, 4)
    
    def test_single_label_forward_pass(self):
        """Test forward pass for single-label classification."""
        task = ClassificationTask(self.single_label_config)
        
        # Mock inputs
        batch_size = 2
        hidden_size = 768  # Default for BERT
        pooled_output = torch.randn(batch_size, hidden_size)
        sequence_output = torch.randn(batch_size, 10, hidden_size)  # seq_len=10
        
        feature = {
            'labels': torch.tensor([1, 2], dtype=torch.long),
            'input_ids': torch.randint(0, 1000, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10)
        }
        
        try:
            output = task.get_forward_output(feature, pooled_output, sequence_output)
            
            self.assertIsInstance(output, TaskOutput)
            self.assertIsInstance(output.loss, (float, torch.Tensor))
            self.assertIn("sentiment_head", output.logits)
            
            # Verify logits shape
            logits = output.logits["sentiment_head"]
            self.assertEqual(logits.shape, (batch_size, 3))
            
        except Exception as e:
            self.fail(f"Single-label forward pass failed: {e}")
    
    def test_multi_label_forward_pass(self):
        """Test forward pass for multi-label classification."""
        task = ClassificationTask(self.multi_label_config)
        
        # Mock inputs
        batch_size = 2
        hidden_size = 768
        pooled_output = torch.randn(batch_size, hidden_size)
        sequence_output = torch.randn(batch_size, 10, hidden_size)
        
        feature = {
            'labels': {
                'sentiment_head': torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float),
                'emotion_head': torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float)
            },
            'input_ids': torch.randint(0, 1000, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10)
        }
        
        try:
            output = task.get_forward_output(feature, pooled_output, sequence_output)
            
            self.assertIsInstance(output, TaskOutput)
            self.assertIsInstance(output.loss, (float, torch.Tensor))
            self.assertIn("sentiment_head", output.logits)
            self.assertIn("emotion_head", output.logits)
            
            # Verify logits shapes
            sentiment_logits = output.logits["sentiment_head"]
            emotion_logits = output.logits["emotion_head"]
            self.assertEqual(sentiment_logits.shape, (batch_size, 3))
            self.assertEqual(emotion_logits.shape, (batch_size, 4))
            
        except Exception as e:
            self.fail(f"Multi-label forward pass failed: {e}")


class TestNERTask(unittest.TestCase):
    """Test NER task implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ner_config = TaskConfig(
            name="ner",
            type="ner",
            data_path="/dummy/path",
            heads=[HeadConfig(name="ner_head", num_labels=9)],  # B-I-O for 4 entity types
            label_maps={"ner_head": {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC", 
                                   5: "B-ORG", 6: "I-ORG", 7: "B-MISC", 8: "I-MISC"}}
        )
    
    def test_ner_task_creation(self):
        """Test NER task creation."""
        task = NERTask(self.ner_config)
        
        self.assertEqual(task.name, "ner")
        self.assertEqual(task.type, "ner")
        self.assertIn("ner_head", task.heads)
        
        # Verify the head is a Linear layer with correct dimensions
        head = task.heads["ner_head"]
        self.assertIsInstance(head, torch.nn.Linear)
        self.assertEqual(head.out_features, 9)
    
    def test_ner_forward_pass(self):
        """Test forward pass for NER."""
        task = NERTask(self.ner_config)
        
        # Mock inputs
        batch_size = 2
        seq_length = 8
        hidden_size = 768
        pooled_output = torch.randn(batch_size, hidden_size)
        sequence_output = torch.randn(batch_size, seq_length, hidden_size)
        
        # Mock NER labels (token-level)
        feature = {
            'labels': [
                [0, 1, 2, 0, 3, 4, 0, 0],  # O, B-PER, I-PER, O, B-LOC, I-LOC, O, O
                [-100, 0, 5, 6, 0, 0, -100, -100]  # -100 for special tokens, O, B-ORG, I-ORG, O, O
            ],
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length)
        }
        
        try:
            output = task.get_forward_output(feature, pooled_output, sequence_output)
            
            self.assertIsInstance(output, TaskOutput)
            self.assertIsInstance(output.loss, (float, torch.Tensor))
            self.assertIn("ner_head", output.logits)
            
            # Verify logits shape (should be sequence-level)
            logits = output.logits["ner_head"]
            self.assertEqual(logits.shape, (batch_size, seq_length, 9))
            
        except Exception as e:
            self.fail(f"NER forward pass failed: {e}")


class TestSentimentAnalysisTask(unittest.TestCase):
    """Test SentimentAnalysisTask implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sentiment_config = TaskConfig(
            name="sentiment_analysis",
            type="sentiment_analysis",
            data_path="/dummy/path",
            heads=[HeadConfig(name="sentiment_head", num_labels=3)]  # Negative, Neutral, Positive
        )
    
    def test_sentiment_task_creation(self):
        """Test sentiment analysis task creation."""
        task = SentimentAnalysisTask(self.sentiment_config)
        
        self.assertEqual(task.name, "sentiment_analysis")
        self.assertEqual(task.type, "sentiment_analysis")
        self.assertIn("sentiment_head", task.heads)
        
        # Verify the head is a Linear layer with correct dimensions
        head = task.heads["sentiment_head"]
        self.assertIsInstance(head, torch.nn.Linear)
        self.assertEqual(head.out_features, 3)
    
    def test_sentiment_forward_pass(self):
        """Test forward pass for sentiment analysis."""
        task = SentimentAnalysisTask(self.sentiment_config)
        
        # Mock inputs
        batch_size = 2
        hidden_size = 768
        pooled_output = torch.randn(batch_size, hidden_size)
        sequence_output = torch.randn(batch_size, 10, hidden_size)
        
        feature = {
            'labels': torch.tensor([0, 2], dtype=torch.long),  # Negative, Positive
            'input_ids': torch.randint(0, 1000, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10)
        }
        
        try:
            output = task.get_forward_output(feature, pooled_output, sequence_output)
            
            self.assertIsInstance(output, TaskOutput)
            self.assertIsInstance(output.loss, (float, torch.Tensor))
            self.assertIn("sentiment_head", output.logits)
            
            # Verify logits shape
            logits = output.logits["sentiment_head"]
            self.assertEqual(logits.shape, (batch_size, 3))
            
        except Exception as e:
            self.fail(f"Sentiment analysis forward pass failed: {e}")


class TestTaskValidation(unittest.TestCase):
    """Test task validation and error handling."""
    
    def test_invalid_task_config(self):
        """Test handling of invalid task configurations."""
        # Test with zero labels
        with self.assertRaises((ValueError, AssertionError)):
            HeadConfig(name="invalid", num_labels=0)
        
        # Test with negative labels
        with self.assertRaises((ValueError, AssertionError)):
            HeadConfig(name="invalid", num_labels=-1)
    
    def test_mismatched_labels_and_heads(self):
        """Test handling of mismatched labels and head configurations."""
        config = TaskConfig(
            name="mismatched",
            type="multi_label_classification",
            data_path="/dummy/path",
            heads=[
                HeadConfig(name="head1", num_labels=2),
                HeadConfig(name="head2", num_labels=3)
            ]
        )
        
        task = ClassificationTask(config)
        
        # Mock inputs with mismatched label structure
        batch_size = 2
        hidden_size = 768
        pooled_output = torch.randn(batch_size, hidden_size)
        sequence_output = torch.randn(batch_size, 10, hidden_size)
        
        # Missing labels for head2
        feature = {
            'labels': {
                'head1': torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
                # Missing 'head2' labels
            },
            'input_ids': torch.randint(0, 1000, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10)
        }
        
        # Should handle missing labels gracefully or raise clear error
        try:
            output = task.get_forward_output(feature, pooled_output, sequence_output)
            # If it succeeds, verify it handled missing labels appropriately
            self.assertIn("head1", output.logits)
        except (KeyError, ValueError) as e:
            # Expected behavior for missing labels
            self.assertIn("head2", str(e).lower())


def run_tests():
    """Run all task tests and generate report."""
    print("=" * 70)
    print("  TASK DEFINITION UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTaskConfigurations))
    suite.addTests(loader.loadTestsFromTestCase(TestBaseTask))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationTask))
    suite.addTests(loader.loadTestsFromTestCase(TestNERTask))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalysisTask))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskValidation))
    
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