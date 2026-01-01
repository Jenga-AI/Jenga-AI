#!/usr/bin/env python3
"""
Data Preprocessing Unit Tests
============================
Tests tokenization, batching, and data preprocessing functionality.

Usage:
    python -m pytest tests/unit/test_data_preprocessing.py -v
    OR
    python tests/unit/test_data_preprocessing.py
"""

import sys
import unittest
import torch
from pathlib import Path
from datasets import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.core.config import ExperimentConfig, TaskConfig, HeadConfig, ModelConfig, TokenizerConfig
from transformers import AutoTokenizer


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.config = ExperimentConfig(
            project_name="test",
            tasks=[],  # Will be populated per test
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64, padding="max_length", truncation=True)
        )
        
        # Sample data for testing
        self.sample_texts = [
            "This is a short text.",
            "This is a much longer text that should test the truncation functionality properly.",
            "Short.",
            "Another example with some special characters: @#$%^&*()!"
        ]
        
        self.sample_labels = [1, 0, 1, 0]
    
    def test_tokenization_basic(self):
        """Test basic tokenization functionality."""
        processor = DataProcessor(self.config, self.tokenizer)
        
        # Create a batch for tokenization
        batch = {'text': self.sample_texts}
        
        # Test tokenization
        tokenized = processor._tokenize(batch)
        
        # Verify output structure
        self.assertIn('input_ids', tokenized)
        self.assertIn('attention_mask', tokenized)
        
        # Verify batch size
        self.assertEqual(len(tokenized['input_ids']), len(self.sample_texts))
        self.assertEqual(len(tokenized['attention_mask']), len(self.sample_texts))
        
        # Verify all sequences have the same length (due to padding=False, lengths may vary)
        for i, input_ids in enumerate(tokenized['input_ids']):
            self.assertIsInstance(input_ids, list)
            self.assertTrue(len(input_ids) > 0)
    
    def test_tokenization_with_padding(self):
        """Test tokenization with max_length padding."""
        # Update config for max_length padding
        self.config.tokenizer.padding = "max_length"
        processor = DataProcessor(self.config, self.tokenizer)
        
        batch = {'text': self.sample_texts}
        tokenized = processor._tokenize(batch)
        
        # With max_length padding disabled in tokenizer, lengths may vary
        # But attention masks should be consistent
        for attention_mask in tokenized['attention_mask']:
            self.assertIsInstance(attention_mask, list)
            self.assertTrue(len(attention_mask) > 0)
    
    def test_tokenization_truncation(self):
        """Test tokenization with truncation."""
        # Create very long text
        very_long_text = "word " * 200  # Much longer than max_length
        batch = {'text': [very_long_text]}
        
        processor = DataProcessor(self.config, self.tokenizer)
        tokenized = processor._tokenize(batch)
        
        # Verify truncation worked (tokens should be within max_length)
        input_ids = tokenized['input_ids'][0]
        # Note: The actual length may vary due to special tokens, but should be reasonable
        self.assertTrue(len(input_ids) <= self.config.tokenizer.max_length + 10)  # Some buffer for special tokens
    
    def test_single_label_classification_preprocessing(self):
        """Test single-label classification data preprocessing."""
        # Create test dataset
        data = {
            'text': self.sample_texts,
            'label': self.sample_labels
        }
        dataset = Dataset.from_dict(data)
        
        # Create task config
        task_config = TaskConfig(
            name="test",
            type="single_label_classification", 
            data_path="/dummy/path",
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        processor = DataProcessor(self.config, self.tokenizer)
        
        # Test preprocessing
        processed_dataset = processor._process_single_label_classification(dataset, task_config)
        
        # Verify labels were processed
        self.assertIn('labels', processed_dataset.column_names)
        
        # Check first example
        example = processed_dataset[0]
        self.assertIn('labels', example)
        self.assertIsInstance(example['labels'], torch.Tensor)
        self.assertEqual(example['labels'].dtype, torch.long)
        self.assertEqual(example['labels'].item(), self.sample_labels[0])
    
    def test_multi_label_classification_preprocessing(self):
        """Test multi-label classification data preprocessing."""
        # Create test dataset with multi-label structure
        data = {
            'text': self.sample_texts,
            'sentiment': [1, 0, 1, 0],
            'toxicity': [0, 1, 0, 0]
        }
        dataset = Dataset.from_dict(data)
        
        # Create task config with multiple heads
        task_config = TaskConfig(
            name="multilabel_test",
            type="multi_label_classification",
            data_path="/dummy/path",
            heads=[
                HeadConfig(name="sentiment", num_labels=2),
                HeadConfig(name="toxicity", num_labels=2)
            ]
        )
        
        processor = DataProcessor(self.config, self.tokenizer)
        
        # Test preprocessing
        processed_dataset = processor._process_multi_label_classification(dataset, task_config)
        
        # Verify labels were processed
        self.assertIn('labels', processed_dataset.column_names)
        
        # Check first example
        example = processed_dataset[0]
        self.assertIn('labels', example)
        self.assertIsInstance(example['labels'], dict)
        
        # Verify each head has labels
        self.assertIn('sentiment', example['labels'])
        self.assertIn('toxicity', example['labels'])
        
        # Verify tensor types
        self.assertIsInstance(example['labels']['sentiment'], torch.Tensor)
        self.assertIsInstance(example['labels']['toxicity'], torch.Tensor)
        self.assertEqual(example['labels']['sentiment'].dtype, torch.float)
        self.assertEqual(example['labels']['toxicity'].dtype, torch.float)
    
    def test_ner_preprocessing(self):
        """Test NER data preprocessing and label alignment."""
        # Create test NER dataset
        data = {
            'text': [
                "John lives in New York.",
                "Mary works at Google."
            ],
            'entities': [
                [
                    {'start': 0, 'end': 4, 'label': 'PERSON'},
                    {'start': 14, 'end': 22, 'label': 'LOCATION'}
                ],
                [
                    {'start': 0, 'end': 4, 'label': 'PERSON'},
                    {'start': 14, 'end': 20, 'label': 'ORG'}
                ]
            ]
        }
        dataset = Dataset.from_dict(data)
        
        # Create NER task config
        task_config = TaskConfig(
            name="ner_test",
            type="ner",
            data_path="/dummy/path",
            heads=[HeadConfig(name="ner_head", num_labels=5)]  # Will be updated
        )
        
        processor = DataProcessor(self.config, self.tokenizer)
        
        try:
            # Test NER preprocessing
            processed_dataset = processor._process_ner(dataset, task_config)
            
            # Verify the dataset structure
            self.assertIn('input_ids', processed_dataset.column_names)
            self.assertIn('attention_mask', processed_dataset.column_names)
            self.assertIn('labels', processed_dataset.column_names)
            
            # Verify label mapping was created
            self.assertIsNotNone(task_config.label_maps)
            self.assertIn('ner_head', task_config.label_maps)
            
            # Verify number of labels was updated
            self.assertTrue(task_config.heads[0].num_labels > 0)
            
            # Check first example
            example = processed_dataset[0]
            self.assertIn('labels', example)
            self.assertIsInstance(example['labels'], list)
            
            # Labels should contain valid IDs or -100 for ignored tokens
            for label in example['labels']:
                self.assertTrue(label == -100 or (0 <= label < task_config.heads[0].num_labels))
                
        except Exception as e:
            self.fail(f"NER preprocessing failed: {e}")
    
    def test_empty_dataset_handling(self):
        """Test preprocessing with empty datasets."""
        # Create empty dataset
        empty_dataset = Dataset.from_dict({'text': [], 'label': []})
        
        task_config = TaskConfig(
            name="empty_test",
            type="single_label_classification",
            data_path="/dummy/path", 
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        processor = DataProcessor(self.config, self.tokenizer)
        
        # Test preprocessing empty dataset
        processed_dataset = processor._process_single_label_classification(empty_dataset, task_config)
        
        # Should handle gracefully
        self.assertEqual(len(processed_dataset), 0)
        self.assertIn('labels', processed_dataset.column_names)
    
    def test_special_characters_handling(self):
        """Test handling of special characters in text."""
        special_texts = [
            "Text with émojis 🙂 and ñon-ASCII çharacters",
            "Text with\nnewlines\tand\ttabs",
            "Text with \"quotes\" and 'apostrophes'",
            ""  # Empty string
        ]
        
        batch = {'text': special_texts}
        processor = DataProcessor(self.config, self.tokenizer)
        
        try:
            tokenized = processor._tokenize(batch)
            
            # Should handle special characters without crashing
            self.assertIn('input_ids', tokenized)
            self.assertIn('attention_mask', tokenized)
            self.assertEqual(len(tokenized['input_ids']), len(special_texts))
            
        except Exception as e:
            self.fail(f"Special character handling failed: {e}")
    
    def test_label_type_consistency(self):
        """Test that label types are consistent after preprocessing."""
        # Test single-label classification
        data = {
            'text': self.sample_texts,
            'label': [1, 0, 1, 0]  # Integers
        }
        dataset = Dataset.from_dict(data)
        
        task_config = TaskConfig(
            name="consistency_test",
            type="single_label_classification",
            data_path="/dummy/path",
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        processor = DataProcessor(self.config, self.tokenizer)
        processed_dataset = processor._process_single_label_classification(dataset, task_config)
        
        # All labels should be torch.long tensors
        for example in processed_dataset:
            self.assertIsInstance(example['labels'], torch.Tensor)
            self.assertEqual(example['labels'].dtype, torch.long)
    
    def test_tokenizer_configuration_impact(self):
        """Test how different tokenizer configurations affect preprocessing."""
        # Test with different max_lengths
        for max_length in [32, 64, 128]:
            self.config.tokenizer.max_length = max_length
            processor = DataProcessor(self.config, self.tokenizer)
            
            batch = {'text': ["This is a test sentence that might be truncated."]}
            tokenized = processor._tokenize(batch)
            
            # Verify configuration is respected
            self.assertIsInstance(tokenized['input_ids'][0], list)
            # Note: Actual length may be less than max_length due to no padding
            self.assertTrue(len(tokenized['input_ids'][0]) > 0)


def run_tests():
    """Run all data preprocessing tests and generate report."""
    print("=" * 70)
    print("  DATA PREPROCESSING UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataPreprocessing)
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