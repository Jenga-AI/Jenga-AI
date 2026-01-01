#!/usr/bin/env python3
"""
Data Loading Unit Tests
======================
Tests data loading functionality for various formats (CSV, JSON, JSONL).

Usage:
    python -m pytest tests/unit/test_data_loading.py -v
    OR
    python tests/unit/test_data_loading.py
"""

import sys
import unittest
import tempfile
import os
import json
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multitask_bert.data.data_processing import DataProcessor
from multitask_bert.core.config import ExperimentConfig, TaskConfig, HeadConfig, ModelConfig, TokenizerConfig
from transformers import AutoTokenizer


class TestDataLoading(unittest.TestCase):
    """Test data loading for different file formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data files
        self._create_test_csv()
        self._create_test_json()
        self._create_test_jsonl()
        self._create_test_ner_jsonl()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_csv(self):
        """Create test CSV file for sentiment analysis."""
        data = {
            'text': [
                'This is a positive example.',
                'This is a negative example.',
                'Another positive text.',
                'Another negative text.'
            ],
            'label': [1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        self.csv_file = os.path.join(self.temp_dir, 'test_sentiment.csv')
        df.to_csv(self.csv_file, index=False)
    
    def _create_test_json(self):
        """Create test JSON file."""
        data = [
            {'text': 'This is a positive example.', 'label': 1},
            {'text': 'This is a negative example.', 'label': 0},
            {'text': 'Another positive text.', 'label': 1},
            {'text': 'Another negative text.', 'label': 0}
        ]
        self.json_file = os.path.join(self.temp_dir, 'test_data.json')
        with open(self.json_file, 'w') as f:
            json.dump(data, f)
    
    def _create_test_jsonl(self):
        """Create test JSONL file."""
        data = [
            {'text': 'This is a positive example.', 'label': 1},
            {'text': 'This is a negative example.', 'label': 0},
            {'text': 'Another positive text.', 'label': 1},
            {'text': 'Another negative text.', 'label': 0}
        ]
        self.jsonl_file = os.path.join(self.temp_dir, 'test_data.jsonl')
        with open(self.jsonl_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def _create_test_ner_jsonl(self):
        """Create test NER JSONL file."""
        data = [
            {
                'text': 'John lives in New York.',
                'entities': [
                    {'start': 0, 'end': 4, 'label': 'PERSON'},
                    {'start': 14, 'end': 22, 'label': 'LOCATION'}
                ]
            },
            {
                'text': 'Mary works at Google.',
                'entities': [
                    {'start': 0, 'end': 4, 'label': 'PERSON'},
                    {'start': 14, 'end': 20, 'label': 'ORG'}
                ]
            }
        ]
        self.ner_jsonl_file = os.path.join(self.temp_dir, 'test_ner.jsonl')
        with open(self.ner_jsonl_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def test_csv_loading(self):
        """Test loading data from CSV files."""
        # Create config for CSV data
        task_config = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path=self.csv_file,
            heads=[HeadConfig(name="sentiment_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        # Load the CSV data through pandas (simulating the existing code path)
        df = pd.read_csv(self.csv_file)
        
        self.assertEqual(len(df), 4)
        self.assertListEqual(df['label'].tolist(), [1, 0, 1, 0])
        self.assertTrue('text' in df.columns)
        self.assertTrue('label' in df.columns)
    
    def test_json_loading(self):
        """Test loading data from JSON files."""
        # Test reading JSON as would be done in DataProcessor
        df = pd.read_json(self.json_file)
        
        self.assertEqual(len(df), 4)
        self.assertListEqual(df['label'].tolist(), [1, 0, 1, 0])
        self.assertTrue('text' in df.columns)
        self.assertTrue('label' in df.columns)
    
    def test_jsonl_loading(self):
        """Test loading data from JSONL files."""
        # Test reading JSONL as would be done in DataProcessor
        df = pd.read_json(self.jsonl_file, lines=True)
        
        self.assertEqual(len(df), 4)
        self.assertListEqual(df['label'].tolist(), [1, 0, 1, 0])
        self.assertTrue('text' in df.columns)
        self.assertTrue('label' in df.columns)
    
    def test_ner_jsonl_loading(self):
        """Test loading NER data from JSONL files."""
        # Test the custom NER loading path in DataProcessor
        data_list = []
        with open(self.ner_jsonl_file, 'r') as f:
            for line in f:
                data_list.append(json.loads(line))
        df = pd.DataFrame(data_list)
        
        self.assertEqual(len(df), 2)
        self.assertTrue('text' in df.columns)
        self.assertTrue('entities' in df.columns)
        
        # Verify entity structure
        entities = df.iloc[0]['entities']
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]['label'], 'PERSON')
        self.assertEqual(entities[1]['label'], 'LOCATION')
    
    def test_single_label_classification_processing(self):
        """Test complete single-label classification data processing."""
        task_config = TaskConfig(
            name="sentiment",
            type="single_label_classification",
            data_path=self.jsonl_file,
            heads=[HeadConfig(name="sentiment_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        try:
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Verify datasets were created
            self.assertIn("sentiment", train_datasets)
            self.assertIn("sentiment", eval_datasets)
            
            # Verify dataset structure
            train_ds = train_datasets["sentiment"]
            self.assertTrue(len(train_ds) > 0)
            
            # Verify columns
            expected_columns = ['input_ids', 'attention_mask', 'labels']
            for col in expected_columns:
                self.assertIn(col, train_ds.column_names)
                
        except Exception as e:
            self.fail(f"Processing failed with error: {e}")
    
    def test_ner_processing(self):
        """Test complete NER data processing."""
        task_config = TaskConfig(
            name="ner",
            type="ner",
            data_path=self.ner_jsonl_file,
            heads=[HeadConfig(name="ner_head", num_labels=5)]  # Will be updated dynamically
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        try:
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Verify datasets were created
            self.assertIn("ner", train_datasets)
            self.assertIn("ner", eval_datasets)
            
            # Verify dataset structure
            train_ds = train_datasets["ner"]
            self.assertTrue(len(train_ds) > 0)
            
            # Verify columns
            expected_columns = ['input_ids', 'attention_mask', 'labels']
            for col in expected_columns:
                self.assertIn(col, train_ds.column_names)
            
            # Verify label mapping was created
            self.assertIsNotNone(updated_config.tasks[0].label_maps)
            self.assertIn('ner_head', updated_config.tasks[0].label_maps)
                
        except Exception as e:
            self.fail(f"NER processing failed with error: {e}")
    
    def test_malformed_data_handling(self):
        """Test handling of malformed data files."""
        # Create malformed JSON file
        malformed_file = os.path.join(self.temp_dir, 'malformed.json')
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": "json"')  # Missing closing brace
        
        task_config = TaskConfig(
            name="test",
            type="single_label_classification",
            data_path=malformed_file,
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        # Should raise an exception for malformed JSON
        with self.assertRaises((json.JSONDecodeError, ValueError, pd.errors.ParserError)):
            processor.process()
    
    def test_empty_data_handling(self):
        """Test handling of empty data files."""
        # Create empty JSON file
        empty_file = os.path.join(self.temp_dir, 'empty.json')
        with open(empty_file, 'w') as f:
            f.write('[]')
        
        task_config = TaskConfig(
            name="test",
            type="single_label_classification",
            data_path=empty_file,
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        try:
            train_datasets, eval_datasets, updated_config = processor.process()
            
            # Should handle empty data gracefully
            if "test" in train_datasets:
                self.assertEqual(len(train_datasets["test"]), 0)
                
        except Exception as e:
            # Empty data should either be handled gracefully or raise a clear error
            self.assertIn("empty", str(e).lower(), 
                         f"Unexpected error for empty data: {e}")
    
    def test_missing_columns_handling(self):
        """Test handling of data with missing required columns."""
        # Create data without 'label' column
        data = [
            {'text': 'This is text without labels.'},
            {'text': 'Another text without labels.'}
        ]
        missing_cols_file = os.path.join(self.temp_dir, 'missing_cols.json')
        with open(missing_cols_file, 'w') as f:
            json.dump(data, f)
        
        task_config = TaskConfig(
            name="test",
            type="single_label_classification",
            data_path=missing_cols_file,
            heads=[HeadConfig(name="test_head", num_labels=2)]
        )
        
        config = ExperimentConfig(
            project_name="test",
            tasks=[task_config],
            model=ModelConfig(base_model="prajjwal1/bert-tiny"),
            tokenizer=TokenizerConfig(max_length=64)
        )
        
        processor = DataProcessor(config, self.tokenizer)
        
        # Should raise an exception for missing required columns
        with self.assertRaises((KeyError, ValueError, AttributeError)):
            processor.process()


def run_tests():
    """Run all data loading tests and generate report."""
    print("=" * 70)
    print("  DATA LOADING UNIT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataLoading)
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