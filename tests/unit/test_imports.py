#!/usr/bin/env python3
"""
Module Import Tests
===================
Tests that all Jenga-AI modules can be imported without errors.

Usage:
    python -m pytest tests/unit/test_imports.py -v
    OR
    python tests/unit/test_imports.py
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestCoreImports(unittest.TestCase):
    """Test imports for core modules."""
    
    def test_import_model(self):
        """Test importing core.model module."""
        try:
            from multitask_bert.core import model
            self.assertIsNotNone(model)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.core.model: {e}")
    
    def test_import_fusion(self):
        """Test importing core.fusion module."""
        try:
            from multitask_bert.core import fusion
            self.assertIsNotNone(fusion)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.core.fusion: {e}")
    
    def test_import_config(self):
        """Test importing core.config module."""
        try:
            from multitask_bert.core import config
            self.assertIsNotNone(config)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.core.config: {e}")
    
    def test_import_registry(self):
        """Test importing core.registry module."""
        try:
            from multitask_bert.core import registry
            self.assertIsNotNone(registry)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.core.registry: {e}")


class TestTaskImports(unittest.TestCase):
    """Test imports for task modules."""
    
    def test_import_base_task(self):
        """Test importing tasks.base module."""
        try:
            from multitask_bert.tasks import base
            self.assertIsNotNone(base)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.base: {e}")
    
    def test_import_classification(self):
        """Test importing tasks.classification module."""
        try:
            from multitask_bert.tasks import classification
            self.assertIsNotNone(classification)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.classification: {e}")
    
    def test_import_ner(self):
        """Test importing tasks.ner module."""
        try:
            from multitask_bert.tasks import ner
            self.assertIsNotNone(ner)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.ner: {e}")
    
    def test_import_question_answering(self):
        """Test importing tasks.question_answering module."""
        try:
            from multitask_bert.tasks import question_answering
            self.assertIsNotNone(question_answering)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.question_answering: {e}")
    
    def test_import_qa_qc(self):
        """Test importing tasks.qa_qc module."""
        try:
            from multitask_bert.tasks import qa_qc
            self.assertIsNotNone(qa_qc)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.qa_qc: {e}")
    
    def test_import_sentiment_analysis(self):
        """Test importing tasks.sentiment_analysis module."""
        try:
            from multitask_bert.tasks import sentiment_analysis
            self.assertIsNotNone(sentiment_analysis)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.sentiment_analysis: {e}")
    
    def test_import_regression(self):
        """Test importing tasks.regression module."""
        try:
            from multitask_bert.tasks import regression
            self.assertIsNotNone(regression)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.tasks.regression: {e}")


class TestDataImports(unittest.TestCase):
    """Test imports for data processing modules."""
    
    def test_import_data_processing(self):
        """Test importing data.data_processing module."""
        try:
            from multitask_bert.data import data_processing
            self.assertIsNotNone(data_processing)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.data.data_processing: {e}")
    
    def test_import_universal(self):
        """Test importing data.universal module."""
        try:
            from multitask_bert.data import universal
            self.assertIsNotNone(universal)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.data.universal: {e}")
    
    def test_import_custom(self):
        """Test importing data.custom module."""
        try:
            from multitask_bert.data import custom
            self.assertIsNotNone(custom)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.data.custom: {e}")


class TestTrainingImports(unittest.TestCase):
    """Test imports for training modules."""
    
    def test_import_trainer(self):
        """Test importing training.trainer module."""
        try:
            from multitask_bert.training import trainer
            self.assertIsNotNone(trainer)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.training.trainer: {e}")
    
    def test_import_callbacks(self):
        """Test importing training.callbacks module."""
        try:
            from multitask_bert.training import callbacks
            self.assertIsNotNone(callbacks)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.training.callbacks: {e}")
    
    def test_import_data(self):
        """Test importing training.data module."""
        try:
            from multitask_bert.training import data
            self.assertIsNotNone(data)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.training.data: {e}")


class TestAnalysisImports(unittest.TestCase):
    """Test imports for analysis modules."""
    
    def test_import_attention(self):
        """Test importing analysis.attention module."""
        try:
            from multitask_bert.analysis import attention
            self.assertIsNotNone(attention)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.analysis.attention: {e}")
    
    def test_import_metrics(self):
        """Test importing analysis.metrics module."""
        try:
            from multitask_bert.analysis import metrics
            self.assertIsNotNone(metrics)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.analysis.metrics: {e}")
    
    def test_import_visualization(self):
        """Test importing analysis.visualization module."""
        try:
            from multitask_bert.analysis import visualization
            self.assertIsNotNone(visualization)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.analysis.visualization: {e}")


class TestDeploymentImports(unittest.TestCase):
    """Test imports for deployment modules."""
    
    def test_import_inference(self):
        """Test importing deployment.inference module."""
        try:
            from multitask_bert.deployment import inference
            self.assertIsNotNone(inference)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.deployment.inference: {e}")
    
    def test_import_export(self):
        """Test importing deployment.export module."""
        try:
            from multitask_bert.deployment import export
            self.assertIsNotNone(export)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.deployment.export: {e}")


class TestUtilImports(unittest.TestCase):
    """Test imports for utility modules."""
    
    def test_import_logging(self):
        """Test importing utils.logging module."""
        try:
            from multitask_bert.utils import logging
            self.assertIsNotNone(logging)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.utils.logging: {e}")
    
    def test_import_metrics(self):
        """Test importing utils.metrics module."""
        try:
            from multitask_bert.utils import metrics
            self.assertIsNotNone(metrics)
        except ImportError as e:
            self.fail(f"Failed to import multitask_bert.utils.metrics: {e}")


class TestLLMFinetuningImports(unittest.TestCase):
    """Test imports for LLM fine-tuning modules."""
    
    def test_import_model_factory(self):
        """Test importing llm_finetuning.model.model_factory module."""
        try:
            from llm_finetuning.model import model_factory
            self.assertIsNotNone(model_factory)
        except ImportError as e:
            self.fail(f"Failed to import llm_finetuning.model.model_factory: {e}")
    
    def test_import_teacher_student(self):
        """Test importing llm_finetuning.model.teacher_student module."""
        try:
            from llm_finetuning.model import teacher_student
            self.assertIsNotNone(teacher_student)
        except ImportError as e:
            self.fail(f"Failed to import llm_finetuning.model.teacher_student: {e}")
    
    def test_import_llm_trainer(self):
        """Test importing llm_finetuning.training.trainer module."""
        try:
            from llm_finetuning.training import trainer
            self.assertIsNotNone(trainer)
        except ImportError as e:
            self.fail(f"Failed to import llm_finetuning.training.trainer: {e}")
    
    def test_import_llm_data_processing(self):
        """Test importing llm_finetuning.data.data_processing module."""
        try:
            from llm_finetuning.data import data_processing
            self.assertIsNotNone(data_processing)
        except ImportError as e:
            self.fail(f"Failed to import llm_finetuning.data.data_processing: {e}")


class TestSeq2SeqImports(unittest.TestCase):
    """Test imports for seq2seq modules."""
    
    def test_import_seq2seq_config(self):
        """Test importing seq2seq_models.core.config module."""
        try:
            from seq2seq_models.core import config
            self.assertIsNotNone(config)
        except ImportError as e:
            self.fail(f"Failed to import seq2seq_models.core.config: {e}")
    
    def test_import_seq2seq_model_factory(self):
        """Test importing seq2seq_models.model.seq2seq_model module."""
        try:
            from seq2seq_models.model import seq2seq_model
            self.assertIsNotNone(seq2seq_model)
        except ImportError as e:
            self.fail(f"Failed to import seq2seq_models.model.seq2seq_model: {e}")


def run_tests():
    """Run all import tests and generate report."""
    print("=" * 70)
    print("  JENGA-AI MODULE IMPORT TESTS")
    print("=" * 70)
    print()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoreImports))
    suite.addTests(loader.loadTestsFromTestCase(TestTaskImports))
    suite.addTests(loader.loadTestsFromTestCase(TestDataImports))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingImports))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisImports))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentImports))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilImports))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMFinetuningImports))
    suite.addTests(loader.loadTestsFromTestCase(TestSeq2SeqImports))
    
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


