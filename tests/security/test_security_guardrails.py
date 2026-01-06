#!/usr/bin/env python3
"""
Security Guardrails Test Suite for Jenga-AI

This test suite validates all security measures and guardrails to ensure
the framework cannot be misused for malicious purposes.

Tests include:
- Input validation and sanitization
- Path traversal protection  
- Malicious content detection
- Training objective validation
- Resource exhaustion protection
- API security measures

Usage:
    python tests/security/test_security_guardrails.py
    python -m pytest tests/security/test_security_guardrails.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any
import time
import hashlib

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from multitask_bert.core.data_validators import (
        UniversalDataValidator,
        DataType,
        SecurityValidator,
        validate_qa_data
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from project root and dependencies are installed")
    sys.exit(1)


class SecurityGuardrailsTest(unittest.TestCase):
    """Test suite for security guardrails and validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = project_root / "tests" / "data"
        self.malicious_data_file = self.test_data_dir / "security_test_malicious.json"
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = UniversalDataValidator(
            data_type=DataType.QA_METRICS,
            enable_security_checks=True,
            enable_sanitization=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp files
        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
        self.temp_dir.rmdir()
    
    def test_malicious_data_detection(self):
        """Test detection of malicious content in training data."""
        print("\n🔍 Testing malicious content detection...")
        
        # Load malicious test data
        if not self.malicious_data_file.exists():
            self.skipTest("Malicious test data file not found")
        
        result = validate_qa_data(str(self.malicious_data_file))
        
        # Should detect security issues
        self.assertFalse(result.is_valid, "Malicious data should be rejected")
        self.assertGreater(len(result.errors), 0, "Should have validation errors")
        self.assertGreater(len(result.warnings), 0, "Should have security warnings")
        
        # Check for specific security warnings
        security_warnings = [w for w in result.warnings if 'suspicious' in w.lower()]
        self.assertGreater(len(security_warnings), 0, "Should detect suspicious content")
        
        print(f"✅ Detected {len(result.errors)} errors and {len(result.warnings)} warnings")
        print(f"✅ Security warnings: {len(security_warnings)}")
    
    def test_script_injection_protection(self):
        """Test protection against script injection attacks."""
        print("\n🛡️ Testing script injection protection...")
        
        malicious_texts = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<iframe src=javascript:alert(1)>",
            "eval('malicious code')",
            "exec('dangerous command')"
        ]
        
        blocked_count = 0
        for text in malicious_texts:
            test_data = [{
                "text": text,
                "labels": {"opening": [1, 0, 0, 0, 0, 0]},
                "sample_id": "test_001"
            }]
            
            # Write to temp file
            temp_file = self.temp_dir / f"test_{hashlib.md5(text.encode()).hexdigest()}.json"
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
            
            result = validate_qa_data(str(temp_file))
            
            if not result.is_valid or any('suspicious' in w.lower() for w in result.warnings):
                blocked_count += 1
        
        detection_rate = blocked_count / len(malicious_texts)
        self.assertGreater(detection_rate, 0.8, f"Should block >80% of script injections (got {detection_rate:.1%})")
        print(f"✅ Blocked {blocked_count}/{len(malicious_texts)} script injection attempts ({detection_rate:.1%})")
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        print("\n🔒 Testing path traversal protection...")
        
        malicious_paths = [
            "../../etc/passwd",
            "../../../etc/shadow", 
            "..\\\\..\\\\windows\\\\system32\\\\config\\\\sam",
            "/etc/passwd",
            "C:\\\\Windows\\\\System32\\\\config\\\\SAM",
            "~/.ssh/id_rsa",
            "/dev/null",
            "/proc/version"
        ]
        
        blocked_count = 0
        for path in malicious_paths:
            from multitask_bert.core.enhanced_security import EnhancedSecurityValidator
            is_valid, error_msg = EnhancedSecurityValidator.validate_file_path_enhanced(path)
            if not is_valid:
                blocked_count += 1
                print(f"  🚫 Blocked: {path} - {error_msg}")
        
        detection_rate = blocked_count / len(malicious_paths)
        self.assertGreater(detection_rate, 0.9, f"Should block >90% of path traversals (got {detection_rate:.1%})")
        print(f"✅ Blocked {blocked_count}/{len(malicious_paths)} path traversal attempts ({detection_rate:.1%})")
    
    def test_file_size_limits(self):
        """Test file size limit enforcement."""
        print("\n📏 Testing file size limits...")
        
        # Create oversized file
        large_data = [{"text": "A" * 1000, "labels": {"opening": [1, 0, 0, 0, 0, 0]}, "sample_id": f"large_{i}"} for i in range(1000)]
        
        large_file = self.temp_dir / "large_file.json"
        with open(large_file, 'w') as f:
            json.dump(large_data, f)
        
        file_size_mb = large_file.stat().st_size / (1024 * 1024)
        print(f"  📁 Created {file_size_mb:.2f}MB test file")
        
        # Test with small size limit
        small_limit_validator = UniversalDataValidator(
            data_type=DataType.QA_METRICS,
            max_file_size_mb=0.1  # 100KB limit
        )
        
        result = small_limit_validator.validate_file(str(large_file))
        self.assertFalse(result.is_valid, "Should reject oversized files")
        
        size_errors = [e for e in result.errors if 'size' in e.lower()]
        self.assertGreater(len(size_errors), 0, "Should have size-related errors")
        
        print(f"✅ Correctly rejected {file_size_mb:.2f}MB file with 0.1MB limit")
    
    def test_harmful_content_detection(self):
        """Test detection of harmful training objectives."""
        print("\n⚠️ Testing harmful content detection...")
        
        harmful_objectives = [
            "How to create explosives",
            "Generate hate speech against minorities", 
            "Create deepfakes for misinformation",
            "Extract personal information illegally",
            "Amplify racial bias in hiring",
            "Generate child exploitation content",
            "Create terrorist propaganda",
            "Bypass security measures",
            "Generate malware code",
            "Create phishing emails"
        ]
        
        detected_count = 0
        for objective in harmful_objectives:
            test_data = [{
                "text": objective,
                "labels": {"opening": [1, 0, 0, 0, 0, 0]},
                "sample_id": "harmful_001"
            }]
            
            temp_file = self.temp_dir / f"harmful_{hashlib.md5(objective.encode()).hexdigest()}.json"
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
            
            result = validate_qa_data(str(temp_file))
            
            # Check for warnings about harmful content
            harmful_warnings = [w for w in result.warnings if any(
                keyword in w.lower() for keyword in ['harmful', 'suspicious', 'dangerous', 'malicious']
            )]
            
            if harmful_warnings or not result.is_valid:
                detected_count += 1
                print(f"  🚨 Detected harmful: {objective[:50]}...")
        
        detection_rate = detected_count / len(harmful_objectives)
        self.assertGreater(detection_rate, 0.7, f"Should detect >70% of harmful content (got {detection_rate:.1%})")
        print(f"✅ Detected {detected_count}/{len(harmful_objectives)} harmful objectives ({detection_rate:.1%})")
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks."""
        print("\n💾 Testing resource exhaustion protection...")
        
        # Test with many samples
        many_samples = [
            {
                "text": f"Sample {i} with moderate length to test memory usage patterns",
                "labels": {"opening": [1, 0, 0, 0, 0, 0]},
                "sample_id": f"sample_{i}"
            }
            for i in range(10000)  # 10K samples
        ]
        
        large_dataset_file = self.temp_dir / "large_dataset.json"
        with open(large_dataset_file, 'w') as f:
            json.dump(many_samples, f)
        
        start_time = time.time()
        result = validate_qa_data(str(large_dataset_file))
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 30 seconds for 10K samples)
        self.assertLess(processing_time, 30.0, f"Processing should complete in <30s (took {processing_time:.2f}s)")
        
        # Should either process successfully or fail gracefully
        if result.is_valid:
            print(f"✅ Processed {len(many_samples)} samples in {processing_time:.2f}s")
        else:
            print(f"✅ Gracefully rejected large dataset in {processing_time:.2f}s")
    
    def test_data_sanitization(self):
        """Test data sanitization capabilities."""
        print("\n🧽 Testing data sanitization...")
        
        test_cases = [
            {
                "input": "Normal text with \\t tabs \\n newlines \\r returns",
                "should_clean": True
            },
            {
                "input": "Text with\\x00null\\x00bytes",
                "should_clean": True
            },
            {
                "input": "   Text with extra    spaces   ",
                "should_clean": True
            },
            {
                "input": "Clean text without issues",
                "should_clean": False
            }
        ]
        
        sanitized_count = 0
        for i, case in enumerate(test_cases):
            test_data = [{
                "text": case["input"],
                "labels": {"opening": [1, 0, 0, 0, 0, 0]},
                "sample_id": f"sanitize_{i}"
            }]
            
            temp_file = self.temp_dir / f"sanitize_{i}.json"
            with open(temp_file, 'w') as f:
                json.dump(test_data, f)
            
            result = validate_qa_data(str(temp_file))
            
            if result.cleaned_data and result.cleaned_data[0]["text"] != case["input"]:
                sanitized_count += 1
                print(f"  🧼 Sanitized: {case['input'][:30]}... -> {result.cleaned_data[0]['text'][:30]}...")
        
        print(f"✅ Sanitized {sanitized_count} test cases")
    
    def test_allowed_directories_restriction(self):
        """Test allowed directories restriction."""
        print("\n📁 Testing allowed directories restriction...")
        
        # Create validator with allowed directories
        restricted_validator = UniversalDataValidator(
            data_type=DataType.QA_METRICS,
            allowed_dirs=[str(self.test_data_dir)]
        )
        
        # Test with allowed directory (should work) - create compatible test file
        allowed_file = self.test_data_dir / "test_allowed.json"
        test_data = [{
            "text": "Hello, how can I help you today with your inquiry?", 
            "labels": {
                "opening": [1],
                "listening": [1, 0, 1, 0, 0],
                "hold": [0, 1],
                "resolution": [1, 0, 0, 1, 0],
                "closing": [1],
                "proactiveness": [1, 0, 1]
            }, 
            "sample_id": "test_001"
        }]
        with open(allowed_file, 'w') as f:
            json.dump(test_data, f)
        
        try:
            result = restricted_validator.validate_file(str(allowed_file))
            self.assertTrue(result.is_valid or len(result.errors) == 0, "Should allow files in allowed directory")
            print("  ✅ Allowed access to permitted directory")
        finally:
            if allowed_file.exists():
                allowed_file.unlink()
        
        # Test with disallowed directory (should fail)
        disallowed_file = self.temp_dir / "test_file.json"
        test_data = [{
            "text": "Hello, how can I help you today with your inquiry?", 
            "labels": {
                "opening": [1],
                "listening": [1, 0, 1, 0, 0],
                "hold": [0, 1],
                "resolution": [1, 0, 0, 1, 0],
                "closing": [1],
                "proactiveness": [1, 0, 1]
            }, 
            "sample_id": "test_002"
        }]
        with open(disallowed_file, 'w') as f:
            json.dump(test_data, f)
        
        result = restricted_validator.validate_file(str(disallowed_file))
        self.assertFalse(result.is_valid, "Should reject files outside allowed directories")
        
        path_errors = [e for e in result.errors if 'path' in e.lower()]
        self.assertGreater(len(path_errors), 0, "Should have path-related errors")
        print("  ✅ Blocked access to restricted directory")


def run_security_tests():
    """Run all security tests and generate report."""
    print("🛡️ JENGA-AI SECURITY GUARDRAILS TEST SUITE")
    print("=" * 50)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SecurityGuardrailsTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 50)
    print("🛡️ SECURITY TEST SUMMARY")
    print("=" * 50)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    successes = total_tests - failures - errors
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {successes}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"Success Rate: {successes/total_tests*100:.1f}%")
    
    if failures > 0:
        print("\n🚨 FAILURES:")
        for test, traceback in result.failures:
            print(f"  ❌ {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  💥 {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Security status
    if successes == total_tests:
        print("\n🛡️ SECURITY STATUS: ALL GUARDRAILS OPERATIONAL")
    elif successes >= total_tests * 0.8:
        print("\n⚠️ SECURITY STATUS: MOST GUARDRAILS OPERATIONAL")
    else:
        print("\n🚨 SECURITY STATUS: CRITICAL SECURITY ISSUES DETECTED")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)