#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
===============================
Runs all tests in the Jenga-AI testing framework and generates detailed reports.

Usage:
    python tests/run_test_suite.py [options]
    
Options:
    --unit-only      Run only unit tests
    --integration    Run only integration tests  
    --performance    Run only performance tests
    --all            Run all available tests (default)
    --report-html    Generate HTML report
    --verbose        Verbose output
    --fast           Skip slow tests
"""

import sys
import unittest
import argparse
import time
import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestResult:
    """Store results for a single test module."""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.skipped = 0
        self.duration = 0.0
        self.failure_messages = []
        self.error_messages = []


class TestSuiteRunner:
    """Comprehensive test runner for Jenga-AI."""
    
    def __init__(self):
        self.results: Dict[str, TestResult] = {}
        self.start_time = time.time()
        
        # Available test modules
        self.test_modules = {
            "unit": [
                "tests.unit.test_imports",
                "tests.unit.test_data_loading",
                "tests.unit.test_data_preprocessing", 
                "tests.unit.test_tasks",
                "tests.unit.test_attention_fusion",
                "tests.unit.test_multitask_model",
                "tests.unit.test_config_validation"
            ],
            "integration": [
                "tests.integration.test_single_task_training",
                "tests.integration.test_sentiment_training",
                # Add more as we create them
            ],
            "performance": [
                # Add performance tests when created
            ],
            "algorithm": [
                "tests.algorithm_validation.test_attention_fusion_performance",
                "tests.algorithm_validation.test_memory_performance",
                "tests.algorithm_validation.test_multitask_benchmark",
                "tests.algorithm_validation.test_round_robin_sampling"
            ]
        }
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print("Checking dependencies...")
        
        required_packages = [
            'torch', 'transformers', 'datasets', 'numpy', 
            'pandas', 'sklearn', 'yaml', 'tqdm'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Install with: pip install torch transformers datasets numpy pandas scikit-learn pyyaml tqdm")
            return False
        
        print("All dependencies available!")
        return True
    
    def run_module_tests(self, module_name: str, verbose: bool = False) -> TestResult:
        """Run tests for a specific module."""
        result = TestResult(module_name)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running: {module_name}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Import and load the test module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run tests with custom result handler
            runner = unittest.TextTestRunner(
                stream=sys.stdout if verbose else open(os.devnull, 'w'),
                verbosity=2 if verbose else 0
            )
            
            test_result = runner.run(suite)
            
            # Store results
            result.passed = test_result.testsRun - len(test_result.failures) - len(test_result.errors)
            result.failed = len(test_result.failures)
            result.errors = len(test_result.errors)
            result.duration = time.time() - start_time
            
            # Store failure and error messages
            for test, message in test_result.failures:
                result.failure_messages.append(f"{test}: {message}")
            
            for test, message in test_result.errors:
                result.error_messages.append(f"{test}: {message}")
                
            if not verbose:
                status = "PASS" if result.failed == 0 and result.errors == 0 else "FAIL"
                print(f"  {status:4} {module_name:50} ({result.duration:.2f}s)")
                
        except Exception as e:
            result.errors = 1
            result.error_messages.append(f"Module import/run error: {str(e)}")
            result.duration = time.time() - start_time
            
            if not verbose:
                print(f"  ERR  {module_name:50} ({result.duration:.2f}s)")
        
        return result
    
    def run_test_category(self, category: str, verbose: bool = False, fast: bool = False) -> None:
        """Run all tests in a category."""
        if category not in self.test_modules:
            print(f"Unknown test category: {category}")
            return
        
        modules = self.test_modules[category]
        if not modules:
            print(f"No tests found for category: {category}")
            return
        
        print(f"\n{'='*70}")
        print(f"  {category.upper()} TESTS")
        print(f"{'='*70}")
        
        for module in modules:
            if fast and "performance" in module.lower():
                print(f"  SKIP {module:50} (fast mode)")
                continue
                
            result = self.run_module_tests(module, verbose)
            self.results[module] = result
    
    def generate_summary_report(self) -> None:
        """Generate summary report of all test results."""
        print(f"\n{'='*70}")
        print("  COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*70}")
        
        total_duration = time.time() - self.start_time
        
        # Calculate totals
        total_passed = sum(r.passed for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        total_errors = sum(r.errors for r in self.results.values())
        total_tests = total_passed + total_failed + total_errors
        
        print(f"\nOverall Results:")
        print(f"  Total Tests:    {total_tests}")
        print(f"  Passed:         {total_passed} ({100*total_passed/total_tests:.1f}%)" if total_tests > 0 else "  Passed:         0")
        print(f"  Failed:         {total_failed}")
        print(f"  Errors:         {total_errors}")
        print(f"  Duration:       {total_duration:.2f}s")
        
        # Module breakdown
        print(f"\nModule Breakdown:")
        print(f"{'Module':<50} {'Status':<8} {'Pass':<4} {'Fail':<4} {'Err':<4} {'Time':<6}")
        print("-" * 80)
        
        for module_name, result in self.results.items():
            status = "PASS" if result.failed == 0 and result.errors == 0 else "FAIL"
            print(f"{module_name:<50} {status:<8} {result.passed:<4} {result.failed:<4} {result.errors:<4} {result.duration:<6.2f}")
        
        # Show failures and errors
        if total_failed > 0 or total_errors > 0:
            print(f"\n{'='*70}")
            print("  FAILURES AND ERRORS")
            print(f"{'='*70}")
            
            for module_name, result in self.results.items():
                if result.failure_messages or result.error_messages:
                    print(f"\n{module_name}:")
                    
                    for msg in result.failure_messages:
                        print(f"  FAIL: {msg[:200]}...")
                    
                    for msg in result.error_messages:
                        print(f"  ERR:  {msg[:200]}...")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("  RECOMMENDATIONS")
        print(f"{'='*70}")
        
        if total_failed == 0 and total_errors == 0:
            print("  🎉 All tests passed! Framework is working correctly.")
        elif total_errors > total_failed:
            print("  🔧 Focus on fixing errors first - these indicate missing dependencies or setup issues.")
        else:
            print("  🐛 Focus on fixing test failures - these indicate functionality issues.")
        
        if total_tests > 0:
            success_rate = total_passed / total_tests
            if success_rate < 0.5:
                print("  ⚠️  Many tests are failing. Check dependency installation and setup.")
            elif success_rate < 0.8:
                print("  📝 Good progress! Focus on remaining failures for full coverage.")
            else:
                print("  ✨ Excellent test coverage! Framework is well-tested.")
    
    def generate_json_report(self, output_file: str = "test_results.json") -> None:
        """Generate JSON report for programmatic analysis."""
        report_data = {
            "timestamp": time.time(),
            "duration": time.time() - self.start_time,
            "summary": {
                "total_passed": sum(r.passed for r in self.results.values()),
                "total_failed": sum(r.failed for r in self.results.values()), 
                "total_errors": sum(r.errors for r in self.results.values()),
                "total_tests": sum(r.passed + r.failed + r.errors for r in self.results.values())
            },
            "modules": {}
        }
        
        for module_name, result in self.results.items():
            report_data["modules"][module_name] = {
                "passed": result.passed,
                "failed": result.failed,
                "errors": result.errors,
                "duration": result.duration,
                "failure_messages": result.failure_messages,
                "error_messages": result.error_messages
            }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nJSON report saved to: {output_file}")
    
    def run_environment_check(self) -> bool:
        """Run environment validation."""
        print("Running environment check...")
        
        try:
            # Try to run the environment check script
            result = self.run_module_tests("tests.environment_check", verbose=True)
            return result.errors == 0 and result.failed == 0
        except Exception as e:
            print(f"Environment check failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Jenga-AI Test Suite Runner")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--algorithm", action="store_true", help="Run only algorithm validation tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--json", type=str, help="Save JSON report to file")
    parser.add_argument("--no-deps", action="store_true", help="Skip dependency check")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific category selected
    if not any([args.unit_only, args.integration, args.performance, args.algorithm]):
        args.all = True
    
    runner = TestSuiteRunner()
    
    print("=" * 70)
    print("  JENGA-AI COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"  Environment: Python {sys.version}")
    print(f"  Working Dir: {os.getcwd()}")
    print("=" * 70)
    
    # Check dependencies
    if not args.no_deps and not runner.check_dependencies():
        print("\n❌ Dependency check failed. Run with --no-deps to skip this check.")
        return 1
    
    # Run selected test categories
    try:
        if args.unit_only or args.all:
            runner.run_test_category("unit", args.verbose, args.fast)
        
        if args.integration or args.all:
            runner.run_test_category("integration", args.verbose, args.fast)
        
        if args.performance or args.all:
            runner.run_test_category("performance", args.verbose, args.fast)
            
        if args.algorithm or args.all:
            runner.run_test_category("algorithm", args.verbose, args.fast)
        
        # Generate reports
        runner.generate_summary_report()
        
        if args.json:
            runner.generate_json_report(args.json)
        
        # Return exit code
        total_failed = sum(r.failed for r in runner.results.values())
        total_errors = sum(r.errors for r in runner.results.values())
        
        return 0 if total_failed == 0 and total_errors == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())