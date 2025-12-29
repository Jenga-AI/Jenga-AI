#!/usr/bin/env python3
"""
Phase 1 Algorithm Validation Test Runner
Comprehensive testing suite for Jenga-AI core algorithms
"""

import sys
import os
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add parent directories to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent.parent))

from test_attention_fusion_performance import run_performance_validation
from test_multitask_benchmark import run_multitask_benchmark
from test_round_robin_sampling import run_round_robin_validation
from test_memory_performance import run_memory_performance_validation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'phase1_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class Phase1ValidationRunner:
    """Comprehensive Phase 1 algorithm validation runner"""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.test_summary = {}
        
        # Create results directory
        self.results_dir = Path("algorithm_validation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Phase 1 Algorithm Validation Runner initialized")
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all Phase 1 validation tests"""
        logger.info("=" * 100)
        logger.info("🚀 STARTING PHASE 1 ALGORITHM VALIDATION")
        logger.info("=" * 100)
        logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📁 Results directory: {self.results_dir}")
        
        # Define test suite
        test_suite = [
            {
                'name': 'attention_fusion',
                'description': 'Attention Fusion Performance Validation',
                'runner': run_performance_validation,
                'critical': True
            },
            {
                'name': 'multitask_benchmark',
                'description': 'Multi-Task vs Single-Task Benchmark',
                'runner': run_multitask_benchmark,
                'critical': True
            },
            {
                'name': 'round_robin_sampling',
                'description': 'Round-Robin Task Sampling Validation',
                'runner': run_round_robin_validation,
                'critical': True
            },
            {
                'name': 'memory_performance',
                'description': 'Memory Efficiency & Performance Analysis',
                'runner': run_memory_performance_validation,
                'critical': True
            }
        ]
        
        # Run each test
        for i, test_config in enumerate(test_suite):
            test_name = test_config['name']
            test_desc = test_config['description']
            test_runner = test_config['runner']
            is_critical = test_config['critical']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 TEST {i+1}/{len(test_suite)}: {test_desc}")
            logger.info(f"{'='*60}")
            
            test_start_time = time.time()
            
            try:
                # Run the test
                result = test_runner()
                test_duration = time.time() - test_start_time
                
                # Store results
                self.results[test_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'result': result,
                    'duration': test_duration,
                    'critical': is_critical,
                    'description': test_desc
                }
                
                if result:
                    logger.info(f"✅ {test_desc} PASSED ({test_duration:.2f}s)")
                else:
                    logger.error(f"❌ {test_desc} FAILED ({test_duration:.2f}s)")
                    
                    if is_critical:
                        logger.error(f"💥 CRITICAL TEST FAILED: {test_name}")
                
            except Exception as e:
                test_duration = time.time() - test_start_time
                logger.error(f"💀 {test_desc} CRASHED: {str(e)}")
                
                self.results[test_name] = {
                    'status': 'CRASHED',
                    'error': str(e),
                    'duration': test_duration,
                    'critical': is_critical,
                    'description': test_desc
                }
                
                if is_critical:
                    logger.error(f"💥 CRITICAL TEST CRASHED: {test_name}")
        
        # Generate final report
        self._generate_final_report()
        
        return self.results
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        total_duration = time.time() - self.start_time
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r['status'] == 'PASSED'])
        failed_tests = len([r for r in self.results.values() if r['status'] == 'FAILED'])
        crashed_tests = len([r for r in self.results.values() if r['status'] == 'CRASHED'])
        critical_failures = len([r for r in self.results.values() 
                               if r['status'] != 'PASSED' and r['critical']])
        
        # Overall status
        overall_status = "PASSED" if critical_failures == 0 else "FAILED"
        
        logger.info("\n" + "=" * 100)
        logger.info("📋 PHASE 1 VALIDATION SUMMARY REPORT")
        logger.info("=" * 100)
        
        logger.info(f"🕐 Total Duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        logger.info(f"📊 Test Results: {total_tests} total, {passed_tests} passed, {failed_tests} failed, {crashed_tests} crashed")
        logger.info(f"🎯 Critical Failures: {critical_failures}")
        logger.info(f"🏆 Overall Status: {overall_status}")
        
        # Detailed test results
        logger.info(f"\n📋 DETAILED TEST RESULTS:")
        for test_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "💀"
            critical_marker = " [CRITICAL]" if result['critical'] else ""
            
            logger.info(f"{status_emoji} {result['description']}: {result['status']} ({result['duration']:.2f}s){critical_marker}")
            
            if result['status'] == 'CRASHED':
                logger.info(f"   💥 Error: {result.get('error', 'Unknown error')}")
        
        # Algorithm-specific insights
        self._generate_algorithm_insights()
        
        # Recommendations
        self._generate_recommendations()
        
        # Save results to file
        self._save_results_to_file()
        
        # Final verdict
        logger.info("\n" + "=" * 100)
        if overall_status == "PASSED":
            logger.info("🎉 PHASE 1 ALGORITHM VALIDATION: ALL CRITICAL TESTS PASSED!")
            logger.info("🚀 Ready to proceed to Phase 2: Comprehensive Testing Suite")
        else:
            logger.info("⚠️  PHASE 1 ALGORITHM VALIDATION: CRITICAL FAILURES DETECTED")
            logger.info("🔧 Fix critical issues before proceeding to Phase 2")
        logger.info("=" * 100)
    
    def _generate_algorithm_insights(self):
        """Generate insights about algorithm performance"""
        logger.info(f"\n🧠 ALGORITHM INSIGHTS:")
        
        # Attention Fusion Analysis
        if 'attention_fusion' in self.results and self.results['attention_fusion']['status'] == 'PASSED':
            logger.info("✅ Attention Fusion: Performance validation completed")
            logger.info("   - Fusion mechanism adds computational overhead but enables task-specific representations")
            logger.info("   - Different tasks produce measurably different attention patterns")
            logger.info("   - Gradient flow is healthy through the fusion mechanism")
        
        # Multi-task Learning Analysis
        if 'multitask_benchmark' in self.results:
            result = self.results['multitask_benchmark']
            if result['status'] == 'PASSED':
                # Extract insights from benchmark results
                benchmark_result = result.get('result', {})
                main_benchmark = benchmark_result.get('main_benchmark', {})
                
                if main_benchmark.get('negative_transfer_detected', False):
                    logger.info("⚠️  Multi-task Learning: Negative transfer detected in some tasks")
                    logger.info("   - Recommendation: Adjust task weights or fusion parameters")
                else:
                    logger.info("✅ Multi-task Learning: No negative transfer detected")
                    logger.info("   - Multi-task learning shows neutral to positive transfer")
                
                # Fusion impact
                fusion_benchmark = benchmark_result.get('fusion_benchmark', {})
                if fusion_benchmark:
                    logger.info("✅ Fusion Impact: Attention fusion evaluation completed")
        
        # Round-Robin Sampling Analysis
        if 'round_robin_sampling' in self.results and self.results['round_robin_sampling']['status'] == 'PASSED':
            logger.info("✅ Round-Robin Sampling: Balanced training validated")
            logger.info("   - Tasks receive equal sampling regardless of dataset sizes")
            logger.info("   - Temporal distribution ensures all tasks appear in all epochs")
        
        # Memory Performance Analysis
        if 'memory_performance' in self.results and self.results['memory_performance']['status'] == 'PASSED':
            logger.info("✅ Memory Performance: Efficiency validation completed")
            logger.info("   - Memory usage scales predictably with batch size and sequence length")
            logger.info("   - Gradient accumulation provides memory-efficient alternative to large batches")
            logger.info("   - CPU optimizations show performance improvements")
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        logger.info(f"\n💡 RECOMMENDATIONS:")
        
        # Performance recommendations
        logger.info("🚀 Performance Optimization:")
        logger.info("   - Use prajjwal1/bert-tiny for development/testing (4.4M params, ~200MB memory)")
        logger.info("   - Implement gradient accumulation for memory-constrained environments")
        logger.info("   - Batch size 2-4 with gradient accumulation for 16GB RAM systems")
        logger.info("   - Sequence length 32-64 for optimal speed/memory tradeoff")
        
        # Memory recommendations
        logger.info("💾 Memory Management:")
        logger.info("   - Monitor memory during training to detect leaks")
        logger.info("   - Use CPU-only training for development and small-scale deployment")
        logger.info("   - Consider mixed precision if supported by hardware")
        
        # Multi-task recommendations
        logger.info("🎯 Multi-task Learning:")
        logger.info("   - Continue monitoring for negative transfer in production")
        logger.info("   - Experiment with different task weights if imbalanced performance")
        logger.info("   - Attention fusion provides task specialization benefits")
        
        # African context recommendations
        logger.info("🌍 African Context:")
        logger.info("   - Framework is CPU-optimized for resource-constrained environments")
        logger.info("   - Round-robin sampling ensures balanced learning across diverse tasks")
        logger.info("   - Ready for African language and cultural context fine-tuning")
    
    def _save_results_to_file(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"phase1_validation_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for test_name, result in self.results.items():
            serializable_result = {
                'status': result['status'],
                'duration': result['duration'],
                'critical': result['critical'],
                'description': result['description']
            }
            
            # Add error if exists
            if 'error' in result:
                serializable_result['error'] = result['error']
            
            # Add simplified result data (avoiding complex objects)
            if 'result' in result and isinstance(result['result'], (bool, dict)):
                if isinstance(result['result'], bool):
                    serializable_result['result'] = result['result']
                else:
                    # For complex dict results, just store success status
                    serializable_result['result'] = result['result'].get('success', True)
            
            serializable_results[test_name] = serializable_result
        
        # Create summary report
        summary_report = {
            'timestamp': timestamp,
            'total_duration': time.time() - self.start_time,
            'jenga_ai_version': 'Phase 1 Algorithm Validation',
            'test_results': serializable_results,
            'summary': {
                'total_tests': len(self.results),
                'passed_tests': len([r for r in self.results.values() if r['status'] == 'PASSED']),
                'failed_tests': len([r for r in self.results.values() if r['status'] == 'FAILED']),
                'crashed_tests': len([r for r in self.results.values() if r['status'] == 'CRASHED']),
                'critical_failures': len([r for r in self.results.values() 
                                        if r['status'] != 'PASSED' and r['critical']]),
                'overall_status': "PASSED" if len([r for r in self.results.values() 
                                                 if r['status'] != 'PASSED' and r['critical']]) == 0 else "FAILED"
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        # Also save a human-readable summary
        summary_file = self.results_dir / f"phase1_validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("JENGA-AI PHASE 1 ALGORITHM VALIDATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {time.time() - self.start_time:.2f} seconds\n")
            f.write(f"Overall Status: {summary_report['summary']['overall_status']}\n\n")
            
            f.write("TEST RESULTS:\n")
            for test_name, result in serializable_results.items():
                f.write(f"- {result['description']}: {result['status']}\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write(f"- Total Tests: {summary_report['summary']['total_tests']}\n")
            f.write(f"- Passed: {summary_report['summary']['passed_tests']}\n")
            f.write(f"- Failed: {summary_report['summary']['failed_tests']}\n")
            f.write(f"- Crashed: {summary_report['summary']['crashed_tests']}\n")
            f.write(f"- Critical Failures: {summary_report['summary']['critical_failures']}\n")
        
        logger.info(f"📄 Summary report saved to: {summary_file}")


def main():
    """Main function to run Phase 1 validation"""
    runner = Phase1ValidationRunner()
    
    try:
        results = runner.run_all_validations()
        
        # Exit with appropriate code
        critical_failures = len([r for r in results.values() 
                               if r['status'] != 'PASSED' and r['critical']])
        
        if critical_failures == 0:
            logger.info("🎉 Phase 1 validation completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"❌ Phase 1 validation failed with {critical_failures} critical failures")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💀 Validation runner crashed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()