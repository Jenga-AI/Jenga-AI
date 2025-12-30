#!/usr/bin/env python3
"""
Quick Algorithm Test - One Command Validation
Run: python3 quick_algorithm_test.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n🧪 JENGA-AI QUICK ALGORITHM TEST")
    print("="*50)
    
    scripts = [
        ("1. Create Test Data", "python3 create_tiny_datasets.py"),
        ("2. Run Full Pipeline", "python3 algorithm_validation_complete.py"),
    ]
    
    for name, command in scripts:
        print(f"\n{name}...")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"   ✅ SUCCESS")
            else:
                print(f"   ❌ FAILED: {result.stderr}")
                return False
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    
    print(f"\n🎉 ALL TESTS PASSED!")
    print(f"📊 Results in: algorithm_runs/")
    print(f"📄 Report: CPU_ALGORITHM_VALIDATION_REPORT.md")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)