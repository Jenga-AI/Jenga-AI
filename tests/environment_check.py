#!/usr/bin/env python3
"""
Jenga-AI Environment Check Script
==================================
Validates system requirements and dependencies for testing the framework.

Usage:
    python tests/environment_check.py
    
This script checks:
- Python version
- System resources (RAM, CPU, disk)
- PyTorch installation and device
- Transformers library
- Other critical dependencies
- Basic import functionality
"""

import sys
import os
import platform
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(check_name, status, details=""):
    """Print a check result with status indicator."""
    status_symbol = "✓" if status else "✗"
    status_text = "PASS" if status else "FAIL"
    print(f"{status_symbol} {check_name:.<50} {status_text}")
    if details:
        print(f"  └─ {details}")


def check_python_version():
    """Check Python version (requires 3.9+)."""
    print_header("Python Environment")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    required_major, required_minor = 3, 9
    
    is_valid = version.major >= required_major and version.minor >= required_minor
    print_result(
        "Python Version",
        is_valid,
        f"Current: {version_str}, Required: {required_major}.{required_minor}+"
    )
    
    print(f"  Python Executable: {sys.executable}")
    print(f"  Platform: {platform.platform()}")
    
    return is_valid


def check_system_resources():
    """Check system RAM, CPU cores, and disk space."""
    print_header("System Resources")
    
    all_checks_pass = True
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        is_ram_sufficient = ram_gb >= 8  # Minimum 8GB, recommended 16GB
        print_result(
            "Total RAM",
            is_ram_sufficient,
            f"{ram_gb:.2f} GB total, {ram_available_gb:.2f} GB available"
        )
        if ram_gb < 16:
            print(f"  ⚠️  Warning: 16GB RAM recommended for optimal performance")
        all_checks_pass &= is_ram_sufficient
    except ImportError:
        print_result("Total RAM", False, "psutil not installed (pip install psutil)")
        all_checks_pass = False
    
    # Check CPU cores
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        print_result(
            "CPU Cores",
            True,
            f"{cpu_physical} physical, {cpu_count} logical cores"
        )
    except:
        cpu_count = os.cpu_count() or 1
        print_result("CPU Cores", True, f"{cpu_count} cores detected")
    
    # Check disk space
    try:
        import psutil
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        is_disk_sufficient = disk_free_gb >= 5  # Need at least 5GB
        print_result(
            "Disk Space",
            is_disk_sufficient,
            f"{disk_free_gb:.2f} GB free (need 5GB+ for models/datasets)"
        )
        all_checks_pass &= is_disk_sufficient
    except:
        print_result("Disk Space", False, "Unable to check disk space")
        all_checks_pass = False
    
    return all_checks_pass


def check_pytorch():
    """Check PyTorch installation and available devices."""
    print_header("PyTorch Installation")
    
    all_checks_pass = True
    
    # Check if PyTorch is installed
    try:
        import torch
        torch_version = torch.__version__
        print_result("PyTorch", True, f"Version {torch_version}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            print_result(
                "CUDA GPU",
                True,
                f"CUDA {cuda_version}, {gpu_count} GPU(s): {gpu_name}"
            )
        else:
            print_result(
                "CUDA GPU",
                True,  # Not a failure - CPU is fine for testing
                "Not available (CPU-only mode, which is fine for testing)"
            )
        
        # Test basic tensor operations
        try:
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)
            print_result("Tensor Operations", True, "Basic operations working")
        except Exception as e:
            print_result("Tensor Operations", False, f"Error: {str(e)}")
            all_checks_pass = False
            
    except ImportError as e:
        print_result("PyTorch", False, "Not installed (pip install torch)")
        all_checks_pass = False
    
    return all_checks_pass


def check_transformers():
    """Check Transformers library installation."""
    print_header("Transformers Library")
    
    all_checks_pass = True
    
    try:
        import transformers
        version = transformers.__version__
        print_result("Transformers", True, f"Version {version}")
        
        # Test loading a tiny model config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("prajjwal1/bert-tiny", trust_remote_code=False)
            print_result(
                "Model Loading Test",
                True,
                "Successfully loaded bert-tiny config"
            )
        except Exception as e:
            print_result(
                "Model Loading Test",
                False,
                f"Failed to load config: {str(e)[:50]}..."
            )
            all_checks_pass = False
            
    except ImportError:
        print_result("Transformers", False, "Not installed (pip install transformers)")
        all_checks_pass = False
    
    return all_checks_pass


def check_other_dependencies():
    """Check other required dependencies."""
    print_header("Other Dependencies")
    
    dependencies = {
        "numpy": "Numerical computing",
        "pandas": "Data manipulation",
        "datasets": "Hugging Face datasets",
        "pyyaml": "YAML config parsing",
        "tqdm": "Progress bars",
        "scikit-learn": "Metrics and utilities",
        "psutil": "System monitoring",
    }
    
    all_installed = True
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            # Get version if available
            try:
                mod = __import__(package)
                version = getattr(mod, "__version__", "unknown")
                print_result(package, True, f"v{version} - {description}")
            except:
                print_result(package, True, description)
        except ImportError:
            print_result(package, False, f"Not installed - {description}")
            all_installed = False
    
    return all_installed


def check_jenga_imports():
    """Check if Jenga-AI modules can be imported."""
    print_header("Jenga-AI Module Imports")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    modules = [
        "multitask_bert.core.model",
        "multitask_bert.core.fusion",
        "multitask_bert.core.config",
        "multitask_bert.tasks.base",
        "multitask_bert.tasks.classification",
        "multitask_bert.tasks.ner",
        "multitask_bert.data.data_processing",
        "multitask_bert.training.trainer",
    ]
    
    all_imported = True
    
    for module in modules:
        try:
            __import__(module)
            print_result(module, True, "")
        except Exception as e:
            error_msg = str(e).split('\n')[0][:60]
            print_result(module, False, f"Error: {error_msg}...")
            all_imported = False
    
    return all_imported


def check_uv_package_manager():
    """Check if UV package manager is installed."""
    print_header("UV Package Manager")
    
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_result("UV", True, version)
            return True
        else:
            print_result("UV", False, "Installed but not working properly")
            return False
    except FileNotFoundError:
        print_result("UV", False, "Not installed (optional but recommended)")
        print("  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False
    except subprocess.TimeoutExpired:
        print_result("UV", False, "Command timed out")
        return False


def generate_report():
    """Generate a summary report."""
    print_header("Summary Report")
    
    all_results = []
    
    # Run all checks
    all_results.append(("Python Version", check_python_version()))
    all_results.append(("System Resources", check_system_resources()))
    all_results.append(("PyTorch", check_pytorch()))
    all_results.append(("Transformers", check_transformers()))
    all_results.append(("Dependencies", check_other_dependencies()))
    all_results.append(("Jenga-AI Modules", check_jenga_imports()))
    all_results.append(("UV Package Manager", check_uv_package_manager()))
    
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    for check_name, result in all_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} {check_name}")
    
    print("\n" + "-" * 70)
    print(f"  Total: {passed}/{total} checks passed")
    print("-" * 70)
    
    if passed == total:
        print("\n🎉 All checks passed! Environment is ready for testing.")
        return 0
    elif passed >= total - 2:
        print("\n⚠️  Most checks passed. Minor issues detected but testing can proceed.")
        return 0
    else:
        print("\n❌ Multiple checks failed. Please fix issues before proceeding.")
        print("\nQuick Fix Commands:")
        print("  pip install torch transformers datasets numpy pandas pyyaml tqdm scikit-learn psutil")
        return 1


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  JENGA-AI ENVIRONMENT CHECK")
    print("  Testing Framework Validation Script")
    print("=" * 70)
    print(f"\n  Date: {platform.node()}")
    print(f"  User: {os.getenv('USER', 'unknown')}")
    print(f"  Working Directory: {os.getcwd()}")
    
    exit_code = generate_report()
    
    print("\n" + "=" * 70)
    print("  For detailed testing plan, see: todo.md")
    print("  For testing summary, see: TESTING_SUMMARY.md")
    print("=" * 70 + "\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


