#!/usr/bin/env python3
"""
JengaHub API Health Check

This script performs comprehensive health checks for the JengaHub API server
including service availability, model loading, and basic functionality.
"""

import sys
import requests
import time
import os
import json
from pathlib import Path

def check_api_health():
    """Check if the API server is responding."""
    try:
        # Get the port from environment or use default
        port = os.environ.get('JENGAHUB_API_PORT', '8000')
        host = 'localhost'
        
        # Basic health endpoint
        health_url = f"http://{host}:{port}/health"
        
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Check if all required services are healthy
            if health_data.get('status') == 'healthy':
                print("✓ API server is healthy")
                return True
            else:
                print(f"✗ API server reports unhealthy status: {health_data}")
                return False
        else:
            print(f"✗ API server returned status code: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server")
        return False
    except requests.exceptions.Timeout:
        print("✗ API server response timeout")
        return False
    except Exception as e:
        print(f"✗ API health check failed: {str(e)}")
        return False

def check_model_status():
    """Check if models are loaded and ready."""
    try:
        port = os.environ.get('JENGAHUB_API_PORT', '8000')
        host = 'localhost'
        
        # Model status endpoint
        models_url = f"http://{host}:{port}/models/status"
        
        response = requests.get(models_url, timeout=15)
        
        if response.status_code == 200:
            models_data = response.json()
            
            # Check if at least one model is loaded
            loaded_models = [
                model for model in models_data.get('models', [])
                if model.get('status') == 'loaded'
            ]
            
            if loaded_models:
                print(f"✓ {len(loaded_models)} model(s) loaded and ready")
                return True
            else:
                print("⚠ No models are currently loaded")
                # This might be acceptable for some deployments
                return True
        else:
            print(f"✗ Model status check failed: {response.status_code}")
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"⚠ Model status check failed: {str(e)}")
        # Model check is less critical than basic API health
        return True
    except Exception as e:
        print(f"⚠ Model status check error: {str(e)}")
        return True

def check_system_resources():
    """Check system resource availability."""
    try:
        import psutil
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        if memory_usage > 95:
            print(f"⚠ High memory usage: {memory_usage}%")
            return False
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        if disk_usage > 95:
            print(f"⚠ High disk usage: {disk_usage}%")
            return False
        
        # Check if CUDA is available and working (if GPU container)
        if os.environ.get('NVIDIA_VISIBLE_DEVICES'):
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✓ CUDA available with {torch.cuda.device_count()} device(s)")
                else:
                    print("⚠ CUDA not available in GPU container")
                    return False
            except ImportError:
                print("⚠ PyTorch not available for CUDA check")
        
        print(f"✓ System resources OK (Memory: {memory_usage}%, Disk: {disk_usage}%)")
        return True
    
    except ImportError:
        print("✓ psutil not available, skipping resource check")
        return True
    except Exception as e:
        print(f"⚠ System resource check failed: {str(e)}")
        return True

def check_file_permissions():
    """Check if necessary directories are writable."""
    try:
        required_dirs = [
            os.environ.get('JENGAHUB_DATA_DIR', '/app/data'),
            os.environ.get('JENGAHUB_LOG_DIR', '/app/logs'),
            os.environ.get('JENGAHUB_CACHE_DIR', '/app/cache'),
            '/app/tmp'
        ]
        
        for dir_path in required_dirs:
            try:
                # Try to create a test file
                test_file = Path(dir_path) / '.health_check_test'
                test_file.write_text('test')
                test_file.unlink()
                
            except (OSError, PermissionError) as e:
                print(f"✗ Cannot write to {dir_path}: {str(e)}")
                return False
        
        print("✓ File permissions OK")
        return True
    
    except Exception as e:
        print(f"⚠ File permission check failed: {str(e)}")
        return True

def main():
    """Main health check function."""
    print("JengaHub Health Check")
    print("=" * 50)
    
    checks = [
        ("API Health", check_api_health),
        ("Model Status", check_model_status),
        ("System Resources", check_system_resources),
        ("File Permissions", check_file_permissions),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name} check failed: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("✓ All health checks passed")
        sys.exit(0)
    else:
        print("✗ Some health checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()