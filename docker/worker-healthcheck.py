#!/usr/bin/env python3
"""
JengaHub Worker Health Check

This script performs health checks specific to worker containers including
training job status, resource monitoring, and distributed training connectivity.
"""

import sys
import os
import time
import json
import socket
from pathlib import Path

def check_worker_status():
    """Check if the worker process is running."""
    try:
        # Check for worker status file
        status_file = Path('/app/tmp/worker_status.json')
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            last_heartbeat = status.get('last_heartbeat', 0)
            current_time = time.time()
            
            # Check if heartbeat is recent (within last 5 minutes)
            if current_time - last_heartbeat < 300:
                worker_type = status.get('worker_type', 'unknown')
                print(f"✓ Worker ({worker_type}) is active")
                return True
            else:
                print(f"✗ Worker heartbeat is stale ({current_time - last_heartbeat:.0f}s ago)")
                return False
        else:
            # If status file doesn't exist, check if this is a new worker
            uptime = time.time() - os.path.getctime('/proc/self')
            if uptime < 120:  # Less than 2 minutes old
                print("✓ New worker starting up")
                return True
            else:
                print("✗ Worker status file missing")
                return False
    
    except Exception as e:
        print(f"✗ Worker status check failed: {str(e)}")
        return False

def check_training_resources():
    """Check training-specific resources."""
    try:
        # Check GPU availability for training workers
        gpu_required = os.environ.get('JENGAHUB_REQUIRE_GPU', 'false').lower() == 'true'
        
        if gpu_required:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    print(f"✓ {gpu_count} GPU(s) available for training")
                    
                    # Check GPU memory
                    for i in range(gpu_count):
                        memory_free = torch.cuda.mem_get_info(i)[0] / 1e9  # GB
                        memory_total = torch.cuda.mem_get_info(i)[1] / 1e9  # GB
                        memory_used_pct = (memory_total - memory_free) / memory_total * 100
                        
                        if memory_used_pct > 95:
                            print(f"⚠ GPU {i} memory usage high: {memory_used_pct:.1f}%")
                        else:
                            print(f"✓ GPU {i} memory: {memory_used_pct:.1f}% used")
                    
                    return True
                else:
                    print("✗ CUDA not available but required for training")
                    return False
            except ImportError:
                print("✗ PyTorch not available for GPU check")
                return False
        else:
            print("✓ CPU-only training mode")
            return True
    
    except Exception as e:
        print(f"⚠ Training resource check failed: {str(e)}")
        return True

def check_distributed_training():
    """Check distributed training connectivity."""
    try:
        # Check if this is a distributed training setup
        world_size = os.environ.get('WORLD_SIZE')
        rank = os.environ.get('RANK')
        master_addr = os.environ.get('MASTER_ADDR')
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        if world_size and rank:
            print(f"✓ Distributed training mode: rank {rank}/{world_size}")
            
            # Check master connectivity
            if master_addr and master_addr != 'localhost':
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((master_addr, int(master_port)))
                    sock.close()
                    
                    if result == 0:
                        print(f"✓ Can connect to master at {master_addr}:{master_port}")
                    else:
                        print(f"⚠ Cannot connect to master at {master_addr}:{master_port}")
                        return False
                except Exception as e:
                    print(f"⚠ Master connectivity check failed: {str(e)}")
                    return False
        else:
            print("✓ Single-node training mode")
        
        return True
    
    except Exception as e:
        print(f"⚠ Distributed training check failed: {str(e)}")
        return True

def check_experiment_tracking():
    """Check experiment tracking services."""
    try:
        # Check MLflow tracking directory
        mlflow_dir = Path(os.environ.get('MLFLOW_TRACKING_URI', 'file:///app/logs/mlflow').replace('file://', ''))
        if mlflow_dir.exists() and mlflow_dir.is_dir():
            print("✓ MLflow tracking directory accessible")
        else:
            print("⚠ MLflow tracking directory not found")
        
        # Check if Weights & Biases is configured
        wandb_dir = Path(os.environ.get('WANDB_DIR', '/app/logs'))
        if wandb_dir.exists() and os.access(wandb_dir, os.W_OK):
            print("✓ Weights & Biases directory accessible")
        else:
            print("⚠ Weights & Biases directory not writable")
        
        return True
    
    except Exception as e:
        print(f"⚠ Experiment tracking check failed: {str(e)}")
        return True

def check_data_availability():
    """Check if training data is available."""
    try:
        data_dir = Path(os.environ.get('JENGAHUB_DATA_DIR', '/app/data'))
        
        if not data_dir.exists():
            print("⚠ Data directory does not exist")
            return False
        
        # Check if data directory has any files
        data_files = list(data_dir.rglob('*'))
        file_count = len([f for f in data_files if f.is_file()])
        
        if file_count > 0:
            print(f"✓ Data directory contains {file_count} files")
        else:
            print("⚠ Data directory is empty")
        
        return True
    
    except Exception as e:
        print(f"⚠ Data availability check failed: {str(e)}")
        return True

def check_checkpoint_storage():
    """Check checkpoint storage availability."""
    try:
        checkpoint_dir = Path(os.environ.get('JENGAHUB_CHECKPOINT_DIR', '/app/checkpoints'))
        
        # Ensure checkpoint directory exists and is writable
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = checkpoint_dir / '.health_check_test'
        test_file.write_text('test')
        test_file.unlink()
        
        print("✓ Checkpoint storage accessible")
        return True
    
    except Exception as e:
        print(f"✗ Checkpoint storage check failed: {str(e)}")
        return False

def main():
    """Main worker health check function."""
    print("JengaHub Worker Health Check")
    print("=" * 50)
    
    checks = [
        ("Worker Status", check_worker_status),
        ("Training Resources", check_training_resources),
        ("Distributed Training", check_distributed_training),
        ("Experiment Tracking", check_experiment_tracking),
        ("Data Availability", check_data_availability),
        ("Checkpoint Storage", check_checkpoint_storage),
    ]
    
    all_passed = True
    critical_failed = False
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        try:
            if not check_func():
                all_passed = False
                # Mark as critical failure for certain checks
                if check_name in ["Worker Status", "Checkpoint Storage"]:
                    critical_failed = True
        except Exception as e:
            print(f"✗ {check_name} check failed: {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if critical_failed:
        print("✗ Critical health checks failed")
        sys.exit(1)
    elif all_passed:
        print("✓ All health checks passed")
        sys.exit(0)
    else:
        print("⚠ Some non-critical health checks failed")
        sys.exit(0)  # Allow container to continue running

if __name__ == "__main__":
    main()