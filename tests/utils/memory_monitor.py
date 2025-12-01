#!/usr/bin/env python3
"""
Memory Monitoring Utility
==========================
Monitors and logs memory usage during training/testing.

Usage:
    # In your training script:
    from tests.utils.memory_monitor import MemoryMonitor
    
    monitor = MemoryMonitor()
    monitor.start()
    
    # ... your training code ...
    
    monitor.log_checkpoint("After model load")
    # ... more code ...
    monitor.log_checkpoint("After first epoch")
    
    monitor.stop()
    monitor.print_report()
"""

import psutil
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional
import sys


class MemoryMonitor:
    """
    Monitors memory usage and provides logging/reporting functionality.
    """
    
    def __init__(self, interval: float = 1.0, name: str = "Memory Monitor"):
        """
        Initialize memory monitor.
        
        Args:
            interval: Sampling interval in seconds
            name: Name for this monitoring session
        """
        self.interval = interval
        self.name = name
        self.process = psutil.Process()
        
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.checkpoints: List[Dict] = []
        self.samples: List[Dict] = []
        self.start_time: Optional[float] = None
        
    def start(self):
        """Start monitoring memory usage."""
        if self.monitoring:
            print("Warning: Monitor already running")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.checkpoints = []
        self.samples = []
        
        # Log initial state
        self.log_checkpoint("Start")
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"✓ {self.name} started")
    
    def stop(self):
        """Stop monitoring memory usage."""
        if not self.monitoring:
            print("Warning: Monitor not running")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Log final state
        self.log_checkpoint("Stop")
        
        print(f"✓ {self.name} stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self._get_memory_info()
                self.samples.append({
                    "timestamp": time.time() - self.start_time,
                    **memory_info
                })
            except Exception as e:
                print(f"Warning: Memory monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def _get_memory_info(self) -> Dict:
        """Get current memory usage information."""
        mem_info = self.process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
            "system_used_mb": virtual_mem.used / (1024 * 1024),
            "system_available_mb": virtual_mem.available / (1024 * 1024),
            "system_percent": virtual_mem.percent,
        }
    
    def log_checkpoint(self, label: str):
        """
        Log a named checkpoint for memory usage.
        
        Args:
            label: Description of this checkpoint
        """
        if not self.start_time:
            self.start_time = time.time()
        
        memory_info = self._get_memory_info()
        checkpoint = {
            "label": label,
            "timestamp": time.time() - self.start_time,
            **memory_info
        }
        self.checkpoints.append(checkpoint)
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.samples and not self.checkpoints:
            return 0.0
        
        all_measurements = self.samples + self.checkpoints
        return max(m["rss_mb"] for m in all_measurements)
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self._get_memory_info()["rss_mb"]
    
    def print_report(self):
        """Print a formatted memory usage report."""
        if not self.checkpoints:
            print("No checkpoints recorded")
            return
        
        print("\n" + "=" * 80)
        print(f"  {self.name.upper()} - MEMORY USAGE REPORT")
        print("=" * 80)
        
        # Checkpoints
        print("\nCheckpoints:")
        print("-" * 80)
        print(f"{'Label':<30} {'Time (s)':<12} {'RSS (MB)':<15} {'System %':<10}")
        print("-" * 80)
        
        for cp in self.checkpoints:
            print(f"{cp['label']:<30} {cp['timestamp']:>10.2f}  "
                  f"{cp['rss_mb']:>12.2f}    {cp['system_percent']:>8.1f}%")
        
        # Statistics
        if len(self.checkpoints) > 1:
            print("\n" + "-" * 80)
            print("Statistics:")
            print("-" * 80)
            
            initial = self.checkpoints[0]["rss_mb"]
            peak = self.get_peak_memory()
            final = self.checkpoints[-1]["rss_mb"]
            increase = final - initial
            
            print(f"  Initial Memory:    {initial:>10.2f} MB")
            print(f"  Peak Memory:       {peak:>10.2f} MB")
            print(f"  Final Memory:      {final:>10.2f} MB")
            print(f"  Memory Increase:   {increase:>10.2f} MB ({increase/initial*100:>5.1f}%)")
            
            duration = self.checkpoints[-1]["timestamp"]
            print(f"\n  Total Duration:    {duration:>10.2f} seconds")
        
        # System info
        virtual_mem = psutil.virtual_memory()
        print("\n" + "-" * 80)
        print("System Memory:")
        print("-" * 80)
        print(f"  Total:             {virtual_mem.total / (1024**3):>10.2f} GB")
        print(f"  Available:         {virtual_mem.available / (1024**3):>10.2f} GB")
        print(f"  Used:              {virtual_mem.used / (1024**3):>10.2f} GB ({virtual_mem.percent:.1f}%)")
        
        print("=" * 80 + "\n")
    
    def save_report(self, filepath: str):
        """Save memory usage data to a file."""
        import json
        
        data = {
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "checkpoints": self.checkpoints,
            "peak_memory_mb": self.get_peak_memory(),
            "system_info": {
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count": psutil.cpu_count(),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Memory report saved to: {filepath}")


def demo():
    """Demo usage of MemoryMonitor."""
    import numpy as np
    
    print("\nMemory Monitor Demo")
    print("=" * 80)
    
    monitor = MemoryMonitor(interval=0.5, name="Demo")
    monitor.start()
    
    # Simulate some work
    print("\n1. Allocating 100MB array...")
    monitor.log_checkpoint("Before allocation")
    arr1 = np.random.rand(100 * 1024 * 1024 // 8)  # ~100MB
    time.sleep(1)
    monitor.log_checkpoint("After 100MB allocation")
    
    print("2. Allocating another 200MB...")
    arr2 = np.random.rand(200 * 1024 * 1024 // 8)  # ~200MB
    time.sleep(1)
    monitor.log_checkpoint("After 200MB allocation")
    
    print("3. Deleting first array...")
    del arr1
    time.sleep(1)
    monitor.log_checkpoint("After deleting 100MB array")
    
    print("4. Cleaning up...")
    del arr2
    time.sleep(1)
    monitor.log_checkpoint("After cleanup")
    
    monitor.stop()
    monitor.print_report()
    
    # Save report
    report_path = "/tmp/memory_demo_report.json"
    monitor.save_report(report_path)


def check_memory_safety(model_size_mb: float, batch_size: int, 
                       sequence_length: int, safety_margin_gb: float = 4.0) -> bool:
    """
    Check if there's enough memory for training.
    
    Args:
        model_size_mb: Estimated model size in MB
        batch_size: Training batch size
        sequence_length: Max sequence length
        safety_margin_gb: Safety margin in GB to keep free
        
    Returns:
        True if memory is sufficient, False otherwise
    """
    virtual_mem = psutil.virtual_memory()
    available_gb = virtual_mem.available / (1024**3)
    
    # Rough estimate: model + gradients + optimizer states + activations
    estimated_usage_mb = model_size_mb * 4  # Conservative estimate
    estimated_usage_mb += batch_size * sequence_length * 0.01  # Activations
    estimated_usage_gb = estimated_usage_mb / 1024
    
    is_safe = (available_gb - estimated_usage_gb) > safety_margin_gb
    
    print(f"\nMemory Safety Check:")
    print(f"  Available Memory:    {available_gb:.2f} GB")
    print(f"  Estimated Usage:     {estimated_usage_gb:.2f} GB")
    print(f"  Safety Margin:       {safety_margin_gb:.2f} GB")
    print(f"  Remaining:           {available_gb - estimated_usage_gb:.2f} GB")
    print(f"  Status:              {'✓ SAFE' if is_safe else '✗ UNSAFE'}")
    
    if not is_safe:
        print("\n⚠️  Warning: Insufficient memory. Consider:")
        print("  - Reducing batch size")
        print("  - Reducing sequence length")
        print("  - Using a smaller model")
        print("  - Enabling gradient accumulation")
    
    return is_safe


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        # Quick memory check
        print("\nSystem Memory Status:")
        print("=" * 80)
        virtual_mem = psutil.virtual_memory()
        print(f"  Total:     {virtual_mem.total / (1024**3):.2f} GB")
        print(f"  Available: {virtual_mem.available / (1024**3):.2f} GB")
        print(f"  Used:      {virtual_mem.used / (1024**3):.2f} GB ({virtual_mem.percent:.1f}%)")
        print(f"  Free:      {virtual_mem.free / (1024**3):.2f} GB")
        print("=" * 80)
        
        # Example safety check for bert-tiny
        print("\nExample: Training bert-tiny")
        check_memory_safety(model_size_mb=20, batch_size=2, sequence_length=64)
    else:
        print("Memory Monitor Utility")
        print("\nUsage:")
        print("  python memory_monitor.py demo    # Run demo")
        print("  python memory_monitor.py check   # Check system memory")
        print("\nOr import in your code:")
        print("  from tests.utils.memory_monitor import MemoryMonitor")


