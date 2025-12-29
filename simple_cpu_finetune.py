#!/usr/bin/env python3
"""
Simplified CPU-Friendly Finetuning Script
Minimal dependencies version for testing with synthetic data
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def process_synthetic_data():
    """Process synthetic test data for training"""
    print("\n=== Processing Synthetic Data ===")
    
    data_path = project_root / "tests/outputs/test_data.json"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return None
    
    # Read and process first 50 samples
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data[:50]:  # Use only 50 samples for CPU efficiency
        text = item.get('text', '')
        if text:
            processed.append({
                'text': text,
                'sample_id': item.get('sample_id', 'unknown'),
                'quality': item.get('quality_level', 'unknown')
            })
    
    # Save processed data
    output_path = project_root / "cpu_training_data_simple.json"
    with open(output_path, 'w') as f:
        json.dump(processed, f, indent=2)
    
    print(f"✓ Processed {len(processed)} samples")
    print(f"✓ Saved to: {output_path}")
    
    return str(output_path)

def setup_mlflow_simple():
    """Simple MLflow setup without full dependencies"""
    print("\n=== Setting up MLflow (Simple) ===")
    
    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    
    # Create experiment directory
    exp_dir = mlruns_dir / "cpu-finetuning-experiment"
    exp_dir.mkdir(exist_ok=True)
    
    # Create simple run tracking
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = exp_dir / f"run_{run_id}"
    run_dir.mkdir(exist_ok=True)
    
    # Log basic info
    info_file = run_dir / "run_info.json"
    run_info = {
        "run_id": run_id,
        "experiment": "cpu-finetuning",
        "start_time": datetime.now().isoformat(),
        "status": "started"
    }
    
    with open(info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    print(f"✓ Created tracking directory: {run_dir}")
    return run_dir

def run_finetuning_simulation(data_path, run_dir):
    """Simulate finetuning process (placeholder for actual training)"""
    print("\n=== Starting Finetuning Simulation ===")
    print("Note: This is a placeholder. Actual training requires full dependencies.")
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} training samples")
    
    # Simulate training steps
    num_epochs = 1
    batch_size = 1
    steps_per_epoch = len(data) // batch_size
    
    metrics = {
        "epochs": num_epochs,
        "batch_size": batch_size,
        "total_samples": len(data),
        "steps_per_epoch": steps_per_epoch
    }
    
    print("\n--- Training Configuration ---")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Save metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Update run status
    info_file = run_dir / "run_info.json"
    with open(info_file, 'r') as f:
        run_info = json.load(f)
    
    run_info["status"] = "completed"
    run_info["end_time"] = datetime.now().isoformat()
    
    with open(info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    return metrics

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("  Simplified CPU Finetuning with Synthetic Data")
    print("="*60)
    
    # Process data
    data_path = process_synthetic_data()
    if not data_path:
        print("❌ Failed to process data")
        return
    
    # Setup tracking
    run_dir = setup_mlflow_simple()
    
    # Run training simulation
    metrics = run_finetuning_simulation(data_path, run_dir)
    
    # Summary
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"\n✓ Data processed: {data_path}")
    print(f"✓ Run tracked in: {run_dir}")
    print(f"✓ Total samples: {metrics['total_samples']}")
    
    print("\n--- Next Steps ---")
    print("1. Install full dependencies:")
    print("   uv pip install torch transformers peft mlflow")
    print("\n2. Run full finetuning:")
    print("   python3 run_cpu_finetuning.py")
    print("\n3. View MLflow UI (when installed):")
    print("   mlflow ui --host 0.0.0.0 --port 5000")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()