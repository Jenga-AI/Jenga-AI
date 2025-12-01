#!/usr/bin/env python3
"""
MLflow Initialization Script for Jenga-AI

This script initializes MLflow for the project, validates the setup,
and creates default experiments.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    import yaml
except ImportError as e:
    print(f"❌ Error: Required package not found: {e}")
    print("\nPlease install MLflow:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def validate_installation():
    """Validate MLflow installation."""
    print_section("Validating MLflow Installation")
    
    try:
        version = mlflow.__version__
        print(f"✓ MLflow version: {version}")
        return True
    except Exception as e:
        print(f"❌ MLflow validation failed: {e}")
        return False


def load_config():
    """Load MLflow configuration."""
    print_section("Loading Configuration")
    
    config_path = project_root / "mlflow_config.yaml"
    
    if not config_path.exists():
        print(f"⚠ Configuration file not found: {config_path}")
        print("  Using default configuration...")
        return {
            "tracking": {
                "uri": "./mlruns",
                "default_experiment": "jenga-ai-experiments",
                "artifact_location": "./mlruns"
            },
            "server": {
                "host": "127.0.0.1",
                "port": 5000
            }
        }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def setup_tracking_uri(config):
    """Set up MLflow tracking URI."""
    print_section("Setting Up Tracking URI")
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["tracking"]["uri"]
    
    # Create mlruns directory if using local storage
    if tracking_uri == "./mlruns" or tracking_uri.startswith("./"):
        mlruns_path = project_root / tracking_uri.lstrip("./")
        mlruns_path.mkdir(exist_ok=True)
        print(f"✓ MLruns directory: {mlruns_path}")
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ Tracking URI set to: {tracking_uri}")
    
    return tracking_uri


def create_default_experiment(config):
    """Create default experiment if it doesn't exist."""
    print_section("Setting Up Default Experiment")
    
    experiment_name = config["tracking"]["default_experiment"]
    artifact_location = config["tracking"].get("artifact_location")
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is not None:
        print(f"✓ Experiment already exists: {experiment_name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        return experiment.experiment_id
    
    # Create new experiment
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=artifact_location
    )
    print(f"✓ Created experiment: {experiment_name}")
    print(f"  Experiment ID: {experiment_id}")
    
    return experiment_id


def test_logging(experiment_name):
    """Test basic MLflow logging."""
    print_section("Testing MLflow Logging")
    
    mlflow.set_experiment(experiment_name)
    
    try:
        with mlflow.start_run(run_name="initialization_test") as run:
            # Log some test parameters
            mlflow.log_param("test_param", "initialization")
            mlflow.log_param("project", "jenga-ai")
            
            # Log some test metrics
            mlflow.log_metric("test_metric", 1.0)
            
            # Log a test artifact
            test_file = project_root / "test_artifact.txt"
            with open(test_file, 'w') as f:
                f.write("MLflow initialization test artifact")
            mlflow.log_artifact(str(test_file))
            test_file.unlink()  # Clean up
            
            print(f"✓ Test run created successfully")
            print(f"  Run ID: {run.info.run_id}")
        
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def list_existing_experiments():
    """List existing experiments."""
    print_section("Existing Experiments")
    
    client = MlflowClient()
    experiments = client.search_experiments()
    
    if not experiments:
        print("  No experiments found.")
        return
    
    for exp in experiments:
        print(f"\n  Name: {exp.name}")
        print(f"  ID: {exp.experiment_id}")
        print(f"  Artifact Location: {exp.artifact_location}")
        print(f"  Lifecycle Stage: {exp.lifecycle_stage}")


def print_next_steps(config):
    """Print next steps for the user."""
    print_section("Next Steps")
    
    print("\n1. Start the MLflow UI server:")
    print(f"   bash scripts/start_mlflow_server.sh")
    print("\n2. Access the MLflow UI:")
    print(f"   http://{config['server']['host']}:{config['server']['port']}")
    print("\n3. Use MLflow in your training scripts:")
    print("   from multitask_bert.utils.mlflow_utils import initialize_mlflow")
    print("   initialize_mlflow()")
    print("\n4. View the example script:")
    print("   python examples/mlflow_example.py")


def main():
    """Main initialization function."""
    print("\n" + "=" * 60)
    print("  MLflow Initialization - Jenga-AI Project")
    print("=" * 60)
    
    # Step 1: Validate installation
    if not validate_installation():
        sys.exit(1)
    
    # Step 2: Load configuration
    config = load_config()
    
    # Step 3: Setup tracking URI
    tracking_uri = setup_tracking_uri(config)
    
    # Step 4: Create default experiment
    experiment_id = create_default_experiment(config)
    
    # Step 5: Test logging
    test_success = test_logging(config["tracking"]["default_experiment"])
    
    # Step 6: List existing experiments
    list_existing_experiments()
    
    # Step 7: Print summary
    print_section("Initialization Summary")
    
    if test_success:
        print("\n✓ MLflow initialization completed successfully!")
        print(f"\n  Tracking URI: {tracking_uri}")
        print(f"  Default Experiment: {config['tracking']['default_experiment']}")
        print(f"  Experiment ID: {experiment_id}")
    else:
        print("\n⚠ MLflow initialization completed with warnings.")
        print("  Please check the errors above.")
    
    # Step 8: Print next steps
    print_next_steps(config)
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
