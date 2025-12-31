
import mlflow
import logging
import socket
from urllib.parse import urlparse
from typing import Dict, Any, Optional
from datetime import datetime
from seq2seq_models.core.config import MLflowConfig

logger = logging.getLogger(__name__)

class MLflowManager:
    """Manages MLflow tracking - centralized, resilient"""
    
    def __init__(self, config: MLflowConfig):
        """Initialize MLflow manager"""
        self.mlflow_config = config
        self.run_id = None
        self.enabled = self.mlflow_config.log_metrics
        
        if not self.enabled:
            logger.info("ℹ️ MLflow logging disabled in config")
            return
        
        logger.info("✅ MLflowManager initialized")
    
    def _check_connectivity(self, uri: str, timeout: int = 5) -> bool:
        """Check if MLflow server is reachable"""
        try:
            parsed = urlparse(uri)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            # Skip check for local file paths or databricks
            if not host: 
                return True

            logger.info(f"🔍 Checking connectivity to {host}:{port}...")
            
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.close()
            
            logger.info(f"✅ MLflow server is reachable")
            return True
            
        except socket.gaierror as e:
            logger.warning(f"⚠️ DNS resolution failed: {e}")
            return False
        except socket.timeout:
            logger.warning(f"⚠️ Connection timeout to MLflow server")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Could not reach MLflow server: {e}")
            return False
    
    def setup_tracking(self):
        """Setup MLflow tracking connection"""
        if not self.enabled:
            return
        
        try:
            tracking_uri = self.mlflow_config.tracking_uri
            experiment_name = self.mlflow_config.experiment_name
            
            # Strip whitespace from URI and experiment name
            tracking_uri = tracking_uri.strip()
            experiment_name = experiment_name.strip()
            
            logger.info(f"🔍 Checking connectivity to {tracking_uri}...")
            
            # Check connectivity before setting up
            if not self._check_connectivity(tracking_uri):
                logger.error(f"❌ Cannot reach MLflow server at {tracking_uri}")
                logger.error("   Will proceed with file logging only (MLflow disabled)")
                self.enabled = False
                return
            
            # Set socket timeout for mlflow operations
            socket.setdefaulttimeout(10)
            
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"✅ MLflow tracking URI set: {tracking_uri}")
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    logger.info(f"✅ Created experiment: {experiment_name}")
                else:
                    logger.info(f"✅ Using experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not manage experiment: {e}")
            
            mlflow.set_experiment(experiment_name)
            
        except Exception as e:
            logger.error(f"❌ MLflow setup failed: {e}")
            self.enabled = False
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start MLflow run"""
        if not self.enabled:
            return None
        
        try:
            if run_name is None:
                run_name = self.mlflow_config.run_name
            
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"translation_{timestamp}"
            
            run = mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            
            logger.info(f"✅ MLflow run started: {run_name}")
            logger.info(f"   Run ID: {self.run_id}")
            logger.info(f"   Tracking URI: {self.mlflow_config.tracking_uri}")
            
            return self.run_id
            
        except Exception as e:
            logger.error(f"❌ Failed to start MLflow run: {e}")
            self.enabled = False
            return None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow (with file fallback)"""
        if not self.enabled or not mlflow.active_run():
            # File logging still happens via logger in the trainer
            return
        
        try:
            valid_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if not (isinstance(value, float) and (value != value or value == float('inf'))):
                        valid_metrics[key] = float(value)
            
            if valid_metrics:
                mlflow.log_metrics(valid_metrics, step=step)
                
        except Exception as e:
            logger.warning(f"⚠️ Error logging metrics to MLflow: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        if not self.enabled or not mlflow.active_run():
            return

        try:
            # Flatten and filter params
            flat_params = {}
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_params[k] = v
                elif v is None:
                    flat_params[k] = "None"
            
            mlflow.log_params(flat_params)
        except Exception as e:
            logger.warning(f"⚠️ Error logging params to MLflow: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: str = ""):
        """Log artifacts directory to MLflow"""
        if not self.enabled or not mlflow.active_run():
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            logger.info(f"✅ Logged artifacts from {local_dir} to MLflow")
        except Exception as e:
            logger.warning(f"⚠️ Could not log artifacts to MLflow: {e}")
    
    def end_run(self):
        """End MLflow run"""
        if not self.enabled:
            return
        
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info(f"✅ MLflow run ended: {self.run_id}")
        except Exception as e:
            logger.error(f"❌ Error ending MLflow run: {e}")
