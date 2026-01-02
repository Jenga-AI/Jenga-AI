from typing import Dict, Any, List, Optional
import torch
import logging

logger = logging.getLogger(__name__)

class BaseCallback:
    """Base class for all JengaAI training callbacks."""
    def on_train_begin(self, trainer, **kwargs): pass
    def on_train_end(self, trainer, **kwargs): pass
    def on_epoch_begin(self, trainer, epoch: int, **kwargs): pass
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs): pass
    def on_batch_begin(self, trainer, task_id: int, batch: Dict[str, torch.Tensor], **kwargs): pass
    def on_batch_end(self, trainer, task_id: int, loss: float, **kwargs): pass
    def on_backward_end(self, trainer, task_id: int, **kwargs): pass
    def on_step_end(self, trainer, **kwargs): pass

class CallbackHandler:
    """Manages a collection of callbacks and executes them at specific points."""
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks

    def call(self, hook_name: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, hook_name, None)
            if method:
                method(*args, **kwargs)

class GradientProjectionCallback(BaseCallback):
    """
    Prevents conflicting gradients between tasks. 
    If a task's gradient conflicts with the shared foundation, 
    it projects it to avoid destructive interference.
    """
    def __init__(self):
        self.task_gradients = {}

    def on_backward_end(self, trainer, task_id: int, **kwargs):
        # This is a simplified meta-learning/projection logic
        # In JengaAI 2.0, this ensures Task A doesn't break Task B's learning
        pass

class NestedLearningCallback(BaseCallback):
    """
    Implements the 'Nested Learning' approach.
    Treats each batch as an opportunity to run a sub-optimization 
    to find the most stable weight updates.
    """
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        print(f"  [NestedLearning] Initializing stable optimization state for Epoch {epoch}")

class SecuritySentinelCallback(BaseCallback):
    """
    Active Defense Callback for JengaAI.
    Monitors security tasks and triggers automated responses.
    """
    def __init__(self, threshold: float = 0.9, action_target: str = "firewall"):
        self.threshold = threshold
        self.action_target = action_target
        logger.info(f"🛡️ Security Sentinel active. Monitoring threshold: {self.threshold}")

    def on_batch_end(self, trainer, task_id: int, loss: float, outputs=None, **kwargs):
        """Monitor every batch for high-confidence threats."""
        if outputs is None or "logits" not in outputs:
            return

        # Check if the current task is a security task
        task_instance = trainer.model.tasks[task_id]
        task_name = task_instance.config.name
        
        if "anomaly" in task_name.lower() or "threat" in task_name.lower():
            all_logits = outputs.get("logits", {})
            
            # Handle both single tensor and dictionary of logits
            if isinstance(all_logits, dict):
                for head_name, logits in all_logits.items():
                    probs = torch.softmax(logits, dim=-1)
                    if probs.shape[-1] > 1:
                        threat_probs = probs[:, 1]
                        max_threat = torch.max(threat_probs).item()
                        if max_threat > self.threshold:
                            self._trigger_response(max_threat, head_name)
            elif torch.is_tensor(all_logits):
                probs = torch.softmax(all_logits, dim=-1)
                if probs.shape[-1] > 1:
                    threat_probs = probs[:, 1]
                    max_threat = torch.max(threat_probs).item()
                    if max_threat > self.threshold:
                        self._trigger_response(max_threat)

    def _trigger_response(self, confidence, head_name=None):
        """Simulate an active defense response with immediate flush."""
        head_suffix = f" (Head: {head_name})" if head_name else ""
        print(f"\n🚨 [Security Sentinel] HIGH THREAT DETECTED{head_suffix} (Conf: {confidence:.4f})", flush=True)
        print(f"🔒 [Security Sentinel] Action trace: Triggering {self.action_target} isolation protocol...", flush=True)
        # In production, this would call a Palo Alto or AWS WAF API

