"""
Distributed Training Support for JengaHub

This module provides comprehensive distributed training capabilities including
multi-GPU, multi-node training with advanced synchronization and optimization.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import os
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import subprocess
import socket

from .trainer import JengaHubTrainer
from ..core.config import MultiModalConfig
from ..core.model import JengaHubMultiModalModel
from ..data.processor import UnifiedDataProcessor


class DistributedTrainer(JengaHubTrainer):
    """
    Extended JengaHub trainer with advanced distributed training capabilities.
    """
    
    def __init__(
        self,
        model: JengaHubMultiModalModel,
        config: MultiModalConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        callbacks: List = None,
        logger = None,
        output_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        local_rank: int = -1,
        find_unused_parameters: bool = False
    ):
        # Initialize distributed training first
        self.local_rank = local_rank
        self.find_unused_parameters = find_unused_parameters
        
        # Setup distributed environment
        self._setup_distributed()
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            callbacks=callbacks,
            logger=logger,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Wrap model with DDP after moving to device
        if self.is_distributed:
            self.model = self._wrap_model_ddp()
        
        self.training_logger.info(f"Distributed trainer initialized on rank {self.rank}/{self.world_size}")
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        
        if self.local_rank == -1:
            # Try to get from environment
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            # Setup backend
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            
            # Initialize distributed training
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                timeout=torch.distributed.get_timeout()
            )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
    
    def _wrap_model_ddp(self) -> DDP:
        """Wrap model with DistributedDataParallel."""
        
        return DDP(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=True  # Memory optimization
        )
    
    def _save_checkpoint(self):
        """Save checkpoint only on rank 0."""
        if self.rank == 0:
            super()._save_checkpoint()
    
    def _save_best_model(self):
        """Save best model only on rank 0."""
        if self.rank == 0:
            super()._save_best_model()


def setup_distributed_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl"
):
    """
    Setup distributed training environment for a specific process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
        backend: Communication backend (nccl, gloo)
    """
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set CUDA device for this process
    if torch.cuda.is_available() and backend == 'nccl':
        torch.cuda.set_device(rank % torch.cuda.device_count())


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedLauncher:
    """Launcher for distributed training jobs."""
    
    def __init__(
        self,
        train_function: Callable,
        config: MultiModalConfig,
        num_gpus: int = None,
        num_nodes: int = 1,
        master_addr: str = "localhost",
        master_port: str = "12355"
    ):
        self.train_function = train_function
        self.config = config
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.num_nodes = num_nodes
        self.master_addr = master_addr
        self.master_port = master_port
        
        self.world_size = self.num_gpus * self.num_nodes
        
        self.logger = logging.getLogger(__name__)
    
    def launch(self):
        """Launch distributed training."""
        
        if self.world_size == 1:
            # Single process training
            self.logger.info("Launching single-process training")
            self.train_function(0, self.config)
        
        elif self.num_nodes == 1:
            # Single-node multi-GPU training
            self.logger.info(f"Launching single-node training on {self.num_gpus} GPUs")
            mp.spawn(
                self._train_worker,
                args=(self.world_size,),
                nprocs=self.num_gpus,
                join=True
            )
        
        else:
            # Multi-node training
            self.logger.info(f"Launching multi-node training: {self.num_nodes} nodes, {self.num_gpus} GPUs per node")
            self._launch_multinode()
    
    def _train_worker(self, rank: int, world_size: int):
        """Training worker for distributed training."""
        
        # Setup distributed environment
        setup_distributed_training(
            rank=rank,
            world_size=world_size,
            master_addr=self.master_addr,
            master_port=self.master_port
        )
        
        try:
            # Run training function
            self.train_function(rank, self.config)
        
        finally:
            # Cleanup
            cleanup_distributed()
    
    def _launch_multinode(self):
        """Launch multi-node distributed training."""
        
        # This would typically be handled by a job scheduler like SLURM
        # For now, provide instructions for manual setup
        
        self.logger.info("Multi-node training setup:")
        self.logger.info(f"Master node: {self.master_addr}:{self.master_port}")
        self.logger.info(f"World size: {self.world_size}")
        self.logger.info("Run the following command on each node:")
        
        command = f"""
        python -m torch.distributed.launch \\
            --nproc_per_node={self.num_gpus} \\
            --nnodes={self.num_nodes} \\
            --node_rank=<NODE_RANK> \\
            --master_addr={self.master_addr} \\
            --master_port={self.master_port} \\
            your_training_script.py
        """
        
        self.logger.info(command)
        
        raise NotImplementedError(
            "Multi-node training requires manual setup or job scheduler integration"
        )


def create_distributed_dataloader(
    dataset: List,
    batch_size: int,
    collate_fn: Callable,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create DataLoader with distributed sampler.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        collate_fn: Collation function
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader with distributed sampler
    """
    
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )
        # Don't shuffle in DataLoader when using DistributedSampler
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )


class DistributedMetrics:
    """Utilities for handling metrics in distributed training."""
    
    @staticmethod
    def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """
        Reduce tensor across all processes.
        
        Args:
            tensor: Tensor to reduce
            average: Whether to average or sum
            
        Returns:
            Reduced tensor
        """
        if not dist.is_initialized():
            return tensor
        
        # Clone tensor to avoid modifying original
        reduced_tensor = tensor.clone()
        
        # All-reduce operation
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
        
        if average:
            reduced_tensor /= dist.get_world_size()
        
        return reduced_tensor
    
    @staticmethod
    def gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Gather tensor from all processes to rank 0.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of tensors from all processes (only valid on rank 0)
        """
        if not dist.is_initialized():
            return [tensor]
        
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_tensors, tensor)
        
        return gathered_tensors
    
    @staticmethod
    def reduce_dict(metric_dict: Dict[str, float], average: bool = True) -> Dict[str, float]:
        """
        Reduce dictionary of metrics across all processes.
        
        Args:
            metric_dict: Dictionary of metrics
            average: Whether to average or sum
            
        Returns:
            Reduced metrics dictionary
        """
        if not dist.is_initialized():
            return metric_dict
        
        reduced_dict = {}
        
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value, device=torch.cuda.current_device())
                reduced_tensor = DistributedMetrics.reduce_tensor(tensor_value, average)
                reduced_dict[key] = reduced_tensor.item()
            else:
                reduced_dict[key] = value
        
        return reduced_dict


def synchronize():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get world size."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class DistributedCheckpoint:
    """Enhanced checkpointing for distributed training."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        global_step: int,
        config: Dict[str, Any]
    ):
        """Save distributed checkpoint."""
        
        if not is_main_process():
            return
        
        # Get model state dict (unwrap DDP if needed)
        if isinstance(model, DDP):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
            'world_size': get_world_size(),
            'rng_states': self._get_rng_states()
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.output_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"Distributed checkpoint saved: {checkpoint_path}")
    
    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, Any]:
        """Load distributed checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state (handle DDP)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore RNG states
        self._set_rng_states(checkpoint.get('rng_states', {}))
        
        self.logger.info(f"Distributed checkpoint loaded: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'config': checkpoint.get('config', {})
        }
    
    def _get_rng_states(self) -> Dict[str, Any]:
        """Get random number generator states."""
        states = {
            'python': None,  # Python RNG state is not serializable
            'numpy': None,   # NumPy state handling would go here
            'torch': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            states['cuda'] = torch.cuda.get_rng_state()
        
        return states
    
    def _set_rng_states(self, states: Dict[str, Any]):
        """Set random number generator states."""
        if 'torch' in states:
            torch.set_rng_state(states['torch'])
        
        if 'cuda' in states and torch.cuda.is_available():
            torch.cuda.set_rng_state(states['cuda'])


def get_distributed_info() -> Dict[str, Any]:
    """Get comprehensive distributed training information."""
    
    info = {
        'distributed': dist.is_initialized(),
        'rank': get_rank(),
        'world_size': get_world_size(),
        'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
        'backend': dist.get_backend() if dist.is_initialized() else None,
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device': torch.cuda.current_device(),
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(),
        })
    
    return info