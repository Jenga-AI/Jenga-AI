"""
Dynamic Model Serving with Smart Caching for JengaHub

This module provides intelligent model serving capabilities with adaptive
caching, load balancing, and automatic model selection based on language,
task, and performance requirements.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import time
import json
import hashlib
import pickle
import redis
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict
import logging
from contextlib import asynccontextmanager
import mlflow
import numpy as np
from collections import defaultdict, OrderedDict
from enum import Enum

from .config import MultiModalConfig
from .memory import ContinuumMemorySystem
from .nested_lora import NestedLoRALinear
from .code_switching import MultimodalCodeSwitchingBridge
from ..data.processor import ProcessedSample, UnifiedDataProcessor


class ModelType(Enum):
    """Types of models in the serving system."""
    BASE = "base"
    SPECIALIZED = "specialized"
    FAST = "fast"
    ACCURATE = "accurate"


class CacheLevel(Enum):
    """Cache levels for different types of data."""
    MEMORY = "memory"      # In-memory cache (fastest)
    REDIS = "redis"        # Redis cache (fast, persistent)
    DISK = "disk"          # Disk cache (slower, large capacity)


@dataclass
class ModelInstance:
    """Container for a model instance with metadata."""
    
    model: nn.Module
    config: MultiModalConfig
    model_type: ModelType
    languages: List[str]
    tasks: List[str]
    performance_metrics: Dict[str, float]
    load_time: float
    memory_usage: float
    last_used: float
    usage_count: int = 0
    is_loaded: bool = True
    device: str = "cpu"


@dataclass
class InferenceRequest:
    """Container for inference request."""
    
    request_id: str
    audio_path: Optional[str] = None
    text: Optional[str] = None
    task: str = "classification"
    language: Optional[str] = None
    priority: int = 0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResult:
    """Container for inference results."""
    
    request_id: str
    predictions: Dict[str, Any]
    confidence: float
    language_detected: Optional[str]
    code_switching_info: Optional[Dict[str, Any]]
    processing_time: float
    model_used: str
    cache_hit: bool = False


class SmartCache:
    """
    Multi-level intelligent cache with automatic eviction and warming.
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        redis_url: Optional[str] = None,
        disk_cache_path: Optional[str] = None,
        ttl: int = 3600  # Time to live in seconds
    ):
        self.memory_size = memory_size
        self.ttl = ttl
        
        # Memory cache (LRU)
        self.memory_cache = OrderedDict()
        self.memory_usage = 0
        
        # Redis cache
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}")
        
        # Disk cache
        self.disk_cache_path = Path(disk_cache_path) if disk_cache_path else None
        if self.disk_cache_path:
            self.disk_cache_path.mkdir(exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def _generate_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            'audio_path': request.audio_path,
            'text': request.text,
            'task': request.task,
            'language': request.language
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self, 
        request: InferenceRequest,
        level: CacheLevel = CacheLevel.MEMORY
    ) -> Optional[InferenceResult]:
        """Get cached result for request."""
        key = self._generate_key(request)
        
        with self._lock:
            # Try memory cache first
            if level == CacheLevel.MEMORY or level == CacheLevel.MEMORY:
                if key in self.memory_cache:
                    # Move to end (LRU)
                    result = self.memory_cache.pop(key)
                    self.memory_cache[key] = result
                    
                    # Check TTL
                    if time.time() - result.timestamp < self.ttl:
                        self.stats['hits'] += 1
                        result.cache_hit = True
                        return result
                    else:
                        # Expired, remove
                        del self.memory_cache[key]
            
            # Try Redis cache
            if self.redis_client and (level == CacheLevel.REDIS or level == CacheLevel.MEMORY):
                try:
                    cached_data = self.redis_client.get(f"jengahub:{key}")
                    if cached_data:
                        result = pickle.loads(cached_data)
                        
                        # Check TTL
                        if time.time() - result.timestamp < self.ttl:
                            self.stats['hits'] += 1
                            result.cache_hit = True
                            
                            # Warm memory cache
                            self._put_memory(key, result)
                            return result
                        else:
                            # Expired, remove
                            self.redis_client.delete(f"jengahub:{key}")
                
                except Exception as e:
                    logging.warning(f"Redis get failed: {e}")
            
            # Try disk cache
            if self.disk_cache_path and level == CacheLevel.DISK:
                cache_file = self.disk_cache_path / f"{key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        
                        # Check TTL
                        if time.time() - result.timestamp < self.ttl:
                            self.stats['hits'] += 1
                            result.cache_hit = True
                            
                            # Warm higher-level caches
                            self._put_memory(key, result)
                            if self.redis_client:
                                self._put_redis(key, result)
                            
                            return result
                        else:
                            # Expired, remove
                            cache_file.unlink()
                    
                    except Exception as e:
                        logging.warning(f"Disk cache read failed: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def put(
        self, 
        request: InferenceRequest, 
        result: InferenceResult,
        levels: List[CacheLevel] = None
    ):
        """Cache result for request."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        
        key = self._generate_key(request)
        result.timestamp = time.time()
        
        with self._lock:
            if CacheLevel.MEMORY in levels:
                self._put_memory(key, result)
            
            if CacheLevel.REDIS in levels and self.redis_client:
                self._put_redis(key, result)
            
            if CacheLevel.DISK in levels and self.disk_cache_path:
                self._put_disk(key, result)
    
    def _put_memory(self, key: str, result: InferenceResult):
        """Put result in memory cache."""
        # Evict if necessary
        while len(self.memory_cache) >= self.memory_size:
            oldest_key, _ = self.memory_cache.popitem(last=False)
            self.stats['evictions'] += 1
        
        self.memory_cache[key] = result
    
    def _put_redis(self, key: str, result: InferenceResult):
        """Put result in Redis cache."""
        try:
            serialized = pickle.dumps(result)
            self.redis_client.setex(
                f"jengahub:{key}", 
                self.ttl, 
                serialized
            )
        except Exception as e:
            logging.warning(f"Redis put failed: {e}")
    
    def _put_disk(self, key: str, result: InferenceResult):
        """Put result in disk cache."""
        try:
            cache_file = self.disk_cache_path / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logging.warning(f"Disk cache write failed: {e}")
    
    def clear(self, levels: List[CacheLevel] = None):
        """Clear cache at specified levels."""
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        
        with self._lock:
            if CacheLevel.MEMORY in levels:
                self.memory_cache.clear()
            
            if CacheLevel.REDIS in levels and self.redis_client:
                try:
                    # Delete all JengaHub keys
                    keys = self.redis_client.keys("jengahub:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logging.warning(f"Redis clear failed: {e}")
            
            if CacheLevel.DISK in levels and self.disk_cache_path:
                try:
                    for cache_file in self.disk_cache_path.glob("*.pkl"):
                        cache_file.unlink()
                except Exception as e:
                    logging.warning(f"Disk cache clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'total_requests': total_requests
        }


class ModelManager:
    """
    Manages multiple model instances with intelligent loading and selection.
    """
    
    def __init__(
        self,
        max_loaded_models: int = 3,
        device_preference: List[str] = None
    ):
        self.max_loaded_models = max_loaded_models
        self.device_preference = device_preference or ["cuda", "cpu"]
        
        # Model registry
        self.models: Dict[str, ModelInstance] = {}
        self.model_paths: Dict[str, str] = {}
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Background model management
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    def register_model(
        self,
        model_id: str,
        model_path: str,
        config: MultiModalConfig,
        model_type: ModelType = ModelType.BASE,
        languages: List[str] = None,
        tasks: List[str] = None
    ):
        """Register a model for serving."""
        self.model_paths[model_id] = model_path
        
        # Create model instance (not loaded yet)
        instance = ModelInstance(
            model=None,
            config=config,
            model_type=model_type,
            languages=languages or ["en"],
            tasks=tasks or ["classification"],
            performance_metrics={},
            load_time=0.0,
            memory_usage=0.0,
            last_used=0.0,
            usage_count=0,
            is_loaded=False
        )
        
        with self._lock:
            self.models[model_id] = instance
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model into memory."""
        if model_id not in self.models:
            return False
        
        with self._lock:
            instance = self.models[model_id]
            
            if instance.is_loaded:
                return True
            
            # Check if we need to unload other models
            loaded_count = sum(1 for m in self.models.values() if m.is_loaded)
            if loaded_count >= self.max_loaded_models:
                self._evict_least_recently_used()
            
            # Load model
            try:
                start_time = time.time()
                
                # Load from MLflow or file path
                model_path = self.model_paths[model_id]
                if model_path.startswith("models:/"):
                    # MLflow model
                    model = mlflow.pytorch.load_model(model_path)
                else:
                    # Local file
                    # This would need proper model reconstruction logic
                    model = torch.load(model_path, map_location='cpu')
                
                # Move to appropriate device
                device = self._select_device()
                model = model.to(device)
                model.eval()
                
                # Update instance
                instance.model = model
                instance.is_loaded = True
                instance.load_time = time.time() - start_time
                instance.device = device
                instance.memory_usage = self._estimate_model_memory(model)
                
                logging.info(f"Loaded model {model_id} on {device} in {instance.load_time:.2f}s")
                return True
                
            except Exception as e:
                logging.error(f"Failed to load model {model_id}: {e}")
                return False
    
    def unload_model(self, model_id: str):
        """Unload a model from memory."""
        with self._lock:
            if model_id in self.models:
                instance = self.models[model_id]
                if instance.is_loaded:
                    del instance.model
                    instance.model = None
                    instance.is_loaded = False
                    torch.cuda.empty_cache()  # Clear GPU memory
                    logging.info(f"Unloaded model {model_id}")
    
    def select_model(
        self,
        request: InferenceRequest,
        performance_weight: float = 0.7
    ) -> Optional[str]:
        """
        Select the best model for a given request.
        
        Args:
            request: Inference request
            performance_weight: Weight for performance vs. other factors
            
        Returns:
            Model ID or None if no suitable model found
        """
        with self._lock:
            suitable_models = []
            
            for model_id, instance in self.models.items():
                # Check language support
                if request.language and request.language not in instance.languages:
                    continue
                
                # Check task support
                if request.task not in instance.tasks:
                    continue
                
                # Calculate score
                score = self._calculate_model_score(instance, request, performance_weight)
                suitable_models.append((model_id, score))
            
            if not suitable_models:
                return None
            
            # Sort by score (higher is better)
            suitable_models.sort(key=lambda x: x[1], reverse=True)
            
            return suitable_models[0][0]
    
    def get_model(self, model_id: str) -> Optional[ModelInstance]:
        """Get a model instance, loading it if necessary."""
        if model_id not in self.models:
            return None
        
        with self._lock:
            instance = self.models[model_id]
            
            if not instance.is_loaded:
                success = self.load_model(model_id)
                if not success:
                    return None
            
            # Update usage statistics
            instance.last_used = time.time()
            instance.usage_count += 1
            
            return instance
    
    def _evict_least_recently_used(self):
        """Evict the least recently used model."""
        loaded_models = [
            (model_id, instance) for model_id, instance in self.models.items()
            if instance.is_loaded
        ]
        
        if not loaded_models:
            return
        
        # Find LRU model
        lru_model_id, _ = min(loaded_models, key=lambda x: x[1].last_used)
        self.unload_model(lru_model_id)
    
    def _select_device(self) -> str:
        """Select the best available device."""
        for device in self.device_preference:
            if device == "cuda" and torch.cuda.is_available():
                return "cuda"
            elif device == "cpu":
                return "cpu"
        
        return "cpu"
    
    def _estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    
    def _calculate_model_score(
        self,
        instance: ModelInstance,
        request: InferenceRequest,
        performance_weight: float
    ) -> float:
        """Calculate a score for model selection."""
        score = 0.0
        
        # Performance factor
        if 'accuracy' in instance.performance_metrics:
            score += performance_weight * instance.performance_metrics['accuracy']
        
        # Speed factor
        if 'inference_time' in instance.performance_metrics:
            speed_score = 1.0 / (1.0 + instance.performance_metrics['inference_time'])
            score += (1.0 - performance_weight) * speed_score
        
        # Type preference based on priority
        type_bonus = {
            ModelType.FAST: 0.1 if request.priority > 5 else 0.0,
            ModelType.ACCURATE: 0.1 if request.priority < 3 else 0.0,
            ModelType.SPECIALIZED: 0.15,
            ModelType.BASE: 0.05
        }
        score += type_bonus.get(instance.model_type, 0.0)
        
        # Usage frequency (recently used models are preferred)
        recency_bonus = 0.1 / (1.0 + (time.time() - instance.last_used) / 3600)
        score += recency_bonus
        
        return score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        with self._lock:
            loaded_models = sum(1 for m in self.models.values() if m.is_loaded)
            total_memory = sum(
                m.memory_usage for m in self.models.values() if m.is_loaded
            )
            
            return {
                'total_models': len(self.models),
                'loaded_models': loaded_models,
                'total_memory_mb': total_memory,
                'model_types': {
                    model_type.value: sum(
                        1 for m in self.models.values() 
                        if m.model_type == model_type
                    )
                    for model_type in ModelType
                }
            }


class JengaHubServingEngine:
    """
    Main serving engine that orchestrates model management, caching, and inference.
    """
    
    def __init__(
        self,
        config: MultiModalConfig,
        cache_config: Dict[str, Any] = None,
        model_manager_config: Dict[str, Any] = None
    ):
        self.config = config
        
        # Initialize components
        self.cache = SmartCache(**(cache_config or {}))
        self.model_manager = ModelManager(**(model_manager_config or {}))
        self.data_processor = UnifiedDataProcessor(config)
        
        # Request queue for async processing
        self.request_queue = asyncio.Queue()
        self.result_futures: Dict[str, Future] = {}
        
        # Performance monitoring
        self.inference_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_inference_time': 0.0,
            'error_rate': 0.0
        }
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._processing_task = None
        self._is_running = False
        
        logging.info("JengaHub Serving Engine initialized")
    
    async def start(self):
        """Start the serving engine."""
        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_requests())
        logging.info("Serving engine started")
    
    async def stop(self):
        """Stop the serving engine."""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        logging.info("Serving engine stopped")
    
    async def infer(
        self,
        request: InferenceRequest
    ) -> InferenceResult:
        """
        Perform inference with caching and model selection.
        
        Args:
            request: Inference request
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(request)
        if cached_result:
            self.inference_stats['cache_hits'] += 1
            return cached_result
        
        # Select best model
        model_id = self.model_manager.select_model(request)
        if not model_id:
            raise ValueError(f"No suitable model found for request: {request}")
        
        # Get model instance
        model_instance = self.model_manager.get_model(model_id)
        if not model_instance:
            raise RuntimeError(f"Failed to load model: {model_id}")
        
        try:
            # Process input data
            processed_sample = await self._process_input(request)
            
            # Run inference
            with torch.no_grad():
                result = await self._run_inference(
                    model_instance, 
                    processed_sample, 
                    request
                )
            
            # Cache result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.model_used = model_id
            
            self.cache.put(request, result)
            
            # Update statistics
            self._update_stats(processing_time)
            
            return result
            
        except Exception as e:
            logging.error(f"Inference failed for model {model_id}: {e}")
            self.inference_stats['error_rate'] += 1
            raise
    
    async def batch_infer(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Perform batch inference with optimizations."""
        
        # Group requests by model requirements
        grouped_requests = self._group_requests_by_model(requests)
        
        results = []
        
        for model_id, batch_requests in grouped_requests.items():
            model_instance = self.model_manager.get_model(model_id)
            if not model_instance:
                # Handle failed model loading
                for req in batch_requests:
                    error_result = InferenceResult(
                        request_id=req.request_id,
                        predictions={'error': f'Model {model_id} unavailable'},
                        confidence=0.0,
                        language_detected=None,
                        code_switching_info=None,
                        processing_time=0.0,
                        model_used=model_id
                    )
                    results.append(error_result)
                continue
            
            # Process batch
            batch_results = await self._run_batch_inference(
                model_instance,
                batch_requests
            )
            results.extend(batch_results)
        
        return results
    
    async def _process_input(self, request: InferenceRequest) -> ProcessedSample:
        """Process input data asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Run data processing in thread pool
        processed_sample = await loop.run_in_executor(
            self.executor,
            self.data_processor.process_multimodal_sample,
            request.audio_path,
            request.text,
            None,  # transcript
            request.task,
            None   # label
        )
        
        return processed_sample
    
    async def _run_inference(
        self,
        model_instance: ModelInstance,
        processed_sample: ProcessedSample,
        request: InferenceRequest
    ) -> InferenceResult:
        """Run model inference asynchronously."""
        
        loop = asyncio.get_event_loop()
        
        # Prepare inputs
        def _inference_fn():
            model = model_instance.model
            device = model_instance.device
            
            # Move data to device
            inputs = {}
            if processed_sample.audio_features is not None:
                inputs['audio_features'] = processed_sample.audio_features.unsqueeze(0).to(device)
            
            if processed_sample.text_input_ids is not None:
                inputs['text_input_ids'] = processed_sample.text_input_ids.unsqueeze(0).to(device)
                inputs['text_attention_mask'] = processed_sample.text_attention_mask.unsqueeze(0).to(device)
            
            # Run model
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs
            if isinstance(outputs, torch.Tensor):
                logits = outputs
                predictions = torch.softmax(logits, dim=-1)
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('prediction_logits'))
                predictions = torch.softmax(logits, dim=-1)
            else:
                raise ValueError("Unsupported model output format")
            
            # Extract predictions
            pred_probs = predictions.cpu().numpy()[0]
            pred_class = int(np.argmax(pred_probs))
            confidence = float(np.max(pred_probs))
            
            # Detect language if applicable
            language_detected = processed_sample.language
            
            # Code-switching analysis (if model supports it)
            code_switching_info = None
            if hasattr(model, 'code_switching_bridge'):
                # Extract code-switching information
                cs_outputs = model.code_switching_bridge(
                    inputs.get('audio_features'),
                    inputs.get('text_features')
                )
                code_switching_info = {
                    'detected_switches': len(cs_outputs.get('switch_points', [])),
                    'languages_detected': cs_outputs.get('languages', [])
                }
            
            return {
                'prediction_class': pred_class,
                'probabilities': pred_probs.tolist(),
                'confidence': confidence,
                'language_detected': language_detected,
                'code_switching_info': code_switching_info
            }
        
        # Run inference in thread pool
        inference_results = await loop.run_in_executor(
            self.executor,
            _inference_fn
        )
        
        # Create result object
        result = InferenceResult(
            request_id=request.request_id,
            predictions=inference_results,
            confidence=inference_results['confidence'],
            language_detected=inference_results['language_detected'],
            code_switching_info=inference_results['code_switching_info'],
            processing_time=0.0,  # Will be set by caller
            model_used=""  # Will be set by caller
        )
        
        return result
    
    async def _run_batch_inference(
        self,
        model_instance: ModelInstance,
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Run batch inference for multiple requests."""
        
        # Process all samples
        processed_samples = []
        for request in requests:
            sample = await self._process_input(request)
            processed_samples.append(sample)
        
        # Create batched inputs
        batch_inputs = self._create_batch_inputs(processed_samples, model_instance.device)
        
        # Run batch inference
        loop = asyncio.get_event_loop()
        
        def _batch_inference_fn():
            model = model_instance.model
            
            with torch.no_grad():
                outputs = model(**batch_inputs)
            
            # Process batch outputs
            if isinstance(outputs, torch.Tensor):
                predictions = torch.softmax(outputs, dim=-1)
            elif isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('prediction_logits'))
                predictions = torch.softmax(logits, dim=-1)
            else:
                raise ValueError("Unsupported model output format")
            
            return predictions.cpu().numpy()
        
        batch_predictions = await loop.run_in_executor(
            self.executor,
            _batch_inference_fn
        )
        
        # Create individual results
        results = []
        for i, request in enumerate(requests):
            pred_probs = batch_predictions[i]
            pred_class = int(np.argmax(pred_probs))
            confidence = float(np.max(pred_probs))
            
            result = InferenceResult(
                request_id=request.request_id,
                predictions={
                    'prediction_class': pred_class,
                    'probabilities': pred_probs.tolist(),
                    'confidence': confidence
                },
                confidence=confidence,
                language_detected=processed_samples[i].language,
                code_switching_info=None,  # TODO: Implement for batch
                processing_time=0.0,
                model_used=model_instance.model.__class__.__name__
            )
            
            results.append(result)
        
        return results
    
    def _group_requests_by_model(
        self,
        requests: List[InferenceRequest]
    ) -> Dict[str, List[InferenceRequest]]:
        """Group requests by optimal model."""
        grouped = defaultdict(list)
        
        for request in requests:
            model_id = self.model_manager.select_model(request)
            if model_id:
                grouped[model_id].append(request)
        
        return dict(grouped)
    
    def _create_batch_inputs(
        self,
        samples: List[ProcessedSample],
        device: str
    ) -> Dict[str, torch.Tensor]:
        """Create batched inputs from processed samples."""
        
        # Use the data processor's collate function
        collate_fn = self.data_processor.create_batch_collator()
        batch = collate_fn(samples)
        
        # Move to device
        batched_inputs = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batched_inputs[key] = value.to(device)
        
        return batched_inputs
    
    def _update_stats(self, processing_time: float):
        """Update inference statistics."""
        self.inference_stats['total_requests'] += 1
        
        # Update average inference time
        total_requests = self.inference_stats['total_requests']
        current_avg = self.inference_stats['avg_inference_time']
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.inference_stats['avg_inference_time'] = new_avg
    
    async def _process_requests(self):
        """Background task to process requests from queue."""
        while self._is_running:
            try:
                # Wait for requests with timeout
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # Process request
                try:
                    result = await self.infer(request)
                    
                    # Set result for waiting future
                    if request.request_id in self.result_futures:
                        future = self.result_futures[request.request_id]
                        if not future.done():
                            future.set_result(result)
                        del self.result_futures[request.request_id]
                
                except Exception as e:
                    # Set exception for waiting future
                    if request.request_id in self.result_futures:
                        future = self.result_futures[request.request_id]
                        if not future.done():
                            future.set_exception(e)
                        del self.result_futures[request.request_id]
                
            except asyncio.TimeoutError:
                # No requests to process
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing request: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive serving statistics."""
        return {
            'inference_stats': self.inference_stats,
            'cache_stats': self.cache.get_stats(),
            'model_manager_stats': self.model_manager.get_stats(),
            'is_running': self._is_running
        }


# Utility functions
def create_serving_engine(
    config: MultiModalConfig,
    model_paths: Dict[str, str],
    redis_url: Optional[str] = None,
    cache_size: int = 1000
) -> JengaHubServingEngine:
    """Create a configured serving engine."""
    
    # Cache configuration
    cache_config = {
        'memory_size': cache_size,
        'redis_url': redis_url,
        'disk_cache_path': str(Path(config.cache_dir) / "serving_cache")
    }
    
    # Model manager configuration
    model_manager_config = {
        'max_loaded_models': 3,
        'device_preference': ["cuda", "cpu"]
    }
    
    # Create engine
    engine = JengaHubServingEngine(
        config=config,
        cache_config=cache_config,
        model_manager_config=model_manager_config
    )
    
    # Register models
    for model_id, model_path in model_paths.items():
        engine.model_manager.register_model(
            model_id=model_id,
            model_path=model_path,
            config=config,
            model_type=ModelType.BASE,
            languages=config.audio.secondary_languages + [config.audio.primary_language],
            tasks=config.text.tasks
        )
    
    return engine


async def run_serving_example():
    """Example of how to use the serving engine."""
    
    # Create dummy config
    from .config import MultiModalConfig
    config = MultiModalConfig()
    
    # Create engine
    engine = create_serving_engine(
        config=config,
        model_paths={
            'base_model': 'path/to/base/model.pth',
            'swahili_specialized': 'path/to/swahili/model.pth'
        }
    )
    
    # Start engine
    await engine.start()
    
    try:
        # Create inference request
        request = InferenceRequest(
            request_id="test_001",
            text="Habari yako? How are you doing?",
            task="classification",
            language="swahili"
        )
        
        # Run inference
        result = await engine.infer(request)
        print(f"Inference result: {result}")
        
        # Get statistics
        stats = engine.get_stats()
        print(f"Engine stats: {stats}")
        
    finally:
        # Stop engine
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(run_serving_example())