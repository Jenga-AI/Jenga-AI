
import os
import logging
import torch
import numpy as np
from typing import Dict, List, Optional
from seq2seq_models.utils.oov_restoration import restore_oov_words_in_translation

logger = logging.getLogger(__name__)

class DomainEvaluator:
    """Handles evaluation on domain-specific data (real call transcriptions)"""
    
    def __init__(self, source_file: str, target_file: str):
        self.source_file = source_file
        self.target_file = target_file
        self.sources = []
        self.targets = []
        self.is_valid = False
        
        self._load_data()
    
    def _load_data(self):
        """Load domain evaluation data"""
        logger.info(f"\n📥 Loading domain evaluation data...")
        
        if not os.path.exists(self.source_file):
            logger.error(f"❌ Domain source file not found: {self.source_file}")
            return
        if not os.path.exists(self.target_file):
            logger.error(f"❌ Domain target file not found: {self.target_file}")
            return
        
        try:
            with open(self.source_file, 'r', encoding='utf-8') as f:
                self.sources = [line.strip() for line in f if line.strip()]
            
            with open(self.target_file, 'r', encoding='utf-8') as f:
                self.targets = [line.strip() for line in f if line.strip()]
            
            if len(self.sources) != len(self.targets):
                logger.error(
                    f"❌ Mismatch in domain data: {len(self.sources)} sources vs {len(self.targets)} targets"
                )
                return
            
            if len(self.sources) == 0:
                logger.error(f"❌ Domain data files are empty!")
                return
            
            self.is_valid = True
            logger.info(f"✅ Loaded {len(self.sources)} domain evaluation pairs")
            
        except Exception as e:
            logger.error(f"❌ Error loading domain data: {e}")
            return
    
    def evaluate(self, model, tokenizer, bleu_metric, chrf_metric, comet_model=None, max_samples=None, batch_size=32):
        """Evaluate model on domain data with batch inference for speed"""
        if not self.is_valid:
            logger.warning("⚠️ Domain evaluator is not valid, skipping evaluation")
            return {}

        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Evaluating on DOMAIN data...")
        logger.info(f"{'='*70}")

        sources = self.sources[:max_samples] if max_samples else self.sources
        targets = self.targets[:max_samples] if max_samples else self.targets

        predictions = []
        model.eval()
        device = next(model.parameters()).device

        num_batches = (len(sources) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(sources))
                batch_sources = sources[start_idx:end_idx]

                try:
                    # Batch tokenization
                    inputs = tokenizer(
                        batch_sources,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Batch generation
                    outputs = model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=5,
                        early_stopping=True
                    )

                    # Decode batch with OOV restoration
                    for i, (source, output) in enumerate(zip(batch_sources, outputs)):
                        prediction = restore_oov_words_in_translation(source, output, tokenizer)
                        predictions.append(prediction)

                except Exception as e:
                    logger.warning(f"  Batch {batch_idx + 1} failed: {e}")
                    # Add empty predictions for failed batch
                    predictions.extend([""] * len(batch_sources))

        if len(predictions) == 0 or all(p == "" for p in predictions):
            logger.error("❌ No successful predictions for domain evaluation")
            return {}
        
        results = {}
        
        # BLEU
        try:
            bleu_score = bleu_metric.compute(
                predictions=predictions,
                references=[[target] for target in targets]
            )
            results['domain_bleu'] = bleu_score['bleu']
        except Exception as e:
            logger.warning(f"⚠️ BLEU computation failed: {e}")
            results['domain_bleu'] = 0.0
        
        # chrF
        try:
            chrf_score = chrf_metric.compute(
                predictions=predictions,
                references=targets
            )
            results['domain_chrf'] = chrf_score['score']
        except Exception as e:
            logger.warning(f"⚠️ chrF computation failed: {e}")
            results['domain_chrf'] = 0.0
        
        # COMET (if available)
        if comet_model is not None:
            try:
                logger.info("  Computing COMET-QE...")
                comet_data = [
                    {"src": src, "mt": pred}
                    for src, pred in zip(sources, predictions)
                ]
                
                model_output = comet_model.predict(
                    comet_data,
                    batch_size=32,
                    gpus=1 if torch.cuda.is_available() else 0
                )
                
                if hasattr(model_output, 'scores'):
                    scores = model_output.scores
                elif isinstance(model_output, list):
                    scores = model_output
                else:
                    scores = [0.0]
                
                results['domain_comet'] = float(np.mean(scores))
                results['domain_comet_std'] = float(np.std(scores))
            except Exception as e:
                logger.warning(f"⚠️ COMET evaluation failed: {e}")
                results['domain_comet'] = 0.0
        
        logger.info(f"\n✅ Domain Evaluation Results:")
        logger.info(f"   BLEU:     {results.get('domain_bleu', 0):.4f}")
        logger.info(f"   chrF:     {results.get('domain_chrf', 0):.2f}")
        if 'domain_comet' in results:
            logger.info(f"   COMET-QE: {results.get('domain_comet', 0):.4f}")
        
        return results
