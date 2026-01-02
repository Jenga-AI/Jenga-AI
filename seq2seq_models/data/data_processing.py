
import os
import json
import logging
import random
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset
from seq2seq_models.core.config import DatasetConfig, TranslationConfig

logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.dataset_config = config.dataset_config
        self.cache_dir = config.system_config.cache_dir
    
    def load_datasets(self) -> DatasetDict:
        """Load and combine datasets"""
        
        logger.info("🔍 Verifying dataset files...")
        for dataset_info in self.dataset_config.custom_datasets:
            path = dataset_info['path']
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Dataset not found: {path}")
            
            # Simple line count check
            with open(path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            logger.info(f"  ✅ {dataset_info.get('name', 'unknown')}: {line_count:,} lines in file")
            
            if line_count < 100:
                raise ValueError(f"❌ Dataset too small: {path} has only {line_count} lines")
        
        all_data = []
        
        for dataset_info in self.dataset_config.custom_datasets:
            logger.info(f"📥 Loading {dataset_info.get('name', 'unknown')} from {dataset_info['path']}")
            data = self._load_jsonl_dataset(dataset_info)
            
            if len(data) == 0:
                raise ValueError(f"❌ No data loaded from {dataset_info['path']}! Check file format.")
            
            all_data.extend(data)
        
        logger.info(f"📊 Total samples loaded: {len(all_data):,}")
        
        if len(all_data) < 1000: # Reduced threshold for testing purposes, user script had 5000
             logger.warning(
                f"⚠️ Dataset small! Only {len(all_data):,} samples loaded.\n"
                f"   For production quality, aim for 50,000+ samples."
            )
        
        filtered_data = self._filter_data(all_data)
        logger.info(f"📊 Samples after filtering: {len(filtered_data):,}")
        
        unique_data = self._remove_duplicates(filtered_data)
        logger.info(f"📊 Samples after deduplication: {len(unique_data):,}")
        
        dataset = Dataset.from_list(unique_data)
        
        if self.dataset_config.shuffle:
            dataset = dataset.shuffle(seed=self.dataset_config.seed)
        
        val_split = self.dataset_config.validation_split
        test_split = self.dataset_config.test_split
        
        train_val = dataset.train_test_split(test_size=test_split, seed=self.dataset_config.seed)
        train_val_split = train_val['train'].train_test_split(
            test_size=val_split/(1-test_split), seed=self.dataset_config.seed
        )
        
        dataset_dict = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],
            'test': train_val['test']
        })
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Dataset splits:")
        logger.info(f"  Train:      {len(dataset_dict['train']):,} samples")
        logger.info(f"  Validation: {len(dataset_dict['validation']):,} samples")
        logger.info(f"  Test:       {len(dataset_dict['test']):,} samples")
        logger.info(f"{'='*70}\n")
        
        return dataset_dict
    
    def _load_jsonl_dataset(self, dataset_info: Dict) -> List[Dict]:
        """Load a JSONL dataset file efficiently using datasets library"""
        path = dataset_info['path']
        weight = dataset_info.get('weight', 1.0)
        name = dataset_info.get('name', 'unknown')

        # Try efficient loading with datasets library first
        try:
            raw_dataset = load_dataset(
                'json',
                data_files=path,
                split='train',
                cache_dir=self.cache_dir
            )
            logger.info(f"  ✅ Loaded {len(raw_dataset):,} samples using datasets library (memory-efficient)")

            # Process and normalize format
            data = []
            for i, item in enumerate(raw_dataset):
                source, target = self._extract_source_target(item)
                if source and target:
                    # Apply weighting
                    full_copies = int(weight)
                    for _ in range(full_copies):
                        data.append({
                            'source': source.strip(),
                            'target': target.strip(),
                            'dataset': name
                        })

                    # Partial weight (probabilistic)
                    if weight % 1 > 0 and random.random() < weight % 1:
                        data.append({
                            'source': source.strip(),
                            'target': target.strip(),
                            'dataset': name
                        })

            logger.info(f"  Loaded {len(data):,} samples from {name} (weight: {weight}x)")
            return data

        except Exception as e:
            logger.warning(f"  ⚠️ datasets library failed: {e}")
            logger.info(f"  Falling back to manual loading...")
            return self._load_jsonl_manual(dataset_info)

    def _extract_source_target(self, item: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract source/target from various JSONL formats"""
        if 'source' in item and 'target' in item:
            return item['source'], item['target']
        elif 'sw' in item and 'en' in item:
            return item['sw'], item['en']
        elif 'text' in item and 'translation' in item:
            return item['text'], item['translation']
        return None, None

    def _load_jsonl_manual(self, dataset_info: Dict) -> List[Dict]:
        """Fallback manual JSONL loading for problematic files"""
        data = []
        weight = dataset_info.get('weight', 1.0)
        name = dataset_info.get('name', 'unknown')

        with open(dataset_info['path'], 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    source, target = self._extract_source_target(item)

                    if source and target:
                        full_copies = int(weight)
                        for _ in range(full_copies):
                            data.append({
                                'source': source.strip(),
                                'target': target.strip(),
                                'dataset': name
                            })

                        if weight % 1 > 0 and random.random() < weight % 1:
                            data.append({
                                'source': source.strip(),
                                'target': target.strip(),
                                'dataset': name
                            })

                except json.JSONDecodeError as e:
                    logger.warning(f"  JSON error at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"  Error at line {line_num}: {e}")

        logger.info(f"  Loaded {len(data):,} samples from {name} (weight: {weight}x)")
        return data
    
    def _filter_data(self, data: List[Dict]) -> List[Dict]:
        """Apply quality filters to the data"""
        filtered = []
        stats = defaultdict(int)
        
        max_len = self.dataset_config.max_length
        max_ratio = self.dataset_config.max_length_ratio
        min_ratio = self.dataset_config.min_length_ratio
        
        for item in data:
            source = item['source']
            target = item['target']
            
            source_words = source.split()
            target_words = target.split()
            
            if len(source_words) == 0 or len(target_words) == 0:
                stats['empty'] += 1
                continue
            
            if len(source_words) > max_len or len(target_words) > max_len:
                stats['too_long'] += 1
                continue
            
            if self.dataset_config.filter_length_ratio:
                ratio = len(target_words) / max(len(source_words), 1)
                if ratio > max_ratio or ratio < min_ratio:
                    stats['bad_ratio'] += 1
                    continue
            
            filtered.append(item)
            stats['kept'] += 1
        
        logger.info(f"Filtering statistics: {dict(stats)}")
        return filtered
    
    def _remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate samples"""
        seen = set()
        unique = []
        
        for item in data:
            key = hashlib.md5(f"{item['source']}|{item['target']}".encode()).hexdigest()
            
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        duplicates_removed = len(data) - len(unique)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed:,} duplicates")

        return unique

    def validate_dataset(self, dataset_dict: DatasetDict, tokenizer) -> None:
        """
        Validate dataset before training to catch issues early.
        """
        logger.info("\n🔍 Validating dataset before training...")
        issues = []
        warnings_list = []

        for split_name, split_data in dataset_dict.items():
            split_issues = []

            # Check for empty samples
            empty_source = 0
            empty_target = 0
            for ex in split_data:
                if not ex['source'].strip():
                    empty_source += 1
                if not ex['target'].strip():
                    empty_target += 1

            if empty_source > 0:
                split_issues.append(f"{empty_source} samples with empty source")
            if empty_target > 0:
                split_issues.append(f"{empty_target} samples with empty target")

            # Test tokenization on samples
            try:
                if len(split_data) > 0:
                    sample = split_data[0]
                    test_input = tokenizer(
                        sample['source'],
                        truncation=True,
                        max_length=256
                    )
                    with tokenizer.as_target_tokenizer():
                        test_output = tokenizer(
                            sample['target'],
                            truncation=True,
                            max_length=256
                        )

                    if len(test_input['input_ids']) == 0:
                        split_issues.append("Tokenizer produced empty input_ids")

            except Exception as e:
                split_issues.append(f"Tokenization failed: {str(e)[:100]}")

            if split_issues:
                for issue in split_issues:
                    issues.append(f"{split_name}: {issue}")

        # Report results
        if issues:
            error_msg = "❌ Dataset validation failed:\n" + "\n".join(f"  - {i}" for i in issues)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ Dataset validation passed")
