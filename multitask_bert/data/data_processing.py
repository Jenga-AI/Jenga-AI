import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, List
from ..core.config import ExperimentConfig, TaskConfig
from datasets import Dataset
import json # Added import

class DataProcessor:
    """
    Processes raw data from files into tokenized features for various tasks,
    driven by a unified ExperimentConfig.
    """
    def __init__(self, config: ExperimentConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def process(self) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], ExperimentConfig]:
        """
        Main processing method. Iterates through tasks, processes data,
        and returns dictionaries of train and eval Datasets.
        """
        train_datasets = {}
        eval_datasets = {}

        for task_config in self.config.tasks:
            print(f"Processing data for task: {task_config.name}")
            
            if task_config.data_path.endswith('.jsonl'):
                df = pd.read_json(task_config.data_path, lines=True)
            else:
                df = pd.read_json(task_config.data_path)

            dataset = Dataset.from_pandas(df)
            
            if task_config.type == "single_head_single_label_classification":
                tokenized_dataset = self._process_single_head_single_label_classification(dataset, task_config)
            elif task_config.type == "classification": # New type for multi-head single-label classification
                tokenized_dataset = self._process_multi_head_single_label_classification(dataset, task_config)
            elif task_config.type == "multi_label_classification" or task_config.type == "qa_qc":
                tokenized_dataset = self._process_multi_label_classification(dataset, task_config)
            elif task_config.type == "ner":
                tokenized_dataset = self._process_ner(dataset, task_config)
            elif task_config.type == "anomaly_detection":
                # Security Task: No tokenizer, just feature scaling
                tokenized_dataset = self._process_anomaly_detection(dataset, task_config)
            else:
                raise ValueError(f"Unknown task type: {task_config.type}")

            # Set format
            columns_to_set = ['input_ids', 'attention_mask']
            # Dynamically add label columns based on task type
            if task_config.type == "ner":
                columns_to_set.append('labels')
            elif task_config.type in ["classification", "multi_label_classification"]:
                for head_config in task_config.heads:
                    columns_to_set.append(f'labels_{head_config.name}')
            elif task_config.type == "anomaly_detection":
                # For security, we use 'features' (the numbers) and 'labels_anomaly'
                columns_to_set = ['features']
                for head_config in task_config.heads:
                    columns_to_set.append(f'labels_{head_config.name}')
            
            tokenized_dataset.set_format(type='torch', columns=columns_to_set)

            # Split dataset
            split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
            train_datasets[task_config.name] = split_dataset['train']
            eval_datasets[task_config.name] = split_dataset['test']
        
        return train_datasets, eval_datasets, self.config

    def _process_single_head_single_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes and tokenizes data for single-head, single-label classification."""
        def process_and_tokenize(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=False,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
            )
            tokenized_inputs['labels'] = [torch.tensor(label, dtype=torch.long) for label in examples['label']]
            return tokenized_inputs
        return dataset.map(process_and_tokenize, batched=True)

    def _process_multi_head_single_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes and tokenizes data for multi-head, single-label classification."""
        def process_and_tokenize(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=False,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
            )
            
            # Instead of nesting under 'labels', add each head's labels as a separate key
            for head_config in task_config.heads:
                head_name = head_config.name
                
                # Get label map and invert it if it exists
                label_map = task_config.label_maps.get(head_name) if task_config.label_maps else None
                label_to_id = {v: k for k, v in label_map.items()} if label_map else None

                # Try common label column names
                label_column = None
                if 'label' in examples:
                    label_column = 'label'
                elif 'quality_level' in examples:
                    label_column = 'quality_level'
                
                if label_column:
                    labels = examples[label_column]
                    
                    # Dynamic label mapping if not provided
                    if label_to_id is None:
                        unique_labels = sorted(list(set(labels)))
                        label_to_id = {l: i for i, l in enumerate(unique_labels)}
                        id_to_label = {i: l for l, i in label_to_id.items()}
                        # Update task config for later use
                        if task_config.label_maps is None:
                            task_config.label_maps = {}
                        task_config.label_maps[head_name] = id_to_label
                        head_config.num_labels = len(unique_labels)
                    
                    # Convert string labels to IDs
                    if isinstance(labels[0], str):
                        labels = [label_to_id.get(l, -100) for l in labels]
                    
                    tokenized_inputs[f'labels_{head_name}'] = [torch.tensor(l, dtype=torch.long) for l in labels]
                else:
                    num_examples = len(examples['text'])
                    tokenized_inputs[f'labels_{head_name}'] = [torch.full((1,), -100, dtype=torch.long) for _ in range(num_examples)]
            
            return tokenized_inputs
        return dataset.map(process_and_tokenize, batched=True)

    def _process_multi_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes and tokenizes data for multi-label classification (multi-head)."""
        import json
        
        def process_and_tokenize(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=False,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
            )
            
            for head_config in task_config.heads:
                head_name = head_config.name
                head_labels_batch = []
                
                for i in range(len(examples['text'])):
                    # Extract label data for this example
                    labels_data = examples['labels'][i]
                    
                    # Parse JSON if it's a string
                    if isinstance(labels_data, str):
                        try:
                            labels_data = json.loads(labels_data)
                        except json.JSONDecodeError:
                            labels_data = {}
                    
                    if not isinstance(labels_data, dict):
                        labels_data = {}
                        
                    labels = labels_data.get(head_name, [])
                    
                    # Ensure labels is a list of numbers
                    if not isinstance(labels, list):
                        labels = [labels] if labels is not None else []
                    
                    # Convert all elements to float/int to avoid "str" error
                    try:
                        labels = [float(l) for l in labels]
                    except (ValueError, TypeError):
                        labels = [0.0] * head_config.num_labels

                    # Ensure it matches expected length
                    if len(labels) < head_config.num_labels:
                        labels = labels + [0.0] * (head_config.num_labels - len(labels))
                    elif len(labels) > head_config.num_labels:
                        labels = labels[:head_config.num_labels]
                        
                    head_labels_batch.append(torch.tensor(labels, dtype=torch.float))
                
                tokenized_inputs[f'labels_{head_name}'] = head_labels_batch
                
            return tokenized_inputs
            
        return dataset.map(process_and_tokenize, batched=True)


    # def _process_ner(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
    #     """Processes data for NER tasks, including complex label alignment."""
        
    #     unique_labels = set(['O'])
    #     for example in dataset:
    #         for entity in example['entities']:
    #             unique_labels.add(entity['label'])
        
    #     label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
    #     id_to_label = {i: label for label, i in label_to_id.items()}
        
    #     task_config.heads[0].num_labels = len(label_to_id)
    #     task_config.label_maps = {'ner_head': id_to_label}

    #     def align_labels_with_tokens(examples):
    #         tokenized_inputs = self.tokenizer(
    #             examples['text'],
    #             padding=False,
    #             truncation=self.config.tokenizer.truncation,
    #             max_length=self.config.tokenizer.max_length,
    #             is_split_into_words=False,
    #             return_offsets_mapping=True
    #         )

    #         labels = []
    #         for i, text in enumerate(examples['text']):
    #             word_ids = tokenized_inputs.word_ids(batch_index=i)
                
    #             # 1. Initialize with 'O' label
    #             current_labels = [label_to_id['O']] * len(word_ids)
                
    #             # 2. Align entities
    #             for entity in examples['entities'][i]:
    #                 start_char, end_char, label = entity['start'], entity['end'], entity['label']
    #                 entity_label_id = label_to_id[label]
                    
    #                 for token_idx, word_id in enumerate(word_ids):
    #                     if word_id is None:
    #                         continue
                        
    #                     offset = tokenized_inputs['offset_mapping'][i][token_idx]
    #                     if offset is None:
    #                         continue
                        
    #                     token_start, token_end = offset
    #                     if token_start >= start_char and token_end <= end_char:
    #                         current_labels[token_idx] = entity_label_id
                
    #             # 3. Set labels of special tokens to -100
    #             for token_idx, word_id in enumerate(word_ids):
    #                 if word_id is None:
    #                     current_labels[token_idx] = -100
                        
    #             labels.append(current_labels)
            
    #         tokenized_inputs["labels"] = labels
    #         return tokenized_inputs

    #     return dataset.map(align_labels_with_tokens, batched=True)

    def _process_ner(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes data for NER tasks - simplified robust version."""
        
        # Build label vocabulary from data
        unique_labels = set(['O'])
        for example in dataset:
            for entity in example['entities']:
                unique_labels.add(entity['label'])
        
        label_to_id = {label: i for i, label in enumerate(sorted(list(unique_labels)))}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        # Update task config
        task_config.heads[0].num_labels = len(label_to_id)
        task_config.label_maps = {'ner_head': id_to_label}

        def align_labels_with_tokens(examples):
            # Use a simpler approach without offset mapping
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=False,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
            )

            labels = []
            for i, text in enumerate(examples['text']):
                # Tokenize the text to get word IDs
                encoding = self.tokenizer(text, truncation=True, max_length=self.config.tokenizer.max_length)
                word_ids = encoding.word_ids()
                
                # Initialize with O labels
                label_ids = [label_to_id['O']] * len(word_ids)
                
                # Create a character-to-word mapping for the original text
                char_to_word = {}
                word_start = 0
                words = text.split()
                for word_idx, word in enumerate(words):
                    for char_idx in range(word_start, word_start + len(word)):
                        char_to_word[char_idx] = word_idx
                    word_start += len(word) + 1  # +1 for space
                
                # Align entities
                for entity in examples['entities'][i]:
                    start_char, end_char, label = entity['start'], entity['end'], entity['label']
                    entity_label_id = label_to_id[label]
                    
                    # Find which words are in this entity span
                    entity_words = set()
                    for char_idx in range(start_char, end_char):
                        if char_idx in char_to_word:
                            entity_words.add(char_to_word[char_idx])
                    
                    # Label tokens that belong to these words
                    for token_idx, word_idx in enumerate(word_ids):
                        if word_idx in entity_words:
                            label_ids[token_idx] = entity_label_id
                
                # Set special tokens to -100
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        label_ids[token_idx] = -100
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        return dataset.map(align_labels_with_tokens, batched=True)

    def _process_anomaly_detection(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """
        Processes tabular data for security/anomaly detection tasks.
        Expects columns like 'packet_size', 'port', 'request_count', etc.
        """
        import numpy as np
        
        # 1. Identify Feature Columns (exclude 'label', 'text', 'id')
        feature_cols = [c for c in dataset.column_names if c not in ['label', 'id', 'text', 'labels']]
        print(f"🔒 [Security Refinery] Detected features: {feature_cols}")
        
        # 2. Update task config so the Backbone knows the input dimension
        # We assume all heads share the same input features for now
        task_config.input_dim = len(feature_cols)

        # 3. Normalization (Simple Z-Score or MinMax equivalent)
        # For simplicity in this demo, we'll convert to float tensor directly.
        # In a real system, you'd fit a StandardScaler here.
        
        def process_tabular(examples):
            # Convert list of dicts (columns) -> list of lists (rows) -> Tensor
            batch_size = len(examples[feature_cols[0]])
            features_batch = []
            labels_batch = []
            
            for i in range(batch_size):
                # Extract features
                row_features = [float(examples[col][i]) for col in feature_cols]
                features_batch.append(torch.tensor(row_features, dtype=torch.float))
                
                # Extract labels
                # We assume a single 'label' column for 0 (Normal) vs 1 (Anomaly)
                # But we map it to the head name expected by the model
                label_val = float(examples['label'][i]) if 'label' in examples else 0.0
                labels_batch.append(torch.tensor(label_val, dtype=torch.long))

            return {
                "features": features_batch,
                f"labels_{task_config.heads[0].name}": labels_batch
            }

        return dataset.map(process_tabular, batched=True, remove_columns=dataset.column_names)