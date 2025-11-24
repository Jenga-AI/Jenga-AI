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
                # Assuming the label column in the dataset is named 'label' for single-label classification
                if 'label' in examples: # Changed from head_config.name in examples
                    tokenized_inputs[f'labels_{head_name}'] = [torch.tensor(l, dtype=torch.long) for l in examples['label']] # Changed from examples[head_name]
                else:
                    num_examples = len(examples['text'])
                    tokenized_inputs[f'labels_{head_name}'] = [torch.full((1,), -100, dtype=torch.long) for _ in range(num_examples)]
            
            return tokenized_inputs
        return dataset.map(process_and_tokenize, batched=True)

    def _process_multi_label_classification(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Processes and tokenizes data for multi-label classification."""
        def process_and_tokenize(examples):
            tokenized_inputs = self.tokenizer(
                examples['text'],
                padding=False,
                truncation=self.config.tokenizer.truncation,
                max_length=self.config.tokenizer.max_length,
            )
            
            # Convert list of labels to multi-hot encoded tensor
            # Assuming labels are provided as a list of strings or integers
            # and task_config.labels contains the full set of possible labels
            num_labels = len(task_config.labels)
            batch_labels = []
            for example_labels in examples['labels']:
                multi_hot_labels = [0] * num_labels
                for label in example_labels:
                    if label in task_config.labels:
                        multi_hot_labels[task_config.labels.index(label)] = 1
                batch_labels.append(multi_hot_labels)
            
            tokenized_inputs['labels'] = [torch.tensor(l, dtype=torch.float) for l in batch_labels]
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