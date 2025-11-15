# Jenga-AI Project: Gemini's Contextual Understanding

This document outlines my understanding of the Jenga-AI project based on a review of its source code and documentation.

## 1. High-Level Goal

Jenga-AI is a Python-based framework for building and training multi-task Natural Language Processing (NLP) models. Its core mission, as stated in the `README.md`, is to democratize advanced NLP in an African context, with a focus on tasks relevant to sustainable development and security (e.g., Swahili sentiment analysis, threat detection).

The framework is built on the `transformers` and `torch` libraries and uses a modular architecture to define, train, and evaluate models that can handle multiple distinct NLP tasks simultaneously.

## 2. Project Architecture and Key Components

The project is primarily organized within the `multitask_bert/` directory.

### a. `multitask_bert/core/` - The Engine

This directory contains the central components of the framework.

-   **`config.py`**: Defines a series of `dataclass` objects (`ExperimentConfig`, `TaskConfig`, `ModelConfig`, etc.) that create a strongly-typed configuration system. This allows for defining entire experiments, from model type to task specifics, in a single YAML file.
-   **`model.py`**: Contains the `MultiTaskModel`. This class uses a shared transformer-based encoder (like BERT) and routes the output to different task-specific "heads". It is designed to handle one task at a time per forward pass, identified by a `task_id`.
-   **`fusion.py`**: Implements the `AttentionFusion` module. This is a key innovation that allows the model to learn task-specific representations by creating task embeddings and using an attention mechanism to re-weight the shared encoder's output for each specific task.

### b. `multitask_bert/tasks/` - Task Definitions

This directory defines the individual NLP tasks.

-   **`base.py`**: Contains the abstract `BaseTask` class. Every specific task (like classification or NER) must inherit from this class. It standardizes the task interface, requiring an implementation for `get_forward_output`, which defines the task-specific model head and loss calculation.
-   **`classification.py`**: Provides concrete implementations for `SingleLabelClassificationTask` and `MultiLabelClassificationTask`. These classes show how to build a task-specific head (a simple `nn.Linear` layer) and compute the appropriate loss (`CrossEntropyLoss` or `BCEWithLogitsLoss`).

### c. `multitask_bert/data/` - Data Handling

This directory is responsible for loading and preparing data.

-   **`data_processing.py`**: Contains the `DataProcessor` class. This class reads an `ExperimentConfig`, iterates through the defined tasks, and uses the `datasets` library to load and process data from files. It includes task-specific logic for formatting labels, such as aligning labels with tokens for NER tasks.
-   **`universal.py`**: This file is currently empty, suggesting data processing is centralized in `data_processing.py`.

### d. `multitask_bert/training/` - Training and Evaluation

This directory manages the model training and evaluation loop.

-   **`trainer.py`**: Contains the main `Trainer` class. It orchestrates the multi-task training process.
    -   **Dataloaders**: It creates separate `DataLoader` instances for each task, using task-specific collate functions to handle batching.
    -   **Training Loop**: It uses a round-robin scheduler to iterate through batches from different tasks, ensuring the model trains on all tasks in each epoch.
    -   **Evaluation**: It calculates and logs metrics for each task head separately.
    -   **Logging**: It supports logging to both MLflow and TensorBoard.
-   **`data.py`**: This file is empty.

### e. `multitask_bert/utils/` - Utilities

This directory contains helper functions.

-   **`metrics.py`**: Provides functions to compute evaluation metrics for different task types, such as `compute_classification_metrics` and `compute_ner_metrics`. These are called by the `Trainer` during the evaluation phase.
-   **`logging.py`**: This file is empty, with logging logic being handled directly in the `Trainer`.

## 3. Summary of Functionality

The end-to-end workflow of the Jenga-AI framework is as follows:

1.  A user defines a complete experiment in a **YAML file**, specifying the project name, one or more tasks (with their types, data paths, and heads), model configuration (including the base model and fusion strategy), and training parameters.
2.  A script (e.g., `run_experiment.py`) loads this YAML into the `ExperimentConfig` dataclasses.
3.  The `DataProcessor` uses this config to load and process the data for all specified tasks, creating tokenized `Dataset` objects.
4.  The `MultiTaskModel` is instantiated with the shared encoder and the defined task heads.
5.  The `Trainer` is initialized with the model, datasets, and configuration. It creates the necessary dataloaders, optimizer, and scheduler.
6.  The `trainer.train()` method is called, which begins the round-robin training loop, feeding batches from each task to the model and updating the weights.
7.  After each epoch, `trainer.evaluate()` computes and logs the performance on each task's evaluation set.
