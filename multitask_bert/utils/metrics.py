from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from typing import Dict, List
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Computes metrics for a single-label classification task.
    """
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_multi_label_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes metrics for a multi-label classification task.
    """
    preds = (predictions > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# def compute_ner_metrics(predictions: List[np.ndarray], labels: List[np.ndarray], label_map: Dict[int, str]) -> Dict[str, float]:
#     """
#     Computes metrics for NER task.
#     Handles both sequence format and flattened format for robustness.
#     """
#     # Handle case where we get flattened arrays instead of sequences
#     if len(predictions) > 0 and not isinstance(predictions[0], (list, np.ndarray)):
#         # Convert flattened arrays back to sequences
#         # This is a fallback - the trainer should be fixed to pass sequences
#         seq_length = len(predictions) // len(labels) if len(labels) > 0 else 0
#         if seq_length > 0:
#             predictions = [predictions[i:i+seq_length] for i in range(0, len(predictions), seq_length)]
#             labels = [labels[i:i+seq_length] for i in range(0, len(labels), seq_length)]
    
#     # Filter out ignore_index (-100) and map to string labels
#     true_labels = []
#     true_predictions = []

#     for seq_preds, seq_labels in zip(predictions, labels):
#         # Convert to lists if they're numpy arrays
#         if hasattr(seq_preds, 'tolist'):
#             seq_preds = seq_preds.tolist()
#         if hasattr(seq_labels, 'tolist'):
#             seq_labels = seq_labels.tolist()
            
#         filtered_seq_labels = []
#         filtered_seq_preds = []
        
#         # Handle case where we might have single elements instead of sequences
#         if not isinstance(seq_preds, (list, np.ndarray)):
#             seq_preds = [seq_preds]
#         if not isinstance(seq_labels, (list, np.ndarray)):
#             seq_labels = [seq_labels]
            
#         for p, l in zip(seq_preds, seq_labels):
#             if l != -100:  # Ignore padding tokens
#                 filtered_seq_labels.append(label_map.get(int(l), 'O'))
#                 filtered_seq_preds.append(label_map.get(int(p), 'O'))
        
#         if filtered_seq_labels:  # Only add non-empty sequences
#             true_labels.append(filtered_seq_labels)
#             true_predictions.append(filtered_seq_preds)

#     # Calculate metrics using seqeval
#     if true_labels and true_predictions:
#         try:
#             # Use individual metrics for better control
#             precision = precision_score(true_labels, true_predictions, average='micro', zero_division=0)
#             recall = recall_score(true_labels, true_predictions, average='micro', zero_division=0)
#             f1 = f1_score(true_labels, true_predictions, average='micro', zero_division=0)
            
#             # Calculate token-level accuracy as fallback
#             flat_preds = [item for sublist in true_predictions for item in sublist]
#             flat_labels = [item for sublist in true_labels for item in sublist]
#             accuracy = accuracy_score(flat_labels, flat_preds)
            
#             return {
#                 "accuracy": accuracy,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1
#             }
#         except Exception as e:
#             print(f"Error in seqeval metrics: {e}")
#             # Fallback to token-level metrics
#             flat_preds = [item for sublist in true_predictions for item in sublist]
#             flat_labels = [item for sublist in true_labels for item in sublist]
#             accuracy = accuracy_score(flat_labels, flat_preds)
#             precision, recall, f1, _ = precision_recall_fscore_support(
#                 flat_labels, flat_preds, average='micro', zero_division=0
#             )
#             return {
#                 "accuracy": accuracy,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1
#             }
#     else:
#         return {
#             "accuracy": 0.0,
#             "precision": 0.0,
#             "recall": 0.0,
#             "f1": 0.0
#         }

def compute_ner_metrics(predictions: List[np.ndarray], labels: List[np.ndarray], label_map: Dict[int, str]) -> Dict[str, float]:
    # Filter out ignore_index (-100) and map to string labels
    true_labels = []
    true_predictions = []

    for seq_preds, seq_labels in zip(predictions, labels):
        filtered_seq_labels = []
        filtered_seq_preds = []
        for p, l in zip(seq_preds, seq_labels):
            if l != -100:
                filtered_seq_labels.append(label_map[l])
                filtered_seq_preds.append(label_map[p])
        true_labels.append(filtered_seq_labels)
        true_predictions.append(filtered_seq_preds)

    # Flatten for token-level metrics
    flat_true_labels = [label for seq in true_labels for label in seq]
    flat_true_predictions = [label for seq in true_predictions for label in seq]
    
    if flat_true_labels:
        # Calculate token-level metrics (more robust for custom tags)
        accuracy = accuracy_score(flat_true_labels, flat_true_predictions)
        
        # Get unique labels (excluding 'O' for entity-specific metrics)
        unique_labels = list(set(flat_true_labels))
        if 'O' in unique_labels:
            unique_labels.remove('O')  # Remove 'O' for entity-focused metrics
        
        if unique_labels:
            # Calculate metrics for entity classes only
            precision, recall, f1, _ = precision_recall_fscore_support(
                flat_true_labels, 
                flat_true_predictions, 
                labels=unique_labels,
                average='weighted',
                zero_division=0
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    else:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }