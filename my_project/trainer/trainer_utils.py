# my_project/utils.py

import torch
import logging
from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np 
import os
import torch.distributed as dist

logger = logging.getLogger(__name__)


def save_model_checkpoint(model: torch.nn.Module, args, path: str): 
    """
    Save the model's state dictionary to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        args: Configuration arguments.
        path (str): Full file path where the model will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the state dictionary directly to the specified path
    torch.save(model.state_dict(), path)
    
    logger.info(f"Model state dict saved to {path}")


def log_metrics(loss: float, trues: np.ndarray, preds: np.ndarray, logs: dict, num_classes: int, pos_label: int) -> Tuple[dict, str]:
    """
    Compute and log various metrics.

    Args:
        loss (float): Current loss value.
        trues (np.ndarray): Ground truth labels.
        preds (np.ndarray): Predicted labels.
        logs (dict): Dictionary to accumulate logs.
        num_classes (int): Number of classes.
        pos_label (int): Label considered as positive for binary classification.

    Returns:
        Tuple[dict, str]: Updated logs and performance string.
    """
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds, labels=list(range(num_classes)))
    tp = cm[pos_label, pos_label]
    tn = sum(cm[i, j] for i in range(num_classes) for j in range(num_classes) if i != pos_label and j != pos_label)
    fp = sum(cm[i, pos_label] for i in range(num_classes) if i != pos_label)
    fn = sum(cm[pos_label, j] for j in range(num_classes) if j != pos_label)

    logs['count'] += 1
    logs['loss'] += loss
    logs['TP'] += tp
    logs['TN'] += tn
    logs['FP'] += fp
    logs['FN'] += fn
    logs['acc'] = acc

    perf_str = f"Loss: {logs['loss'] / logs['count']:.4f}, Accuracy: {acc:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
    return logs, perf_str



def random_mask_tokens(inputs, mlm_probability=0.15, ntoken=9, special_tokens=[0,1,2,3,4, 5, 6], mask_token=6, pad_token=0):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # mlm_probability defaults to 0.15 in Bert
    #inputs , labels = torch.tensor(inputs), torch.tensor(labels)
    special_tokens_mask = torch.isin(labels, torch.tensor(special_tokens) )
    padding_mask = labels.eq(pad_token)
    
    probability_matrix = torch.full(labels.shape, mlm_probability) #, device=labels.device) 0.15
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # We only compute loss on masked tokens
    # torch ignores -100 tokens for loss computation - not masked

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100 

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(ntoken, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels ## inputs and mask labels


def safe_barrier():
    """
    Ensure synchronization across distributed processes.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()