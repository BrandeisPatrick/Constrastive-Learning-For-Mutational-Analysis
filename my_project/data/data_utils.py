# my_project/data/datasets.py

import torch
from typing import Tuple

def random_mask_tokens(inputs: torch.Tensor,
                       mlm_probability: float = 0.15,
                       ntoken: int = 9,
                       special_tokens: list = [0, 1, 2, 3, 4, 5, 6],
                       mask_token: int = 6,
                       pad_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling:
    80% MASK, 10% random, 10% original.

    Args:
        inputs (torch.Tensor): Input token IDs.
        mlm_probability (float): Probability of masking tokens.
        ntoken (int): Number of tokens in the vocabulary.
        special_tokens (list): List of special token IDs to exclude from masking.
        mask_token (int): Token ID used for masking.
        pad_token (int): Token ID used for padding.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Masked inputs and labels.
    """
    labels = inputs.clone()
    
    # Create masks for special tokens and padding
    special_tokens_mask = torch.isin(labels, torch.tensor(special_tokens))
    padding_mask = labels.eq(pad_token)
    
    # Initialize probability matrix
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    
    # Determine which tokens to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Ignore non-masked tokens in loss
    
    # 80% of the time, replace masked input tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token
    
    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(ntoken, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    # The rest 10% of the time, keep the masked input tokens unchanged
    
    return inputs, labels
