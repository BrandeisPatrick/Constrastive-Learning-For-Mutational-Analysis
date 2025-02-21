# my_project/data/collate.py

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional, Dict, Any
from functools import partial

import torch
from typing import List, Tuple

def pad_batches(src1: List[torch.Tensor], src2: List[torch.Tensor],
                            length_limit: int = 1201, pad_token: int = -100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads and truncates batches of src1 and src2 sequences to a fixed length.

    Args:
        src1 (List[torch.Tensor]): List of src1 sequences.
        src2 (List[torch.Tensor]): List of src2 sequences.
        length_limit (int, optional): Fixed sequence length. Defaults to 1201.
        pad_token (int, optional): Padding token ID. Defaults to -100.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Padded src1 and src2 tensors with fixed length.
    """
    # Helper function to pad or truncate a single sequence
    def pad_or_truncate(seq: torch.Tensor, length: int, pad_val: int) -> torch.Tensor:
        if len(seq) < length:
            padding = torch.full((length - len(seq),), pad_val, dtype=seq.dtype)
            return torch.cat((seq, padding), dim=0)
        else:
            return seq[:length]
    
    # Apply padding/truncation to all sequences in src1 and src2
    src1_fixed = [pad_or_truncate(seq, length_limit, pad_token) for seq in src1]
    src2_fixed = [pad_or_truncate(seq, length_limit, pad_token) for seq in src2]
    
    # Stack into tensors
    src1_padded = torch.stack(src1_fixed)
    src2_padded = torch.stack(src2_fixed)
    
    return src1_padded, src2_padded

def collate_fn_model_classifier_seq(batch: List[Dict[str, Any]],
                                    tokenizer: object,
                                    device: str = 'cpu',
                                    pad_token: int = -100,
                                    training: bool = False,
                                    args: Optional[object] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for Sequence-based Model Classifier.

    Args:
        batch (List[Dict[str, Any]]): Batch data containing sequences and class labels.
        tokenizer (object): Tokenizer object with a `tokenize` method.
        device (str, optional): Device to move tensors to. Defaults to 'cpu'.
        pad_token (int, optional): Padding token ID. Defaults to -100.
        training (bool, optional): Flag indicating training mode. Defaults to False.
        args (Optional[object], optional): Additional arguments. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded src1, src2, and class labels.
    """
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided to the collate function.")
    if args is None:
        raise ValueError("Arguments (`args`) must be provided to the collate function.")

    # Extract sequences and labels from each sample in the batch
    seq_x = [sample['seq_x'] for sample in batch]
    seq_y = [sample['seq_y'] for sample in batch]
    cl = [sample['sameFunc'] for sample in batch]

    # Optionally apply length limiting (if not already done in the Dataset)
    if hasattr(args, 'max_seq_len') and args.max_seq_len:
        seq_x = [s[:args.max_seq_len] for s in seq_x]
        seq_y = [s[:args.max_seq_len] for s in seq_y]

    # Tokenize sequences
    tokens_x = tokenizer.tokenize(seq_x)
    tokens_y = tokenizer.tokenize(seq_y)

    # Pad sequences
    src1_padded, src2_padded = pad_batches(tokens_x, tokens_y,
                                           length_limit=args.max_seq_len,
                                           pad_token=pad_token)

    # Convert class labels to tensor
    cl_tensor = torch.tensor(cl, dtype=torch.long)

    # Optionally add noise for data augmentation during training
    if training and hasattr(args, 'add_noise') and args.add_noise > 0:
        src1_padded = src1_padded + torch.randn_like(src1_padded) * args.add_noise
        src2_padded = src2_padded + torch.randn_like(src2_padded) * args.add_noise

    return src1_padded.to(device), src2_padded.to(device), cl_tensor.to(device)