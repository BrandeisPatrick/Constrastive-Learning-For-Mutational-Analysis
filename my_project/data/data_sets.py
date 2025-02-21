# my_project/data/datasets.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob
from typing import List, Optional, Tuple, Union


def limit_by_length(sequences: List[np.ndarray], length_limit: Optional[int] = None) -> List[np.ndarray]:
    """
    Truncate sequences to a maximum length.

    Args:
        sequences (List[np.ndarray]): List of sequences.
        length_limit (Optional[int], optional): Maximum allowed length. Defaults to None.

    Returns:
        List[np.ndarray]: Truncated sequences.
    """
    if length_limit is None:
        return sequences
    return [seq if len(seq) < length_limit else seq[:length_limit, :] for seq in sequences]


class EmbeddDataset(Dataset):
    """
    Dataset for loading embedding files.

    Args:
        folder_path (str): Path to the folder containing .npz files.
        n_files (Optional[int], optional): Number of files to load. Defaults to None.
        length_limit (Optional[int], optional): Maximum sequence length. Defaults to None.
    """

    def __init__(self, folder_path: str, n_files: Optional[int] = None, length_limit: Optional[int] = None):
        self.folder_path = folder_path
        self.files = sorted(glob(os.path.join(self.folder_path, "*.npz")))
        
        if n_files:
            self.files = self.files[:min(len(self.files), n_files)]
        self.length_limit = length_limit
        print(f'Number of files: {len(self.files)}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        embds = data['embds']
        e1 = limit_by_length(embds[data['ind1']], self.length_limit)
        e2 = limit_by_length(embds[data['ind2']], self.length_limit)
        cl = data['cl']
        return e1, e2, cl


class EmbeddDatasetSeq(Dataset):
    """
    Dataset for loading sequence data from TSV files.

    Args:
        folder_path (Union[str, pd.DataFrame]): Path to the TSV file or a DataFrame.
        n_files (Optional[int], optional): Number of file splits. Defaults to None.
        length_limit (Optional[int], optional): Maximum sequence length. Defaults to None.
        batch_size (int, optional): Number of samples per split. Defaults to 32.
    """

    def __init__(self, folder_path: Union[str, pd.DataFrame], n_files: Optional[int] = None,
                 length_limit: Optional[int] = None, batch_size: int = 32):
        if isinstance(folder_path, str):
            self.folder_path = folder_path
            if folder_path.endswith('.tsv'):
                self.data = pd.read_csv(folder_path, sep='\t', usecols=['seq_x', 'seq_y', 'sameFunc'])
            else:
                self.data = pd.read_csv(os.path.join(folder_path, 'data.tsv'), sep='\t', usecols=['seq_x', 'seq_y', 'sameFunc'])
        elif isinstance(folder_path, pd.DataFrame):
            self.data = folder_path
            self.folder_path = 'DataFrame'
        else:
            raise ValueError("folder_path must be a string or a pandas DataFrame.")

        # Split data into batches
        self.files = list(np.array_split(self.data, np.ceil(len(self.data) / batch_size)))

        if n_files:
            self.files = self.files[:min(len(self.files), n_files)]

        self.length_limit = int(length_limit) if length_limit else None
        print(f'Number of file splits: {len(self.files)}')

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> List[Tuple[str, str, int]]:
        batch = self.files[idx]
        samples = batch[['seq_x', 'seq_y', 'sameFunc']].values.tolist()
        # Truncate sequences if necessary
        if self.length_limit:
            samples = [
                (x if len(x) < self.length_limit else x[:self.length_limit],
                 y if len(y) < self.length_limit else y[:self.length_limit],
                 z)
                for x, y, z in samples
            ]
        return samples