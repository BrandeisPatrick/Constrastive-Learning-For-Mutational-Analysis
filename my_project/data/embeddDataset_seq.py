from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
import os
from .collate import collate_fn_model_classifier_seq 
from my_project.tokenizers import BioSeqTokenizer 

class EmbeddDatasetSeq(Dataset):
    def __init__(self, folder_path, dataset_type, args):
        if isinstance(folder_path, str):
            try:
                if folder_path.endswith('.tsv'):
                    self.data = pd.read_csv(folder_path, sep='\t', usecols=['seq_x', 'seq_y', 'sameFunc'])
                else:
                    self.data = pd.read_csv(os.path.join(folder_path, 'data.tsv'), sep='\t', usecols=['seq_x', 'seq_y', 'sameFunc'])
            except Exception as e:
                raise ValueError(f"Error loading data from {folder_path}: {e}")
        elif isinstance(folder_path, pd.DataFrame):
            self.data = folder_path
        else:
            raise ValueError("folder_path must be a string path or a pandas DataFrame.")

        if 'sameFunc' not in self.data.columns:
            raise ValueError(f"'sameFunc' column missing in the {dataset_type} dataset.")

        self.length_limit = args.max_seq_len
        self.tokenizer = BioSeqTokenizer(
            window=args.window,
            stride=args.stride,
            model=args.model
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        seq_x = sample['seq_x'][:self.length_limit] if self.length_limit else sample['seq_x']
        seq_y = sample['seq_y'][:self.length_limit] if self.length_limit else sample['seq_y']
        same_func = sample['sameFunc']
        return {
            'seq_x': seq_x,
            'seq_y': seq_y,
            'sameFunc': same_func
        }

def create_dataloaders(args):
    tokenizer = BioSeqTokenizer(
        window=args.window,
        stride=args.stride,
        model=args.model
    )

    collate_fn_train = partial(
        collate_fn_model_classifier_seq,
        tokenizer=tokenizer,
        device=args.devices,  # Ensure args.devices is correctly set
        pad_token=args.pad_token,
        training=True, 
        args=args 
    )

    collate_fn_eval = partial(
        collate_fn_model_classifier_seq,
        tokenizer=tokenizer,
        device=args.devices,
        pad_token=args.pad_token,
        training=False, 
        args=args 
    )

    train_dataset = EmbeddDatasetSeq(args.folder_train, "training", args)
    val_dataset = EmbeddDatasetSeq(args.folder_val, "validation", args)
    test_dataset = EmbeddDatasetSeq(args.folder_test, "test", args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
        # pin_memory=(args.default == 'cuda'), 
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval,
        # pin_memory=(args.default == 'cuda'),
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_eval,
        # pin_memory=(args.default == 'cuda'),
        shuffle=False
    )

    return train_loader, val_loader, test_loader