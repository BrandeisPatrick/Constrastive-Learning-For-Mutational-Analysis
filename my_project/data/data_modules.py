# # my_project/data/data_modules.py

# import torch
# from torch.utils.data import DataLoader, distributed
# from functools import partial
# from typing import Tuple, Optional, Type, Any, Callable

# from .data_sets import EmbeddDataset, EmbeddDatasetSeq
# from .collate import  collate_fn_model_classifier_seq


# class BaseDataModule:
#     """
#     Base class for Data Modules.

#     Args:
#         args (object): Configuration arguments.
#         dataset_cls (Type): Dataset class to use.
#         collate_fn_factory (Callable): A factory function that returns a collate function.
#     """

#     def __init__(
#         self,
#         args: Any,
#         dataset_cls: Type,
#         collate_fn_factory: Callable[[bool], Callable]
#     ):
#         self.args = args
#         self.dataset_cls = dataset_cls
#         self.collate_fn_factory = collate_fn_factory

#         self.train_set: Optional[Any] = None
#         self.val_set: Optional[Any] = None
#         self.test_set: Optional[Any] = None

#         self.train_loader: Optional[DataLoader] = None
#         self.val_loader: Optional[DataLoader] = None
#         self.test_loader: Optional[DataLoader] = None

#     def prepare_data(self):
#         """
#         Method to prepare data. Can be overridden by subclasses.
#         """
#         pass

#     def setup(self, stage: Optional[str] = None):
#         """
#         Sets up the datasets for training, validation, and testing.

#         Args:
#             stage (Optional[str], optional): Stage of setup. Defaults to None.
#         """
#         n_files = 200 if getattr(self.args, 'mock', 0) == 1000 else None

#         # Initialize datasets with common parameters
#         self.train_set = self.dataset_cls(
#             folder_path=self.args.folder_train,
#             n_files=n_files,
#             length_limit=self.get_length_limit(),
#             **self.get_additional_dataset_kwargs()
#         )
#         self.val_set = self.dataset_cls(
#             folder_path=self.args.folder_val,
#             n_files=n_files,
#             length_limit=self.get_length_limit(),
#             **self.get_additional_dataset_kwargs()
#         )
#         self.test_set = self.dataset_cls(
#             folder_path=self.args.folder_test,
#             n_files=n_files,
#             length_limit=self.get_length_limit(),
#             **self.get_additional_dataset_kwargs()
#         )

#         print(
#             f'Loaded {self.dataset_cls.__name__}: '
#             f'Train={len(self.train_set)}, Val={len(self.val_set)}, Test={len(self.test_set)}'
#         )

#     def get_length_limit(self) -> int:
#         """
#         Returns the length limit for the datasets. Can be overridden by subclasses.

#         Returns:
#             int: Length limit.
#         """
#         return self.args.max_seq_len

#     def get_additional_dataset_kwargs(self) -> dict:
#         """
#         Returns additional keyword arguments for dataset initialization.
#         Can be overridden by subclasses.

#         Returns:
#             dict: Additional kwargs.
#         """
#         return {}

#     def create_dataloader(self, dataset: Any, training: bool = True) -> DataLoader:
#         """
#         Creates a DataLoader for a given dataset.

#         Args:
#             dataset (Any): Dataset object.
#             training (bool, optional): Flag indicating training mode. Defaults to True.

#         Returns:
#             DataLoader: Configured DataLoader.
#         """
#         collate_fn = self.collate_fn_factory(training)

#         sampler = None
#         if training and getattr(self.args, 'run_mode', '') == 'torch_distributed_gpu':
#             sampler = distributed.DistributedSampler(
#                 dataset,
#                 seed=self.args.random_seed,
#                 num_replicas=self.args.world_size,
#                 rank=self.args.rank
#             )

#         return DataLoader(
#             dataset,
#             batch_size=1,
#             num_workers=self.args.num_workers,
#             collate_fn=collate_fn,
#             sampler=sampler,
#             shuffle=(sampler is None)
#         )

#     def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
#         """
#         Returns train, validation, and test DataLoaders.

#         Returns:
#             Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and testing.
#         """
#         if not all([self.train_set, self.val_set, self.test_set]):
#             raise RuntimeError("Datasets not initialized. Call `setup()` before getting dataloaders.")

#         self.train_loader = self.create_dataloader(self.train_set, training=True)
#         self.val_loader = self.create_dataloader(self.val_set, training=False)
#         self.test_loader = self.create_dataloader(self.test_set, training=False)
#         return self.train_loader, self.val_loader, self.test_loader


# class EmbeddDataModule(BaseDataModule):
#     """
#     DataModule for EmbeddDataset.

#     Args:
#         args (object): Configuration arguments.
#     """

#     def __init__(self, args: Any):
#         super().__init__(
#             args=args,
#             dataset_cls=EmbeddDataset,
#             collate_fn_factory=self.get_collate_fn
#         )

#     def get_collate_fn(self, training: bool) -> Callable:
#         """
#         Returns the collate function for EmbeddDataset.

#         Args:
#             training (bool): Indicates if it's for training.

#         Returns:
#             Callable: Collate function.
#         """
#         return partial(
#             collate_fn_model_classifier,
#             device=self.args.default_device,
#             pad_token=self.args.pad_token,
#             args=self.args,
#             training=training
#         )


# class EmbeddSeqDataModule(BaseDataModule):
#     """
#     DataModule for EmbeddDatasetSeq.

#     Args:
#         args (object): Configuration arguments.
#         tokenizer (object): Tokenizer object.
#     """

#     def __init__(self, args: Any, tokenizer: Any):
#         self.tokenizer = tokenizer
#         super().__init__(
#             args=args,
#             dataset_cls=EmbeddDatasetSeq,
#             collate_fn_factory=self.get_collate_fn
#         )

#     def get_length_limit(self) -> int:
#         """
#         Overrides the length limit for EmbeddDatasetSeq.

#         Returns:
#             int: Adjusted length limit.
#         """
#         return self.args.max_seq_len - 1

#     def get_additional_dataset_kwargs(self) -> dict:
#         """
#         Provides additional kwargs specific to EmbeddDatasetSeq.

#         Returns:
#             dict: Additional kwargs.
#         """
#         return {'batch_size': self.args.batch_size}

#     def get_collate_fn(self, training: bool) -> Callable:
#         """
#         Returns the collate function for EmbeddDatasetSeq.

#         Args:
#             training (bool): Indicates if it's for training.

#         Returns:
#             Callable: Collate function.
#         """
#         return partial(
#             collate_fn_model_classifier_seq,
#             tokenizer=self.tokenizer,
#             device=self.args.default_device,
#             pad_token=self.args.pad_token,
#             args=self.args,
#             training=training
#         )