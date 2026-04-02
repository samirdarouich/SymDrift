from __future__ import annotations

import os.path as osp
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from tspath.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class GeometricInMemoryDataModule(pl.LightningDataModule):
    """Lightning datamodule for PyG InMemoryDataset split handling.

    This module loads a single base dataset and builds train/val/test subset views
    from the split npz file. Each split can use its own transform at access time.
    """

    def __init__(
        self,
        dataset: InMemoryDataset,
        split_identifier: Optional[str] = None,
        train_transform: Optional[List[Callable]] = None,
        val_transform: Optional[List[Callable]] = None,
        test_transform: Optional[List[Callable]] = None,
        train_batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        follow_batch: Optional[List[str]] = None,
        shuffle_train: bool = True,
        drop_last_train: bool = False,
    ):
        super().__init__()
        self.base_dataset = dataset
        self.split_identifier = split_identifier

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size or train_batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False
        self.follow_batch = follow_batch
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._split_cache: Optional[Dict[str, np.ndarray]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._build_split_dataset(
                "train", self.train_transform
            )
            self.val_dataset = self._build_split_dataset("val", self.val_transform)
        if stage in (None, "test"):
            self.test_dataset = self._build_split_dataset("test", self.test_transform)
        elif stage == "train":
            self.train_dataset = self._build_split_dataset(
                "train", self.train_transform
            )
        elif stage == "val":
            self.val_dataset = self._build_split_dataset("val", self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle_train,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            follow_batch=self.follow_batch,
        )

    def _build_split_dataset(self, split: str, transform: Optional[BaseTransform]):
        indices = self._resolve_split_indices(split)
        logger.info(f"Building '{split}' split with {len(indices)} samples")
        split_dataset = self.base_dataset.index_select(indices)
        split_dataset.transform = transform
        return split_dataset

    def _resolve_split_indices(self, split: str) -> List[int]:
        split_values = self._load_split_array(split)

        # In case the base dataset has a mapping from identifiers saved in the split to
        # indices, use it to convert split values to indices
        mapping = getattr(self.base_dataset, "split_identifier_to_index", None)
        if mapping is not None:
            return [int(mapping[s]) for s in split_values]

        return np.asarray(split_values, dtype=np.int64).tolist()

    def _load_split_array(self, split: str) -> np.ndarray:
        if self._split_cache is None:
            self._split_cache = self._load_split_file()

        if split not in self._split_cache:
            raise ValueError(f"Split '{split}' not found in split file.")

        return self._split_cache[split]

    def _load_split_file(self) -> Dict[str, np.ndarray]:
        source = getattr(self.base_dataset, "source", "")
        split_suffix = self.split_identifier
        if split_suffix:
            filename = f"split_{source}_{split_suffix}.npz"
        else:
            filename = f"split_{source}.npz"

        split_path = osp.join(self.base_dataset.raw_dir, filename)
        return np.load(split_path)
