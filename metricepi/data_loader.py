#!/usr/bin/env python3
"""
Sharded Data Loader

For loading data from sharded HDF5 files for training. Features:
- On-demand loading: Dynamically load shards during training
- Memory efficient: Does not load all data at once
- Random sampling: Supports cross-shard random sampling
- Multi-worker support: Supports PyTorch multi-process loading

Usage:
    from metricepi import ShardedEPIDataset, get_sharded_dataloader

    dataset = ShardedEPIDataset('datasets/GM12878/processed')
    dataloader = get_sharded_dataloader(dataset, batch_size=32)
"""

import os
import json
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict
import warnings
import multiprocessing as mp
from functools import lru_cache


class ShardedEPIDataset(Dataset):
    """
    Sharded dataset

    Supports loading data from multiple HDF5 shard files with multi-process support
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        cache_size: int = 2,
        preload_metadata: bool = True
    ):
        """
        Args:
            data_dir: Sharded data directory
            transform: Optional data transformation function
            cache_size: Number of shards to cache (each worker has independent cache)
            preload_metadata: Whether to preload metadata
        """
        self.data_dir = data_dir
        self.transform = transform
        self.cache_size = cache_size

        # Load metadata
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.total_samples = self.metadata['total_samples']
        self.num_shards = self.metadata['num_shards']
        self.shard_size = self.metadata['shard_size']
        self.enhancer_length = self.metadata['enhancer_length']
        self.promoter_length = self.metadata['promoter_length']
        self.num_channels = self.metadata['num_channels']
        self.feature_names = self.metadata['feature_names']

        # Build mapping from global index to shard
        self.shard_files = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith('shard_') and f.endswith('.h5')
        ])

        # Validate shard count
        if len(self.shard_files) != self.num_shards:
            warnings.warn(f"Expected {self.num_shards} shards, found {len(self.shard_files)}")

        # Build sample indices: (shard_idx, local_idx)
        # Build complete index on preload, otherwise build lazily
        if preload_metadata:
            self._build_sample_indices()
        else:
            self.sample_indices = None

        # Shard cache (independent per worker)
        self._cache = {}
        self._cache_order = []

        # Only print in main process
        if mp.current_process().name == 'MainProcess':
            print(f"ShardedEPIDataset loaded:")
            print(f"  - Total samples: {self.total_samples:,}")
            print(f"  - Number of shards: {self.num_shards}")
            print(f"  - Enhancer length: {self.enhancer_length}")
            print(f"  - Promoter length: {self.promoter_length}")
            print(f"  - Channels: {self.num_channels}")

    def _build_sample_indices(self):
        """Build sample indices"""
        self.sample_indices = []
        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as hf:
                num_samples = hf['label'].shape[0]
                for local_idx in range(num_samples):
                    self.sample_indices.append((shard_idx, local_idx))

    def _get_sample_indices(self):
        """Lazily get sample indices"""
        if self.sample_indices is None:
            self._build_sample_indices()
        return self.sample_indices

    def _load_shard(self, shard_idx: int) -> Dict:
        """Load shard data to cache"""
        if shard_idx in self._cache:
            return self._cache[shard_idx]

        # If cache is full, remove the oldest
        while len(self._cache) >= self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]

        # Load new shard
        shard_file = self.shard_files[shard_idx]
        with h5py.File(shard_file, 'r') as hf:
            shard_data = {
                'enhancer': hf['enhancer'][:],
                'promoter': hf['promoter'][:],
                'label': hf['label'][:]
            }
            # Check if distance data exists (backwards compatible)
            if 'distance' in hf:
                shard_data['distance'] = hf['distance'][:]
            else:
                # If no distance data, use default value
                shard_data['distance'] = np.full(hf['label'].shape, 50000.0, dtype=np.float32)

            self._cache[shard_idx] = shard_data

        self._cache_order.append(shard_idx)
        return self._cache[shard_idx]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
        sample_indices = self._get_sample_indices()
        shard_idx, local_idx = sample_indices[idx]

        shard_data = self._load_shard(shard_idx)

        enhancer = shard_data['enhancer'][local_idx]
        promoter = shard_data['promoter'][local_idx]
        label = shard_data['label'][local_idx]
        distance = shard_data['distance'][local_idx]

        if self.transform:
            enhancer, promoter = self.transform(enhancer, promoter)

        return enhancer, promoter, label, distance

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'num_samples': self.total_samples,
            'num_shards': self.num_shards,
            'enhancer_length': self.enhancer_length,
            'promoter_length': self.promoter_length,
            'num_channels': self.num_channels,
            'feature_names': self.feature_names
        }

    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
        self._cache_order.clear()


class ShardedEPICollate:
    """Custom collate function"""

    def __call__(self, batch):
        # Support both formats: with and without distance
        if len(batch[0]) == 4:
            # New format: (enhancer, promoter, label, distance)
            enhancers, promoters, labels, distances = zip(*batch)
            enhancers = torch.tensor(np.array(enhancers), dtype=torch.float32)
            promoters = torch.tensor(np.array(promoters), dtype=torch.float32)
            labels = torch.tensor(np.array(labels), dtype=torch.long)
            distances = torch.tensor(np.array(distances), dtype=torch.float32)
            return enhancers, promoters, labels, distances
        else:
            # Old format: (enhancer, promoter, label)
            enhancers, promoters, labels = zip(*batch)
            enhancers = torch.tensor(np.array(enhancers), dtype=torch.float32)
            promoters = torch.tensor(np.array(promoters), dtype=torch.float32)
            labels = torch.tensor(np.array(labels), dtype=torch.long)
            return enhancers, promoters, labels


class ShardedSampler(Sampler):
    """
    Shard-aware sampler

    Tries to access shards in order within an epoch to reduce IO overhead from random access
    But shuffles within each shard
    """

    def __init__(
        self,
        dataset: ShardedEPIDataset,
        shuffle: bool = True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        # Organize indices by shard
        shard_to_indices = defaultdict(list)
        sample_indices = self.dataset._get_sample_indices()

        # Check format and process
        if len(sample_indices) > 0 and isinstance(sample_indices[0], tuple):
            first_item = sample_indices[0]
            if len(first_item) == 3:
                # SubsetShardedDataset: [(subset_idx, shard_idx, local_idx), ...]
                for subset_idx, shard_idx, local_idx in sample_indices:
                    shard_to_indices[shard_idx].append(subset_idx)
            elif len(first_item) == 2:
                # ShardedEPIDataset: [(shard_idx, local_idx), ...]
                for global_idx, (shard_idx, local_idx) in enumerate(sample_indices):
                    shard_to_indices[shard_idx].append(global_idx)
            else:
                # Fallback
                for i in range(len(self.dataset)):
                    shard_to_indices[0].append(i)
        else:
            # Fallback: normal order
            for i in range(len(self.dataset)):
                shard_to_indices[0].append(i)

        # Generate sampling order
        all_indices = []

        # Shuffle shard order
        shard_order = list(shard_to_indices.keys())
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shard_order)

        for shard_idx in shard_order:
            indices = shard_to_indices[shard_idx]
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(indices)
            all_indices.extend(indices)

        return iter(all_indices)

    def __len__(self) -> int:
        return len(self.dataset)


def get_sharded_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_size: int = 2,
    seed: int = 42
) -> DataLoader:
    """
    Create sharded data loader

    Args:
        data_dir: Sharded data directory
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        cache_size: Number of shards to cache
        seed: Random seed

    Returns:
        PyTorch DataLoader
    """
    dataset = ShardedEPIDataset(data_dir, cache_size=cache_size)
    sampler = ShardedSampler(dataset, shuffle=shuffle, seed=seed)
    collate = ShardedEPICollate()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive to reduce startup overhead
    )


def split_sharded_dataset(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split dataset into train and validation indices

    Returns:
        (train_indices, val_indices)
    """
    random.seed(seed)

    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    total_samples = metadata['total_samples']
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)

    val_size = int(total_samples * val_ratio)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    return train_indices, val_indices


class SubsetShardedDataset(Dataset):
    """Subset of a sharded dataset"""

    def __init__(
        self,
        dataset: ShardedEPIDataset,
        indices: List[int]
    ):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, float]:
        return self.dataset[self.indices[idx]]


def get_train_val_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    cache_size: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders

    Args:
        data_dir: Sharded data directory
        batch_size: Batch size
        val_ratio: Validation set ratio
        num_workers: Number of data loading workers
        cache_size: Number of shards to cache
        seed: Random seed

    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = ShardedEPIDataset(data_dir, cache_size=cache_size)

    # Split indices
    train_indices, val_indices = split_sharded_dataset(data_dir, val_ratio, seed)

    # Create subsets
    train_dataset = SubsetShardedDataset(full_dataset, train_indices)
    val_dataset = SubsetShardedDataset(full_dataset, val_indices)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # Create DataLoaders
    collate = ShardedEPICollate()

    # Multi-process optimization
    persistent_workers = num_workers > 0

    # Training set: use normal shuffle, rely on large cache to avoid thrashing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Validation set: use single process to share cache
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single process shared cache
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=None
    )

    return train_loader, val_loader


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Test loading
    print(f"\n{'='*60}")
    print("Testing Sharded Data Loader")
    print(f"{'='*60}\n")

    # Use convenience function
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir,
        batch_size=4,
        val_ratio=0.2,
        cache_size=2
    )

    # Test iteration
    print("\nTesting train loader...")
    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 4:
            enhancers, promoters, labels, distances = batch
            print(f"Batch {batch_idx}: enhancer={enhancers.shape}, promoter={promoters.shape}, labels={labels.shape}")
        else:
            enhancers, promoters, labels = batch
            print(f"Batch {batch_idx}: enhancer={enhancers.shape}, promoter={promoters.shape}, labels={labels.shape}")
        if batch_idx >= 2:
            break

    print("\nTesting val loader...")
    for batch_idx, batch in enumerate(val_loader):
        if len(batch) == 4:
            enhancers, promoters, labels, distances = batch
            print(f"Batch {batch_idx}: enhancer={enhancers.shape}, promoter={promoters.shape}, labels={labels.shape}")
        else:
            enhancers, promoters, labels = batch
            print(f"Batch {batch_idx}: enhancer={enhancers.shape}, promoter={promoters.shape}, labels={labels.shape}")
        if batch_idx >= 1:
            break

    print("\nTest completed!")
