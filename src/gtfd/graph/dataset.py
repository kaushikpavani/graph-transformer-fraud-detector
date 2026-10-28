"""Dataset and DataLoader for graph snapshots."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import NeighborLoader

from .builder import GraphSnapshot


class GraphDataset(torch.utils.data.Dataset):
    """Dataset wrapper for graph snapshots.

    Provides PyTorch Dataset interface for temporal graph snapshots
    with optional neighbor sampling for large graphs.
    """

    def __init__(
        self,
        snapshots: list[GraphSnapshot],
        use_neighbor_sampler: bool = False,
        num_neighbors: list[int] = [15, 10, 5],
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        """Initialize the dataset.

        Args:
            snapshots: List of GraphSnapshot objects.
            use_neighbor_sampler: Whether to use NeighborLoader for large graphs.
            num_neighbors: Number of neighbors to sample per layer.
            batch_size: Batch size (for neighbor sampler).
            shuffle: Whether to shuffle snapshots.
        """
        self.snapshots = snapshots
        self.use_neighbor_sampler = use_neighbor_sampler
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data | HeteroData:
        return self.snapshots[idx].data

    def get_loader(self) -> Union[PyGDataLoader, NeighborLoader]:
        """Get a DataLoader for this dataset."""
        if self.use_neighbor_sampler and len(self.snapshots) > 0:
            # Use first snapshot to infer node types for HeteroData
            sample_data = self.snapshots[0].data
            if isinstance(sample_data, HeteroData):
                return NeighborLoader(
                    sample_data,
                    num_neighbors=self.num_neighbors,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    input_nodes=None,  # All nodes
                )
        return PyGDataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[Data | HeteroData]) -> list[Data | HeteroData]:
        """Custom collate that returns list (graphs may have different sizes)."""
        return batch


class GraphDataLoader:
    """High-level DataLoader for graph snapshots with temporal batching."""

    def __init__(
        self,
        snapshots: list[GraphSnapshot],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        use_neighbor_sampler: bool = False,
        num_neighbors: list[int] = [15, 10, 5],
    ):
        """Initialize the data loader.

        Args:
            snapshots: List of GraphSnapshot objects.
            batch_size: Number of snapshots per batch.
            shuffle: Whether to shuffle snapshots.
            drop_last: Whether to drop the last incomplete batch.
            use_neighbor_sampler: Use NeighborLoader for large graphs.
            num_neighbors: Neighbors per layer for neighbor sampling.
        """
        self.dataset = GraphDataset(
            snapshots=snapshots,
            use_neighbor_sampler=use_neighbor_sampler,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.loader = self.dataset.get_loader()

    def __iter__(self) -> Iterator[list[Data | HeteroData]]:
        return iter(self.loader)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def snapshots(self) -> list[GraphSnapshot]:
        return self.dataset.snapshots