"""Graph augmentation for self-supervised pretraining and adversarial robustness."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import dropout_edge, to_undirected

from ..config import PretrainingConfig


class GraphAugmenter:
    """Graph augmentation for contrastive learning and pretraining.

    Provides stochastic augmentations: edge dropout, feature masking,
    subgraph sampling, and attribute perturbation.
    """

    def __init__(
        self,
        edge_dropout: float = 0.1,
        feature_mask_prob: float = 0.1,
        feature_noise_std: float = 0.01,
        subgraph_ratio: float = 0.9,
    ):
        """Initialize the augmenter.

        Args:
            edge_dropout: Probability of dropping edges.
            feature_mask_prob: Probability of masking node features.
            feature_noise_std: Std of Gaussian noise added to features.
            subgraph_ratio: Ratio of nodes to keep in subgraph sampling.
        """
        self.edge_dropout = edge_dropout
        self.feature_mask_prob = feature_mask_prob
        self.feature_noise_std = feature_noise_std
        self.subgraph_ratio = subgraph_ratio

    def augment(self, data: Data | HeteroData) -> Data | HeteroData:
        """Apply augmentations to a graph.

        Args:
            data: Input graph data.

        Returns:
            Augmented graph data (new object).
        """
        if isinstance(data, HeteroData):
            return self._augment_hetero(data)
        return self._augment_homo(data)

    def _augment_homo(self, data: Data) -> Data:
        """Augment homogeneous graph."""
        data = data.clone()

        # Edge dropout
        if self.edge_dropout > 0:
            edge_index, edge_attr = dropout_edge(
                data.edge_index,
                data.edge_attr,
                p=self.edge_dropout,
                force_undirected=False,
                training=True,
            )
            data.edge_index = edge_index
            if edge_attr is not None:
                data.edge_attr = edge_attr

        # Feature masking
        if self.feature_mask_prob > 0 and data.x is not None:
            mask = torch.rand_like(data.x) < self.feature_mask_prob
            data.x = data.x.clone()
            data.x[mask] = 0

        # Feature noise
        if self.feature_noise_std > 0 and data.x is not None:
            noise = torch.randn_like(data.x) * self.feature_noise_std
            data.x = data.x + noise

        return data

    def _augment_hetero(self, data: HeteroData) -> HeteroData:
        """Augment heterogeneous graph."""
        data = data.clone()

        for edge_type in data.edge_types:
            if self.edge_dropout > 0:
                edge_index = data[edge_type].edge_index
                edge_attr = data[edge_type].edge_attr if hasattr(data[edge_type], "edge_attr") else None
                edge_index, edge_attr = dropout_edge(
                    edge_index,
                    edge_attr,
                    p=self.edge_dropout,
                    force_undirected=False,
                    training=True,
                )
                data[edge_type].edge_index = edge_index
                if edge_attr is not None:
                    data[edge_type].edge_attr = edge_attr

        for node_type in data.node_types:
            if self.feature_mask_prob > 0 and hasattr(data[node_type], "x") and data[node_type].x is not None:
                x = data[node_type].x
                mask = torch.rand_like(x) < self.feature_mask_prob
                data[node_type].x = x.clone()
                data[node_type].x[mask] = 0

            if self.feature_noise_std > 0 and hasattr(data[node_type], "x") and data[node_type].x is not None:
                noise = torch.randn_like(data[node_type].x) * self.feature_noise_std
                data[node_type].x = data[node_type].x + noise

        return data

    def augment_pair(self, data: Data | HeteroData) -> tuple[Data | HeteroData, Data | HeteroData]:
        """Generate two augmented views for contrastive learning.

        Args:
            data: Input graph data.

        Returns:
            Tuple of two independently augmented views.
        """
        view1 = self.augment(data)
        view2 = self.augment(data)
        return view1, view2


class AdversarialAugmenter:
    """Adversarial graph augmentation for robust pretraining.

    Implements PGD-style attacks on node features and edge structure
    to generate hard negative samples for adversarial training.
    """

    def __init__(self, config: PretrainingConfig):
        """Initialize with pretraining configuration.

        Args:
            config: Pretraining configuration with adversarial parameters.
        """
        self.epsilon = config.adv_epsilon
        self.steps = config.adv_steps
        self.edge_dropout = config.edge_dropout
        self.step_size = self.epsilon / self.steps

    def attack_features(
        self,
        data: Data | HeteroData,
        model: torch.nn.Module,
        loss_fn: callable,
        target_nodes: Optional[torch.Tensor] = None,
    ) -> Data | HeteroData:
        """PGD attack on node features.

        Args:
            data: Input graph data.
            model: Model to attack (provides gradients).
            loss_fn: Loss function to maximize.
            target_nodes: Specific nodes to attack (None = all).

        Returns:
            Adversarially perturbed graph data.
        """
        data = data.clone()
        model.eval()

        # Enable gradients for node features
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                if hasattr(data[node_type], "x") and data[node_type].x is not None:
                    data[node_type].x.requires_grad_(True)
        else:
            if data.x is not None:
                data.x.requires_grad_(True)

        for step in range(self.steps):
            model.zero_grad()
            out = model(data)
            loss = loss_fn(out, data)
            loss.backward()

            # Update features with gradient sign
            if isinstance(data, HeteroData):
                for node_type in data.node_types:
                    if hasattr(data[node_type], "x") and data[node_type].x is not None and data[node_type].x.grad is not None:
                        grad = data[node_type].x.grad.sign()
                        data[node_type].x = data[node_type].x + self.step_size * grad
                        # Project to epsilon ball
                        data[node_type].x = torch.clamp(
                            data[node_type].x,
                            min=data[node_type].x - self.epsilon,
                            max=data[node_type].x + self.epsilon,
                        )
                        data[node_type].x.grad.zero_()
            else:
                if data.x is not None and data.x.grad is not None:
                    grad = data.x.grad.sign()
                    data.x = data.x + self.step_size * grad
                    data.x = torch.clamp(
                        data.x, min=data.x - self.epsilon, max=data.x + self.epsilon
                    )
                    data.x.grad.zero_()

        # Detach
        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                if hasattr(data[node_type], "x"):
                    data[node_type].x = data[node_type].x.detach()
        else:
            if data.x is not None:
                data.x = data.x.detach()

        return data

    def attack_structure(self, data: Data | HeteroData) -> Data | HeteroData:
        """Adversarial edge dropout (random for now, can be gradient-based)."""
        return GraphAugmenter(edge_dropout=self.edge_dropout).augment(data)


def pgd_attack(
    data: Data | HeteroData,
    model: torch.nn.Module,
    loss_fn: callable,
    epsilon: float = 0.1,
    steps: int = 3,
    step_size: Optional[float] = None,
) -> Data | HeteroData:
    """Projected Gradient Descent attack on node features.

    Standalone function for adversarial evaluation.

    Args:
        data: Input graph data.
        model: Model to attack.
        loss_fn: Loss function to maximize.
        epsilon: L-infinity perturbation budget.
        steps: Number of PGD steps.
        step_size: Step size (defaults to epsilon/steps).

    Returns:
        Adversarially perturbed graph data.
    """
    if step_size is None:
        step_size = epsilon / steps

    data = data.clone()
    model.eval()

    if isinstance(data, HeteroData):
        for node_type in data.node_types:
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                data[node_type].x.requires_grad_(True)
    else:
        if data.x is not None:
            data.x.requires_grad_(True)

    for _ in range(steps):
        model.zero_grad()
        out = model(data)
        loss = loss_fn(out, data)
        loss.backward()

        if isinstance(data, HeteroData):
            for node_type in data.node_types:
                if hasattr(data[node_type], "x") and data[node_type].x.grad is not None:
                    grad = data[node_type].x.grad.sign()
                    data[node_type].x = data[node_type].x + step_size * grad
                    data[node_type].x = torch.clamp(
                        data[node_type].x,
                        min=data[node_type].x - epsilon,
                        max=data[node_type].x + epsilon,
                    )
                    data[node_type].x.grad.zero_()
        else:
            if data.x is not None and data.x.grad is not None:
                grad = data.x.grad.sign()
                data.x = data.x + step_size * grad
                data.x = torch.clamp(data.x, min=data.x - epsilon, max=data.x + epsilon)
                data.x.grad.zero_()

    if isinstance(data, HeteroData):
        for node_type in data.node_types:
            if hasattr(data[node_type], "x"):
                data[node_type].x = data[node_type].x.detach()
    else:
        if data.x is not None:
            data.x = data.x.detach()

    return data


def edge_dropout_attack(
    data: Data | HeteroData,
    dropout_rate: float = 0.2,
) -> Data | HeteroData:
    """Random edge dropout attack for structural robustness testing.

    Args:
        data: Input graph data.
        dropout_rate: Fraction of edges to drop.

    Returns:
        Graph with randomly dropped edges.
    """
    return GraphAugmenter(edge_dropout=dropout_rate).augment(data)