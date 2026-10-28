"""Graph builder for constructing payment graphs from transaction data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

from ..config import GraphConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphSnapshot:
    """A single temporal snapshot of the payment graph."""

    data: Data | HeteroData
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    node_mapping: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def num_nodes(self) -> int:
        """Return total number of nodes."""
        if isinstance(self.data, HeteroData):
            return sum(self.data[node_type].num_nodes for node_type in self.data.node_types)
        return self.data.num_nodes

    def num_edges(self) -> int:
        """Return total number of edges."""
        if isinstance(self.data, HeteroData):
            return sum(self.data[edge_type].num_edges for edge_type in self.data.edge_types)
        return self.data.num_edges


class PaymentGraphBuilder:
    """Builds payment graphs from transaction DataFrames.

    Constructs heterogeneous graphs with accounts and merchants as nodes,
    transactions as edges with rich attributes. Supports temporal snapshots
    for dynamic graph analysis.

    Example:
        >>> builder = PaymentGraphBuilder(config)
        >>> snapshots = builder.build_snapshots(transactions_df)
        >>> for snap in snapshots:
        ...     print(f"Nodes: {snap.num_nodes()}, Edges: {snap.num_edges()}")
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        """Initialize the graph builder.

        Args:
            config: Graph construction configuration. Uses defaults if None.
        """
        self.config = config or GraphConfig()
        self._account_id_map: dict[str, int] = {}
        self._merchant_id_map: dict[str, int] = {}
        self._next_account_id = 0
        self._next_merchant_id = 0

    def reset_mappings(self) -> None:
        """Reset ID mappings for fresh graph construction."""
        self._account_id_map.clear()
        self._merchant_id_map.clear()
        self._next_account_id = 0
        self._next_merchant_id = 0

    def _get_account_id(self, account: str) -> int:
        """Get or create integer ID for account."""
        if account not in self._account_id_map:
            self._account_id_map[account] = self._next_account_id
            self._next_account_id += 1
        return self._account_id_map[account]

    def _get_merchant_id(self, merchant: str) -> int:
        """Get or create integer ID for merchant."""
        if merchant not in self._merchant_id_map:
            self._merchant_id_map[merchant] = self._next_merchant_id
            self._next_merchant_id += 1
        return self._merchant_id_map[merchant]

    def build_graph(
        self,
        transactions: pd.DataFrame,
        account_col: str = "account_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        tx_type_col: str = "tx_type",
        label_col: Optional[str] = "is_fraud",
    ) -> Data:
        """Build a single static payment graph from transactions.

        Args:
            transactions: DataFrame with transaction data.
            account_col: Column name for sender account ID.
            merchant_col: Column name for receiver merchant ID.
            amount_col: Column name for transaction amount.
            timestamp_col: Column name for transaction timestamp.
            tx_type_col: Column name for transaction type.
            label_col: Column name for fraud label (optional).

        Returns:
            PyG Data object with node features and edge attributes.
        """
        self.reset_mappings()

        # Map accounts and merchants to integer IDs
        src_ids = transactions[account_col].apply(self._get_account_id).values
        dst_ids = transactions[merchant_col].apply(self._get_merchant_id).values

        num_accounts = len(self._account_id_map)
        num_merchants = len(self._merchant_id_map)
        num_nodes = num_accounts + num_merchants

        # Build edge index (bipartite: accounts -> merchants)
        edge_index = torch.tensor(
            [src_ids, dst_ids + num_accounts], dtype=torch.long
        )

        # Edge attributes
        amounts = transactions[amount_col].values.astype(np.float32)
        timestamps = pd.to_datetime(transactions[timestamp_col]).values
        tx_types = transactions[tx_type_col].values if tx_type_col in transactions.columns else np.zeros(len(transactions), dtype=str)

        # Discretize amounts into bins
        amount_bins = np.linspace(
            amounts.min(), amounts.max(), self.config.amount_bins + 1
        )
        amount_binned = np.digitize(amounts, amount_bins) - 1
        amount_binned = np.clip(amount_binned, 0, self.config.amount_bins - 1)

        # One-hot encode transaction types
        unique_types = np.unique(tx_types)
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        type_encoded = np.array([type_to_idx.get(t, 0) for t in tx_types])

        edge_attr = torch.tensor(
            np.column_stack([
                amounts,
                amount_binned.astype(np.float32),
                type_encoded.astype(np.float32),
                np.arange(len(transactions), dtype=np.float32),  # temporal position
            ]),
            dtype=torch.float,
        )

        # Node features
        node_features = self._compute_node_features(
            transactions, src_ids, dst_ids, num_accounts, num_merchants,
            amounts, timestamps, tx_types, label_col
        )

        # Labels
        y = None
        if label_col and label_col in transactions.columns:
            # Edge-level labels
            edge_labels = transactions[label_col].values.astype(np.float32)
            # Node-level labels (aggregate: max fraud label for incident edges)
            node_labels = np.zeros(num_nodes, dtype=np.float32)
            for i, (src, dst) in enumerate(zip(src_ids, dst_ids + num_accounts)):
                label = edge_labels[i]
                node_labels[src] = max(node_labels[src], label)
                node_labels[dst] = max(node_labels[dst], label)
            y = torch.tensor(node_labels, dtype=torch.float).unsqueeze(1)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
        )

        # Store metadata
        data.account_id_map = self._account_id_map
        data.merchant_id_map = self._merchant_id_map
        data.num_accounts = num_accounts
        data.num_merchants = num_merchants

        logger.info(f"Built graph: {num_nodes} nodes ({num_accounts} accounts, {num_merchants} merchants), {edge_index.shape[1]} edges")
        return data

    def _compute_node_features(
        self,
        transactions: pd.DataFrame,
        src_ids: np.ndarray,
        dst_ids: np.ndarray,
        num_accounts: int,
        num_merchants: int,
        amounts: np.ndarray,
        timestamps: np.ndarray,
        tx_types: np.ndarray,
        label_col: Optional[str],
    ) -> torch.Tensor:
        """Compute node features from transaction data."""
        num_nodes = num_accounts + num_merchants

        # Initialize feature matrix
        # Features: [degree, total_amount, avg_amount, tx_count, unique_merchants/accounts,
        #            time_since_first, time_since_last, fraud_ratio (if labels)]
        n_features = 8 if label_col and label_col in transactions.columns else 7
        features = np.zeros((num_nodes, n_features), dtype=np.float32)

        # Account features
        for acc_name, acc_idx in self._account_id_map.items():
            mask = src_ids == acc_idx
            if not mask.any():
                continue
            acc_amounts = amounts[mask]
            acc_times = timestamps[mask]
            features[acc_idx, 0] = mask.sum()  # out-degree
            features[acc_idx, 1] = acc_amounts.sum()  # total amount sent
            features[acc_idx, 2] = acc_amounts.mean()  # avg amount
            features[acc_idx, 3] = mask.sum()  # tx count
            features[acc_idx, 4] = len(np.unique(dst_ids[mask]))  # unique merchants
            features[acc_idx, 5] = (acc_times.max() - acc_times.min()).astype('timedelta64[s]').astype(float)  # time span
            features[acc_idx, 6] = (pd.Timestamp.now() - pd.Timestamp(acc_times.max())).total_seconds()  # time since last

            if n_features == 8 and label_col and label_col in transactions.columns:
                acc_labels = transactions.loc[mask, label_col].values
                features[acc_idx, 7] = acc_labels.mean()  # fraud ratio

        # Merchant features
        for mer_name, mer_idx in self._merchant_id_map.items():
            node_idx = mer_idx + num_accounts
            mask = dst_ids == mer_idx
            if not mask.any():
                continue
            mer_amounts = amounts[mask]
            mer_times = timestamps[mask]
            features[node_idx, 0] = mask.sum()  # in-degree
            features[node_idx, 1] = mer_amounts.sum()  # total amount received
            features[node_idx, 2] = mer_amounts.mean()  # avg amount
            features[node_idx, 3] = mask.sum()  # tx count
            features[node_idx, 4] = len(np.unique(src_ids[mask]))  # unique accounts
            features[node_idx, 5] = (mer_times.max() - mer_times.min()).astype('timedelta64[s]').astype(float)
            features[node_idx, 6] = (pd.Timestamp.now() - pd.Timestamp(mer_times.max())).total_seconds()

            if n_features == 8 and label_col and label_col in transactions.columns:
                mer_labels = transactions.loc[mask, label_col].values
                features[node_idx, 7] = mer_labels.mean()

        # Normalize features (log for amounts, standardize others)
        features[:, 1:3] = np.log1p(np.abs(features[:, 1:3])) * np.sign(features[:, 1:3])
        for i in range(n_features):
            col = features[:, i]
            if col.std() > 0:
                features[:, i] = (col - col.mean()) / (col.std() + 1e-8)

        return torch.tensor(features, dtype=torch.float)

    def build_snapshots(
        self,
        transactions: pd.DataFrame,
        timestamp_col: str = "timestamp",
        **kwargs,
    ) -> list[GraphSnapshot]:
        """Build temporal graph snapshots using rolling windows.

        Args:
            transactions: DataFrame with transaction data.
            timestamp_col: Column name for timestamp.
            **kwargs: Additional arguments passed to build_graph.

        Returns:
            List of GraphSnapshot objects for each time window.
        """
        transactions = transactions.copy()
        transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
        transactions = transactions.sort_values(timestamp_col)

        min_time = transactions[timestamp_col].min()
        max_time = transactions[timestamp_col].max()

        snapshots = []
        window = pd.Timedelta(days=self.config.time_window_days)
        step = window / 2  # 50% overlap

        current_start = min_time
        while current_start < max_time:
            current_end = current_start + window
            mask = (transactions[timestamp_col] >= current_start) & (transactions[timestamp_col] < current_end)
            window_txns = transactions[mask]

            if len(window_txns) > 0:
                self.reset_mappings()
                data = self.build_graph(window_txns, timestamp_col=timestamp_col, **kwargs)
                snapshot = GraphSnapshot(
                    data=data,
                    timestamp_start=current_start,
                    timestamp_end=current_end,
                    node_mapping={
                        "account": np.array(list(self._account_id_map.keys())),
                        "merchant": np.array(list(self._merchant_id_map.keys())),
                    },
                    metadata={"num_transactions": len(window_txns)},
                )
                snapshots.append(snapshot)

            current_start += step

        logger.info(f"Built {len(snapshots)} temporal snapshots")
        return snapshots

    def build_hetero_graph(
        self,
        transactions: pd.DataFrame,
        account_col: str = "account_id",
        merchant_col: str = "merchant_id",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp",
        tx_type_col: str = "tx_type",
        label_col: Optional[str] = "is_fraud",
    ) -> HeteroData:
        """Build a heterogeneous graph with separate node types.

        Args:
            transactions: DataFrame with transaction data.
            account_col: Column name for sender account ID.
            merchant_col: Column name for receiver merchant ID.
            amount_col: Column name for transaction amount.
            timestamp_col: Column name for transaction timestamp.
            tx_type_col: Column name for transaction type.
            label_col: Column name for fraud label (optional).

        Returns:
            PyG HeteroData object.
        """
        self.reset_mappings()

        # Map IDs
        src_ids = transactions[account_col].apply(self._get_account_id).values
        dst_ids = transactions[merchant_col].apply(self._get_merchant_id).values

        num_accounts = len(self._account_id_map)
        num_merchants = len(self._merchant_id_map)

        hetero_data = HeteroData()

        # Account node features
        acc_features = self._compute_node_features(
            transactions, src_ids, dst_ids, num_accounts, num_merchants,
            transactions[amount_col].values.astype(np.float32),
            pd.to_datetime(transactions[timestamp_col]).values,
            transactions[tx_type_col].values if tx_type_col in transactions.columns else np.zeros(len(transactions), dtype=str),
            label_col,
        )[:num_accounts]
        hetero_data["account"].x = acc_features

        # Merchant node features
        mer_features = self._compute_node_features(
            transactions, src_ids, dst_ids, num_accounts, num_merchants,
            transactions[amount_col].values.astype(np.float32),
            pd.to_datetime(transactions[timestamp_col]).values,
            transactions[tx_type_col].values if tx_type_col in transactions.columns else np.zeros(len(transactions), dtype=str),
            label_col,
        )[num_accounts:]
        hetero_data["merchant"].x = mer_features

        # Edge: account -> transaction -> merchant
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
        hetero_data["account", "transacts", "merchant"].edge_index = edge_index

        # Edge attributes
        amounts = transactions[amount_col].values.astype(np.float32)
        amount_bins = np.linspace(amounts.min(), amounts.max(), self.config.amount_bins + 1)
        amount_binned = np.digitize(amounts, amount_bins) - 1
        amount_binned = np.clip(amount_binned, 0, self.config.amount_bins - 1)

        tx_types = transactions[tx_type_col].values if tx_type_col in transactions.columns else np.zeros(len(transactions), dtype=str)
        unique_types = np.unique(tx_types)
        type_to_idx = {t: i for i, t in enumerate(unique_types)}
        type_encoded = np.array([type_to_idx.get(t, 0) for t in tx_types])

        edge_attr = torch.tensor(
            np.column_stack([amounts, amount_binned.astype(np.float32), type_encoded.astype(np.float32)]),
            dtype=torch.float,
        )
        hetero_data["account", "transacts", "merchant"].edge_attr = edge_attr

        # Labels
        if label_col and label_col in transactions.columns:
            edge_labels = transactions[label_col].values.astype(np.float32)
            hetero_data["account", "transacts", "merchant"].y = torch.tensor(edge_labels, dtype=torch.float)

            # Node labels
            node_labels_acc = np.zeros(num_accounts, dtype=np.float32)
            node_labels_mer = np.zeros(num_merchants, dtype=np.float32)
            for i, (src, dst) in enumerate(zip(src_ids, dst_ids)):
                label = edge_labels[i]
                node_labels_acc[src] = max(node_labels_acc[src], label)
                node_labels_mer[dst] = max(node_labels_mer[dst], label)
            hetero_data["account"].y = torch.tensor(node_labels_acc, dtype=torch.float).unsqueeze(1)
            hetero_data["merchant"].y = torch.tensor(node_labels_mer, dtype=torch.float).unsqueeze(1)

        logger.info(f"Built hetero graph: {num_accounts} accounts, {num_merchants} merchants, {edge_index.shape[1]} edges")
        return hetero_data