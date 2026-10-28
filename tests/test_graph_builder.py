"""Tests for graph builder."""

import pytest
import pandas as pd
import torch
from torch_geometric.data import Data

from gtfd.graph.builder import PaymentGraphBuilder, GraphSnapshot
from gtfd.config import GraphConfig


class TestPaymentGraphBuilder:
    """Tests for PaymentGraphBuilder."""

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction DataFrame."""
        return pd.DataFrame({
            "account_id": ["ACC_001", "ACC_002", "ACC_001", "ACC_003"],
            "merchant_id": ["MER_001", "MER_001", "MER_002", "MER_001"],
            "amount": [100.0, 50.0, 200.0, 75.0],
            "timestamp": pd.to_datetime([
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 12:00:00",
                "2024-01-01 13:00:00",
            ]),
            "tx_type": ["purchase", "purchase", "withdrawal", "payment"],
            "is_fraud": [0, 0, 1, 0],
        })

    @pytest.fixture
    def builder(self):
        """Create graph builder with default config."""
        return PaymentGraphBuilder(GraphConfig())

    def test_build_graph_basic(self, builder, sample_transactions):
        """Test basic graph construction."""
        data = builder.build_graph(sample_transactions)

        assert isinstance(data, Data)
        assert data.num_nodes == 5  # 3 accounts + 2 merchants
        assert data.num_edges == 4
        assert data.x is not None
        assert data.x.shape[0] == 5
        assert data.edge_index.shape == (2, 4)
        assert data.edge_attr is not None
        assert data.edge_attr.shape[0] == 4
        assert data.y is not None  # Fraud labels

    def test_build_graph_node_counts(self, builder, sample_transactions):
        """Test correct node counts."""
        data = builder.build_graph(sample_transactions)
        assert data.num_accounts == 3
        assert data.num_merchants == 2

    def test_build_graph_features(self, builder, sample_transactions):
        """Test node features are computed."""
        data = builder.build_graph(sample_transactions)
        # Features: degree, total_amount, avg_amount, tx_count, unique_counterparts, time_span, time_since_last, fraud_ratio
        assert data.x.shape[1] == 8  # 7 base + 1 fraud_ratio

    def test_build_graph_labels(self, builder, sample_transactions):
        """Test fraud labels are propagated to nodes."""
        data = builder.build_graph(sample_transactions)
        # ACC_001 has one fraud txn (index 2), ACC_002 and ACC_003 have none
        # MER_001 has no fraud, MER_002 has one fraud
        assert data.y.shape == (5, 1)
        # ACC_001 (index 0) should have fraud label
        assert data.y[0, 0] == 1.0
        # MER_002 (index 4) should have fraud label
        assert data.y[4, 0] == 1.0

    def test_build_snapshots(self, builder):
        """Test temporal snapshot creation."""
        # Create transactions over multiple days
        dates = pd.date_range("2024-01-01", periods=20, freq="12h")
        txns = pd.DataFrame({
            "account_id": ["ACC_001"] * 20,
            "merchant_id": ["MER_001"] * 20,
            "amount": [100.0] * 20,
            "timestamp": dates,
            "tx_type": ["purchase"] * 20,
            "is_fraud": [0] * 20,
        })

        snapshots = builder.build_snapshots(txns, time_window_days=1)
        assert len(snapshots) > 0
        for snap in snapshots:
            assert isinstance(snap, GraphSnapshot)
            assert snap.data.num_nodes > 0

    def test_build_hetero_graph(self, builder, sample_transactions):
        """Test heterogeneous graph construction."""
        hetero_data = builder.build_hetero_graph(sample_transactions)

        assert "account" in hetero_data.node_types
        assert "merchant" in hetero_data.node_types
        assert ("account", "transacts", "merchant") in hetero_data.edge_types
        assert hetero_data["account"].x is not None
        assert hetero_data["merchant"].x is not None
        assert hetero_data["account", "transacts", "merchant"].edge_index is not None

    def test_reset_mappings(self, builder, sample_transactions):
        """Test ID mapping reset."""
        builder.build_graph(sample_transactions)
        assert len(builder._account_id_map) == 3
        builder.reset_mappings()
        assert len(builder._account_id_map) == 0
        assert len(builder._merchant_id_map) == 0


class TestGraphSnapshot:
    """Tests for GraphSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation and properties."""
        data = Data(x=torch.randn(5, 8), edge_index=torch.randint(0, 5, (2, 10)))
        snap = GraphSnapshot(
            data=data,
            timestamp_start=pd.Timestamp("2024-01-01"),
            timestamp_end=pd.Timestamp("2024-01-02"),
        )
        assert snap.num_nodes() == 5
        assert snap.num_edges() == 10