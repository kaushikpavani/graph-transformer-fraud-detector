"""Tests for encoders."""

import pytest
import torch
from torch_geometric.data import Data

from gtfd.encoder.graph_encoder import GraphAttentionEncoder, GraphEncoderWrapper
from gtfd.encoder.temporal_encoder import GatedTemporalEncoder, TransformerBlock
from gtfd.encoder.fusion import CrossModalFusion, CrossAttentionFusion
from gtfd.config import GraphEncoderConfig, TemporalEncoderConfig, FusionConfig


class TestGraphAttentionEncoder:
    """Tests for GraphAttentionEncoder."""

    @pytest.fixture
    def config(self):
        return GraphEncoderConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            edge_dim=16,
            use_edge_attr=True,
        )

    @pytest.fixture
    def encoder(self, config):
        return GraphAttentionEncoder(config, in_channels=16, out_channels=64)

    def test_forward_homo(self, encoder):
        """Test forward pass on homogeneous graph."""
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 16)

        out = encoder(x, edge_index, edge_attr)
        assert out.shape == (10, 64)

    def test_forward_no_edge_attr(self, encoder):
        """Test forward without edge attributes."""
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))

        out = encoder(x, edge_index, edge_attr=None)
        assert out.shape == (10, 64)

    def test_residual_connection(self, encoder):
        """Test residual connections work."""
        x = torch.randn(10, 16)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, 16)

        out = encoder(x, edge_index, edge_attr)
        # Output should be different from input projection
        assert not torch.allclose(out, encoder.input_proj(x))


class TestGraphEncoderWrapper:
    """Tests for GraphEncoderWrapper."""

    @pytest.fixture
    def wrapper(self):
        config = GraphEncoderConfig(hidden_dim=64, num_layers=2, num_heads=4)
        return GraphEncoderWrapper(config, in_channels=16, out_channels=64)

    def test_forward_homo(self, wrapper):
        """Test forward on homogeneous data."""
        data = Data(x=torch.randn(10, 16), edge_index=torch.randint(0, 10, (2, 20)))
        out = wrapper(data)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10, 64)

    def test_forward_hetero(self, wrapper):
        """Test forward on heterogeneous data."""
        from torch_geometric.data import HeteroData
        data = HeteroData()
        data["account"].x = torch.randn(5, 16)
        data["merchant"].x = torch.randn(3, 16)
        data["account", "transacts", "merchant"].edge_index = torch.randint(0, 5, (2, 10))

        out = wrapper(data)
        assert isinstance(out, dict)
        assert "account" in out
        assert "merchant" in out


class TestGatedTemporalEncoder:
    """Tests for GatedTemporalEncoder."""

    @pytest.fixture
    def config(self):
        return TemporalEncoderConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            max_seq_length=32,
            d_ff=256,
            use_rope=True,
            gated_residual=True,
        )

    @pytest.fixture
    def encoder(self, config):
        return GatedTemporalEncoder(config, num_tx_types=5, num_mcc_codes=10)

    def test_forward_batched(self, encoder):
        """Test forward pass with batched sequences."""
        batch_size = 4
        seq_len = 16
        sequences = {
            "amounts": torch.randint(0, 50, (batch_size, seq_len)),
            "tx_types": torch.randint(0, 5, (batch_size, seq_len)),
            "mcc_codes": torch.randint(0, 10, (batch_size, seq_len)),
            "timestamps": torch.randint(1000000000, 2000000000, (batch_size, seq_len)),
        }
        lengths = torch.tensor([seq_len] * batch_size)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        out = encoder(sequences, lengths, padding_mask)
        assert out.shape == (batch_size, 64)

    def test_forward_single(self, encoder):
        """Test forward pass for single sequence."""
        seq_len = 10
        out = encoder.forward_single(
            amounts=torch.randint(0, 50, (seq_len,)),
            tx_types=torch.randint(0, 5, (seq_len,)),
            mcc_codes=torch.randint(0, 10, (seq_len,)),
            timestamps=torch.randint(1000000000, 2000000000, (seq_len,)),
        )
        assert out.shape == (64,)


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_forward(self):
        """Test transformer block forward."""
        block = TransformerBlock(
            hidden_dim=64,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
            use_rope=True,
            gated_residual=True,
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_forward_with_mask(self):
        """Test with attention mask."""
        block = TransformerBlock(64, 4, 256, 0.1)
        x = torch.randn(2, 16, 64)
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        out = block(x, mask=mask)
        assert out.shape == (2, 16, 64)


class TestCrossModalFusion:
    """Tests for CrossModalFusion."""

    @pytest.fixture
    def fusion(self):
        config = FusionConfig(hidden_dim=64, num_heads=4, dropout=0.1, gate_type="sigmoid")
        return CrossModalFusion(config)

    def test_forward_homo(self, fusion):
        """Test fusion on homogeneous embeddings."""
        h_graph = torch.randn(10, 64)
        h_temporal = torch.randn(10, 64)

        out = fusion(h_graph, h_temporal)
        assert out.shape == (10, 64)

    def test_forward_batched(self, fusion):
        """Test fusion on batched embeddings."""
        h_graph = torch.randn(2, 10, 64)
        h_temporal = torch.randn(2, 10, 64)

        out = fusion(h_graph, h_temporal)
        assert out.shape == (2, 10, 64)

    def test_fusion_gate_types(self):
        """Test different gate types."""
        for gate_type in ["sigmoid", "softmax", "none"]:
            config = FusionConfig(hidden_dim=64, num_heads=4, gate_type=gate_type)
            fusion = CrossModalFusion(config)
            h_graph = torch.randn(10, 64)
            h_temporal = torch.randn(10, 64)
            out = fusion(h_graph, h_temporal)
            assert out.shape == (10, 64)

    def test_hetero_fusion(self):
        """Test fusion with heterogeneous inputs."""
        config = FusionConfig(hidden_dim=64, num_heads=4)
        fusion = CrossModalFusion(config)

        h_graph = {"account": torch.randn(5, 64), "merchant": torch.randn(3, 64)}
        h_temporal = {"account": torch.randn(5, 64), "merchant": torch.randn(3, 64)}

        out = fusion(h_graph, h_temporal)
        assert isinstance(out, dict)
        assert "account" in out
        assert "merchant" in out