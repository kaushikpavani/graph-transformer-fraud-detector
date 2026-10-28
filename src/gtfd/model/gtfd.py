"""Main GTFD model composition and downstream heads."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from ..config import ModelConfig, GraphConfig, GraphEncoderConfig, TemporalEncoderConfig, FusionConfig
from .graph_encoder import GraphAttentionEncoder, GraphEncoderWrapper
from .temporal_encoder import GatedTemporalEncoder
from .fusion import CrossModalFusion


class GTFDModel(nn.Module):
    """Graph-Transformer Fraud Detection model.

    Combines graph structural encoder, temporal sequence encoder,
    and cross-modal fusion for fraud detection with conformal risk control.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Graph encoder
        self.graph_encoder = GraphEncoderWrapper(
            config.graph_encoder,
            in_channels=config.hidden_dim,  # Will be set after first forward
            out_channels=config.hidden_dim,
            node_types=config.graph.node_types,
        )

        # Temporal encoder
        self.temporal_encoder = GatedTemporalEncoder(config.temporal_encoder)

        # Cross-modal fusion
        self.fusion = CrossModalFusion(config.fusion)

        # Edge prediction heads (for pretraining)
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.edge_attr_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.graph_encoder.edge_dim),
        )

        # Downstream heads (added later)
        self.fraud_head: Optional[nn.Module] = None
        self.anomaly_head: Optional[nn.Module] = None
        self.type_head: Optional[nn.Module] = None

        # Initialize lazy modules
        self._initialized = False

    def _lazy_init(self, in_channels: int):
        """Lazy initialization based on input feature dimension."""
        if self._initialized:
            return
        self.graph_encoder = GraphEncoderWrapper(
            self.config.graph_encoder,
            in_channels=in_channels,
            out_channels=self.config.hidden_dim,
            node_types=self.config.graph.node_types,
        )
        self._initialized = True

    def encode(self, data: Data | HeteroData) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Encode graph to fused embeddings (graph + temporal).

        Args:
            data: Input graph data.

        Returns:
            Fused node embeddings.
        """
        # Lazy init
        if not self._initialized:
            in_channels = data.x.shape[1] if hasattr(data, "x") and data.x is not None else self.config.hidden_dim
            self._lazy_init(in_channels)

        # Graph embeddings
        h_graph = self.graph_encoder(data)

        # Temporal embeddings (would need sequences per node)
        # For now, return graph embeddings as placeholder
        # Full implementation requires temporal sequences per node
        h_temporal = h_graph  # Placeholder

        # Fusion
        h_fused = self.fusion(h_graph, h_temporal)
        return h_fused

    def forward(self, data: Data | HeteroData) -> torch.Tensor:
        """Forward pass for downstream tasks.

        Args:
            data: Input graph data.

        Returns:
            Logits for downstream task.
        """
        h_fused = self.encode(data)

        if self.fraud_head is not None:
            if isinstance(h_fused, dict):
                # Heterogeneous: use account nodes
                h = h_fused.get("account", next(iter(h_fused.values())))
            else:
                h = h_fused
            return self.fraud_head(h)

        return h_fused

    def predict_edges(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge existence from node embeddings.

        Args:
            h_src: Source node embeddings.
            h_dst: Destination node embeddings.
            edge_index: Edge indices [2, num_edges].

        Returns:
            Edge existence logits [num_edges].
        """
        src_emb = h_src[edge_index[0]]
        dst_emb = h_dst[edge_index[1]]
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.edge_predictor(combined).squeeze(-1)

    def predict_edge_attrs(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge attributes."""
        src_emb = h_src[edge_index[0]]
        dst_emb = h_dst[edge_index[1]]
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.edge_attr_predictor(combined)

    def add_fraud_head(self, num_classes: int = 2, hidden_dim: Optional[int] = None):
        """Add fraud classification head."""
        h_dim = hidden_dim or self.config.hidden_dim
        self.fraud_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h_dim, num_classes),
        )

    def add_anomaly_head(self, hidden_dim: Optional[int] = None):
        """Add anomaly scoring head."""
        h_dim = hidden_dim or self.config.hidden_dim
        self.anomaly_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h_dim, 1),
        )

    def add_type_head(self, num_types: int, hidden_dim: Optional[int] = None):
        """Add fraud type classification head."""
        h_dim = hidden_dim or self.config.hidden_dim
        self.type_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h_dim, num_types),
        )

    def get_embeddings(self, data: Data | HeteroData) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Get fused embeddings for downstream use."""
        return self.encode(data)


class FraudClassificationHead(nn.Module):
    """Fraud classification head."""

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AnomalyScoringHead(nn.Module):
    """Anomaly scoring head (regression)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class FraudTypeClassificationHead(nn.Module):
    """Fraud type classification head."""

    def __init__(self, hidden_dim: int, num_types: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)