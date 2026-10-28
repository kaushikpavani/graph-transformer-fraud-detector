"""Cross-modal fusion layer combining graph and temporal embeddings."""

from __future__ import annotations

from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from ..config import FusionConfig


class CrossAttentionFusion(nn.Module):
    """Cross-attention based fusion of graph and temporal embeddings.

    Computes cross-attention between graph and temporal views,
    then applies a gating mechanism to combine them.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.gate_type = config.gate_type
        self.cross_attention = config.cross_attention

        # Cross-attention: graph -> temporal and temporal -> graph
        if config.cross_attention:
            self.graph_to_temporal = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
            )
            self.temporal_to_graph = nn.MultiheadAttention(
                config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            self.norm2 = nn.LayerNorm(config.hidden_dim)

        # Gating network
        if config.gate_type != "none":
            gate_input_dim = config.hidden_dim * 4  # concat, diff, prod
            self.gate_net = nn.Sequential(
                nn.Linear(gate_input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Sigmoid() if config.gate_type == "sigmoid" else nn.Softmax(dim=-1),
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        h_graph: torch.Tensor,
        h_temporal: torch.Tensor,
        graph_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse graph and temporal embeddings.

        Args:
            h_graph: Graph embeddings [num_nodes, hidden_dim] or [batch, num_nodes, hidden_dim].
            h_temporal: Temporal embeddings [num_nodes, hidden_dim] or [batch, num_nodes, hidden_dim].
            graph_mask: Optional mask for graph nodes.
            temporal_mask: Optional mask for temporal nodes.

        Returns:
            Fused embeddings [num_nodes, hidden_dim] or [batch, num_nodes, hidden_dim].
        """
        # Handle both batched and unbatched inputs
        is_batched = h_graph.dim() == 3
        if not is_batched:
            h_graph = h_graph.unsqueeze(0)
            h_temporal = h_temporal.unsqueeze(0)
            if graph_mask is not None:
                graph_mask = graph_mask.unsqueeze(0)
            if temporal_mask is not None:
                temporal_mask = temporal_mask.unsqueeze(0)

        batch_size, num_nodes, _ = h_graph.shape

        # Cross-attention
        if self.cross_attention:
            # Graph attends to temporal
            h_graph_norm = self.norm1(h_graph)
            h_temporal_norm = self.norm1(h_temporal)

            graph_attended, _ = self.graph_to_temporal(
                h_graph_norm, h_temporal_norm, h_temporal_norm,
                key_padding_mask=temporal_mask,
                need_weights=False,
            )

            # Temporal attends to graph
            temporal_attended, _ = self.temporal_to_graph(
                h_temporal_norm, h_graph_norm, h_graph_norm,
                key_padding_mask=graph_mask,
                need_weights=False,
            )

            h_graph = h_graph + graph_attended
            h_temporal = h_temporal + temporal_attended

        # Gating
        if self.gate_type != "none":
            # Compute gate input: [h_graph, h_temporal, h_graph - h_temporal, h_graph * h_temporal]
            gate_input = torch.cat([
                h_graph,
                h_temporal,
                h_graph - h_temporal,
                h_graph * h_temporal,
            ], dim=-1)

            gate = self.gate_net(gate_input)  # [batch, num_nodes, hidden_dim]

            if self.gate_type == "sigmoid":
                # Sigmoid gate: alpha * h_graph + (1 - alpha) * h_temporal
                fused = gate * h_graph + (1 - gate) * h_temporal
            else:
                # Softmax gate: alpha * h_graph + beta * h_temporal
                # gate has shape [..., hidden_dim], treat as per-feature weights
                fused = gate * h_graph + (1 - gate) * h_temporal
        else:
            # Simple average
            fused = (h_graph + h_temporal) / 2

        # Output projection
        fused = self.output_proj(fused)

        if not is_batched:
            fused = fused.squeeze(0)

        return fused


class CrossModalFusion(nn.Module):
    """Main fusion module for GTFD.

    Combines graph structural embeddings and temporal sequence embeddings
    using cross-attention and adaptive gating.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.fusion = CrossAttentionFusion(config)

    def forward(
        self,
        graph_embeddings: Union[torch.Tensor, dict[str, torch.Tensor]],
        temporal_embeddings: Union[torch.Tensor, dict[str, torch.Tensor]],
        graph_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse graph and temporal embeddings.

        Args:
            graph_embeddings: Graph encoder output.
            temporal_embeddings: Temporal encoder output.
            graph_mask: Optional mask for graph nodes.
            temporal_mask: Optional mask for temporal nodes.

        Returns:
            Fused embeddings (same structure as input).
        """
        if isinstance(graph_embeddings, dict) and isinstance(temporal_embeddings, dict):
            # Heterogeneous case - fuse per node type
            fused = {}
            for node_type in graph_embeddings:
                if node_type in temporal_embeddings:
                    fused[node_type] = self.fusion(
                        graph_embeddings[node_type],
                        temporal_embeddings[node_type],
                        graph_mask=graph_mask.get(node_type) if graph_mask else None,
                        temporal_mask=temporal_mask.get(node_type) if temporal_mask else None,
                    )
                else:
                    fused[node_type] = graph_embeddings[node_type]
            return fused
        else:
            # Homogeneous case
            return self.fusion(
                graph_embeddings,
                temporal_embeddings,
                graph_mask=graph_mask,
                temporal_mask=temporal_mask,
            )


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion baseline."""

    def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        h_graph: torch.Tensor,
        h_temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate and project."""
        combined = torch.cat([h_graph, h_temporal], dim=-1)
        return self.proj(combined)