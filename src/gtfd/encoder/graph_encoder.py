"""Multi-head Graph Attention Network encoder for structural evidence."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, GATv2Conv

from ..config import GraphEncoderConfig


class GraphAttentionEncoder(nn.Module):
    """Multi-head Graph Attention Network encoder.

    Encodes structural evidence from the payment graph using
    multi-head attention with residual connections and optional
    edge attribute conditioning.

    Args:
        config: Graph encoder configuration.
        in_channels: Input feature dimension (set dynamically).
        out_channels: Output embedding dimension (defaults to hidden_dim).
    """

    def __init__(
        self,
        config: GraphEncoderConfig,
        in_channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.hidden_dim = config.hidden_dim
        self.out_channels = out_channels or config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.residual = config.residual
        self.concat_heads = config.concat_heads

        # Input projection
        self.input_proj = nn.Linear(in_channels, config.hidden_dim)

        # GAT layers
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            in_dim = config.hidden_dim if i > 0 else config.hidden_dim
            out_dim = config.hidden_dim

            if config.use_edge_attr:
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // config.num_heads if config.concat_heads else out_dim,
                    heads=config.num_heads,
                    concat=config.concat_heads,
                    dropout=config.attention_dropout,
                    add_self_loops=True,
                    edge_dim=config.edge_dim,
                    bias=True,
                )
            else:
                conv = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // config.num_heads if config.concat_heads else out_dim,
                    heads=config.num_heads,
                    concat=config.concat_heads,
                    dropout=config.attention_dropout,
                    add_self_loops=True,
                    bias=True,
                )
            self.layers.append(conv)

            # Layer norm and residual projection
            if config.residual and i > 0:
                self.layers.append(nn.LayerNorm(config.hidden_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, self.out_channels),
            nn.Dropout(config.dropout),
        )

        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through GAT encoder.

        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            edge_attr: Edge attributes [num_edges, edge_dim] (optional).
            batch: Batch assignment [num_nodes] (optional).

        Returns:
            Node embeddings [num_nodes, out_channels].
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout_layer(h)

        # GAT layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (GATConv, GATv2Conv)):
                h_residual = h
                if self.config.use_edge_attr and edge_attr is not None:
                    h = layer(h, edge_index, edge_attr)
                else:
                    h = layer(h, edge_index)

                if self.residual and h_residual.shape == h.shape:
                    h = h + h_residual

                h = F.relu(h)
                h = self.dropout_layer(h)
            else:
                # LayerNorm
                h = layer(h)

        # Output projection
        h = self.output_proj(h)
        return h

    def forward_hetero(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: Optional[dict[tuple, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for heterogeneous graphs.

        Note: This is a simplified version that processes each
        node type independently. For full hetero GAT, consider
        using torch_geometric.nn.HeteroConv.

        Args:
            x_dict: Node features per node type.
            edge_index_dict: Edge indices per edge type.
            edge_attr_dict: Edge attributes per edge type.

        Returns:
            Node embeddings per node type.
        """
        # For now, process each node type with its own encoder
        # In practice, would use HeteroConv with message passing
        out_dict = {}
        for node_type, x in x_dict.items():
            # Get edges incident to this node type
            relevant_edges = {}
            for (src, rel, dst), edge_index in edge_index_dict.items():
                if src == node_type or dst == node_type:
                    relevant_edges[(src, rel, dst)] = edge_index

            if not relevant_edges:
                # No edges, just project
                out_dict[node_type] = self.output_proj(F.relu(self.input_proj(x)))
                continue

            # Use first relevant edge type (simplified)
            first_edge = next(iter(relevant_edges.values()))
            first_attr = None
            if edge_attr_dict:
                first_edge_key = next(iter(relevant_edges.keys()))
                first_attr = edge_attr_dict.get(first_edge_key)

            out_dict[node_type] = self.forward(x, first_edge, first_attr)

        return out_dict


class GraphEncoderWrapper(nn.Module):
    """Wrapper that handles both homogeneous and heterogeneous graphs."""

    def __init__(
        self,
        config: GraphEncoderConfig,
        in_channels: int,
        out_channels: Optional[int] = None,
        node_types: Optional[list[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.node_types = node_types
        self.encoder = GraphAttentionEncoder(config, in_channels, out_channels)

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass."""
        if isinstance(data, HeteroData):
            x_dict = {nt: data[nt].x for nt in data.node_types if hasattr(data[nt], "x")}
            edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
            edge_attr_dict = {}
            for et in data.edge_types:
                if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                    edge_attr_dict[et] = data[et].edge_attr
            return self.encoder.forward_hetero(x_dict, edge_index_dict, edge_attr_dict)
        else:
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            return self.encoder.forward(data.x, data.edge_index, edge_attr, data.batch)