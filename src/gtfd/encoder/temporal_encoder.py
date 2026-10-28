"""Gated Transformer encoder for temporal transaction sequences."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TemporalEncoderConfig


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positional encoding for absolute timestamps."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Generate time encodings.

        Args:
            timestamps: Tensor of shape [batch, seq_len] or [seq_len]
                containing Unix timestamps or relative times in seconds.

        Returns:
            Time encodings [batch, seq_len, dim] or [seq_len, dim].
        """
        device = timestamps.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timestamps.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Embedding (RoPE) for Transformer."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, device: torch.device):
        if seq_len != self._seq_len_cached:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()
            self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Input tensor [..., seq_len, dim].
            seq_dim: Dimension representing sequence length.

        Returns:
            Tensor with RoPE applied.
        """
        seq_len = x.shape[seq_dim]
        self._update_cache(seq_len, x.device)

        cos = self._cos_cached[:seq_len]
        sin = self._sin_cached[:seq_len]

        # Reshape for broadcasting
        cos = cos.view(*([1] * (x.ndim - 2)), seq_len, self.dim)
        sin = sin.view(*([1] * (x.ndim - 2)), seq_len, self.dim)

        # Apply rotary embedding
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)


class TransactionEmbedding(nn.Module):
    """Embedding layer for transaction features."""

    def __init__(
        self,
        amount_bins: int,
        num_tx_types: int,
        num_mcc_codes: int,
        hidden_dim: int,
        time_encoding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.amount_embedding = nn.Embedding(amount_bins, hidden_dim // 4)
        self.type_embedding = nn.Embedding(num_tx_types, hidden_dim // 4)
        self.mcc_embedding = nn.Embedding(num_mcc_codes, hidden_dim // 4)
        self.time_encoding = SinusoidalTimeEncoding(time_encoding_dim)
        self.projection = nn.Linear(hidden_dim // 4 * 3 + time_encoding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        amounts: torch.Tensor,  # [batch, seq_len] - binned amounts
        tx_types: torch.Tensor,  # [batch, seq_len]
        mcc_codes: torch.Tensor,  # [batch, seq_len]
        timestamps: torch.Tensor,  # [batch, seq_len] - Unix timestamps
    ) -> torch.Tensor:
        """Embed transaction features.

        Args:
            amounts: Binned transaction amounts.
            tx_types: Transaction type indices.
            mcc_codes: MCC code indices.
            timestamps: Unix timestamps.

        Returns:
            Transaction embeddings [batch, seq_len, hidden_dim].
        """
        amt_emb = self.amount_embedding(amounts)
        type_emb = self.type_embedding(tx_types)
        mcc_emb = self.mcc_embedding(mcc_codes)
        time_emb = self.time_encoding(timestamps)

        combined = torch.cat([amt_emb, type_emb, mcc_emb, time_emb], dim=-1)
        out = self.projection(combined)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out


class GatedResidual(nn.Module):
    """Gated residual connection for Transformer layers."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply gated residual: gate * x + (1 - gate) * residual."""
        gate_input = torch.cat([x, residual], dim=-1)
        gate = self.gate(gate_input)
        return gate * x + (1 - gate) * residual


class TransformerBlock(nn.Module):
    """Transformer block with optional RoPE and gated residual."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = True,
        gated_residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_rope = use_rope
        self.gated_residual = gated_residual

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, hidden_dim),
            nn.Dropout(dropout),
        )

        if gated_residual:
            self.gate1 = GatedResidual(hidden_dim)
            self.gate2 = GatedResidual(hidden_dim)
        else:
            self.gate1 = None
            self.gate2 = None

        if use_rope:
            self.rope = RoPEPositionalEncoding(hidden_dim)
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        residual = x
        x = self.norm1(x)

        # Apply RoPE if enabled
        if self.rope is not None:
            x = self.rope(x)

        # Self-attention
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Gated residual or standard residual
        if self.gate1 is not None:
            x = self.gate1(attn_out, residual)
        else:
            x = attn_out + residual

        # FFN
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)

        if self.gate2 is not None:
            x = self.gate2(ffn_out, residual)
        else:
            x = ffn_out + residual

        return x


class GatedTemporalEncoder(nn.Module):
    """Gated Transformer encoder for per-node transaction sequences.

    Encodes temporal evidence from ordered transaction sequences
    using a Transformer with sinusoidal time encoding, optional RoPE,
    and gated residual connections.
    """

    def __init__(self, config: TemporalEncoderConfig, num_tx_types: int = 10, num_mcc_codes: int = 100):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_seq_length = config.max_seq_length

        # Transaction embedding
        self.tx_embedding = TransactionEmbedding(
            amount_bins=50,  # Default, can be configured
            num_tx_types=num_tx_types,
            num_mcc_codes=num_mcc_codes,
            hidden_dim=config.hidden_dim,
            time_encoding_dim=config.time_encoding_dim,
            dropout=config.dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_rope=config.use_rope,
                gated_residual=config.gated_residual,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        sequences: dict[str, torch.Tensor],
        lengths: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for batched sequences.

        Args:
            sequences: Dict with keys 'amounts', 'tx_types', 'mcc_codes', 'timestamps'.
                Each tensor of shape [batch, seq_len].
            lengths: Actual sequence lengths [batch].
            padding_mask: Boolean mask [batch, seq_len] (True = padding).

        Returns:
            Sequence embeddings [batch, hidden_dim] (mean pooled).
        """
        # Embed transactions
        x = self.tx_embedding(
            sequences["amounts"],
            sequences["tx_types"],
            sequences["mcc_codes"],
            sequences["timestamps"],
        )

        # Create causal mask
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        # Combine with padding mask
        if padding_mask is not None:
            attn_mask = causal_mask | padding_mask.unsqueeze(1)
        else:
            attn_mask = causal_mask

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask=attn_mask, key_padding_mask=padding_mask)

        # Mean pooling over sequence (excluding padding)
        if padding_mask is not None:
            # Mask out padding positions
            mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Output projection
        x = self.output_proj(x)
        return x

    def forward_single(
        self,
        amounts: torch.Tensor,
        tx_types: torch.Tensor,
        mcc_codes: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for a single sequence (no batching).

        Args:
            amounts: [seq_len] binned amounts.
            tx_types: [seq_len] transaction types.
            mcc_codes: [seq_len] MCC codes.
            timestamps: [seq_len] Unix timestamps.

        Returns:
            Sequence embedding [hidden_dim].
        """
        # Add batch dimension
        sequences = {
            "amounts": amounts.unsqueeze(0),
            "tx_types": tx_types.unsqueeze(0),
            "mcc_codes": mcc_codes.unsqueeze(0),
            "timestamps": timestamps.unsqueeze(0),
        }
        lengths = torch.tensor([amounts.shape[0]], device=amounts.device)
        return self.forward(sequences, lengths).squeeze(0)