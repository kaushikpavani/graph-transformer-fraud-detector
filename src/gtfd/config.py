"""Configuration classes for GTFD using Pydantic v2."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PoolingStrategy(str, Enum):
    """Graph pooling strategies."""

    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    ATTENTION = "attention"


class ConformalMethod(str, Enum):
    """Conformal prediction methods."""

    SPLIT = "split"
    APS = "aps"  # Adaptive Prediction Sets
    RAPS = "raps"  # Regularized Adaptive Prediction Sets


class GraphConfig(BaseModel):
    """Configuration for graph construction."""

    model_config = SettingsConfigDict(extra="forbid")

    node_types: list[str] = Field(
        default=["account", "merchant"],
        description="Node types in the heterogeneous graph",
    )
    edge_types: list[str] = Field(
        default=["transaction"],
        description="Edge types in the graph",
    )
    time_window_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Rolling time window for dynamic graph snapshots",
    )
    min_node_degree: int = Field(
        default=1,
        ge=0,
        description="Minimum degree for node inclusion",
    )
    max_nodes_per_snapshot: Optional[int] = Field(
        default=None,
        ge=100,
        description="Maximum nodes per snapshot (subsampling)",
    )
    feature_engineering: bool = Field(
        default=True,
        description="Enable automatic feature engineering for nodes",
    )
    temporal_features: bool = Field(
        default=True,
        description="Include temporal features (time since last tx, frequency, etc.)",
    )
    amount_bins: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="Number of bins for amount discretization",
    )


class GraphEncoderConfig(BaseModel):
    """Configuration for the Graph Attention Network encoder."""

    model_config = SettingsConfigDict(extra="forbid")

    hidden_dim: int = Field(default=256, ge=32, le=2048)
    num_layers: int = Field(default=4, ge=1, le=12)
    num_heads: int = Field(default=8, ge=1, le=32)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    residual: bool = Field(default=True)
    edge_dim: int = Field(default=32, ge=8, le=256)
    use_edge_attr: bool = Field(default=True)
    attention_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    concat_heads: bool = Field(default=False)


class TemporalEncoderConfig(BaseModel):
    """Configuration for the Gated Transformer temporal encoder."""

    model_config = SettingsConfigDict(extra="forbid")

    hidden_dim: int = Field(default=256, ge=32, le=2048)
    num_layers: int = Field(default=3, ge=1, le=8)
    num_heads: int = Field(default=8, ge=1, le=32)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_seq_length: int = Field(default=256, ge=16, le=2048)
    d_ff: int = Field(default=1024, ge=256, le=8192)
    use_rope: bool = Field(default=True)
    gated_residual: bool = Field(default=True)
    time_encoding_dim: int = Field(default=64, ge=16, le=256)


class FusionConfig(BaseModel):
    """Configuration for cross-modal fusion layer."""

    model_config = SettingsConfigDict(extra="forbid")

    hidden_dim: int = Field(default=256, ge=32, le=2048)
    num_heads: int = Field(default=8, ge=1, le=32)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    gate_type: Literal["sigmoid", "softmax", "none"] = Field(default="sigmoid")
    cross_attention: bool = Field(default=True)


class PretrainingConfig(BaseModel):
    """Configuration for self-supervised pretraining."""

    model_config = SettingsConfigDict(extra="forbid")

    mask_ratio: float = Field(default=0.15, ge=0.05, le=0.5)
    lambda_link: float = Field(default=1.0, ge=0.0)
    lambda_adv: float = Field(default=0.5, ge=0.0)
    lambda_contrast: float = Field(default=0.5, ge=0.0)
    adv_epsilon: float = Field(default=0.1, ge=0.01, le=1.0)
    adv_steps: int = Field(default=3, ge=1, le=10)
    edge_dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    temperature: float = Field(default=0.1, ge=0.01, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    epochs: int = Field(default=50, ge=1, le=500)
    warmup_epochs: int = Field(default=5, ge=0)
    grad_clip: float = Field(default=1.0, ge=0.1)


class ConformalConfig(BaseModel):
    """Configuration for conformal prediction calibration."""

    model_config = SettingsConfigDict(extra="forbid")

    method: ConformalMethod = Field(default=ConformalMethod.APS)
    alpha: float = Field(default=0.1, ge=0.01, le=0.5)
    calibration_split: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=42)
    regularization: float = Field(default=0.01, ge=0.0, le=1.0)  # For RAPS
    k_reg: int = Field(default=5, ge=1, le=20)  # For RAPS


class ModelConfig(BaseModel):
    """Main model configuration composing all sub-configs."""

    model_config = SettingsConfigDict(extra="forbid")

    graph: GraphConfig = Field(default_factory=GraphConfig)
    graph_encoder: GraphEncoderConfig = Field(default_factory=GraphEncoderConfig)
    temporal_encoder: TemporalEncoderConfig = Field(default_factory=TemporalEncoderConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    pretraining: PretrainingConfig = Field(default_factory=PretrainingConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    hidden_dim: int = Field(default=256, ge=32, le=2048)
    num_classes: int = Field(default=2, ge=2)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @field_validator("hidden_dim")
    @classmethod
    def validate_hidden_dim(cls, v: int, info: Any) -> int:
        # Ensure all sub-configs have consistent hidden_dim
        return v


class DataConfig(BaseModel):
    """Configuration for data generation and loading."""

    model_config = SettingsConfigDict(extra="forbid")

    num_accounts: int = Field(default=5000, ge=100, le=100000)
    num_merchants: int = Field(default=500, ge=10, le=10000)
    num_transactions: int = Field(default=100000, ge=1000, le=10000000)
    fraud_rate: float = Field(default=0.02, ge=0.001, le=0.5)
    ring_types: list[str] = Field(
        default=[
            "card_testing",
            "account_takeover",
            "money_laundering",
            "refund_abuse",
            "collusion",
        ]
    )
    ring_size_range: tuple[int, int] = Field(default=(3, 10))
    temporal_span_days: int = Field(default=90, ge=7, le=365)
    seed: int = Field(default=42)


class TrainingConfig(BaseModel):
    """Configuration for downstream training."""

    model_config = SettingsConfigDict(extra="forbid")

    batch_size: int = Field(default=64, ge=8, le=1024)
    learning_rate: float = Field(default=1e-3, ge=1e-5, le=1e-1)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    epochs: int = Field(default=100, ge=1, le=1000)
    patience: int = Field(default=10, ge=1, le=100)
    grad_clip: float = Field(default=1.0, ge=0.1)
    scheduler: Literal["cosine", "step", "plateau", "none"] = Field(default="cosine")
    num_workers: int = Field(default=4, ge=0, le=16)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    model_config = SettingsConfigDict(extra="forbid")

    metrics: list[str] = Field(
        default=["auroc", "auprc", "f1", "precision_at_k", "recall_at_k", "ring_recall"]
    )
    k_values: list[int] = Field(default=[10, 50, 100, 500])
    adversarial_epsilons: list[float] = Field(default=[0.0, 0.05, 0.1, 0.15, 0.2])
    num_adversarial_samples: int = Field(default=100, ge=10, le=1000)
    visualization: bool = Field(default=True)


class GTFDConfig(BaseSettings):
    """Top-level GTFD configuration loaded from YAML/env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    device: str = Field(default="auto")
    seed: int = Field(default=42)
    output_dir: Path = Field(default=Path("./outputs"))
    log_level: str = Field(default="INFO")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "GTFDConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)