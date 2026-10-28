"""GTFD: Graph-Transformer Fraud Detection with Conformal Risk Control.

A production-ready framework for detecting coordinated fraud rings in corporate
transaction networks using graph-transformer fusion, self-supervised pretraining,
and conformal prediction for calibrated anomaly scores.

References:
- GTFD: Graph-Transformer Fraud Detection with Self-Supervised Pretraining
  and Conformal Risk Control (arXiv:2609.14234, JP Morgan, Sep 2026)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "GTFD Contributors"
__license__ = "MIT"

from .config import (
    GTFDConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    GraphConfig,
    GraphEncoderConfig,
    TemporalEncoderConfig,
    FusionConfig,
    PretrainingConfig,
    ConformalConfig,
    ConformalMethod,
    PoolingStrategy,
)

__all__ = [
    "GTFDConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "GraphConfig",
    "GraphEncoderConfig",
    "TemporalEncoderConfig",
    "FusionConfig",
    "PretrainingConfig",
    "ConformalConfig",
    "ConformalMethod",
    "PoolingStrategy",
    "__version__",
]