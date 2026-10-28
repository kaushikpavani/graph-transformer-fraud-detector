"""Model package for GTFD."""

from __future__ import annotations

from .gtfd import GTFDModel
from .heads import FraudClassificationHead, AnomalyScoringHead, FraudTypeClassificationHead

__all__ = [
    "GTFDModel",
    "FraudClassificationHead",
    "AnomalyScoringHead",
    "FraudTypeClassificationHead",
]