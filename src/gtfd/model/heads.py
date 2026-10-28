"""Downstream heads for GTFD (moved to model/gtfd.py for simplicity)."""

# This module is kept for API compatibility
# All head classes are defined in gtfd/model/gtfd.py

from .gtfd import FraudClassificationHead, AnomalyScoringHead, FraudTypeClassificationHead

__all__ = [
    "FraudClassificationHead",
    "AnomalyScoringHead",
    "FraudTypeClassificationHead",
]