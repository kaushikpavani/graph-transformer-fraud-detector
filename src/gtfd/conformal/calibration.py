"""Conformal prediction calibration for GTFD.

Implements split conformal prediction and Adaptive Prediction Sets (APS)
for finite-sample coverage guarantees on anomaly scores.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


class NonConformityScorer:
    """Base class for non-conformity score functions."""

    def __call__(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute non-conformity scores.

        Args:
            logits: Model output logits [batch, num_classes] or [batch].
            labels: True labels [batch] (optional, for some scores).

        Returns:
            Non-conformity scores [batch] (higher = more anomalous).
        """
        raise NotImplementedError


class FraudScore(NonConformityScorer):
    """Non-conformity score for fraud detection: 1 - p(fraud)."""

    def __call__(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score = 1 - softmax(fraud_class)."""
        if logits.dim() == 1:
            # Binary logits
            prob_fraud = torch.sigmoid(logits)
        else:
            # Multi-class logits, assume class 1 is fraud
            prob_fraud = F.softmax(logits, dim=-1)[:, 1]
        return 1.0 - prob_fraud


class AnomalyScore(NonConformityScorer):
    """Non-conformity score based on anomaly/reconstruction loss."""

    def __call__(self, logits: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score = anomaly score (already calibrated)."""
        # Assume logits are already anomaly scores (higher = more anomalous)
        return logits.squeeze(-1) if logits.dim() > 1 else logits


class SplitConformal:
    """Split Conformal Prediction for calibrated prediction sets.

    Provides finite-sample coverage guarantee: P(y in C(x)) >= 1 - alpha
    where C(x) is the prediction set.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        calibration_split: float = 0.2,
        random_state: int = 42,
        scorer: Optional[NonConformityScorer] = None,
    ):
        """Initialize split conformal predictor.

        Args:
            alpha: Miscoverage level (1 - alpha = coverage).
            calibration_split: Fraction of data for calibration.
            random_state: Random seed for split.
            scorer: Non-conformity scorer.
        """
        self.alpha = alpha
        self.calibration_split = calibration_split
        self.random_state = random_state
        self.scorer = scorer or FraudScore()
        self.quantile = None
        self.calibration_scores = None
        self.is_fitted = False

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Calibrate on held-out data.

        Args:
            logits: Model logits on calibration set.
            labels: True labels on calibration set.

        Returns:
            Calibration quantile.
        """
        n = len(logits)
        n_cal = int(n * self.calibration_split)

        # Random split
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]

        cal_logits = logits[cal_idx]
        cal_labels = labels[cal_idx]

        # Compute non-conformity scores on calibration set
        scores = self.scorer(cal_logits, cal_labels)

        # For split conformal, use (1 - alpha) quantile with correction
        # Quantile level: (n_cal + 1) * (1 - alpha) / n_cal
        quantile_level = min(1.0, (n_cal + 1) * (1 - self.alpha) / n_cal)
        self.quantile = torch.quantile(scores, quantile_level)
        self.calibration_scores = scores
        self.is_fitted = True

        return self.quantile.item()

    def predict_set(
        self,
        logits: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Generate prediction sets.

        For binary classification:
        - If score <= quantile: include class 1 (fraud) in set
        - If 1 - score <= quantile: include class 0 (legit) in set

        Args:
            logits: Test logits.
            return_scores: Whether to return non-conformity scores.

        Returns:
            Prediction sets as boolean tensor [batch, num_classes]
            (and optionally scores).
        """
        if not self.is_fitted:
            warnings.warn("Predictor not calibrated, using default quantile=0.5")
            self.quantile = torch.tensor(0.5)

        scores = self.scorer(logits)

        # Binary case: set contains class 1 if score <= quantile
        # Set contains class 0 if (1 - score) <= quantile
        pred_sets = torch.zeros(logits.shape[0], 2, dtype=torch.bool, device=logits.device)
        pred_sets[:, 1] = scores <= self.quantile  # fraud class
        pred_sets[:, 0] = (1 - scores) <= self.quantile  # legit class

        # Ensure non-empty sets (at least one class)
        empty_mask = ~pred_sets.any(dim=1)
        if empty_mask.any():
            # Add the class with lower non-conformity
            pred_sets[empty_mask, 1] = scores[empty_mask] <= (1 - scores[empty_mask])
            pred_sets[empty_mask, 0] = ~pred_sets[empty_mask, 1]

        if return_scores:
            return pred_sets, scores
        return pred_sets

    def predict_anomaly_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated anomaly scores.

        Returns p-value: proportion of calibration scores >= test score.
        """
        if not self.is_fitted:
            warnings.warn("Predictor not calibrated")
            return self.scorer(logits)

        scores = self.scorer(logits)
        # p-value = (1 + sum(cal_score >= test_score)) / (n_cal + 1)
        n_cal = len(self.calibration_scores)
        p_values = []
        for s in scores:
            p = (1 + (self.calibration_scores >= s).sum().item()) / (n_cal + 1)
            p_values.append(p)
        return torch.tensor(p_values, device=scores.device)

    def coverage(self, pred_sets: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute empirical coverage.

        Args:
            pred_sets: Prediction sets [batch, num_classes] (bool).
            labels: True labels [batch].

        Returns:
            Coverage rate.
        """
        correct = pred_sets[torch.arange(len(labels)), labels].float().mean()
        return correct.item()

    def avg_set_size(self, pred_sets: torch.Tensor) -> float:
        """Compute average prediction set size."""
        return pred_sets.float().sum(dim=1).mean().item()


class AdaptivePredictionSets:
    """Adaptive Prediction Sets (APS) for smaller, more informative sets.

    APS orders classes by softmax probability and includes classes
    until cumulative probability exceeds threshold.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        calibration_split: float = 0.2,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.calibration_split = calibration_split
        self.random_state = random_state
        self.quantile = None
        self.calibration_scores = None
        self.is_fitted = False

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Calibrate APS on held-out data."""
        n = len(logits)
        n_cal = int(n * self.calibration_split)

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]

        cal_logits = logits[cal_idx]
        cal_labels = labels[cal_idx]

        # APS scores: cumulative softmax probability of true class
        probs = F.softmax(cal_logits, dim=-1)
        # Sort classes by probability descending
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        # Cumulative sum
        cumsum = sorted_probs.cumsum(dim=-1)
        # Find rank of true label
        ranks = (sorted_indices == cal_labels.unsqueeze(-1)).nonzero()[:, 1]
        # APS score = cumsum at rank of true class
        scores = cumsum[torch.arange(n_cal), ranks]

        quantile_level = min(1.0, (n_cal + 1) * (1 - self.alpha) / n_cal)
        self.quantile = torch.quantile(scores, quantile_level)
        self.calibration_scores = scores
        self.is_fitted = True

        return self.quantile.item()

    def predict_set(
        self,
        logits: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Generate APS prediction sets."""
        if not self.is_fitted:
            warnings.warn("APS not calibrated")
            self.quantile = torch.tensor(1.0)

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)

        # Include classes until cumsum > quantile
        # This is vectorized: for each sample, find first index where cumsum > quantile
        mask = cumsum <= self.quantile
        # Always include at least the top class
        mask[:, 0] = True

        # Map back to original class ordering
        pred_sets = torch.zeros_like(logits, dtype=torch.bool)
        pred_sets.scatter_(1, sorted_indices, mask)

        if return_scores:
            # APS score for each sample (cumsum at threshold)
            ranks = (mask.cumsum(dim=-1) == 1).nonzero()[:, 1]
            scores = cumsum[torch.arange(len(logits)), ranks]
            return pred_sets, scores
        return pred_sets


class RegularizedAPS(AdaptivePredictionSets):
    """Regularized APS (RAPS) with regularization for better finite-sample performance."""

    def __init__(
        self,
        alpha: float = 0.1,
        calibration_split: float = 0.2,
        random_state: int = 42,
        regularization: float = 0.01,
        k_reg: int = 5,
    ):
        super().__init__(alpha, calibration_split, random_state)
        self.regularization = regularization
        self.k_reg = k_reg

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Calibrate RAPS with regularization."""
        n = len(logits)
        n_cal = int(n * self.calibration_split)

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n)
        cal_idx = indices[:n_cal]

        cal_logits = logits[cal_idx]
        cal_labels = labels[cal_idx]

        probs = F.softmax(cal_logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

        # Regularization: add penalty for including more classes
        k = torch.arange(1, sorted_probs.shape[1] + 1, device=logits.device, dtype=torch.float)
        reg = self.regularization * torch.maximum(k - self.k_reg, torch.tensor(0.0))

        cumsum = sorted_probs.cumsum(dim=-1) + reg.unsqueeze(0)

        ranks = (sorted_indices == cal_labels.unsqueeze(-1)).nonzero()[:, 1]
        scores = cumsum[torch.arange(n_cal), ranks]

        quantile_level = min(1.0, (n_cal + 1) * (1 - self.alpha) / n_cal)
        self.quantile = torch.quantile(scores, quantile_level)
        self.calibration_scores = scores
        self.is_fitted = True

        return self.quantile.item()

    def predict_set(
        self,
        logits: torch.Tensor,
        return_scores: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Generate RAPS prediction sets."""
        if not self.is_fitted:
            warnings.warn("RAPS not calibrated")
            self.quantile = torch.tensor(1.0)

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

        k = torch.arange(1, sorted_probs.shape[1] + 1, device=logits.device, dtype=torch.float)
        reg = self.regularization * torch.maximum(k - self.k_reg, torch.tensor(0.0))

        cumsum = sorted_probs.cumsum(dim=-1) + reg.unsqueeze(0)

        mask = cumsum <= self.quantile
        mask[:, 0] = True

        pred_sets = torch.zeros_like(logits, dtype=torch.bool)
        pred_sets.scatter_(1, sorted_indices, mask)

        if return_scores:
            ranks = (mask.cumsum(dim=-1) == 1).nonzero()[:, 1]
            scores = cumsum[torch.arange(len(logits)), ranks]
            return pred_sets, scores
        return pred_sets


def compute_conformal_metrics(
    pred_sets: torch.Tensor,
    labels: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Compute conformal prediction metrics.

    Args:
        pred_sets: Prediction sets [batch, num_classes] (bool).
        labels: True labels [batch].
        scores: Non-conformity scores [batch] (optional).

    Returns:
        Dictionary of metrics.
    """
    coverage = pred_sets[torch.arange(len(labels)), labels].float().mean().item()
    avg_size = pred_sets.float().sum(dim=1).mean().item()
    singleton_rate = (pred_sets.float().sum(dim=-1) == 1).float().mean().item()
    empty_rate = (pred_sets.float().sum(dim=-1) == 0).float().mean().item()

    metrics = {
        "coverage": coverage,
        "avg_set_size": avg_size,
        "singleton_rate": singleton_rate,
        "empty_rate": empty_rate,
    }

    if scores is not None:
        metrics["mean_score"] = scores.mean().item()
        metrics["std_score"] = scores.std().item()

    return metrics