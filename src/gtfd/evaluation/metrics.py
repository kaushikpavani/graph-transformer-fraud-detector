"""Evaluation metrics for fraud detection."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    auc,
)


def compute_auroc(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """Compute AUROC score."""
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    return float(roc_auc_score(y_true, y_score, multi_class=multi_class))


def compute_auprc(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
) -> float:
    """Compute AUPRC (Average Precision) score."""
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)
    return float(average_precision_score(y_true, y_score))


def compute_f1(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    average: str = "binary",
) -> float:
    """Compute F1 score."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(f1_score(y_true, y_pred, average=average))


def compute_precision_at_k(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
    k: int,
) -> float:
    """Compute Precision@K."""
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    # Get top-K indices by score
    top_k_idx = np.argsort(y_score)[-k:]
    if len(top_k_idx) == 0:
        return 0.0
    return float(y_true[top_k_idx].mean())


def compute_recall_at_k(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
    k: int,
) -> float:
    """Compute Recall@K."""
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0

    top_k_idx = np.argsort(y_score)[-k:]
    tp = y_true[top_k_idx].sum()
    return float(tp / n_positives)


def compute_ring_recall(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    ring_labels: torch.Tensor | np.ndarray,
) -> float:
    """Compute ring-level recall (fraction of fraud rings detected).

    A ring is detected if at least one of its transactions is flagged.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ring_labels = _to_numpy(ring_labels)

    unique_rings = np.unique(ring_labels[ring_labels >= 0])
    if len(unique_rings) == 0:
        return 1.0  # No rings = perfect

    detected = 0
    for ring_id in unique_rings:
        ring_mask = ring_labels == ring_id
        if y_pred[ring_mask].any():
            detected += 1

    return detected / len(unique_rings)


def compute_coverage(
    pred_sets: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> float:
    """Compute conformal prediction coverage."""
    pred_sets = _to_numpy(pred_sets)
    labels = _to_numpy(labels)

    # pred_sets is [batch, num_classes] bool
    # labels is [batch] int
    correct = pred_sets[np.arange(len(labels)), labels].mean()
    return float(correct)


def compute_avg_set_size(pred_sets: torch.Tensor | np.ndarray) -> float:
    """Compute average prediction set size."""
    pred_sets = _to_numpy(pred_sets)
    return float(pred_sets.sum(axis=1).mean())


def compute_calibration_curve(
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram)."""
    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_true = []
    bin_pred = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.any():
            bin_true.append(y_true[mask].mean())
            bin_pred.append(y_prob[mask].mean())

    return np.array(bin_pred), np.array(bin_true)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)