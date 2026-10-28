"""Evaluation package for GTFD."""

from __future__ import annotations

from .metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ring_recall,
    compute_coverage,
    compute_avg_set_size,
)
from .adversarial import AdversarialEvaluator, pgd_attack, edge_dropout_attack
from .visualization import (
    plot_calibration_curve,
    plot_attention_heatmap,
    plot_embedding_tsne,
    plot_graph_substructure,
    plot_adversarial_robustness,
)

__all__ = [
    "compute_auroc",
    "compute_auprc",
    "compute_f1",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_ring_recall",
    "compute_coverage",
    "compute_avg_set_size",
    "AdversarialEvaluator",
    "pgd_attack",
    "edge_dropout_attack",
    "plot_calibration_curve",
    "plot_attention_heatmap",
    "plot_embedding_tsne",
    "plot_graph_substructure",
    "plot_adversarial_robustness",
]