"""Visualization utilities for GTFD."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def plot_calibration_curve(
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Calibration Curve",
) -> plt.Figure:
    """Plot reliability diagram (calibration curve)."""
    from ..evaluation.metrics import compute_calibration_curve

    y_true = _to_numpy(y_true)
    y_prob = _to_numpy(y_prob)

    bin_pred, bin_true = compute_calibration_curve(y_true, y_prob, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_pred, bin_true, "o-", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    node_labels: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Attention Weights",
) -> plt.Figure:
    """Plot attention weight heatmap."""
    attn = _to_numpy(attention_weights)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)

    if node_labels:
        ax.set_xticks(range(len(node_labels)))
        ax.set_yticks(range(len(node_labels)))
        ax.set_xticklabels(node_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(node_labels, fontsize=6)

    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_embedding_tsne(
    embeddings: torch.Tensor | np.ndarray,
    labels: Optional[torch.Tensor | np.ndarray] = None,
    perplexity: int = 30,
    save_path: Optional[Path] = None,
    title: str = "t-SNE Embedding Visualization",
) -> plt.Figure:
    """Plot t-SNE visualization of embeddings."""
    if not SKLEARN_AVAILABLE:
        warnings.warn("sklearn not available, cannot compute t-SNE")
        return None

    emb = _to_numpy(embeddings)
    if emb.shape[0] > 5000:
        # Subsample for speed
        idx = np.random.choice(emb.shape[0], 5000, replace=False)
        emb = emb[idx]
        if labels is not None:
            labels = _to_numpy(labels)[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        labels = _to_numpy(labels)
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=10)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_embedding_umap(
    embeddings: torch.Tensor | np.ndarray,
    labels: Optional[torch.Tensor | np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "UMAP Embedding Visualization",
) -> plt.Figure:
    """Plot UMAP visualization of embeddings."""
    try:
        import umap
    except ImportError:
        warnings.warn("umap-learn not available")
        return None

    emb = _to_numpy(embeddings)
    if emb.shape[0] > 10000:
        idx = np.random.choice(emb.shape[0], 10000, replace=False)
        emb = emb[idx]
        if labels is not None:
            labels = _to_numpy(labels)[idx]

    reducer = umap.UMAP(random_state=42)
    emb_2d = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        labels = _to_numpy(labels)
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.6, s=10)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_graph_substructure(
    edge_index: torch.Tensor,
    node_features: Optional[torch.Tensor] = None,
    node_labels: Optional[torch.Tensor] = None,
    highlight_nodes: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None,
    title: str = "Graph Substructure",
) -> plt.Figure:
    """Plot graph substructure using networkx."""
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("networkx not available")
        return None

    G = nx.Graph()
    edge_index = _to_numpy(edge_index)
    G.add_edges_from(edge_index.T)

    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)

    # Draw nodes
    if node_labels is not None:
        labels = _to_numpy(node_labels)
        colors = plt.cm.tab10(labels / (labels.max() + 1))
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=50, ax=ax)

    # Highlight nodes
    if highlight_nodes is not None:
        highlight = _to_numpy(highlight_nodes)
        nx.draw_networkx_nodes(G, pos, nodelist=np.where(highlight)[0],
                              node_color="red", node_size=100, ax=ax)

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_adversarial_robustness(
    epsilons: list[float],
    metrics: dict[float, dict[str, float]],
    metric_names: list[str] = ["accuracy", "auroc", "f1"],
    save_path: Optional[Path] = None,
    title: str = "Adversarial Robustness",
) -> plt.Figure:
    """Plot adversarial robustness curves."""
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))

    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        values = [metrics[eps].get(metric_name, 0.0) for eps in epsilons]
        ax.plot(epsilons, values, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Perturbation Magnitude (ε)")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{metric_name.upper()} vs ε")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_loss_curves(
    history: list[dict[str, float]],
    save_path: Optional[Path] = None,
    title: str = "Training Loss Curves",
) -> plt.Figure:
    """Plot training loss curves."""
    if not history:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    for key in history[0].keys():
        values = [h.get(key, 0.0) for h in history]
        ax.plot(values, label=key, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)