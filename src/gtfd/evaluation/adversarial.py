"""Adversarial evaluation for robustness testing."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..graph.augment import pgd_attack, edge_dropout_attack


class AdversarialEvaluator:
    """Evaluates model robustness under adversarial attacks."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilons: list[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
        num_samples: int = 100,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            device: Device for computation.
            epsilons: List of perturbation magnitudes.
            num_samples: Number of samples per epsilon.
        """
        self.model = model.to(device)
        self.device = device
        self.epsilons = epsilons
        self.num_samples = num_samples

    def evaluate_feature_attack(
        self,
        data,
        loss_fn: callable,
        metric_fn: callable,
    ) -> dict[float, dict[str, float]]:
        """Evaluate under PGD feature attacks.

        Args:
            data: Test graph data.
            loss_fn: Loss function for attack.
            metric_fn: Metric function (e.g., accuracy, AUROC).

        Returns:
            Dictionary mapping epsilon to metrics.
        """
        self.model.eval()
        results = {}

        for eps in self.epsilons:
            if eps == 0.0:
                # Clean evaluation
                with torch.no_grad():
                    out = self.model(data)
                    metrics = metric_fn(out, data)
                results[eps] = metrics
                continue

            # Adversarial attack
            data_adv = pgd_attack(
                data, self.model, loss_fn,
                epsilon=eps, steps=5, step_size=eps / 5
            )

            with torch.no_grad():
                out = self.model(data_adv)
                metrics = metric_fn(out, data_adv)
            results[eps] = metrics

        return results

    def evaluate_structure_attack(
        self,
        data,
        metric_fn: callable,
        dropout_rates: Optional[list[float]] = None,
    ) -> dict[float, dict[str, float]]:
        """Evaluate under edge dropout attacks.

        Args:
            data: Test graph data.
            metric_fn: Metric function.
            dropout_rates: Edge dropout rates to test.

        Returns:
            Dictionary mapping dropout rate to metrics.
        """
        if dropout_rates is None:
            dropout_rates = self.epsilons

        self.model.eval()
        results = {}

        for rate in dropout_rates:
            if rate == 0.0:
                with torch.no_grad():
                    out = self.model(data)
                    metrics = metric_fn(out, data)
                results[rate] = metrics
                continue

            data_adv = edge_dropout_attack(data, dropout_rate=rate)

            with torch.no_grad():
                out = self.model(data_adv)
                metrics = metric_fn(out, data_adv)
            results[rate] = metrics

        return results


def compute_robustness_auc(
    epsilons: list[float],
    metrics: dict[float, dict[str, float]],
    metric_name: str = "accuracy",
) -> float:
    """Compute Area Under Robustness Curve (AURC).

    Integrates metric performance over epsilon range.
    """
    values = [metrics[eps].get(metric_name, 0.0) for eps in epsilons]
    # Trapezoidal integration
    auc = 0.0
    for i in range(len(epsilons) - 1):
        auc += (values[i] + values[i + 1]) / 2 * (epsilons[i + 1] - epsilons[i])
    return auc / (epsilons[-1] - epsilons[0]) if epsilons[-1] > epsilons[0] else values[0]