"""Tests for adversarial evaluation."""

import pytest
import torch
import torch.nn as nn

from gtfd.evaluation.adversarial import AdversarialEvaluator, compute_robustness_auc, pgd_attack
from gtfd.model.gtfd import GTFDModel
from gtfd.config import ModelConfig, GraphConfig, GraphEncoderConfig


class TestPGDAttack:
    """Tests for PGD attack."""

    @pytest.fixture
    def model(self):
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=16, num_layers=1, num_heads=2),
        )
        model = GTFDModel(config)
        model.add_fraud_head(num_classes=2)
        model.eval()
        return model

    def test_pgd_attack_homo(self, model):
        """Test PGD on homogeneous graph."""
        from torch_geometric.data import Data
        data = Data(
            x=torch.randn(10, 16),
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.randint(0, 2, (10,)).float(),
        )

        def loss_fn(out, d):
            return nn.functional.cross_entropy(out, d.y.long())

        data_adv = pgd_attack(data, model, loss_fn, epsilon=0.1, steps=3, step_size=0.03)

        # Features should be perturbed
        assert not torch.allclose(data.x, data_adv.x)
        # Perturbation should be bounded
        diff = (data_adv.x - data.x).abs().max().item()
        assert diff <= 0.1 + 1e-5


class TestAdversarialEvaluator:
    """Tests for AdversarialEvaluator."""

    @pytest.fixture
    def model(self):
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=16, num_layers=1, num_heads=2),
        )
        model = GTFDModel(config)
        model.add_fraud_head(num_classes=2)
        model.eval()
        return model

    @pytest.fixture
    def data(self):
        from torch_geometric.data import Data
        return Data(
            x=torch.randn(20, 16),
            edge_index=torch.randint(0, 20, (2, 40)),
            y=torch.randint(0, 2, (20,)).float(),
        )

    def test_evaluate_feature_attack(self, model, data):
        """Test feature attack evaluation."""
        evaluator = AdversarialEvaluator(model, torch.device("cpu"), epsilons=[0.0, 0.1, 0.2])

        def loss_fn(out, d):
            return nn.functional.cross_entropy(out, d.y.long())

        def metric_fn(out, d):
            probs = torch.softmax(out, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            return {"accuracy": (preds == d.y.long()).float().mean().item()}

        results = evaluator.evaluate_feature_attack(data, loss_fn, metric_fn)

        assert 0.0 in results
        assert 0.1 in results
        assert 0.2 in results
        for eps, metrics in results.items():
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_structure_attack(self, model, data):
        """Test structure attack evaluation."""
        evaluator = AdversarialEvaluator(model, torch.device("cpu"), epsilons=[0.0, 0.1, 0.2])

        def metric_fn(out, d):
            probs = torch.softmax(out, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            return {"accuracy": (preds == d.y.long()).float().mean().item()}

        results = evaluator.evaluate_structure_attack(data, metric_fn)

        assert 0.0 in results
        assert 0.1 in results
        assert 0.2 in results


class TestRobustnessAUC:
    """Tests for robustness AUC computation."""

    def test_compute_robustness_auc(self):
        """Test AURC computation."""
        epsilons = [0.0, 0.1, 0.2]
        metrics = {
            0.0: {"accuracy": 0.95},
            0.1: {"accuracy": 0.85},
            0.2: {"accuracy": 0.70},
        }

        auc = compute_robustness_auc(epsilons, metrics, "accuracy")
        # Trapezoidal: (0.95+0.85)/2 * 0.1 + (0.85+0.70)/2 * 0.1 = 0.09 + 0.0775 = 0.1675
        # Normalized: 0.1675 / 0.2 = 0.8375
        expected = ((0.95 + 0.85) / 2 * 0.1 + (0.85 + 0.70) / 2 * 0.1) / 0.2
        assert abs(auc - expected) < 1e-4

    def test_constant_accuracy(self):
        """Test AURC with constant accuracy."""
        epsilons = [0.0, 0.1, 0.2]
        metrics = {eps: {"accuracy": 0.9} for eps in epsilons}
        auc = compute_robustness_auc(epsilons, metrics, "accuracy")
        assert abs(auc - 0.9) < 1e-4