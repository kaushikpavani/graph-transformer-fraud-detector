"""Tests for conformal prediction."""

import pytest
import torch

from gtfd.conformal.calibration import (
    SplitConformal,
    AdaptivePredictionSets,
    RegularizedAPS,
    FraudScore,
    compute_conformal_metrics,
)


class TestFraudScore:
    """Tests for FraudScore non-conformity scorer."""

    def test_binary_logits(self):
        """Test with binary logits."""
        scorer = FraudScore()
        logits = torch.tensor([0.0, 1.0, -1.0, 2.0])  # Various confidences
        scores = scorer(logits)

        # Score = 1 - sigmoid(logit)
        expected = 1.0 - torch.sigmoid(logits)
        assert torch.allclose(scores, expected)

    def test_multiclass_logits(self):
        """Test with multi-class logits."""
        scorer = FraudScore()
        logits = torch.tensor([
            [0.0, 1.0],   # Class 1 more likely
            [2.0, 0.0],   # Class 0 more likely
            [1.0, 1.0],   # Equal
        ])
        scores = scorer(logits)

        # Score = 1 - softmax[:, 1]
        probs = torch.softmax(logits, dim=-1)
        expected = 1.0 - probs[:, 1]
        assert torch.allclose(scores, expected)


class TestSplitConformal:
    """Tests for SplitConformal calibration."""

    @pytest.fixture
    def calibrator(self):
        return SplitConformal(alpha=0.1, calibration_split=0.5, random_state=42)

    def test_calibrate(self, calibrator):
        """Test calibration computes quantile."""
        # Create logits where class 1 is fraud
        logits = torch.randn(100, 2)
        logits[:, 1] += 2.0  # Make fraud class more likely for some
        labels = torch.randint(0, 2, (100,))

        quantile = calibrator.calibrate(logits, labels)
        assert calibrator.is_fitted
        assert isinstance(quantile, float)
        assert 0 <= quantile <= 1

    def test_predict_set(self, calibrator):
        """Test prediction set generation."""
        logits = torch.randn(20, 2)
        labels = torch.randint(0, 2, (20,))
        calibrator.calibrate(logits, labels)

        pred_sets = calibrator.predict_set(logits)
        assert pred_sets.shape == (20, 2)
        assert pred_sets.dtype == torch.bool
        # No empty sets
        assert pred_sets.any(dim=1).all()

    def test_coverage(self, calibrator):
        """Test empirical coverage is approximately 1-alpha."""
        torch.manual_seed(42)
        logits = torch.randn(200, 2)
        logits[:, 1] += 1.0
        labels = torch.randint(0, 2, (200,))

        calibrator.calibrate(logits[:100], labels[:100])
        pred_sets = calibrator.predict_set(logits[100:])
        coverage = calibrator.coverage(pred_sets, labels[100:])

        # Should be close to 0.9 (1 - alpha)
        assert 0.8 <= coverage <= 1.0

    def test_predict_anomaly_score(self, calibrator):
        """Test calibrated anomaly scores (p-values)."""
        logits = torch.randn(50, 2)
        labels = torch.randint(0, 2, (50,))
        calibrator.calibrate(logits, labels)

        p_values = calibrator.predict_anomaly_score(logits)
        assert p_values.shape == (50,)
        assert (p_values >= 0).all() and (p_values <= 1).all()


class TestAdaptivePredictionSets:
    """Tests for APS."""

    @pytest.fixture
    def aps(self):
        return AdaptivePredictionSets(alpha=0.1, calibration_split=0.5, random_state=42)

    def test_calibrate(self, aps):
        """Test APS calibration."""
        logits = torch.randn(100, 3)
        logits[:, 1] += 1.5
        labels = torch.randint(0, 3, (100,))

        quantile = aps.calibrate(logits, labels)
        assert aps.is_fitted
        assert isinstance(quantile, float)

    def test_predict_set(self, aps):
        """Test APS prediction sets."""
        logits = torch.randn(20, 3)
        labels = torch.randint(0, 3, (20,))
        aps.calibrate(logits, labels)

        pred_sets = aps.predict_set(logits)
        assert pred_sets.shape == (20, 3)
        assert pred_sets.dtype == torch.bool
        # No empty sets
        assert pred_sets.any(dim=1).all()

    def test_aps_vs_split(self):
        """Test APS produces smaller sets than split conformal."""
        logits = torch.randn(200, 3)
        logits[:, 1] += 2.0
        labels = torch.randint(0, 3, (200,))

        split = SplitConformal(alpha=0.1, calibration_split=0.5, random_state=42)
        aps = AdaptivePredictionSets(alpha=0.1, calibration_split=0.5, random_state=42)

        split.calibrate(logits[:100], labels[:100])
        aps.calibrate(logits[:100], labels[:100])

        split_sets = split.predict_set(logits[100:])
        aps_sets = aps.predict_set(logits[100:])

        split_avg = split_sets.float().sum(dim=1).mean().item()
        aps_avg = aps_sets.float().sum(dim=1).mean().item()

        # APS should produce smaller or equal sets on average
        assert aps_avg <= split_avg + 0.1  # Allow small variance


class TestRegularizedAPS:
    """Tests for RAPS."""

    @pytest.fixture
    def raps(self):
        return RegularizedAPS(alpha=0.1, calibration_split=0.5, random_state=42, regularization=0.01, k_reg=3)

    def test_calibrate(self, raps):
        """Test RAPS calibration."""
        logits = torch.randn(100, 3)
        logits[:, 1] += 1.5
        labels = torch.randint(0, 3, (100,))

        quantile = raps.calibrate(logits, labels)
        assert raps.is_fitted

    def test_regularization_effect(self):
        """Test regularization reduces set sizes for confident predictions."""
        logits = torch.randn(200, 5)
        logits[:, 1] += 3.0  # Very confident on class 1
        labels = torch.ones(200, dtype=torch.long)

        aps = AdaptivePredictionSets(alpha=0.1, calibration_split=0.5, random_state=42)
        raps = RegularizedAPS(alpha=0.1, calibration_split=0.5, random_state=42, regularization=0.1, k_reg=2)

        aps.calibrate(logits[:100], labels[:100])
        raps.calibrate(logits[:100], labels[:100])

        aps_sets = aps.predict_set(logits[100:])
        raps_sets = raps.predict_set(logits[100:])

        aps_avg = aps_sets.float().sum(dim=1).mean().item()
        raps_avg = raps_sets.float().sum(dim=1).mean().item()

        # RAPS with regularization should have smaller sets for confident predictions
        assert raps_avg <= aps_avg


class TestConformalMetrics:
    """Tests for conformal metrics computation."""

    def test_compute_metrics(self):
        """Test metric computation."""
        pred_sets = torch.tensor([
            [True, False],   # {0}
            [True, True],    # {0, 1}
            [False, True],   # {1}
            [True, False],   # {0}
        ])
        labels = torch.tensor([0, 1, 1, 0])
        scores = torch.tensor([0.1, 0.3, 0.2, 0.15])

        metrics = compute_conformal_metrics(pred_sets, labels, scores)

        assert "coverage" in metrics
        assert "avg_set_size" in metrics
        assert "singleton_rate" in metrics
        assert "empty_rate" in metrics
        assert metrics["coverage"] == 1.0  # All correct
        assert metrics["avg_set_size"] == 1.25
        assert metrics["singleton_rate"] == 0.75
        assert metrics["empty_rate"] == 0.0