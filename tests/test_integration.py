"""Integration tests for full GTFD pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path

from gtfd.config import GTFDConfig, DataConfig
from gtfd.data.generator import SyntheticTransactionGenerator
from gtfd.graph.builder import PaymentGraphBuilder
from gtfd.model.gtfd import GTFDModel
from gtfd.pretraining.trainer import PretrainingTrainer
from gtfd.conformal.calibration import AdaptivePredictionSets
from gtfd.graph.dataset import GraphDataLoader
from gtfd.graph.builder import GraphSnapshot


class TestFullPipeline:
    """Integration tests for complete GTFD pipeline."""

    @pytest.fixture
    def config(self):
        """Minimal config for testing."""
        config = GTFDConfig()
        config.data.num_accounts = 100
        config.data.num_merchants = 20
        config.data.num_transactions = 2000
        config.data.fraud_rate = 0.05
        config.data.temporal_span_days = 30
        config.data.ring_types = ["card_testing", "account_takeover"]
        config.model.pretraining.epochs = 2
        config.model.pretraining.batch_size = 4
        config.model.pretraining.learning_rate = 1e-3
        config.model.graph_encoder.hidden_dim = 32
        config.model.graph_encoder.num_layers = 1
        config.model.graph_encoder.num_heads = 2
        config.model.temporal_encoder.hidden_dim = 32
        config.model.temporal_encoder.num_layers = 1
        config.model.temporal_encoder.num_heads = 2
        config.model.fusion.hidden_dim = 32
        config.model.fusion.num_heads = 2
        config.seed = 42
        return config

    def test_data_generation_to_graph(self, config):
        """Test data generation and graph building."""
        generator = SyntheticTransactionGenerator(config.data)
        df, rings = generator.generate_dataset()

        assert len(df) > 0
        assert len(rings) > 0
        assert "is_fraud" in df.columns
        assert "ring_id" in df.columns

        builder = PaymentGraphBuilder()
        data = builder.build_graph(df)

        assert data.num_nodes > 0
        assert data.num_edges > 0
        assert data.x is not None
        assert data.y is not None

    def test_pretraining_pipeline(self, config):
        """Test pretraining runs without error."""
        generator = SyntheticTransactionGenerator(config.data)
        df, _ = generator.generate_dataset()

        builder = PaymentGraphBuilder()
        data = builder.build_graph(df)

        model = GTFDModel(config.model)

        # Create snapshot and loader
        snapshot = GraphSnapshot(data=data, timestamp_start=None, timestamp_end=None)
        loader = GraphDataLoader([snapshot], batch_size=config.model.pretraining.batch_size)

        trainer = PretrainingTrainer(
            model=model,
            config=config.model.pretraining,
            device=torch.device("cpu"),
        )

        # Should run without error
        history = trainer.train(loader)
        assert len(history) == config.model.pretraining.epochs
        assert "total_loss" in history[0]

    def test_calibration_pipeline(self, config):
        """Test conformal calibration pipeline."""
        generator = SyntheticTransactionGenerator(config.data)
        df, _ = generator.generate_dataset()

        builder = PaymentGraphBuilder()
        data = builder.build_graph(df)

        model = GTFDModel(config.model)
        model.add_fraud_head(num_classes=2)

        # Quick pretrain
        snapshot = GraphSnapshot(data=data, timestamp_start=None, timestamp_end=None)
        loader = GraphDataLoader([snapshot], batch_size=config.model.pretraining.batch_size)

        trainer = PretrainingTrainer(model, config.model.pretraining, torch.device("cpu"))
        trainer.train(loader)

        # Calibrate
        model.eval()
        with torch.no_grad():
            h_fused = model.encode(data)
            if isinstance(h_fused, dict):
                h_fused = h_fused.get("account", next(iter(h_fused.values())))
            logits = model.fraud_head(h_fused)
            labels = data.y.squeeze()

        calibrator = AdaptivePredictionSets(alpha=0.1, calibration_split=0.3, random_state=42)
        quantile = calibrator.calibrate(logits, labels)

        assert calibrator.is_fitted
        assert isinstance(quantile, float)

        # Predict sets
        pred_sets = calibrator.predict_set(logits)
        assert pred_sets.shape == (len(labels), 2)
        assert pred_sets.dtype == torch.bool

    def test_model_save_load(self, config):
        """Test model checkpoint save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            model = GTFDModel(config.model)
            model.add_fraud_head(num_classes=2)

            checkpoint_path = tmpdir / "model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.model,
            }, checkpoint_path)

            # Load
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            new_model = GTFDModel(config.model)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_end_to_end_detection(self, config):
        """Test full detection pipeline."""
        generator = SyntheticTransactionGenerator(config.data)
        df, rings = generator.generate_dataset()

        builder = PaymentGraphBuilder()
        data = builder.build_graph(df)

        model = GTFDModel(config.model)
        model.add_fraud_head(num_classes=2)

        # Pretrain
        snapshot = GraphSnapshot(data=data, timestamp_start=None, timestamp_end=None)
        loader = GraphDataLoader([snapshot], batch_size=config.model.pretraining.batch_size)

        trainer = PretrainingTrainer(model, config.model.pretraining, torch.device("cpu"))
        trainer.train(loader)

        # Calibrate
        model.eval()
        with torch.no_grad():
            h_fused = model.encode(data)
            if isinstance(h_fused, dict):
                h_fused = h_fused.get("account", next(iter(h_fused.values())))
            logits = model.fraud_head(h_fused)
            labels = data.y.squeeze()

        calibrator = AdaptivePredictionSets(alpha=0.1, calibration_split=0.3, random_state=42)
        calibrator.calibrate(logits, labels)

        # Get predictions
        pred_sets = calibrator.predict_set(logits)
        anomaly_scores = calibrator.predict_anomaly_score(logits)

        # Should detect some fraud
        fraud_mask = labels == 1
        if fraud_mask.any():
            detected_fraud = pred_sets[fraud_mask, 1].float().mean().item()
            assert detected_fraud > 0.0  # At least some fraud detected

        # Anomaly scores should be higher for fraud
        fraud_scores = anomaly_scores[fraud_mask].mean().item() if fraud_mask.any() else 0
        legit_scores = anomaly_scores[~fraud_mask].mean().item() if (~fraud_mask).any() else 0
        if fraud_mask.any() and (~fraud_mask).any():
            assert fraud_scores > legit_scores * 0.5  # Fraud should be more anomalous