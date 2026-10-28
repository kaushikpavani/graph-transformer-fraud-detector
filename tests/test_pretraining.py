"""Tests for pretraining objectives."""

import pytest
import torch
import torch.nn as nn

from gtfd.pretraining.objectives import (
    LinkMaskLoss,
    AdversarialLoss,
    ContrastiveLoss,
    PretrainingObjectives,
)
from gtfd.config import PretrainingConfig
from gtfd.model.gtfd import GTFDModel
from gtfd.config import ModelConfig, GraphConfig, GraphEncoderConfig


class TestLinkMaskLoss:
    """Tests for LinkMaskLoss."""

    @pytest.fixture
    def config(self):
        return PretrainingConfig(mask_ratio=0.15)

    @pytest.fixture
    def model(self):
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=32, num_layers=2, num_heads=2),
        )
        return GTFDModel(config)

    def test_forward_homo(self, config, model):
        """Test link mask loss on homogeneous graph."""
        loss_fn = LinkMaskLoss(config)

        # Create simple graph
        data = torch.load("/dev/null") if False else None
        # We'll create a minimal test graph inline
        from torch_geometric.data import Data
        data = Data(
            x=torch.randn(10, 32),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 32),
            y=torch.zeros(10),
        )

        h_fused = torch.randn(10, 32)
        loss, losses = loss_fn(model, data, h_fused)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "link_mask_loss" in losses
        assert "attr_loss" in losses


class TestAdversarialLoss:
    """Tests for AdversarialLoss."""

    @pytest.fixture
    def config(self):
        return PretrainingConfig()

    @pytest.fixture
    def model(self):
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=32, num_layers=2, num_heads=2),
        )
        return GTFDModel(config)

    def test_forward(self, config, model):
        """Test adversarial loss computation."""
        loss_fn = AdversarialLoss(config)

        h_fused = torch.randn(10, 32)
        h_adv = h_fused + torch.randn_like(h_fused) * 0.1

        loss, losses = loss_fn(model, None, h_fused, h_adv)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "adv_loss" in losses


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    @pytest.fixture
    def config(self):
        return PretrainingConfig(temperature=0.1)

    def test_forward_homo(self, config):
        """Test contrastive loss on homogeneous embeddings."""
        loss_fn = ContrastiveLoss(config)

        h_clean = torch.randn(16, 32)
        h_aug = h_clean + torch.randn_like(h_clean) * 0.1

        loss, losses = loss_fn(h_clean, h_aug)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "contrastive_loss" in losses

    def test_forward_hetero(self, config):
        """Test contrastive loss on heterogeneous embeddings."""
        loss_fn = ContrastiveLoss(config)

        h_clean = {"account": torch.randn(10, 32), "merchant": torch.randn(5, 32)}
        h_aug = {
            "account": h_clean["account"] + torch.randn_like(h_clean["account"]) * 0.1,
            "merchant": h_clean["merchant"] + torch.randn_like(h_clean["merchant"]) * 0.1,
        }

        loss, losses = loss_fn(h_clean, h_aug)

        assert isinstance(loss, torch.Tensor)
        assert "contrastive_account" in losses
        assert "contrastive_merchant" in losses


class TestPretrainingObjectives:
    """Tests for combined PretrainingObjectives."""

    @pytest.fixture
    def config(self):
        return PretrainingConfig(
            mask_ratio=0.1,
            lambda_link=1.0,
            lambda_adv=0.5,
            lambda_contrast=0.5,
        )

    @pytest.fixture
    def model(self):
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=32, num_layers=2, num_heads=2),
        )
        return GTFDModel(config)

    def test_forward_all_losses(self, config, model):
        """Test combined loss with all components."""
        objectives = PretrainingObjectives(config)

        from torch_geometric.data import Data
        data = Data(
            x=torch.randn(10, 32),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 32),
        )

        h_fused = torch.randn(10, 32)
        h_adv = h_fused + torch.randn_like(h_fused) * 0.1
        h_aug = h_fused + torch.randn_like(h_fused) * 0.1

        loss, losses = objectives(model, data, h_fused, h_adv=h_adv, h_aug=h_aug)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "total_loss" in losses
        assert "link_mask_loss" in losses
        assert "adv_loss" in losses
        assert "contrastive_loss" in losses


class TestPretrainingTrainer:
    """Tests for PretrainingTrainer (integration)."""

    def test_trainer_creation(self):
        """Test trainer can be instantiated."""
        config = ModelConfig(
            graph=GraphConfig(),
            graph_encoder=GraphEncoderConfig(hidden_dim=16, num_layers=1, num_heads=2),
        )
        model = GTFDModel(config)
        pretrain_config = PretrainingConfig(epochs=1, batch_size=2, learning_rate=1e-3)

        trainer = PretrainingTrainer(
            model=model,
            config=pretrain_config,
            device=torch.device("cpu"),
        )

        assert trainer is not None
        assert trainer.epoch == 0
        assert trainer.step == 0