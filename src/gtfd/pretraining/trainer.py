"""Pretraining trainer for GTFD."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config import PretrainingConfig, GTFDConfig
from ..graph.augment import GraphAugmenter, AdversarialAugmenter
from .objectives import PretrainingObjectives

logger = logging.getLogger(__name__)


class PretrainingTrainer:
    """Trainer for self-supervised pretraining of GTFD.

    Handles the full pretraining loop with link masking,
    adversarial augmentation, and contrastive learning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PretrainingConfig,
        device: torch.device,
        output_dir: Optional[Path] = None,
    ):
        """Initialize the trainer.

        Args:
            model: GTFD model to pretrain.
            config: Pretraining configuration.
            device: Training device.
            output_dir: Directory for checkpoints and logs.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir or Path("./checkpoints")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )

        # Augmenters
        self.graph_augmenter = GraphAugmenter(
            edge_dropout=0.1,
            feature_mask_prob=0.1,
            feature_noise_std=0.01,
        )
        self.adv_augmenter = AdversarialAugmenter(config)

        # Objectives
        self.objectives = PretrainingObjectives(config)

        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float("inf")
        self.history = []

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding graph batches.

        Returns:
            Dictionary of average losses.
        """
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device) if hasattr(batch, "to") else batch
            self.optimizer.zero_grad()

            # Clean forward
            h_fused = self.model.encode(batch)

            # Augmented view for contrastive
            batch_aug = self.graph_augmenter.augment(batch)
            batch_aug = batch_aug.to(self.device) if hasattr(batch_aug, "to") else batch_aug
            h_aug = self.model.encode(batch_aug)

            # Adversarial view
            batch_adv = self.adv_augmenter.attack_structure(batch)
            batch_adv = batch_adv.to(self.device) if hasattr(batch_adv, "to") else batch_adv
            h_adv = self.model.encode(batch_adv)

            # Compute losses
            loss, losses = self.objectives(
                self.model, batch, h_fused, h_adv=h_adv, h_aug=h_aug
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Accumulate
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1
            self.step += 1

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    def train(self, dataloader, epochs: Optional[int] = None) -> list[dict[str, float]]:
        """Run full pretraining.

        Args:
            dataloader: Training DataLoader.
            epochs: Number of epochs (overrides config).

        Returns:
            List of epoch loss dictionaries.
        """
        epochs = epochs or self.config.epochs
        logger.info(f"Starting pretraining for {epochs} epochs on {self.device}")

        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()

            epoch_losses = self.train_epoch(dataloader)
            self.scheduler.step()

            elapsed = time.time() - start_time
            total_loss = epoch_losses.get("total_loss", 0.0)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {total_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Log individual losses
            for k, v in epoch_losses.items():
                if k != "total_loss":
                    logger.info(f"  {k}: {v:.4f}")

            self.history.append(epoch_losses)

            # Save checkpoint
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.save_checkpoint(self.output_dir / "best_pretrain.pt")

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(self.output_dir / f"pretrain_epoch_{epoch + 1}.pt")

        logger.info("Pretraining complete")
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
            "history": self.history,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint.get("history", [])
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")