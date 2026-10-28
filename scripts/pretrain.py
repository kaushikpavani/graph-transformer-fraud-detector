#!/usr/bin/env python3
"""CLI script for self-supervised pretraining."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gtfd.config import GTFDConfig, PretrainingConfig
from gtfd.graph.builder import PaymentGraphBuilder, GraphConfig
from gtfd.graph.dataset import GraphDataLoader
from gtfd.model.gtfd import GTFDModel
from gtfd.pretraining.trainer import PretrainingTrainer
from gtfd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for GTFD")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML file")
    parser.add_argument("--data", type=Path, required=True, help="Input graph data (PyG format)")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/pretrained.pt"), help="Output checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config and args.config.exists():
        config = GTFDConfig.from_yaml(args.config)
    else:
        config = GTFDConfig()

    # Override from CLI
    if args.epochs:
        config.model.pretraining.epochs = args.epochs
    if args.lr:
        config.model.pretraining.learning_rate = args.lr
    config.model.pretraining.batch_size = args.batch_size
    config.seed = args.seed
    config.device = args.device

    # Set seed
    set_seed(config.seed)

    # Device
    device = torch.device(config.device)

    # Load data
    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, map_location=device, weights_only=False)

    # Create model
    model = GTFDModel(config.model).to(device)

    # Create dataloader
    # For pretraining, we use the graph snapshots directly
    if hasattr(data, "snapshots"):
        # List of snapshots
        loader = GraphDataLoader(
            data.snapshots,
            batch_size=config.model.pretraining.batch_size,
            shuffle=True,
        )
    else:
        # Single graph - wrap in list
        from gtfd.graph.builder import GraphSnapshot
        snapshot = GraphSnapshot(data=data, timestamp_start=None, timestamp_end=None)
        loader = GraphDataLoader(
            [snapshot],
            batch_size=config.model.pretraining.batch_size,
            shuffle=True,
        )

    # Trainer
    trainer = PretrainingTrainer(
        model=model,
        config=config.model.pretraining,
        device=device,
        output_dir=args.output.parent,
    )

    # Train
    trainer.train(loader, epochs=args.epochs)

    # Save final model
    trainer.save_checkpoint(args.output)
    print(f"Saved pretrained model to {args.output}")


if __name__ == "__main__":
    main()