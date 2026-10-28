#!/usr/bin/env python3
"""CLI script for conformal calibration."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gtfd.config import GTFDConfig
from gtfd.conformal.calibration import SplitConformal, AdaptivePredictionSets, RegularizedAPS, ConformalMethod
from gtfd.model.gtfd import GTFDModel
from gtfd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Calibrate conformal prediction for GTFD")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML file")
    parser.add_argument("--model", type=Path, required=True, help="Pretrained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Calibration data (PyG format)")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/calibrated.pt"), help="Output calibrator")
    parser.add_argument("--method", type=str, default="aps", choices=["split", "aps", "raps"], help="Conformal method")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level")
    parser.add_argument("--cal-split", type=float, default=0.2, help="Calibration split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config and args.config.exists():
        config = GTFDConfig.from_yaml(args.config)
    else:
        config = GTFDConfig()

    config.model.conformal.method = ConformalMethod(args.method)
    config.model.conformal.alpha = args.alpha
    config.model.conformal.calibration_split = args.cal_split
    config.seed = args.seed

    set_seed(config.seed)

    device = torch.device(config.device)

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = GTFDModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load calibration data
    print(f"Loading calibration data from {args.data}...")
    data = torch.load(args.data, map_location=device, weights_only=False)

    # Get embeddings and predictions
    with torch.no_grad():
        h_fused = model.encode(data)
        if isinstance(h_fused, dict):
            h_fused = h_fused.get("account", next(iter(h_fused.values())))
        logits = model.fraud_head(h_fused) if model.fraud_head else h_fused
        labels = data.y.squeeze() if hasattr(data, "y") and data.y is not None else torch.zeros(h_fused.shape[0])

    # Select conformal method
    if args.method == "split":
        calibrator = SplitConformal(alpha=args.alpha, calibration_split=args.cal_split, random_state=args.seed)
    elif args.method == "aps":
        calibrator = AdaptivePredictionSets(alpha=args.alpha, calibration_split=args.cal_split, random_state=args.seed)
    elif args.method == "raps":
        calibrator = RegularizedAPS(alpha=args.alpha, calibration_split=args.cal_split, random_state=args.seed)

    # Calibrate
    print("Calibrating...")
    quantile = calibrator.calibrate(logits, labels)
    print(f"Calibration quantile: {quantile:.4f}")

    # Save calibrator
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "calibrator_state": calibrator.__dict__,
        "method": args.method,
        "alpha": args.alpha,
        "quantile": quantile,
    }, args.output)
    print(f"Saved calibrator to {args.output}")


if __name__ == "__main__":
    main()