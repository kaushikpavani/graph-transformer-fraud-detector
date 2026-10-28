#!/usr/bin/env python3
"""CLI script for evaluation."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from gtfd.config import GTFDConfig
from gtfd.conformal.calibration import SplitConformal, AdaptivePredictionSets, RegularizedAPS
from gtfd.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_f1,
    compute_precision_at_k, compute_recall_at_k, compute_ring_recall,
    compute_coverage, compute_avg_set_size,
)
from gtfd.model.gtfd import GTFDModel
from gtfd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate GTFD model")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML file")
    parser.add_argument("--model", type=Path, required=True, help="Trained model checkpoint")
    parser.add_argument("--calibrator", type=Path, default=None, help="Conformal calibrator checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Test data (PyG format)")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"), help="Output results")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 50, 100, 500], help="K values for P@K/R@K")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    if args.config and args.config.exists():
        config = GTFDConfig.from_yaml(args.config)
    else:
        config = GTFDConfig()

    config.seed = args.seed
    set_seed(config.seed)

    device = torch.device(config.device)

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = GTFDModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load calibrator if provided
    calibrator = None
    if args.calibrator and args.calibrator.exists():
        print(f"Loading calibrator from {args.calibrator}...")
        cal_checkpoint = torch.load(args.calibrator, map_location=device, weights_only=False)
        method = cal_checkpoint.get("method", "aps")
        if method == "split":
            calibrator = SplitConformal()
        elif method == "aps":
            calibrator = AdaptivePredictionSets()
        elif method == "raps":
            calibrator = RegularizedAPS()
        calibrator.__dict__ = cal_checkpoint["calibrator_state"]

    # Load test data
    print(f"Loading test data from {args.data}...")
    data = torch.load(args.data, map_location=device, weights_only=False)

    # Evaluate
    with torch.no_grad():
        h_fused = model.encode(data)
        if isinstance(h_fused, dict):
            h_fused = h_fused.get("account", next(iter(h_fused.values())))

        # Fraud classification
        if model.fraud_head:
            logits = model.fraud_head(h_fused)
        else:
            logits = h_fused

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs > 0.5).long()

        labels = data.y.squeeze() if hasattr(data, "y") and data.y is not None else torch.zeros(h_fused.shape[0])

        # Ring labels (if available)
        ring_labels = data.ring_id if hasattr(data, "ring_id") else torch.full_like(labels, -1)

        # Compute metrics
        results = {
            "auroc": compute_auroc(labels, probs),
            "auprc": compute_auprc(labels, probs),
            "f1": compute_f1(labels, preds),
        }

        # Precision/Recall at K
        for k in args.k_values:
            results[f"precision_at_{k}"] = compute_precision_at_k(labels, probs, k)
            results[f"recall_at_{k}"] = compute_recall_at_k(labels, probs, k)

        # Ring recall
        if (ring_labels >= 0).any():
            results["ring_recall"] = compute_ring_recall(labels, preds, ring_labels)

        print("Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        # Conformal evaluation
        if calibrator:
            pred_sets = calibrator.predict_set(logits)
            results["conformal_coverage"] = compute_coverage(pred_sets, labels)
            results["conformal_avg_set_size"] = compute_avg_set_size(pred_sets)
            print(f"  conformal_coverage: {results['conformal_coverage']:.4f}")
            print(f"  conformal_avg_set_size: {results['conformal_avg_set_size']:.4f}")

    # Save results
    import json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()