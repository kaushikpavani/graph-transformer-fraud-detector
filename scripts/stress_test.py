#!/usr/bin/env python3
"""CLI script for adversarial stress testing."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from gtfd.config import GTFDConfig
from gtfd.evaluation.adversarial import AdversarialEvaluator, compute_robustness_auc
from gtfd.model.gtfd import GTFDModel
from gtfd.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Adversarial stress testing for GTFD")
    parser.add_argument("--config", type=Path, default=None, help="Config YAML file")
    parser.add_argument("--model", type=Path, required=True, help="Trained model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Test data (PyG format)")
    parser.add_argument("--output", type=Path, default=Path("results/adversarial.json"), help="Output results")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.15, 0.2], help="Perturbation magnitudes")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples per epsilon")
    parser.add_argument("--attack-type", type=str, default="feature", choices=["feature", "structure"], help="Attack type")
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

    # Load test data
    print(f"Loading test data from {args.data}...")
    data = torch.load(args.data, map_location=device, weights_only=False)

    # Define metric function
    def metric_fn(out, data):
        if isinstance(out, dict):
            out = out.get("account", next(iter(out.values())))
        if hasattr(data, "y") and data.y is not None:
            labels = data.y.squeeze()
            probs = torch.softmax(out, dim=-1)[:, 1] if out.shape[-1] > 1 else torch.sigmoid(out)
            preds = (probs > 0.5).long()
            return {
                "accuracy": (preds == labels).float().mean().item(),
                "auroc": compute_auroc(labels, probs),
                "f1": compute_f1(labels, preds),
            }
        return {"accuracy": 0.0, "auroc": 0.0, "f1": 0.0}

    def loss_fn(out, data):
        if isinstance(out, dict):
            out = out.get("account", next(iter(out.values())))
        if hasattr(data, "y") and data.y is not None:
            labels = data.y.squeeze()
            return torch.nn.functional.cross_entropy(out, labels)
        return torch.tensor(0.0, device=device)

    # Evaluator
    evaluator = AdversarialEvaluator(
        model=model,
        device=device,
        epsilons=args.epsilons,
        num_samples=args.num_samples,
    )

    # Run evaluation
    if args.attack_type == "feature":
        print("Running feature-space PGD attacks...")
        results = evaluator.evaluate_feature_attack(data, loss_fn, metric_fn)
    else:
        print("Running structure attacks...")
        results = evaluator.evaluate_structure_attack(data, metric_fn)

    # Compute robustness AUC
    for metric in ["accuracy", "auroc", "f1"]:
        auc = compute_robustness_auc(args.epsilons, results, metric)
        results[f"robustness_auc_{metric}"] = auc

    print("Results:")
    for eps, metrics in results.items():
        if isinstance(metrics, dict):
            print(f"  ε={eps}: {metrics}")
        else:
            print(f"  {eps}: {metrics:.4f}")

    # Save results
    import json
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        # Convert to serializable
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = {k2: float(v2) for k2, v2 in v.items()}
            else:
                serializable[k] = float(v)
        json.dump(serializable, f, indent=2)
    print(f"Saved results to {args.output}")


# Import metrics here to avoid circular import
from gtfd.evaluation.metrics import compute_auroc, compute_auprc, compute_f1


if __name__ == "__main__":
    main()