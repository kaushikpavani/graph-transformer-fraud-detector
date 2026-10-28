#!/usr/bin/env python3
"""Quickstart example for GTFD.

This script demonstrates the complete pipeline:
1. Generate synthetic data with fraud rings
2. Build payment graph
3. Self-supervised pretraining
4. Conformal calibration
5. Downstream fraud detection
6. Evaluation with adversarial stress testing
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from gtfd.config import GTFDConfig
from gtfd.data.generator import SyntheticTransactionGenerator
from gtfd.graph.builder import PaymentGraphBuilder, GraphConfig
from gtfd.model.gtfd import GTFDModel
from gtfd.pretraining.trainer import PretrainingTrainer
from gtfd.conformal.calibration import AdaptivePredictionSets
from gtfd.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_f1,
    compute_precision_at_k, compute_recall_at_k, compute_ring_recall,
)
from gtfd.evaluation.adversarial import AdversarialEvaluator
from gtfd.utils.seed import set_seed


def main():
    print("=" * 60)
    print("GTFD Quickstart: Graph-Transformer Fraud Detection")
    print("=" * 60)

    # Configuration
    config = GTFDConfig()
    config.data.num_accounts = 2000
    config.data.num_merchants = 200
    config.data.num_transactions = 50000
    config.data.fraud_rate = 0.03
    config.model.pretraining.epochs = 5  # Quick demo
    config.model.pretraining.batch_size = 16
    config.seed = 42
    set_seed(config.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Generate synthetic data
        print("\n[1/6] Generating synthetic transaction data...")
        generator = SyntheticTransactionGenerator(config.data)
        df, rings = generator.generate_dataset()
        print(f"  Generated {len(df)} transactions, {len(rings)} fraud rings")

        # Step 2: Build payment graph
        print("\n[2/6] Building payment graph...")
        graph_config = GraphConfig()
        builder = PaymentGraphBuilder(graph_config)
        data = builder.build_graph(df)
        data = data.to(device)
        print(f"  Graph: {data.num_nodes} nodes, {data.num_edges} edges")

        # Step 3: Initialize model
        print("\n[3/6] Initializing GTFD model...")
        model = GTFDModel(config.model).to(device)
        model.add_fraud_head(num_classes=2)
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Step 4: Self-supervised pretraining
        print("\n[4/6] Self-supervised pretraining...")
        from gtfd.graph.dataset import GraphDataLoader
        from gtfd.graph.builder import GraphSnapshot

        snapshot = GraphSnapshot(data=data, timestamp_start=None, timestamp_end=None)
        loader = GraphDataLoader([snapshot], batch_size=config.model.pretraining.batch_size, shuffle=True)

        trainer = PretrainingTrainer(
            model=model,
            config=config.model.pretraining,
            device=device,
            output_dir=tmpdir / "checkpoints",
        )
        trainer.train(loader)

        # Step 5: Conformal calibration
        print("\n[5/6] Conformal calibration...")
        model.eval()
        with torch.no_grad():
            h_fused = model.encode(data)
            if isinstance(h_fused, dict):
                h_fused = h_fused.get("account", next(iter(h_fused.values())))
            logits = model.fraud_head(h_fused)
            labels = data.y.squeeze() if hasattr(data, "y") and data.y is not None else torch.zeros(h_fused.shape[0], device=device)

        calibrator = AdaptivePredictionSets(alpha=0.1, calibration_split=0.2, random_state=42)
        quantile = calibrator.calibrate(logits, labels)
        print(f"  Calibration quantile: {quantile:.4f}")

        # Step 6: Evaluation
        print("\n[6/6] Evaluating...")
        pred_sets = calibrator.predict_set(logits)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = (probs > 0.5).long()

        results = {
            "auroc": compute_auroc(labels, probs),
            "auprc": compute_auprc(labels, probs),
            "f1": compute_f1(labels, preds),
            "precision_at_100": compute_precision_at_k(labels, probs, 100),
            "recall_at_100": compute_recall_at_k(labels, probs, 100),
            "conformal_coverage": compute_ring_recall(labels, preds, data.ring_id if hasattr(data, "ring_id") else torch.full_like(labels, -1)),
            "conformal_avg_set_size": pred_sets.float().sum(dim=1).mean().item(),
        }

        # Ring recall
        if hasattr(data, "ring_id"):
            results["ring_recall"] = compute_ring_recall(labels, preds, data.ring_id)

        print("\nResults:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

        # Adversarial stress test (lightweight)
        print("\n[Bonus] Adversarial stress test...")
        evaluator = AdversarialEvaluator(model, device, epsilons=[0.0, 0.1, 0.2])

        def metric_fn(out, d):
            if isinstance(out, dict):
                out = out.get("account", next(iter(out.values())))
            probs = torch.softmax(out, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            labels = d.y.squeeze() if hasattr(d, "y") and d.y is not None else torch.zeros(out.shape[0], device=device)
            return {"accuracy": (preds == labels).float().mean().item()}

        def loss_fn(out, d):
            if isinstance(out, dict):
                out = out.get("account", next(iter(out.values())))
            labels = d.y.squeeze() if hasattr(d, "y") and d.y is not None else torch.zeros(out.shape[0], device=device)
            return nn.functional.cross_entropy(out, labels)

        adv_results = evaluator.evaluate_feature_attack(data, loss_fn, metric_fn)
        print("  Adversarial accuracy:")
        for eps, metrics in adv_results.items():
            print(f"    ε={eps}: {metrics['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run full pretraining: python scripts/pretrain.py --config configs/pretrain.yaml --data data/graph.pt")
    print("  - Calibrate: python scripts/calibrate.py --model checkpoints/pretrained.pt --data data/cal.pt")
    print("  - Evaluate: python scripts/evaluate.py --model checkpoints/model.pt --calibrator checkpoints/calibrated.pt --data data/test.pt")
    print("  - Stress test: python scripts/stress_test.py --model checkpoints/model.pt --data data/test.pt")


if __name__ == "__main__":
    main()