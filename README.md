# Graph-Transformer Fraud Detector (GTFD)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready Python library for detecting coordinated fraud rings in corporate transaction networks using **Graph-Transformer fusion**, **self-supervised pretraining**, and **conformal risk control**.

## Overview

Traditional fraud detection systems treat transactions in isolation, missing coordinated fraud rings that exploit structural and temporal patterns in payment networks. GTFD addresses this by:

1. **Graph Structural Encoder** — Multi-head Graph Attention Network (GAT) capturing network topology
2. **Temporal Sequence Encoder** — Gated Transformer with time-aware positional embeddings for per-node transaction sequences
3. **Cross-Modal Fusion** — Cross-attention gating layer combining graph and temporal evidence
4. **Self-Supervised Pretraining** — Link masking + adversarial augmentation + contrastive learning on unlabeled graphs
5. **Conformal Risk Control** — Split conformal / Adaptive Prediction Sets (APS) / RAPS for calibrated anomaly scores with finite-sample coverage guarantees
6. **Adversarial Robustness** — PGD feature attacks and structural edge dropout for stress testing

Based on: **GTFD: Graph-Transformer Fraud Detection with Self-Supervised Pretraining and Conformal Risk Control** (arXiv:2609.14234, JP Morgan, Sep 2026)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRAPH-TRANSFORMER FRAUD DETECTOR                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Transaction Data ──▶ Graph Builder ──▶ Payment Graph (Dynamic Snapshots)  │
│                           │                        │                         │
│                    ┌──────┴──────┐         ┌──────┴──────┐                 │
│                    ▼             ▼         ▼             ▼                 │
│            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                 │
│            │ Graph       │ │ Temporal    │ │ Cross-Modal │                 │
│            │ Encoder     │ │ Encoder     │ │ Fusion      │                 │
│            │ (GAT)       │ │ (Gated      │ │ (Cross-Attn │                 │
│            │             │ │  Transformer)│ │  + Gating)  │                 │
│            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                 │
│                   │               │               │                         │
│                   └───────────────┼───────────────┘                         │
│                                   ▼                                        │
│                         ┌─────────────────────┐                            │
│                         │   Fused Embeddings  │                            │
│                         └──────────┬──────────┘                            │
│                                    │                                      │
│                    ┌────────────────┼────────────────┐                    │
│                    ▼                ▼                ▼                    │
│            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│            │ Fraud       │ │ Anomaly     │ │ Fraud Type  │                │
│            │ Classification│ │ Scoring     │ │ Classification              │
│            └─────────────┘ └─────────────┘ └─────────────┘                │
│                    │                │                │                    │
│                    └────────────────┼────────────────┘                    │
│                                     ▼                                    │
│                         ┌─────────────────────┐                           │
│                         │ Conformal           │                           │
│                         │ Calibration         │                           │
│                         │ (Split/APS/RAPS)    │                           │
│                         └─────────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Heterogeneous Payment Graph Construction** — Accounts, merchants, transactions with rich attributes
- **Multi-Head GAT Encoder** — Edge-aware attention with residual connections
- **Gated Temporal Transformer** — Sinusoidal time encoding, RoPE, gated residuals
- **Cross-Modal Fusion** — Cross-attention + adaptive gating between graph and temporal views
- **Self-Supervised Pretraining** — Link masking, adversarial augmentation, contrastive learning
- **Conformal Prediction** — Split conformal, APS, RAPS with coverage guarantees
- **Synthetic Data Generator** — Realistic corporate transactions with 5 fraud ring types
- **Comprehensive Evaluation** — AUROC, AUPRC, ring-level recall, adversarial robustness
- **Production Ready** — Type hints, docstrings, config management, logging, visualization

## Fraud Ring Types Supported

| Type | Description | Pattern |
|------|-------------|---------|
| `card_testing` | Rapid small-amount transactions testing stolen cards | Burst of $0.50-$5.00 txns to same merchant |
| `account_takeover` | Geo/device change + high-value purchases | Reconnaissance → exploitation phases |
| `money_laundering` | Structuring (below threshold) + layering (circular) | Amounts ~$9,900, account-to-account cycles |
| `refund_abuse` | Legitimate purchase → immediate fraudulent refund | Purchase + negative-amount refund pairs |
| `collusion` | Coordinated cliques between account groups | Dense subgraphs with synchronized timing |

## Installation

```bash
# Clone the repository
git clone https://github.com/kaushikpavani/graph-transformer-fraud-detector.git
cd graph-transformer-fraud-detector

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[dev,viz]"
```

### Requirements

- Python 3.10+
- PyTorch 2.3+
- PyTorch Geometric 2.5+
- See `requirements.txt` for full pinned versions

## Quickstart

```python
from gtfd import GTFDConfig, SyntheticTransactionGenerator, PaymentGraphBuilder
from gtfd.model.gtfd import GTFDModel
from gtfd.pretraining.trainer import PretrainingTrainer
from gtfd.conformal.calibration import AdaptivePredictionSets
import torch

# 1. Configure
config = GTFDConfig()
config.data.num_accounts = 5000
config.data.num_transactions = 100000
config.data.fraud_rate = 0.02

# 2. Generate synthetic data with fraud rings
generator = SyntheticTransactionGenerator(config.data)
df, rings = generator.generate_dataset()

# 3. Build payment graph
builder = PaymentGraphBuilder()
data = builder.build_graph(df)

# 4. Initialize model
model = GTFDModel(config.model)
model.add_fraud_head(num_classes=2)

# 5. Self-supervised pretraining
trainer = PretrainingTrainer(model, config.model.pretraining, device="cuda")
trainer.train(dataloader)  # Your graph DataLoader

# 6. Conformal calibration
calibrator = AdaptivePredictionSets(alpha=0.1)
calibrator.calibrate(logits, labels)

# 7. Get calibrated predictions
pred_sets = calibrator.predict_set(test_logits)
anomaly_scores = calibrator.predict_anomaly_score(test_logits)
```

See `examples/quickstart.py` for a complete runnable example.

## CLI Scripts

```bash
# Generate synthetic data
python scripts/generate_data.py --n-accounts 5000 --n-transactions 100000 --output data/train.pt

# Self-supervised pretraining
python scripts/pretrain.py --config configs/pretrain.yaml --data data/train.pt --output checkpoints/pretrained.pt

# Conformal calibration
python scripts/calibrate.py --model checkpoints/pretrained.pt --data data/cal.pt --method aps

# Evaluate
python scripts/evaluate.py --model checkpoints/model.pt --calibrator checkpoints/calibrated.pt --data data/test.pt

# Adversarial stress testing
python scripts/stress_test.py --model checkpoints/model.pt --data data/test.pt --epsilons 0.0 0.1 0.2
```

## Configuration

All settings via YAML config files or Python:

```yaml
# configs/default.yaml
model:
  hidden_dim: 256
  graph_encoder:
    num_layers: 4
    num_heads: 8
  temporal_encoder:
    num_layers: 3
    use_rope: true
  fusion:
    gate_type: "sigmoid"
  pretraining:
    epochs: 50
    lambda_link: 1.0
    lambda_adv: 0.5
  conformal:
    method: "aps"
    alpha: 0.1
```

```python
from gtfd.config import GTFDConfig
config = GTFDConfig.from_yaml("configs/default.yaml")
# Or override programmatically
config.model.pretraining.epochs = 100
```

## Project Structure

```
graph-transformer-fraud-detector/
├── src/gtfd/
│   ├── config.py              # Pydantic configuration classes
│   ├── graph/
│   │   ├── builder.py         # Payment graph construction
│   │   ├── dataset.py         # PyG Dataset/DataLoader
│   │   └── augment.py         # Graph augmentation & adversarial
│   ├── encoder/
│   │   ├── graph_encoder.py   # Multi-head GAT
│   │   ├── temporal_encoder.py # Gated Transformer
│   │   └── fusion.py          # Cross-modal fusion
│   ├── pretraining/
│   │   ├── objectives.py      # Link mask, adversarial, contrastive losses
│   │   └── trainer.py         # Pretraining loop
│   ├── conformal/
│   │   ├── calibration.py     # Split/APS/RAPS conformal
│   │   └── scores.py          # Non-conformity scores
│   ├── model/
│   │   ├── gtfd.py            # Main GTFD model
│   │   └── heads.py           # Downstream heads
│   ├── data/
│   │   ├── generator.py       # Synthetic data with fraud rings
│   │   └── datasets.py        # Real dataset loaders (stubs)
│   ├── evaluation/
│   │   ├── metrics.py         # AUROC, AUPRC, ring recall, etc.
│   │   ├── adversarial.py     # PGD, edge dropout evaluation
│   │   └── visualization.py   # Calibration curves, t-SNE, attention
│   └── utils/
│       ├── logging.py         # Loguru setup
│       └── seed.py            # Reproducibility
├── tests/                     # Unit & integration tests
├── scripts/                   # CLI entry points
├── configs/                   # YAML configurations
├── examples/                  # Quickstart example
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Extending GTFD

### Custom Fraud Ring

```python
from gtfd.data.generator import FraudRingGenerator

class CustomRingGenerator(FraudRingGenerator):
    def generate_my_fraud_ring(self, accounts, merchants, start_time):
        # Implement your pattern
        pass
```

### Custom Conformal Score

```python
from gtfd.conformal.calibration import NonConformityScorer

class MyScore(NonConformityScorer):
    def __call__(self, logits, labels=None):
        # Your scoring logic
        return scores
```

### Custom Encoder

```python
from gtfd.encoder.graph_encoder import GraphAttentionEncoder

class MyEncoder(GraphAttentionEncoder):
    def forward(self, x, edge_index, edge_attr=None):
        # Your custom message passing
        pass
```

## Evaluation Metrics

- **Standard**: AUROC, AUPRC, F1, Precision@K, Recall@K
- **Fraud-Specific**: Ring-level recall (fraction of fraud rings with ≥1 detected transaction)
- **Conformal**: Coverage, average set size, singleton rate
- **Robustness**: Accuracy/AUROC vs. adversarial perturbation magnitude (AURC)

## References

- **GTFD**: Graph-Transformer Fraud Detection with Self-Supervised Pretraining and Conformal Risk Control (arXiv:2609.14234)
- **Conformal Prediction**: Vovk et al., "Algorithmic Learning in a Random World" (2005)
- **APS**: Romano et al., "Classification with Valid and Adaptive Coverage" (NeurIPS 2020)
- **RAPS**: Angelopoulos et al., "Uncertainty Sets for Image Classifiers using Conformal Prediction" (ICLR 2021)
- **GAT**: Veličković et al., "Graph Attention Networks" (ICLR 2018)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{gtfd2026,
  title={Graph-Transformer Fraud Detector (GTFD): Self-Supervised Pretraining and Conformal Risk Control for Coordinated Fraud Ring Detection},
  author={GTFD Contributors},
  year={2026},
  url={https://github.com/kaushikpavani/graph-transformer-fraud-detector}
}
```