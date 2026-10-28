#!/usr/bin/env python3
"""CLI script to generate synthetic transaction data."""

import argparse
from pathlib import Path

import pandas as pd

from gtfd.config import DataConfig, GTFDConfig
from gtfd.data.generator import SyntheticTransactionGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data with fraud rings")
    parser.add_argument("--n-accounts", type=int, default=5000, help="Number of accounts")
    parser.add_argument("--n-merchants", type=int, default=500, help="Number of merchants")
    parser.add_argument("--n-transactions", type=int, default=100000, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.02, help="Fraud rate")
    parser.add_argument("--span-days", type=int, default=90, help="Temporal span in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_transactions.parquet"), help="Output file")
    parser.add_argument("--ring-types", nargs="+", default=None, help="Fraud ring types to include")

    args = parser.parse_args()

    config = DataConfig(
        num_accounts=args.n_accounts,
        num_merchants=args.n_merchants,
        num_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        temporal_span_days=args.span_days,
        seed=args.seed,
        ring_types=args.ring_types or [
            "card_testing",
            "account_takeover",
            "money_laundering",
            "refund_abuse",
            "collusion",
        ],
    )

    generator = SyntheticTransactionGenerator(config)
    df, rings = generator.generate_dataset()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} transactions to {args.output}")

    # Also save ring metadata
    ring_df = pd.DataFrame([
        {
            "ring_id": r.ring_id,
            "ring_type": r.ring_type,
            "n_accounts": len(r.account_ids),
            "n_merchants": len(r.merchant_ids),
            "n_transactions": len(r.transactions),
            "start_time": r.start_time,
            "end_time": r.end_time,
        }
        for r in rings
    ])
    ring_path = args.output.with_name(args.output.stem + "_rings.parquet")
    ring_df.to_parquet(ring_path)
    print(f"Saved {len(rings)} fraud rings to {ring_path}")


if __name__ == "__main__":
    main()