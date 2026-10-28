"""Synthetic data generator for corporate transaction graphs with fraud rings."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from ..config import DataConfig


@dataclass
class FraudRing:
    """Represents a coordinated fraud ring."""

    ring_id: int
    ring_type: str
    account_ids: list[str]
    merchant_ids: list[str]
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    transactions: list[dict[str, Any]]
    metadata: dict[str, Any]


class FraudRingGenerator:
    """Generates coordinated fraud ring patterns."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.ring_counter = 0

    def generate_card_testing_ring(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        duration_hours: int = 24,
    ) -> FraudRing:
        """Generate card testing fraud ring.

        Pattern: Rapid small-amount transactions from compromised cards
        to the same merchant (testing if cards work).
        """
        self.ring_counter += 1
        ring_size = random.randint(*self.config.ring_size_range)
        ring_accounts = random.sample(accounts, min(ring_size, len(accounts)))
        target_merchant = random.choice(merchants)

        transactions = []
        current_time = start_time
        for acc in ring_accounts:
            # 5-20 small transactions per card
            n_tx = random.randint(5, 20)
            for _ in range(n_tx):
                amount = round(random.uniform(0.5, 5.0), 2)  # Small amounts
                tx = {
                    "account_id": acc,
                    "merchant_id": target_merchant,
                    "amount": amount,
                    "timestamp": current_time,
                    "tx_type": "purchase",
                    "is_fraud": 1,
                    "ring_id": self.ring_counter,
                    "ring_type": "card_testing",
                }
                transactions.append(tx)
                current_time += pd.Timedelta(minutes=random.randint(1, 30))

        return FraudRing(
            ring_id=self.ring_counter,
            ring_type="card_testing",
            account_ids=ring_accounts,
            merchant_ids=[target_merchant],
            start_time=start_time,
            end_time=current_time,
            transactions=transactions,
            metadata={"target_merchant": target_merchant, "avg_amount": np.mean([t["amount"] for t in transactions])},
        )

    def generate_account_takeover_ring(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        duration_hours: int = 48,
    ) -> FraudRing:
        """Generate account takeover fraud ring.

        Pattern: Sudden location/device change followed by high-value purchases.
        """
        self.ring_counter += 1
        ring_size = random.randint(1, min(5, len(accounts)))  # Usually single account
        ring_accounts = random.sample(accounts, ring_size)

        transactions = []
        current_time = start_time
        for acc in ring_accounts:
            # First: small transactions from new location (reconnaissance)
            for _ in range(random.randint(2, 5)):
                amount = round(random.uniform(10, 50), 2)
                tx = {
                    "account_id": acc,
                    "merchant_id": random.choice(merchants),
                    "amount": amount,
                    "timestamp": current_time,
                    "tx_type": "purchase",
                    "is_fraud": 1,
                    "ring_id": self.ring_counter,
                    "ring_type": "account_takeover",
                }
                transactions.append(tx)
                current_time += pd.Timedelta(hours=random.randint(1, 6))

            # Then: high-value purchases
            for _ in range(random.randint(3, 10)):
                amount = round(random.uniform(200, 2000), 2)
                tx = {
                    "account_id": acc,
                    "merchant_id": random.choice(merchants),
                    "amount": amount,
                    "timestamp": current_time,
                    "tx_type": "purchase",
                    "is_fraud": 1,
                    "ring_id": self.ring_counter,
                    "ring_type": "account_takeover",
                }
                transactions.append(tx)
                current_time += pd.Timedelta(hours=random.randint(1, 12))

        return FraudRing(
            ring_id=self.ring_counter,
            ring_type="account_takeover",
            account_ids=ring_accounts,
            merchant_ids=[t["merchant_id"] for t in transactions],
            start_time=start_time,
            end_time=current_time,
            transactions=transactions,
            metadata={"phases": ["reconnaissance", "exploitation"]},
        )

    def generate_money_laundering_ring(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        duration_hours: int = 168,  # 1 week
    ) -> FraudRing:
        """Generate money laundering (structuring/layering) ring.

        Pattern: Amounts just below reporting thresholds, circular flows.
        """
        self.ring_counter += 1
        ring_size = random.randint(*self.config.ring_size_range)
        ring_accounts = random.sample(accounts, min(ring_size, len(accounts)))

        transactions = []
        current_time = start_time
        threshold = 10000  # Reporting threshold

        # Structuring: many transactions just below threshold
        for i, acc in enumerate(ring_accounts):
            n_tx = random.randint(10, 30)
            for _ in range(n_tx):
                # Amounts just below threshold (structuring)
                amount = round(random.uniform(threshold * 0.7, threshold * 0.99), 2)
                # Layering: send to next account in ring
                dst_acc = ring_accounts[(i + 1) % len(ring_accounts)]
                tx = {
                    "account_id": acc,
                    "merchant_id": dst_acc,  # Account-to-account
                    "amount": amount,
                    "timestamp": current_time,
                    "tx_type": "transfer",
                    "is_fraud": 1,
                    "ring_id": self.ring_counter,
                    "ring_type": "money_laundering",
                }
                transactions.append(tx)
                current_time += pd.Timedelta(hours=random.randint(1, 24))

        return FraudRing(
            ring_id=self.ring_counter,
            ring_type="money_laundering",
            account_ids=ring_accounts,
            merchant_ids=ring_accounts,  # Account-to-account
            start_time=start_time,
            end_time=current_time,
            transactions=transactions,
            metadata={"threshold": threshold, "pattern": "structuring+layering"},
        )

    def generate_refund_abuse_ring(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        duration_hours: int = 72,
    ) -> FraudRing:
        """Generate refund abuse ring.

        Pattern: Legitimate purchase followed by immediate refund request.
        """
        self.ring_counter += 1
        ring_size = random.randint(2, min(5, len(accounts)))
        ring_accounts = random.sample(accounts, ring_size)

        transactions = []
        current_time = start_time
        for acc in ring_accounts:
            merchant = random.choice(merchants)
            # Legitimate purchase
            purchase_amount = round(random.uniform(100, 500), 2)
            tx_purchase = {
                "account_id": acc,
                "merchant_id": merchant,
                "amount": purchase_amount,
                "timestamp": current_time,
                "tx_type": "purchase",
                "is_fraud": 0,  # Purchase itself is legit
                "ring_id": self.ring_counter,
                "ring_type": "refund_abuse",
            }
            transactions.append(tx_purchase)
            current_time += pd.Timedelta(hours=random.randint(1, 24))

            # Refund request (fraudulent)
            tx_refund = {
                "account_id": acc,
                "merchant_id": merchant,
                "amount": -purchase_amount,  # Negative for refund
                "timestamp": current_time,
                "tx_type": "refund",
                "is_fraud": 1,
                "ring_id": self.ring_counter,
                "ring_type": "refund_abuse",
            }
            transactions.append(tx_refund)
            current_time += pd.Timedelta(hours=random.randint(1, 48))

        return FraudRing(
            ring_id=self.ring_counter,
            ring_type="refund_abuse",
            account_ids=ring_accounts,
            merchant_ids=[t["merchant_id"] for t in transactions if t["tx_type"] == "purchase"],
            start_time=start_time,
            end_time=current_time,
            transactions=transactions,
            metadata={"pattern": "purchase_then_refund"},
        )

    def generate_collusion_ring(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        duration_hours: int = 168,
    ) -> FraudRing:
        """Generate collusion ring (coordinated group).

        Pattern: Coordinated cycles between account groups (k-cliques).
        """
        self.ring_counter += 1
        ring_size = random.randint(*self.config.ring_size_range)
        ring_accounts = random.sample(accounts, min(ring_size, len(accounts)))
        ring_merchants = random.sample(merchants, min(3, len(merchants)))

        transactions = []
        current_time = start_time

        # Create dense subgraph of transactions
        for _ in range(random.randint(50, 200)):
            src = random.choice(ring_accounts)
            dst = random.choice(ring_merchants)
            amount = round(random.uniform(20, 500), 2)
            # Coordinated timing: bursts of activity
            tx = {
                "account_id": src,
                "merchant_id": dst,
                "amount": amount,
                "timestamp": current_time,
                "tx_type": "purchase",
                "is_fraud": 1,
                "ring_id": self.ring_counter,
                "ring_type": "collusion",
            }
            transactions.append(tx)
            current_time += pd.Timedelta(minutes=random.randint(5, 120))

        return FraudRing(
            ring_id=self.ring_counter,
            ring_type="collusion",
            account_ids=ring_accounts,
            merchant_ids=ring_merchants,
            start_time=start_time,
            end_time=current_time,
            transactions=transactions,
            metadata={"pattern": "coordinated_clique", "density": len(transactions) / (ring_size * len(ring_merchants))},
        )

    def generate_ring(self, ring_type: str, **kwargs) -> FraudRing:
        """Generate a fraud ring of specified type."""
        generators = {
            "card_testing": self.generate_card_testing_ring,
            "account_takeover": self.generate_account_takeover_ring,
            "money_laundering": self.generate_money_laundering_ring,
            "refund_abuse": self.generate_refund_abuse_ring,
            "collusion": self.generate_collusion_ring,
        }
        if ring_type not in generators:
            raise ValueError(f"Unknown ring type: {ring_type}")
        return generators[ring_type](**kwargs)


class SyntheticTransactionGenerator:
    """Generates synthetic corporate transaction graphs with fraud rings."""

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize the generator.

        Args:
            config: Data generation configuration.
        """
        self.config = config or DataConfig()
        self.rng = np.random.RandomState(self.config.seed)
        random.seed(self.config.seed)

    def generate_accounts(self, n: int) -> list[str]:
        """Generate account IDs."""
        return [f"ACC_{i:06d}" for i in range(n)]

    def generate_merchants(self, n: int) -> list[str]:
        """Generate merchant IDs."""
        categories = ["retail", "grocery", "gas", "restaurant", "online", "travel", "utility", "healthcare"]
        return [f"MER_{cat}_{i:04d}" for i, cat in enumerate(self.rng.choice(categories, n, replace=True))]

    def generate_legitimate_transactions(
        self,
        accounts: list[str],
        merchants: list[str],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        n_transactions: int,
    ) -> list[dict[str, Any]]:
        """Generate legitimate transaction patterns."""
        transactions = []

        # Salary deposits (monthly)
        salary_accounts = self.rng.choice(accounts, size=int(len(accounts) * 0.3), replace=False)
        for acc in salary_accounts:
            current = start_time
            while current < end_time:
                amount = round(self.rng.uniform(3000, 15000), 2)
                transactions.append({
                    "account_id": acc,
                    "merchant_id": "EMPLOYER_PAYROLL",
                    "amount": amount,
                    "timestamp": current,
                    "tx_type": "deposit",
                    "is_fraud": 0,
                    "ring_id": -1,
                    "ring_type": "legitimate",
                })
                current += pd.Timedelta(days=30)

        # Recurring payments (rent, utilities, subscriptions)
        recurring_accounts = self.rng.choice(accounts, size=int(len(accounts) * 0.5), replace=False)
        for acc in recurring_accounts:
            n_recurring = self.rng.randint(1, 5)
            for _ in range(n_recurring):
                merchant = self.rng.choice([m for m in merchants if "utility" in m or "online" in m])
                amount = round(self.rng.uniform(20, 300), 2)
                day_of_month = self.rng.randint(1, 28)
                current = start_time.replace(day=day_of_month)
                while current < end_time:
                    transactions.append({
                        "account_id": acc,
                        "merchant_id": merchant,
                        "amount": amount,
                        "timestamp": current,
                        "tx_type": "payment",
                        "is_fraud": 0,
                        "ring_id": -1,
                        "ring_type": "legitimate",
                    })
                    current += pd.Timedelta(days=30)

        # Regular purchases
        remaining = n_transactions - len(transactions)
        for _ in range(max(0, remaining)):
            acc = self.rng.choice(accounts)
            mer = self.rng.choice(merchants)
            amount = round(self.rng.lognormal(3, 1.5), 2)  # Log-normal for realistic amounts
            amount = min(amount, 2000)  # Cap
            timestamp = start_time + pd.Timedelta(
                seconds=self.rng.uniform(0, (end_time - start_time).total_seconds())
            )
            tx_type = self.rng.choice(["purchase", "payment", "withdrawal"], p=[0.7, 0.2, 0.1])
            transactions.append({
                "account_id": acc,
                "merchant_id": mer,
                "amount": amount,
                "timestamp": timestamp,
                "tx_type": tx_type,
                "is_fraud": 0,
                "ring_id": -1,
                "ring_type": "legitimate",
            })

        return transactions

    def generate_dataset(
        self,
        num_accounts: Optional[int] = None,
        num_merchants: Optional[int] = None,
        num_transactions: Optional[int] = None,
        fraud_rate: Optional[float] = None,
        ring_types: Optional[list[str]] = None,
        temporal_span_days: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[pd.DataFrame, list[FraudRing]]:
        """Generate complete synthetic dataset.

        Args:
            num_accounts: Number of accounts (overrides config).
            num_merchants: Number of merchants (overrides config).
            num_transactions: Target number of transactions (overrides config).
            fraud_rate: Target fraud rate (overrides config).
            ring_types: Types of fraud rings to include.
            temporal_span_days: Time span in days.
            seed: Random seed.

        Returns:
            Tuple of (transactions DataFrame, list of FraudRing objects).
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            random.seed(seed)

        n_accounts = num_accounts or self.config.num_accounts
        n_merchants = num_merchants or self.config.num_merchants
        n_transactions = num_transactions or self.config.num_transactions
        f_rate = fraud_rate or self.config.fraud_rate
        types = ring_types or self.config.ring_types
        span_days = temporal_span_days or self.config.temporal_span_days

        accounts = self.generate_accounts(n_accounts)
        merchants = self.generate_merchants(n_merchants)
        employers = ["EMPLOYER_PAYROLL"]  # Special merchant for salary

        all_merchants = merchants + employers

        start_time = pd.Timestamp("2024-01-01")
        end_time = start_time + pd.Timedelta(days=span_days)

        # Generate legitimate transactions
        legit_txns = self.generate_legitimate_transactions(
            accounts, merchants, start_time, end_time, n_transactions
        )

        # Generate fraud rings
        ring_generator = FraudRingGenerator(self.config)
        rings = []
        fraud_transactions = []

        target_fraud_txns = int(n_transactions * f_rate)
        current_fraud_txns = 0

        for ring_type in types:
            while current_fraud_txns < target_fraud_txns:
                ring = ring_generator.generate_ring(
                    ring_type,
                    accounts=accounts,
                    merchants=merchants,
                    start_time=start_time + pd.Timedelta(
                        days=self.rng.uniform(0, span_days * 0.8)
                    ),
                )
                rings.append(ring)
                fraud_transactions.extend(ring.transactions)
                current_fraud_txns = len(fraud_transactions)
                if len(rings) > 100:  # Safety limit
                    break

        # Combine
        all_transactions = legit_txns + fraud_transactions
        df = pd.DataFrame(all_transactions)
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"Generated {len(df)} transactions ({len(legit_txns)} legit, {len(fraud_transactions)} fraud)")
        print(f"Fraud rate: {len(fraud_transactions) / len(df):.4f}")
        print(f"Fraud rings: {len(rings)}")
        for ring in rings:
            print(f"  Ring {ring.ring_id}: {ring.ring_type}, {len(ring.transactions)} txns, "
                  f"{len(ring.account_ids)} accounts")

        return df, rings

    def to_pytorch_geometric(self, df: pd.DataFrame) -> tuple:
        """Convert DataFrame to PyG format (requires graph builder)."""
        from ..graph.builder import PaymentGraphBuilder
        from ..config import GraphConfig

        builder = PaymentGraphBuilder(GraphConfig())
        data = builder.build_graph(df)
        return data