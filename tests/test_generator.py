"""Tests for data generator."""

import pytest
import pandas as pd

from gtfd.data.generator import SyntheticTransactionGenerator, FraudRingGenerator, FraudRing
from gtfd.config import DataConfig


class TestSyntheticTransactionGenerator:
    """Tests for SyntheticTransactionGenerator."""

    @pytest.fixture
    def config(self):
        return DataConfig(
            num_accounts=100,
            num_merchants=20,
            num_transactions=2000,
            fraud_rate=0.05,
            ring_types=["card_testing", "account_takeover"],
            temporal_span_days=30,
            seed=42,
        )

    @pytest.fixture
    def generator(self, config):
        return SyntheticTransactionGenerator(config)

    def test_generate_accounts(self, generator):
        """Test account ID generation."""
        accounts = generator.generate_accounts(10)
        assert len(accounts) == 10
        assert all(a.startswith("ACC_") for a in accounts)
        assert len(set(accounts)) == 10  # All unique

    def test_generate_merchants(self, generator):
        """Test merchant ID generation."""
        merchants = generator.generate_merchants(10)
        assert len(merchants) == 10
        assert all(m.startswith("MER_") for m in merchants)
        assert len(set(merchants)) == 10

    def test_generate_legitimate_transactions(self, generator):
        """Test legitimate transaction generation."""
        accounts = generator.generate_accounts(10)
        merchants = generator.generate_merchants(5)
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        txns = generator.generate_legitimate_transactions(accounts, merchants, start, end, 100)

        assert len(txns) == 100
        for tx in txns:
            assert tx["is_fraud"] == 0
            assert tx["ring_id"] == -1
            assert tx["ring_type"] == "legitimate"
            assert tx["account_id"] in accounts
            assert tx["merchant_id"] in merchants or tx["merchant_id"] == "EMPLOYER_PAYROLL"

    def test_generate_dataset(self, generator):
        """Test full dataset generation."""
        df, rings = generator.generate_dataset()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "account_id" in df.columns
        assert "merchant_id" in df.columns
        assert "amount" in df.columns
        assert "timestamp" in df.columns
        assert "tx_type" in df.columns
        assert "is_fraud" in df.columns
        assert "ring_id" in df.columns
        assert "ring_type" in df.columns

        # Check fraud rate approximately matches
        fraud_rate = (df["is_fraud"] == 1).mean()
        assert abs(fraud_rate - 0.05) < 0.03  # Within 3% tolerance

        # Check rings generated
        assert len(rings) > 0
        for ring in rings:
            assert isinstance(ring, FraudRing)
            assert ring.ring_type in ["card_testing", "account_takeover"]

    def test_reproducibility(self, config):
        """Test deterministic generation with same seed."""
        gen1 = SyntheticTransactionGenerator(config)
        df1, rings1 = gen1.generate_dataset()

        gen2 = SyntheticTransactionGenerator(config)
        df2, rings2 = gen2.generate_dataset()

        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
        assert len(rings1) == len(rings2)


class TestFraudRingGenerator:
    """Tests for FraudRingGenerator."""

    @pytest.fixture
    def config(self):
        return DataConfig(
            num_accounts=100,
            num_merchants=20,
            ring_size_range=(3, 5),
        )

    @pytest.fixture
    def ring_gen(self, config):
        return FraudRingGenerator(config)

    def test_card_testing_ring(self, ring_gen):
        """Test card testing ring generation."""
        accounts = [f"ACC_{i:03d}" for i in range(20)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        ring = ring_gen.generate_card_testing_ring(accounts, merchants, start_time)

        assert ring.ring_type == "card_testing"
        assert len(ring.account_ids) >= 3
        assert len(ring.merchant_ids) == 1
        assert len(ring.transactions) > 0
        for tx in ring.transactions:
            assert tx["is_fraud"] == 1
            assert tx["ring_type"] == "card_testing"
            assert tx["amount"] < 10  # Small amounts

    def test_account_takeover_ring(self, ring_gen):
        """Test account takeover ring generation."""
        accounts = [f"ACC_{i:03d}" for i in range(10)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        ring = ring_gen.generate_account_takeover_ring(accounts, merchants, start_time)

        assert ring.ring_type == "account_takeover"
        assert len(ring.account_ids) >= 1
        assert len(ring.transactions) > 0
        # Should have both small and large transactions
        amounts = [tx["amount"] for tx in ring.transactions]
        assert min(amounts) < 100
        assert max(amounts) > 200

    def test_money_laundering_ring(self, ring_gen):
        """Test money laundering ring generation."""
        accounts = [f"ACC_{i:03d}" for i in range(10)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        ring = ring_gen.generate_money_laundering_ring(accounts, merchants, start_time)

        assert ring.ring_type == "money_laundering"
        assert len(ring.account_ids) >= 3
        # Transactions should be account-to-account (merchant_id == account_id)
        for tx in ring.transactions:
            assert tx["tx_type"] == "transfer"
            assert tx["merchant_id"] in ring.account_ids
            assert 7000 <= tx["amount"] <= 9999  # Structuring near threshold

    def test_refund_abuse_ring(self, ring_gen):
        """Test refund abuse ring generation."""
        accounts = [f"ACC_{i:03d}" for i in range(10)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        ring = ring_gen.generate_refund_abuse_ring(accounts, merchants, start_time)

        assert ring.ring_type == "refund_abuse"
        # Should have purchase + refund pairs
        purchases = [tx for tx in ring.transactions if tx["tx_type"] == "purchase"]
        refunds = [tx for tx in ring.transactions if tx["tx_type"] == "refund"]
        assert len(purchases) == len(refunds)
        for p, r in zip(purchases, refunds):
            assert p["amount"] == -r["amount"]  # Refund matches purchase

    def test_collusion_ring(self, ring_gen):
        """Test collusion ring generation."""
        accounts = [f"ACC_{i:03d}" for i in range(10)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        ring = ring_gen.generate_collusion_ring(accounts, merchants, start_time)

        assert ring.ring_type == "collusion"
        assert len(ring.transactions) >= 50
        # Should involve multiple accounts and merchants
        unique_accounts = set(tx["account_id"] for tx in ring.transactions)
        unique_merchants = set(tx["merchant_id"] for tx in ring.transactions)
        assert len(unique_accounts) >= 3
        assert len(unique_merchants) >= 2

    def test_generate_ring_dispatch(self, ring_gen):
        """Test generate_ring dispatches correctly."""
        accounts = [f"ACC_{i:03d}" for i in range(10)]
        merchants = [f"MER_{i:03d}" for i in range(5)]
        start_time = pd.Timestamp("2024-01-01")

        for ring_type in ["card_testing", "account_takeover", "money_laundering", "refund_abuse", "collusion"]:
            ring = ring_gen.generate_ring(ring_type, accounts=accounts, merchants=merchants, start_time=start_time)
            assert ring.ring_type == ring_type

        # Invalid type should raise
        with pytest.raises(ValueError):
            ring_gen.generate_ring("invalid_type", accounts=accounts, merchants=merchants, start_time=start_time)