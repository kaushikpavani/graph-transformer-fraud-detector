"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: Whether to enforce deterministic operations (may impact performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch 1.8+
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Worker init function for DataLoader reproducibility."""
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)