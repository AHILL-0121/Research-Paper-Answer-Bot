"""Small helpers shared across the pipeline."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so experiment runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
