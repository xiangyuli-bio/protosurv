"""Deterministic seed and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def resolve_device(requested: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training") from exc
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

