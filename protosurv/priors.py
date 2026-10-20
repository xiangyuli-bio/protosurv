"""Deterministic construction of soft histopathologic priors.

The training code consumes patch embeddings and five-class priors. This module
provides the deterministic conversion from image/text embeddings to priors;
encoder loading and image preprocessing remain explicit upstream steps so that
their versions and licensing terms can be recorded with each experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_CLASSES = ("Tumor", "Stroma", "Immune Infiltration", "Necrosis", "Others")


def normalize_class_embeddings(prompt_embeddings: np.ndarray) -> np.ndarray:
    """Average normalized prompt embeddings within each class.

    Input shape is [C, P, D] or [C, D]. Output shape is [C, D] with unit rows.
    """

    values = np.asarray(prompt_embeddings, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError("prompt_embeddings must have shape [C, P, D] or [C, D]")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("prompt embeddings must be non-zero")
    normalized = values / norms
    class_embeddings = normalized.mean(axis=1)
    class_norms = np.linalg.norm(class_embeddings, axis=-1, keepdims=True)
    return class_embeddings / np.maximum(class_norms, 1e-12)


def soft_priors(
    image_embeddings: np.ndarray,
    class_embeddings: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert patch/text cosine similarities into row-normalized priors."""

    images = np.asarray(image_embeddings, dtype=np.float32)
    classes = normalize_class_embeddings(class_embeddings)
    if images.ndim != 2 or images.shape[1] != classes.shape[1]:
        raise ValueError("image_embeddings and class_embeddings have incompatible shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    norms = np.linalg.norm(images, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("image embeddings must be non-zero")
    similarities = (images / norms) @ classes.T
    logits = similarities / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def save_prior_file(path: str | Path, coords: np.ndarray, priors: np.ndarray) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to save prior files") from exc
    coords = np.asarray(coords, dtype=np.float32)
    priors = np.asarray(priors, dtype=np.float32)
    if coords.shape[0] != priors.shape[0]:
        raise ValueError("coords and priors must contain the same number of patches")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=coords, compression="gzip")
        handle.create_dataset("priors", data=priors, compression="gzip")
        handle.attrs["prior_classes"] = np.asarray(DEFAULT_CLASSES, dtype="S")
