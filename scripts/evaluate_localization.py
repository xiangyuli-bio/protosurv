"""Evaluate patch-aligned prototype attention against independent annotations.

Each prediction file is an ``.npz`` exported by ``export_attention`` and contains
``attention`` with shape ``[num_prototypes_total, N]`` and ``coords`` with shape
``[N, 2]``. The corresponding annotation file contains a boolean ``tumor``
vector aligned to those retained patches and an optional boolean ``tissue``
vector. This contract keeps rasterization and tissue-coverage decisions
explicit upstream of the metric calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_scores(
    path: Path,
    *,
    class_index: int,
    num_prototypes: int,
    score_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if score_key not in archive or "coords" not in archive:
            raise KeyError(f"{path} must contain '{score_key}' and 'coords'")
        raw_scores = np.asarray(archive[score_key], dtype=np.float32)
        coords = np.asarray(archive["coords"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{path} coords must have shape [N, 2]")
    if score_key in {"priors", "class_attention"} and raw_scores.ndim == 2:
        scores = raw_scores[:, class_index]
    elif raw_scores.ndim == 2 and raw_scores.shape[1] == coords.shape[0]:
        start = class_index * num_prototypes
        stop = start + num_prototypes
        if start < 0 or stop > raw_scores.shape[0]:
            raise ValueError(f"{path} does not contain the requested prototype class")
        scores = raw_scores[start:stop].mean(axis=0)
    elif raw_scores.ndim == 2 and raw_scores.shape[0] == coords.shape[0]:
        if not 0 <= class_index < raw_scores.shape[1]:
            raise ValueError(f"{path} does not contain the requested class index")
        scores = raw_scores[:, class_index]
    elif raw_scores.ndim == 1:
        scores = raw_scores
    else:
        raise ValueError(f"{path} scores must have shape [N] or [M, N]")
    if scores.shape[0] != coords.shape[0] or not np.isfinite(scores).all():
        raise ValueError(f"{path} scores and coords have incompatible or non-finite values")
    return scores, coords


def _load_annotation(path: Path, n_patches: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        if "tumor" not in archive:
            raise KeyError(f"{path} must contain a patch-aligned 'tumor' vector")
        tumor = np.asarray(archive["tumor"], dtype=np.float64)
        area = np.asarray(archive["patch_area"], dtype=np.float64) if "patch_area" in archive else np.ones(n_patches)
        tissue = (
            np.asarray(archive["tissue"], dtype=bool)
            if "tissue" in archive
            else np.ones(n_patches, dtype=bool)
        )
    if tumor.ndim != 1 or tissue.ndim != 1 or tumor.shape[0] != n_patches:
        raise ValueError(f"{path} annotation vectors must have length {n_patches}")
    if tissue.shape[0] != n_patches:
        raise ValueError(f"{path} tissue vector must have length {n_patches}")
    if not np.isfinite(tumor).all() or np.any((tumor < 0) | (tumor > 1)):
        raise ValueError("tumor must contain area fractions between zero and one")
    if area.shape != (n_patches,) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("patch_area must contain a positive finite area per patch")
    tumor = tumor * tissue
    return tumor, tissue, area


def _normalise_scores(scores: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    values = scores[tissue]
    if values.size == 0:
        raise ValueError("tissue mask contains no retained patches")
    lower, upper = float(values.min()), float(values.max())
    normalised = np.zeros_like(scores, dtype=np.float32)
    if upper > lower:
        normalised[tissue] = (values - lower) / (upper - lower)
    return normalised


def _slide_metrics(scores: np.ndarray, tumor: np.ndarray, tissue: np.ndarray, threshold: float, area=None):
    area = np.ones(len(scores)) if area is None else area
    predicted = (scores >= threshold) & tissue
    positive = bool(tumor.any())
    result: dict[str, float | bool] = {"positive": positive}
    if positive:
        intersection = float(np.sum(area * predicted * tumor))
        union = float(np.sum(area * predicted) + np.sum(area * tumor) - intersection)
        result["iou"] = float(intersection / union) if union else 0.0
        ranked = np.flatnonzero(tissue)[np.argsort(scores[tissue])[::-1]]
        budget = 0.20 * np.sum(area[tissue])
        selected_area = np.clip(budget - np.r_[0.0, np.cumsum(area[ranked])[:-1]], 0.0, area[ranked])
        result["precision_at_20"] = float(np.sum(selected_area * tumor[ranked]) / budget)
    else:
        result["predicted_positive_fraction"] = float(
            np.sum(area[predicted]) / np.sum(area[tissue])
        )
    return result


def _evaluate_directory(
    predictions: Path,
    annotations: Path,
    *,
    class_index: int,
    num_prototypes: int,
    score_key: str,
    threshold: float,
) -> list[dict[str, float | bool | str]]:
    rows = []
    prediction_files = sorted(predictions.glob("*.npz"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction .npz files found in {predictions}")
    for prediction_path in prediction_files:
        annotation_path = annotations / prediction_path.name
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file for {prediction_path.name}")
        raw_scores, prediction_coords = _load_scores(
            prediction_path,
            class_index=class_index,
            num_prototypes=num_prototypes,
            score_key=score_key,
        )
        tumor, tissue, area = _load_annotation(annotation_path, raw_scores.shape[0])
        with np.load(annotation_path) as reference:
            if "coords" in reference and not np.array_equal(reference["coords"], prediction_coords):
                raise ValueError(f"Reference patch coordinates/order differ for {prediction_path.stem}")
        scores = raw_scores if score_key == "class_attention" else _normalise_scores(raw_scores, tissue)
        row = _slide_metrics(scores, tumor, tissue, threshold, area)
        row["slide_id"] = prediction_path.stem
        with np.load(annotation_path) as reference:
            row["patient_id"] = str(reference["patient_id"].item()) if "patient_id" in reference else prediction_path.stem
        rows.append(row)
    return rows


def _select_threshold(
    predictions: Path,
    annotations: Path,
    *,
    class_index: int,
    num_prototypes: int,
    score_key: str,
) -> float:
    best_threshold, best_score = 0.05, -np.inf
    for threshold in np.arange(0.05, 0.951, 0.05):
        rows = _evaluate_directory(
            predictions,
            annotations,
            class_index=class_index,
            num_prototypes=num_prototypes,
            score_key=score_key,
            threshold=float(threshold),
        )
        positive_iou = [row["iou"] for row in rows if row["positive"]]
        score = float(np.mean(positive_iou)) if positive_iou else -np.inf
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_threshold, best_score = float(threshold), score
    if not np.isfinite(best_score):
        raise ValueError("Calibration annotations contain no tumor-positive slides")
    return best_threshold


def _bootstrap_interval(rows, metric, repeats=1000, seed=1):
    """Cluster on supplied patient IDs; otherwise use explicitly slide-level resampling."""
    eligible = [row for row in rows if metric in row]
    if not eligible:
        return None
    groups = {}
    for row in eligible:
        groups.setdefault(row["patient_id"], []).append(row[metric])
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    samples = [np.mean([v for i in rng.integers(0, len(values), len(values)) for v in values[i]])
               for _ in range(repeats)]
    return np.percentile(samples, [2.5, 97.5]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate patch-aligned prototype attention against tumor annotations"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--calibration-predictions", default=None)
    parser.add_argument("--calibration-annotations", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--class-index", type=int, default=0)
    parser.add_argument("--num-prototypes", type=int, default=8)
    parser.add_argument("--score-key", default="class_attention")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap count must be positive")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie strictly between zero and one")
    if args.num_prototypes <= 0 or args.class_index < 0:
        raise ValueError("class-index must be non-negative and num-prototypes must be positive")
    if args.threshold is None:
        if not args.calibration_predictions or not args.calibration_annotations:
            raise ValueError(
                "provide --threshold or both calibration prediction and annotation directories"
            )
        if Path(args.calibration_predictions).resolve() == Path(args.predictions).resolve():
            raise ValueError("Calibration and test prediction directories must be distinct")
        threshold = _select_threshold(
            Path(args.calibration_predictions),
            Path(args.calibration_annotations),
            class_index=args.class_index,
            num_prototypes=args.num_prototypes,
            score_key=args.score_key,
        )
    else:
        threshold = args.threshold
    rows = _evaluate_directory(
        Path(args.predictions),
        Path(args.annotations),
        class_index=args.class_index,
        num_prototypes=args.num_prototypes,
        score_key=args.score_key,
        threshold=threshold,
    )
    positive = [row for row in rows if row["positive"]]
    negative = [row for row in rows if not row["positive"]]
    result = {
        "threshold": threshold,
        "n_slides": len(rows),
        "n_positive_slides": len(positive),
        "n_negative_slides": len(negative),
        "macro_iou": float(np.mean([row["iou"] for row in positive])) if positive else None,
        "precision_at_20": (
            float(np.mean([row["precision_at_20"] for row in positive])) if positive else None
        ),
        "negative_slide_positive_fraction": (
            float(np.mean([row["predicted_positive_fraction"] for row in negative]))
            if negative
            else None
        ),
        "slides": rows,
        "bootstrap_95ci": {metric: _bootstrap_interval(rows, metric, args.bootstrap, args.seed)
                            for metric in ("iou", "precision_at_20", "predicted_positive_fraction")},
        "resampling_unit": "patient_id if provided in annotations; otherwise slide_id",
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
