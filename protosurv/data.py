"""Patient-level metadata and slide feature loading.

The loader deliberately accepts a small, documented HDF5 contract. It does
not depend on a particular WSI vendor or on PyTorch Geometric serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SlideRecord:
    patient_id: str
    slide_id: str
    survival_time: float
    event: int
    feature_path: Path


@dataclass
class SlideData:
    slide_id: str
    patient_id: str
    features: np.ndarray
    coords: np.ndarray
    priors: np.ndarray
    edge_index: np.ndarray | None = None


@dataclass
class PatientRecord:
    patient_id: str
    survival_time: float
    event: int
    slides: list[SlideRecord]


def read_metadata(
    csv_path: str | Path,
    *,
    patient_column: str = "patient_id",
    slide_column: str = "slide_id",
    time_column: str = "survival_time",
    event_column: str = "event",
    feature_path_column: str = "feature_path",
) -> pd.DataFrame:
    """Read and validate one row per slide from a metadata CSV."""

    frame = pd.read_csv(csv_path, dtype={patient_column: str, slide_column: str})
    required = [patient_column, slide_column, time_column, event_column, feature_path_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
    if frame[patient_column].isna().any() or frame[slide_column].isna().any():
        raise ValueError("patient_id and slide_id must not contain missing values")
    if frame[slide_column].duplicated().any():
        raise ValueError("slide_id must be unique within a cohort")
    if frame[feature_path_column].isna().any():
        raise ValueError("feature_path must not contain missing values")
    frame = frame.copy()
    frame[patient_column] = frame[patient_column].astype(str)
    frame[slide_column] = frame[slide_column].astype(str)
    metadata_root = Path(csv_path).expanduser().resolve().parent
    def resolve_feature_path(value: str) -> str:
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else (metadata_root / path).resolve())

    frame[feature_path_column] = frame[feature_path_column].astype(str).map(resolve_feature_path)
    frame[time_column] = pd.to_numeric(frame[time_column], errors="raise").astype(float)
    numeric_events = pd.to_numeric(frame[event_column], errors="raise")
    if not numeric_events.isin([0, 1]).all():
        raise ValueError("event must be encoded as 0 (censored) or 1 (observed event)")
    frame[event_column] = numeric_events.astype(int)
    if not np.isfinite(frame[time_column]).all() or (frame[time_column] <= 0).any():
        raise ValueError("survival_time must be positive")
    return frame


def to_slide_records(frame: pd.DataFrame, **columns: str) -> list[SlideRecord]:
    patient_column = columns.get("patient_column", "patient_id")
    slide_column = columns.get("slide_column", "slide_id")
    time_column = columns.get("time_column", "survival_time")
    event_column = columns.get("event_column", "event")
    feature_path_column = columns.get("feature_path_column", "feature_path")
    return [
        SlideRecord(
            patient_id=str(row[patient_column]),
            slide_id=str(row[slide_column]),
            survival_time=float(row[time_column]),
            event=int(row[event_column]),
            feature_path=Path(str(row[feature_path_column])),
        )
        for _, row in frame.iterrows()
    ]


def group_patients(records: Iterable[SlideRecord]) -> list[PatientRecord]:
    grouped: dict[str, list[SlideRecord]] = {}
    for record in records:
        grouped.setdefault(record.patient_id, []).append(record)
    patients: list[PatientRecord] = []
    for patient_id, slides in sorted(grouped.items()):
        times = {record.survival_time for record in slides}
        events = {record.event for record in slides}
        if len(times) != 1 or len(events) != 1:
            raise ValueError(f"Patient {patient_id} has inconsistent survival labels across slides")
        patients.append(PatientRecord(patient_id, times.pop(), events.pop(), slides))
    return patients


def _read_hdf5(path: Path, feature_key: str, coord_key: str, prior_key: str, graph_key: str):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read slide feature files") from exc
    with h5py.File(path, "r") as handle:
        for key in (feature_key, coord_key, prior_key):
            if key not in handle:
                raise KeyError(f"{path} does not contain required dataset '{key}'")
        features = np.asarray(handle[feature_key], dtype=np.float32)
        coords = np.asarray(handle[coord_key], dtype=np.float32)
        priors = np.asarray(handle[prior_key], dtype=np.float32)
        edge_index = np.asarray(handle[graph_key], dtype=np.int64) if graph_key in handle else None
    return features, coords, priors, edge_index


def load_slide(
    record: SlideRecord,
    *,
    feature_key: str = "features",
    coord_key: str = "coords",
    prior_key: str = "priors",
    graph_key: str = "edge_index",
    num_classes: int = 5,
) -> SlideData:
    features, coords, priors, edge_index = _read_hdf5(
        record.feature_path, feature_key, coord_key, prior_key, graph_key
    )
    if features.ndim != 2 or coords.ndim != 2 or priors.ndim != 2:
        raise ValueError(f"Slide {record.slide_id} arrays must all be two-dimensional")
    if features.shape[0] != coords.shape[0] or features.shape[0] != priors.shape[0]:
        raise ValueError(f"Slide {record.slide_id} has inconsistent patch counts")
    if not len(features):
        raise ValueError(f"Slide {record.slide_id} has no retained tissue patches")
    if coords.shape[1] != 2:
        raise ValueError(f"Slide {record.slide_id} coordinates must have shape [N, 2]")
    if priors.shape[1] != num_classes:
        raise ValueError(
            f"Slide {record.slide_id} priors must have {num_classes} classes, got {priors.shape[1]}"
        )
    if not np.isfinite(features).all() or not np.isfinite(coords).all() or not np.isfinite(priors).all():
        raise ValueError(f"Slide {record.slide_id} contains non-finite values")
    if np.any(priors < 0):
        raise ValueError(f"Slide {record.slide_id} contains negative prior values")
    row_sums = priors.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError(f"Slide {record.slide_id} contains an all-zero prior row")
    priors = priors / row_sums
    if edge_index is not None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Slide {record.slide_id} edge_index must have shape [2, E]")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= features.shape[0]):
            raise ValueError(f"Slide {record.slide_id} edge_index contains an invalid node index")
    return SlideData(record.slide_id, record.patient_id, features, coords, priors, edge_index)
