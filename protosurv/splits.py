"""Patient-level stratified outer folds with an internal validation subset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

def patient_folds(
    metadata: pd.DataFrame,
    *,
    n_splits: int = 5,
    validation_fraction: float = 0.2,
    seed: int = 1,
    patient_column: str = "patient_id",
    event_column: str = "event",
) -> list[dict[str, np.ndarray]]:
    """Return disjoint patient-id arrays for each outer fold."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    label_counts = metadata.groupby(patient_column)[event_column].nunique()
    if (label_counts > 1).any():
        bad = label_counts[label_counts > 1].index.astype(str).tolist()
        raise ValueError(f"patient labels are inconsistent in metadata: {bad[:5]}")
    patients = metadata[[patient_column, event_column]].drop_duplicates(patient_column)
    try:
        from sklearn.model_selection import StratifiedKFold, train_test_split
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for patient-level splits") from exc
    ids = patients[patient_column].astype(str).to_numpy()
    labels = patients[event_column].astype(int).to_numpy()
    if np.bincount(labels, minlength=2).min() < n_splits:
        raise ValueError("each event stratum must contain at least n_splits patients")
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[dict[str, np.ndarray]] = []
    for fold, (development, test) in enumerate(outer.split(ids, labels)):
        development_labels = labels[development]
        train, validation = train_test_split(
            development,
            test_size=validation_fraction,
            random_state=seed + fold,
            stratify=development_labels,
        )
        folds.append(
            {
                "train": np.sort(ids[train]),
                "validation": np.sort(ids[validation]),
                "test": np.sort(ids[test]),
            }
        )
    return folds


def write_fold_csvs(metadata: pd.DataFrame, output_dir: str | Path, **kwargs) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    folds = patient_folds(metadata, **kwargs)
    for fold_index, split in enumerate(folds):
        rows = []
        for split_name, patient_ids in split.items():
            for patient_id in patient_ids:
                rows.append({"patient_id": patient_id, "split": split_name})
        pd.DataFrame(rows).to_csv(output / f"fold_{fold_index}.csv", index=False)


def load_fold(path: str | Path) -> dict[str, set[str]]:
    frame = pd.read_csv(path)
    required = {"patient_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError("Fold CSV must contain patient_id and split columns")
    result: dict[str, set[str]] = {}
    for split_name, group in frame.groupby("split"):
        result[str(split_name)] = set(group["patient_id"].astype(str))
    if set().union(*result.values()) and sum(map(len, result.values())) != len(
        set().union(*result.values())
    ):
        raise ValueError("Fold CSV contains a patient in more than one split")
    return result
