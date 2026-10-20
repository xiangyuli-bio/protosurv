from __future__ import annotations

import argparse

from protosurv.config import ExperimentConfig
from protosurv.data import read_metadata
from protosurv.splits import write_fold_csvs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level stratified CV folds")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    metadata = read_metadata(
        args.metadata,
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    write_fold_csvs(
        metadata,
        args.output_dir,
        n_splits=args.splits,
        validation_fraction=args.validation_fraction,
        seed=cfg.seed if args.seed is None else args.seed,
        patient_column=cfg.patient_column,
        event_column=cfg.event_column,
    )


if __name__ == "__main__":
    main()

