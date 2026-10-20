from __future__ import annotations

import argparse

from protosurv.training import train_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the prior-guided model with patient-level CV")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Directory containing fixed fold_0.csv ... fold_{splits-1}.csv files",
    )
    args = parser.parse_args()
    train_cross_validation(
        args.metadata,
        args.config,
        args.output_dir,
        n_splits=args.splits,
        splits_dir=args.splits_dir,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
