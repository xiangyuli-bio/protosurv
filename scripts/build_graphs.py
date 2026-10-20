from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from protosurv.config import ExperimentConfig
from protosurv.data import read_metadata
from protosurv.graph import (
    coordinate_knn_edges,
    coordinate_radius_edges,
    edge_index_to_hdf5,
    physical_coordinates,
    select_degree_matched_radius,
)
from protosurv.splits import load_fold


def _read_coords(path: Path, cfg: ExperimentConfig, pixel_spacing, *, require_spacing: bool):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for graph construction") from exc
    with h5py.File(path, "r") as handle:
        if cfg.coord_key not in handle:
            raise KeyError(f"{path} does not contain '{cfg.coord_key}'")
        coords = np.asarray(handle[cfg.coord_key], dtype=np.float32)
        spacing = pixel_spacing
        if spacing is None:
            for key in ("pixel_spacing", "mpp", "microns_per_pixel"):
                if key in handle.attrs:
                    spacing = np.asarray(handle.attrs[key], dtype=np.float32)
                    break
        if require_spacing and spacing is None:
            raise ValueError(
                f"{path} has no pixel-spacing metadata; pass --pixel-spacing for physical radius matching"
            )
    return physical_coordinates(coords, spacing) if spacing is not None else coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Add coordinate graphs to feature HDF5 files")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph-type", choices=("knn", "radius"), default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Outer fold used for degree matching")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument(
        "--degree-match",
        action="store_true",
        help="Select a physical radius on training slides to match mean degree of kNN",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Fixed fold directory used to identify training slides for --degree-match",
    )
    parser.add_argument("--radius-output", default=None, help="Write selected radius metadata as JSON")
    parser.add_argument(
        "--pixel-spacing",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Pixel spacing used when HDF5 files do not provide physical-spacing metadata",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    frame = read_metadata(
        args.metadata,
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    graph_type = args.graph_type or cfg.graph_type
    radius = args.radius if args.radius is not None else cfg.graph_radius
    if args.degree_match:
        if graph_type != "radius":
            raise ValueError("--degree-match requires --graph-type radius")
        if args.splits_dir is None:
            raise ValueError("--degree-match requires --splits-dir")
        if args.fold is None:
            raise ValueError("--degree-match requires --fold so radius selection is fold-specific")
        fold_path = Path(args.splits_dir) / f"fold_{args.fold}.csv"
        if not fold_path.exists():
            raise FileNotFoundError(fold_path)
        fold = load_fold(fold_path)
        if "train" not in fold:
            raise ValueError(f"{fold_path} has no train split")
        train_ids = fold["train"]
        train_frame = frame[frame[cfg.patient_column].astype(str).isin(train_ids)]
        training_coordinates = [
            _read_coords(
                Path(feature_path),
                cfg,
                args.pixel_spacing,
                require_spacing=True,
            )
            for feature_path in train_frame[cfg.feature_path_column].astype(str).unique()
        ]
        radius = select_degree_matched_radius(training_coordinates, k=args.k or cfg.knn_k)
        if args.radius_output:
            output_path = Path(args.radius_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "radius": radius,
                        "reference_graph": "symmetrized_knn",
                        "reference_k": args.k or cfg.knn_k,
                        "training_slides": len(training_coordinates),
                        "coordinate_units": "physical",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if graph_type == "radius" and radius is None:
        raise ValueError("--radius, graph_radius, or --degree-match is required for a radius graph")

    for feature_path in frame[cfg.feature_path_column].astype(str).unique():
        path = Path(feature_path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for graph construction") from exc
        with h5py.File(path, "r") as handle:
            if cfg.coord_key not in handle:
                raise KeyError(f"{path} does not contain '{cfg.coord_key}'")
            coords = np.asarray(handle[cfg.coord_key], dtype=np.float32)
            if cfg.graph_key in handle and not args.overwrite:
                continue
        if graph_type == "radius":
            coords = _read_coords(
                path,
                cfg,
                args.pixel_spacing,
                require_spacing=args.degree_match,
            )
        if graph_type == "knn":
            edge_index = coordinate_knn_edges(coords, k=args.k or cfg.knn_k)
        else:
            edge_index = coordinate_radius_edges(coords, radius=radius)
        edge_index_to_hdf5(path, edge_index, cfg.graph_key)


if __name__ == "__main__":
    main()
