from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protosurv.priors import soft_priors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cached image/text embeddings into five-class soft priors"
    )
    parser.add_argument("--image-embeddings", required=True, help="NumPy array with shape [N, D]")
    parser.add_argument(
        "--class-embeddings",
        required=True,
        help="NumPy array with shape [5, P, D] or [5, D]",
    )
    parser.add_argument(
        "--source-h5",
        default=None,
        help="Optional feature HDF5 to copy into the output before writing priors",
    )
    parser.add_argument("--output", required=True, help="HDF5 output containing priors")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--coords", default=None, help="Optional NumPy array with shape [N, 2]")
    args = parser.parse_args()
    image = np.load(args.image_embeddings)
    classes = np.load(args.class_embeddings)
    priors = soft_priors(image, classes, temperature=args.temperature)
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to write prior files") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.source_h5 is not None and Path(args.source_h5).expanduser().resolve() == output.resolve():
        raise ValueError("--source-h5 and --output must be different files")
    if args.source_h5 is not None:
        with h5py.File(args.source_h5, "r") as source:
            for key in ("features", "coords"):
                if key in source and source[key].shape[0] != priors.shape[0]:
                    raise ValueError("cached image embeddings and source HDF5 patch counts differ")
    with h5py.File(output, "w") as handle:
        if args.source_h5 is not None:
            with h5py.File(args.source_h5, "r") as source:
                for name, dataset in source.items():
                    source.copy(dataset, handle, name=name)
                for name, value in source.attrs.items():
                    handle.attrs[name] = value
        if "priors" in handle:
            del handle["priors"]
        handle.create_dataset("priors", data=priors.astype(np.float32), compression="gzip")
        if args.coords is not None:
            coords = np.asarray(np.load(args.coords), dtype=np.float32)
            if coords.shape != (priors.shape[0], 2):
                raise ValueError("coords must have shape [N, 2] matching image embeddings")
            if "coords" in handle:
                del handle["coords"]
            handle.create_dataset("coords", data=coords, compression="gzip")
        handle.attrs["temperature"] = float(args.temperature)
        handle.attrs["class_order"] = np.asarray(
            ["Tumor", "Stroma", "Immune Infiltration", "Necrosis", "Others"], dtype="S"
        )


if __name__ == "__main__":
    main()
