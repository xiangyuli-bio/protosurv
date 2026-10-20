"""Convert independent raster masks to area-preserving patch annotations.

Masks must be exported from independent pathologist annotations; model-derived
pseudo-labels are not reference annotations. Coordinates are patch centres.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def patch_annotations(coords, tumor_mask, valid_mask, patch_size, downsample):
    if patch_size <= 0 or downsample <= 0:
        raise ValueError("patch size and mask downsample must be positive")
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite patch centres with shape [N, 2]")
    if tumor_mask.shape != valid_mask.shape or tumor_mask.ndim != 2:
        raise ValueError("tumor and validity masks must have the same two-dimensional shape")
    from scipy.spatial import cKDTree
    pairs = cKDTree(coords).query_pairs(patch_size * 2 ** 0.5, output_type="ndarray")
    if len(pairs) and np.any(np.all(np.abs(coords[pairs[:, 0]]-coords[pairs[:, 1]]) < patch_size-1e-6, axis=1)):
        raise ValueError("patch footprints overlap; area-additive evaluation requires non-overlapping patches")
    fractions, areas, tissue = [], [], []
    height, width = tumor_mask.shape
    for x, y in coords:
        x0, y0 = np.rint((np.array([x, y])-patch_size/2)/downsample).astype(int)
        x1, y1 = np.rint((np.array([x, y])+patch_size/2)/downsample).astype(int)
        full = 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        usable = full and bool(valid_mask[y0:y1, x0:x1].all())
        area = max(1, (x1-x0)*(y1-y0))
        fractions.append(float(tumor_mask[y0:y1, x0:x1].mean()) if usable else 0.0)
        areas.append(area)
        tissue.append(usable)
    return np.asarray(fractions), np.asarray(areas), np.asarray(tissue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--tumor-mask", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--valid-mask")
    group.add_argument("--fully-annotated", action="store_true")
    parser.add_argument("--patch-size-level0", type=float, required=True)
    parser.add_argument("--mask-downsample", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import h5py
    from PIL import Image
    with h5py.File(args.features) as handle:
        coords = np.asarray(handle["coords"])
    tumor = np.asarray(Image.open(args.tumor_mask).convert("L")) > 0
    valid = np.asarray(Image.open(args.valid_mask).convert("L")) > 0 if args.valid_mask else np.ones_like(tumor)
    fractions, areas, tissue = patch_annotations(coords, tumor, valid, args.patch_size_level0, args.mask_downsample)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, tumor=fractions, patch_area=areas, tissue=tissue, coords=coords)
    provenance = {"reference_tumor_mask": str(Path(args.tumor_mask).resolve()),
                  "reference_validity_mask": args.valid_mask,
                  "fully_annotated": args.fully_annotated,
                  "patch_size_level0": args.patch_size_level0,
                  "mask_downsample": args.mask_downsample, "coordinate_convention": "centroid",
                  "n_patches": len(coords), "n_evaluated_patches": int(tissue.sum())}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")

if __name__ == "__main__":
    main()
