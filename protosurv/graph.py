"""Coordinate-defined sparse graph construction."""

from __future__ import annotations

import numpy as np


def physical_coordinates(coords: np.ndarray, pixel_spacing) -> np.ndarray:
    """Convert pixel coordinates to physical units using scalar or ``(x, y)`` spacing."""

    values = np.asarray(pixel_spacing, dtype=np.float32)
    if values.ndim == 0:
        values = np.repeat(values, 2)
    if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("pixel_spacing must be a positive scalar or a pair of values")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [N, 2]")
    return coords * values[None, :]


def mean_degree(edge_index: np.ndarray, num_nodes: int) -> float:
    """Return the mean directed degree represented by an edge list."""

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    return float(edges.shape[1] / num_nodes)


def _coordinates(coords):
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("coordinates must be a finite array with shape [N, 2]")
    return values


def select_degree_matched_radius(coordinate_sets, k: int = 8) -> float:
    """Match mean non-self degree using training slides in micrometres only."""
    from scipy.spatial import cKDTree
    clean = [_coordinates(c) for c in coordinate_sets if len(c) >= 2]
    if not clean or k < 1:
        raise ValueError("at least one slide with two patches and positive k is required")
    trees = [cKDTree(c) for c in clean]
    target = np.mean([mean_degree(coordinate_knn_edges(c, min(k, len(c)-1)), len(c)) for c in clean])
    low, high = 0.0, max(float(np.linalg.norm(np.ptp(c, axis=0))) for c in clean)
    if high <= 0:
        raise ValueError("coordinates must span a positive spatial extent")
    best_radius, best_error = high, float("inf")
    for _ in range(32):
        radius = (low + high) / 2.0
        degree = np.mean([(tree.count_neighbors(tree, radius) - tree.n) / tree.n for tree in trees])
        error = abs(degree-target)
        if error < best_error:
            best_radius, best_error = radius, error
        if degree < target:
            low = radius
        else:
            high = radius
    return float(best_radius)


def coordinate_knn_edges(coords: np.ndarray, k: int = 8, symmetrize: bool = True) -> np.ndarray:
    """Euclidean spatial kNN via a KD-tree; self-loops are added by the GCN.

    A KD-tree avoids an N by N distance matrix. The edge list is sorted and
    deduplicated. Equal-distance neighbour ties follow SciPy's KD-tree order.
    """
    from scipy.spatial import cKDTree
    values = _coordinates(coords)
    n = len(values)
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < N")
    _, indices = cKDTree(values).query(values, k=k+1, workers=1)
    pairs = []
    for source, neighbours in enumerate(indices):
        targets = neighbours[neighbours != source][:k]
        pairs.extend((source, int(target)) for target in targets)
        if symmetrize:
            pairs.extend((int(target), source) for target in targets)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0).T


def coordinate_radius_edges(coords: np.ndarray, radius: float, symmetrize: bool = True) -> np.ndarray:
    """Radius graph in the coordinate units supplied, without self-loops."""
    from scipy.spatial import cKDTree
    values = _coordinates(coords)
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")
    if len(values) < 2:
        return np.empty((2, 0), dtype=np.int64)
    pairs = cKDTree(values).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return np.empty((2, 0), dtype=np.int64)
    if symmetrize:
        pairs = np.concatenate((pairs, pairs[:, ::-1]))
    return np.unique(pairs, axis=0).T.astype(np.int64)


def edge_index_to_hdf5(path, edge_index: np.ndarray, key: str = "edge_index") -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to write graph files") from exc
    with h5py.File(path, "a") as handle:
        if key in handle:
            del handle[key]
        handle.create_dataset(key, data=np.asarray(edge_index, dtype=np.int64), compression="gzip")
