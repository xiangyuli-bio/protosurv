from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import platform

from protosurv.config import ExperimentConfig
from protosurv.graph import coordinate_knn_edges
from protosurv.model import PriorGuidedSurvival
from protosurv.reproducibility import resolve_device, seed_everything


def _load_hdf5(path: Path, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for an HDF5 benchmark input") from exc
    with h5py.File(path, "r") as handle:
        required = (cfg.feature_key, cfg.prior_key, cfg.graph_key)
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(
                f"{path} must contain preconstructed datasets: {', '.join(missing)}"
            )
        features = np.asarray(handle[cfg.feature_key], dtype=np.float32)
        priors = np.asarray(handle[cfg.prior_key], dtype=np.float32)
        edge_index = np.asarray(handle[cfg.graph_key], dtype=np.int64)
    if features.ndim != 2 or priors.ndim != 2 or edge_index.ndim != 2:
        raise ValueError("features, priors and edge_index must be two-dimensional")
    if features.shape[0] != priors.shape[0] or edge_index.shape[0] != 2:
        raise ValueError("HDF5 benchmark arrays have incompatible shapes")
    if features.shape[1] != cfg.input_dim or priors.shape[1] != cfg.num_classes:
        raise ValueError("HDF5 benchmark arrays do not match the configured dimensions")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= features.shape[0]):
        raise ValueError("HDF5 edge_index contains an invalid node index")
    return features, priors, edge_index


def _synthetic_input(
    patch_count: int, cfg: ExperimentConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a smoke-test input with the graph constructed outside timing."""

    features = torch.randn(patch_count, cfg.input_dim)
    priors = torch.softmax(torch.randn(patch_count, cfg.num_classes), dim=-1)
    coords = np.random.default_rng(cfg.seed + patch_count).random((patch_count, 2)).astype(
        np.float32
    ) * 10000.0
    edge_np = coordinate_knn_edges(coords, k=min(cfg.knn_k, patch_count - 1))
    return features, priors, torch.as_tensor(edge_np, dtype=torch.long)


def _prefix_graph(edge_index: np.ndarray, patch_count: int) -> np.ndarray:
    if edge_index.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    keep = (edge_index[0] < patch_count) & (edge_index[1] < patch_count)
    return edge_index[:, keep]


def _measure(
    model: PriorGuidedSurvival,
    features: torch.Tensor,
    priors: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float | None, np.ndarray]:
    features = features.to(device)
    priors = priors.to(device)
    edge_index = edge_index.to(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(features, priors, edge_index=edge_index, return_attention=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(features, priors, edge_index=edge_index, return_attention=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
        memory = (
            torch.cuda.max_memory_allocated(device) / 10**9 if device.type == "cuda" else None
        )
    return float(np.median(durations)), memory, np.asarray(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark downstream inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-h5", default=None, help="HDF5 slide with a preconstructed graph")
    parser.add_argument("--patch-counts", nargs="+", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    cfg = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = PriorGuidedSurvival(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        prototype_dim=cfg.prototype_dim,
        projection_dim=cfg.projection_dim,
        num_classes=cfg.num_classes,
        num_prototypes=cfg.num_prototypes,
        num_gcn_layers=cfg.num_gcn_layers,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        epsilon=cfg.epsilon,
    ).to(device).eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    results = []
    if args.input_h5:
        features_np, priors_np, edges_np = _load_hdf5(Path(args.input_h5), cfg)
        counts = args.patch_counts or [features_np.shape[0]]
        for patch_count in counts:
            if not 1 <= patch_count <= features_np.shape[0]:
                raise ValueError("patch counts must be within the supplied HDF5 slide")
            features = torch.as_tensor(features_np[:patch_count])
            priors = torch.as_tensor(priors_np[:patch_count])
            edge_index = torch.as_tensor(_prefix_graph(edges_np, patch_count))
            latency, memory, durations = _measure(
                model, features, priors, edge_index, device, args.warmup, args.repeats
            )
            results.append(
                {
                    "patches": patch_count,
                    "median_latency_seconds": latency,
                    "peak_memory_gb": memory,
                    "latency_iqr_seconds": np.percentile(durations, [25, 75]).tolist(),
                    "timed_latencies_seconds": durations.tolist(),
                    "graph_edges": int(edge_index.shape[1]),
                }
            )
        source = "hdf5"
    else:
        counts = args.patch_counts or [5000, 10000, 20000, 40000]
        for patch_count in counts:
            if patch_count < 2:
                raise ValueError("synthetic patch counts must be at least two")
            features, priors, edge_index = _synthetic_input(patch_count, cfg)
            latency, memory, durations = _measure(
                model, features, priors, edge_index, device, args.warmup, args.repeats
            )
            results.append(
                {
                    "patches": patch_count,
                    "median_latency_seconds": latency,
                    "peak_memory_gb": memory,
                    "latency_iqr_seconds": np.percentile(durations, [25, 75]).tolist(),
                    "timed_latencies_seconds": durations.tolist(),
                    "graph_edges": int(edge_index.shape[1]),
                }
            )
        source = "synthetic"
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "python_version": platform.python_version(),
                "config": cfg.to_dict(),
                "checkpoint": args.checkpoint,
                "weights": "trained_checkpoint" if args.checkpoint else "random_initialization",
                "memory_units": "GB",
                "excluded_from_timing": ["feature_extraction", "graph_construction", "host_to_device_transfer"],
                "input_source": source,
                "warmup_runs": args.warmup,
                "timed_runs": args.repeats,
                "latency_statistic": "median",
                "results": results,
            },
            handle,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
