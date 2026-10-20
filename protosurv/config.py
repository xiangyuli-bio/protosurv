"""Configuration loading with explicit, typed defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1
    device: str = "auto"
    input_dim: int = 1024
    hidden_dim: int = 256
    prototype_dim: int = 512
    projection_dim: int = 256
    num_classes: int = 5
    num_prototypes: int = 8
    num_gcn_layers: int = 2
    num_attention_heads: int = 8
    knn_k: int = 8
    graph_type: str = "knn"
    graph_radius: float | None = None
    dropout: float = 0.2
    temperature: float = 1.0
    prior_mode: str = "soft"
    epsilon: float = 1e-8
    alpha: float = 0.01
    beta: float = 0.1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 10
    min_delta: float = 1e-4
    batch_size: int = 1
    num_workers: int = 0
    feature_key: str = "features"
    coord_key: str = "coords"
    prior_key: str = "priors"
    graph_key: str = "edge_index"
    patient_column: str = "patient_id"
    slide_column: str = "slide_id"
    time_column: str = "survival_time"
    event_column: str = "event"
    feature_path_column: str = "feature_path"

    def __post_init__(self) -> None:
        if self.prior_mode not in {"soft", "hard"}:
            raise ValueError("prior_mode must be 'soft' or 'hard'")
        if self.graph_type not in {"knn", "radius"}:
            raise ValueError("graph_type must be 'knn' or 'radius'")
        if self.graph_radius is not None and self.graph_radius <= 0:
            raise ValueError("graph_radius must be positive when provided")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
        return cls(**dict(values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("Configuration YAML must contain a mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
