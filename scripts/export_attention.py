from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protosurv.config import ExperimentConfig
from protosurv.data import group_patients, read_metadata, to_slide_records
from protosurv.model import PriorGuidedSurvival
from protosurv.reproducibility import resolve_device
from protosurv.training import _forward_patients


def normalized_class_attention(attention, num_classes, num_prototypes):
    """Average prototypes, normalize each class within one model, then ensemble."""
    grouped = attention.reshape(num_classes, num_prototypes, -1).mean(dim=1)
    lower = grouped.amin(dim=1, keepdim=True)
    span = grouped.amax(dim=1, keepdim=True) - lower
    values = torch.where(span > 0, (grouped - lower) / span.clamp_min(1e-12),
                         torch.zeros_like(grouped))
    return values.transpose(0, 1)


def _build_model(cfg: ExperimentConfig, device):
    return PriorGuidedSurvival(
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
    ).to(device)


def _load_checkpoint_model(path: str, cfg: ExperimentConfig, device):
    model = _build_model(cfg, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prototype-to-context attention maps")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoint", default=None, help="One checkpoint (legacy spelling)")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="One or more checkpoints; attention maps are averaged per slide",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    frame = read_metadata(
        args.metadata,
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    records = to_slide_records(
        frame,
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    patients = group_patients(records)
    device = resolve_device(cfg.device)
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint is not None:
        checkpoints.append(args.checkpoint)
    if not checkpoints:
        parser.error("provide --checkpoint or --checkpoints")
    attention_by_slide: dict[str, dict] = {}
    for checkpoint_path in checkpoints:
        model = _load_checkpoint_model(checkpoint_path, cfg, device)
        result = _forward_patients(model, patients, cfg, device, collect_attention=True)
        for record in result[-1]:
            slide_id = str(record["slide_id"])
            class_attention = normalized_class_attention(
                record["attention"], cfg.num_classes, cfg.num_prototypes)
            if slide_id not in attention_by_slide:
                attention_by_slide[slide_id] = {
                    **record,
                    "attention": record["attention"].clone(),
                    "class_attention": class_attention.clone(),
                    "checkpoint_count": 1,
                }
            else:
                previous = attention_by_slide[slide_id]
                if previous["attention"].shape != record["attention"].shape:
                    raise ValueError(f"attention shape mismatch for slide {slide_id}")
                if not torch.equal(previous["coords"], record["coords"]):
                    raise ValueError(f"coordinate mismatch for slide {slide_id}")
                previous["attention"] += record["attention"]
                previous["class_attention"] += class_attention
                previous["checkpoint_count"] += 1
    expected_count = len(checkpoints)
    result_records = []
    for record in attention_by_slide.values():
        if record["checkpoint_count"] != expected_count:
            raise ValueError(f"slide {record['slide_id']} is missing a checkpoint prediction")
        record["attention"] /= float(expected_count)
        record["class_attention"] /= float(expected_count)
        result_records.append(record)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for record in result_records:
        filename = f"{record['slide_id']}.npz"
        np.savez_compressed(
            output / filename,
            attention=record["attention"].numpy(),
            class_attention=record["class_attention"].numpy(),
            coords=record["coords"].numpy(),
            priors=record["priors"].numpy(),
            hard_tumor_prior=np.argmax(record["priors"].numpy(), axis=1).astype(np.int64) == 0,
        )
        index.append(
            {
                "patient_id": record["patient_id"],
                "slide_id": record["slide_id"],
                "file": filename,
                "checkpoint_count": expected_count,
            }
        )
    with (output / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)


if __name__ == "__main__":
    main()
