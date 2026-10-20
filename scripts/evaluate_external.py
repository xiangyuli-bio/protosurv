from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from protosurv.config import ExperimentConfig
from protosurv.data import group_patients, read_metadata, to_slide_records
from protosurv.metrics import bootstrap_cindex, concordance_index, reverse_km_median
from protosurv.model import PriorGuidedSurvival
from protosurv.reproducibility import resolve_device
from protosurv.training import _forward_patients


def _load_model(path: Path, cfg: ExperimentConfig, device):
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
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed source-cohort checkpoints externally")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
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
    checkpoint_risks = []
    time = np.asarray([patient.survival_time for patient in patients], dtype=float)
    event = np.asarray([patient.event for patient in patients], dtype=int)
    for checkpoint in args.checkpoints:
        model = _load_model(Path(checkpoint), cfg, device)
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source_ids = checkpoint_data.get("source_patient_ids")
        if source_ids is None:
            raise ValueError("Checkpoint lacks source_patient_ids; supply a provenance-complete checkpoint")
        if set(source_ids) & {patient.patient_id for patient in patients}:
            raise ValueError("External patient identifiers overlap the source cohort")
        for key, value in checkpoint_data.get("config", {}).items():
            if key not in {"seed", "device"} and cfg.to_dict().get(key) != value:
                raise ValueError(f"External configuration differs from source checkpoint: {key}")
        if "source_risk_mean" not in checkpoint_data or "source_risk_std" not in checkpoint_data:
            raise ValueError(
                f"Checkpoint {checkpoint} lacks source-training risk normalization statistics"
            )
        result = _forward_patients(model, patients, cfg, device)
        risk = result[0].detach().cpu().numpy()
        checkpoint_risks.append(
            (risk - float(checkpoint_data["source_risk_mean"]))
            / max(float(checkpoint_data["source_risk_std"]), 1e-8)
        )
    ensemble_risk = np.mean(np.stack(checkpoint_risks), axis=0)
    cindex = concordance_index(time, ensemble_risk, event)
    bootstrap_mean, lower, upper = bootstrap_cindex(
        time, ensemble_risk, event, n_bootstrap=args.bootstrap, seed=cfg.seed
    )
    result = {
        "n_patients": len(patients),
        "n_slides": len(records),
        "n_events": int(event.sum()),
        "event_fraction": float(event.mean()),
        "observed_time_median": float(np.median(time)),
        "observed_time_range": [float(time.min()), float(time.max())],
        "median_followup_reverse_km": reverse_km_median(time, event),
        "cindex": cindex,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_95ci": [lower, upper],
        "checkpoints": args.checkpoints,
    }
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    pd.DataFrame({"patient_id": [p.patient_id for p in patients], "survival_time": time,
                  "event": event, "risk": ensemble_risk}).to_csv(
        Path(args.output).with_suffix(".patients.csv"), index=False)


if __name__ == "__main__":
    main()
