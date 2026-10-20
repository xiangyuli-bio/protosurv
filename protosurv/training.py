"""Patient-level training and evaluation loops."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import PatientRecord, group_patients, load_slide, read_metadata, to_slide_records
from .losses import combined_patient_loss
from .metrics import concordance_index
from .model import PriorGuidedSurvival
from .reproducibility import resolve_device, seed_everything
from .splits import load_fold, patient_folds


def _records_for_patients(frame: pd.DataFrame, patient_ids: set[str], cfg: ExperimentConfig):
    records = to_slide_records(
        frame[frame[cfg.patient_column].astype(str).isin(patient_ids)],
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    return group_patients(records)


def _model_input(slide, cfg: ExperimentConfig, device):
    features = torch.as_tensor(slide.features, dtype=torch.float32, device=device)
    priors = torch.as_tensor(slide.priors, dtype=torch.float32, device=device)
    if cfg.prior_mode == "hard":
        priors = F.one_hot(priors.argmax(dim=-1), cfg.num_classes).to(torch.float32)
    coords = torch.as_tensor(slide.coords, dtype=torch.float32, device=device)
    edges = slide.edge_index
    if edges is None:
        from .graph import coordinate_knn_edges
        if cfg.graph_type == "radius":
            raise ValueError("Radius experiments require a preconstructed physical-coordinate graph")
        edges = coordinate_knn_edges(slide.coords, min(cfg.knn_k, len(slide.coords) - 1))
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    return features, priors, coords, edge_tensor


def _forward_patients(
    model: PriorGuidedSurvival,
    patients: list[PatientRecord],
    cfg: ExperimentConfig,
    device,
    *,
    collect_attention: bool = False,
):
    risks: list[torch.Tensor] = []
    times: list[float] = []
    events: list[int] = []
    alignments: list[torch.Tensor] = []
    orthogonalities: list[torch.Tensor] = []
    attention_records: list[dict[str, Any]] = []
    for patient in patients:
        slide_embeddings: list[torch.Tensor] = []
        for record in patient.slides:
            slide = load_slide(
                record,
                feature_key=cfg.feature_key,
                coord_key=cfg.coord_key,
                prior_key=cfg.prior_key,
                graph_key=cfg.graph_key,
                num_classes=cfg.num_classes,
            )
            features, priors, coords, edge_index = _model_input(slide, cfg, device)
            output = model(
                features,
                priors,
                coords=coords,
                edge_index=edge_index,
                return_attention=collect_attention,
            )
            slide_embeddings.append(output.slide_embedding)
            alignments.append(output.alignment_loss)
            orthogonalities.append(output.orthogonality_loss)
            if collect_attention:
                attention_records.append(
                    {
                        "patient_id": patient.patient_id,
                        "slide_id": slide.slide_id,
                        "attention": output.attention.detach().cpu(),
                        "coords": torch.as_tensor(slide.coords),
                        "priors": priors.detach().cpu(),
                    }
                )
        patient_embedding = torch.stack(slide_embeddings, dim=0).mean(dim=0)
        risks.append(model.risk_head(patient_embedding).reshape(()))
        times.append(patient.survival_time)
        events.append(patient.event)
    return (
        torch.stack(risks),
        torch.tensor(times, dtype=torch.float32, device=device),
        torch.tensor(events, dtype=torch.float32, device=device),
        alignments,
        orthogonalities,
        attention_records,
    )


def run_epoch(
    model: PriorGuidedSurvival,
    patients: list[PatientRecord],
    cfg: ExperimentConfig,
    device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    collect_attention: bool = False,
) -> dict[str, Any]:
    if not patients:
        raise ValueError("Cannot run an epoch on an empty patient split")
    training = optimizer is not None
    model.train(training)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        risk, times, events, alignments, orthogonalities, attention = _forward_patients(
            model, patients, cfg, device, collect_attention=collect_attention
        )
        loss = combined_patient_loss(
            risk,
            times,
            events,
            alignments,
            orthogonalities,
            alpha=cfg.alpha,
            beta=cfg.beta,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    risk_np = risk.detach().cpu().numpy()
    time_np = times.detach().cpu().numpy()
    event_np = events.detach().cpu().numpy()
    return {
        "loss": float(loss.detach().cpu()),
        "cox_cindex": concordance_index(time_np, risk_np, event_np),
        "risk": risk_np,
        "survival_time": time_np,
        "event": event_np,
        "attention": attention,
    }


def _checkpoint(model, optimizer, cfg, epoch, validation):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg.to_dict(),
        "epoch": epoch,
        "validation": validation,
    }


def fit_fold(
    metadata: pd.DataFrame,
    split: dict[str, set[str]],
    cfg: ExperimentConfig,
    output_dir: str | Path,
    *,
    fold_index: int,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    device = resolve_device(cfg.device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_patients = _records_for_patients(metadata, split["train"], cfg)
    validation_patients = _records_for_patients(metadata, split["validation"], cfg)
    test_patients = _records_for_patients(metadata, split["test"], cfg)
    if not any(patient.event for patient in train_patients):
        raise ValueError("Training split contains no observed events")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_cindex = -np.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg.max_epochs):
        train_result = run_epoch(model, train_patients, cfg, device, optimizer=optimizer)
        validation_result = run_epoch(model, validation_patients, cfg, device)
        validation_cindex = validation_result["cox_cindex"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_cindex": train_result["cox_cindex"],
                "validation_loss": validation_result["loss"],
                "validation_cindex": validation_cindex,
            }
        )
        torch.save(
            _checkpoint(model, optimizer, cfg, epoch, validation_result),
            output / f"fold_{fold_index}_last.pt",
        )
        if np.isfinite(validation_cindex) and validation_cindex > best_cindex + cfg.min_delta:
            best_cindex = validation_cindex
            best_epoch = epoch
            stale = 0
            torch.save(
                _checkpoint(model, optimizer, cfg, epoch, validation_result),
                output / f"fold_{fold_index}_best.pt",
            )
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    checkpoint_path = output / f"fold_{fold_index}_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output / f"fold_{fold_index}_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    source_result = run_epoch(model, train_patients, cfg, device)
    source_risk = np.asarray(source_result["risk"], dtype=float)
    checkpoint["source_risk_mean"] = float(source_risk.mean())
    checkpoint["source_risk_std"] = float(max(source_risk.std(), 1e-8))
    checkpoint["source_patient_ids"] = sorted(set().union(*split.values()))
    torch.save(checkpoint, checkpoint_path)
    test_result = run_epoch(model, test_patients, cfg, device)
    pd.DataFrame({
        "patient_id": [patient.patient_id for patient in test_patients],
        "survival_time": test_result["survival_time"],
        "event": test_result["event"],
        "risk": test_result["risk"],
    }).to_csv(output / f"fold_{fold_index}_test_predictions.csv", index=False)
    summary = {
        "fold": fold_index,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_cindex": float(best_cindex),
        "test_cindex": float(test_result["cox_cindex"]),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    with (output / f"fold_{fold_index}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def train_cross_validation(
    metadata_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    n_splits: int = 5,
    splits_dir: str | Path | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    if seeds is not None and (not seeds or len(seeds) != len(set(seeds))):
        raise ValueError("training seeds must be non-empty and unique")
    cfg = ExperimentConfig.from_yaml(config_path)
    metadata = read_metadata(
        metadata_path,
        patient_column=cfg.patient_column,
        slide_column=cfg.slide_column,
        time_column=cfg.time_column,
        event_column=cfg.event_column,
        feature_path_column=cfg.feature_path_column,
    )
    if splits_dir is None:
        folds = patient_folds(
            metadata,
            n_splits=n_splits,
            seed=cfg.seed,
            patient_column=cfg.patient_column,
            event_column=cfg.event_column,
        )
    else:
        metadata_patients = set(metadata[cfg.patient_column].astype(str))
        folds = []
        split_root = Path(splits_dir)
        for fold_index in range(n_splits):
            path = split_root / f"fold_{fold_index}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded = load_fold(path)
            required = {"train", "validation", "test"}
            if set(loaded) != required:
                raise ValueError(
                    f"{path} must contain exactly train, validation, and test splits"
                )
            parts = [loaded[name] for name in ("train", "validation", "test")]
            if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
                raise ValueError(f"{path} assigns a patient to more than one split")
            assigned = set().union(*parts)
            if assigned != metadata_patients:
                missing = sorted(metadata_patients - assigned)
                extra = sorted(assigned - metadata_patients)
                raise ValueError(
                    f"{path} does not match metadata patients; missing={missing[:3]}, extra={extra[:3]}"
                )
            folds.append(loaded)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)
    summaries = []
    for seed in seeds or [cfg.seed]:
        seed_cfg = replace(cfg, seed=seed)
        seed_output = output / f"seed_{seed}" if seeds else output
        for index, split in enumerate(folds):
            summaries.append(fit_fold(metadata, split, seed_cfg, seed_output,
                                      fold_index=index, seed=seed))
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=False)
    fold_means = pd.DataFrame(summaries).groupby("fold")["test_cindex"].mean()
    aggregate = {"seed_values": seeds or [cfg.seed], "n_folds": len(fold_means),
                 "fold_seed_mean_cindices": fold_means.tolist(),
                 "mean_cindex": float(fold_means.mean()),
                 "sample_sd_over_folds": float(fold_means.std(ddof=1))}
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False)+"\n")
    return summaries
