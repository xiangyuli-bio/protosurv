"""Retrain graph variants on fixed patient folds with training-only radius selection."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protosurv.config import ExperimentConfig
from protosurv.data import read_metadata
from protosurv.graph import (coordinate_knn_edges, coordinate_radius_edges,
                            edge_index_to_hdf5, mean_degree, select_degree_matched_radius)
from protosurv.splits import load_fold
from protosurv.training import fit_fold
from scripts.build_graphs import _read_coords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--splits-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--variants', nargs='+', default=['knn4', 'knn8', 'knn12', 'knn16', 'radius'])
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    frame = read_metadata(args.metadata, patient_column=cfg.patient_column,
                          slide_column=cfg.slide_column, time_column=cfg.time_column,
                          event_column=cfg.event_column, feature_path_column=cfg.feature_path_column)
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for variant in args.variants:
        if variant not in {'knn4', 'knn8', 'knn12', 'knn16', 'radius'}:
            raise ValueError('unsupported graph variant')
        summaries, manifests = [], []
        for fold_index in range(args.n_folds):
            split = load_fold(Path(args.splits_dir)/f'fold_{fold_index}.csv')
            parts = [split[name] for name in ('train', 'validation', 'test')]
            if any(parts[i] & parts[j] for i in range(3) for j in range(i+1, 3)):
                raise ValueError('patient overlap between splits')
            if set().union(*parts) != set(frame[cfg.patient_column]):
                raise ValueError('fold patient IDs differ from cohort metadata')
            graph_key = f'edges_revision_{variant}_fold_{fold_index}'
            radius = None
            if variant == 'radius':
                training = frame[frame[cfg.patient_column].isin(split['train'])]
                radius = select_degree_matched_radius([
                    _read_coords(Path(path), cfg, None, require_spacing=True)
                    for path in training[cfg.feature_path_column].unique()], k=8)
            degrees = []
            for path in frame[cfg.feature_path_column].unique():
                coords = _read_coords(Path(path), cfg, None, require_spacing=variant == 'radius')
                edges = (coordinate_radius_edges(coords, radius) if radius is not None else
                         coordinate_knn_edges(coords, min(int(variant[3:]), len(coords)-1)))
                edge_index_to_hdf5(path, edges, graph_key)
                degrees.append(mean_degree(edges, len(coords)))
            fold_cfg = replace(cfg, graph_key=graph_key,
                               graph_type='radius' if radius is not None else 'knn',
                               graph_radius=radius, knn_k=8 if radius is not None else int(variant[3:]))
            manifests.append({'fold': fold_index, 'radius_micrometres': radius,
                              'mean_nonself_degree': float(np.mean(degrees)),
                              'sd_nonself_degree': float(np.std(degrees)), 'graph_key': graph_key})
            for seed in args.seeds:
                seed_cfg = replace(fold_cfg, seed=seed)
                summaries.append(fit_fold(frame, split, seed_cfg, root/variant/f'seed_{seed}',
                                          fold_index=fold_index, seed=seed))
        fold_values = pd.DataFrame(summaries).groupby('fold')['test_cindex'].mean()
        all_results[variant] = {'mean_cindex': float(fold_values.mean()),
                               'sample_sd_over_folds': float(fold_values.std(ddof=1)),
                               'fold_seed_mean_cindices': fold_values.tolist(),
                               'seeds': args.seeds, 'graph_manifests': manifests,
                               'runs': summaries}
        (root/'graph_sensitivity.json').write_text(json.dumps(all_results, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
