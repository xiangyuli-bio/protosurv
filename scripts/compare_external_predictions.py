"""Paired patient bootstrap for a prespecified survival-model comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protosurv.metrics import concordance_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    a, b = (pd.read_csv(p, dtype={'patient_id': str}).set_index('patient_id') for p in
            [args.reference, args.candidate])
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError('both methods must have exactly the same unique patients')
    b = b.loc[a.index]
    if not np.array_equal(a[['survival_time', 'event']].values, b[['survival_time', 'event']].values):
        raise ValueError('patient outcome labels differ')
    time, event = a.survival_time.values, a.event.values
    arisk, brisk = a.risk.values, b.risk.values
    difference = concordance_index(time, brisk, event)-concordance_index(time, arisk, event)
    rng = np.random.default_rng(args.seed); samples = []
    for _ in range(args.bootstrap):
        i = rng.integers(0, len(a), len(a))
        delta = concordance_index(time[i], brisk[i], event[i])-concordance_index(time[i], arisk[i], event[i])
        if np.isfinite(delta): samples.append(delta)
    if not samples: raise ValueError('no comparable survival pairs in bootstrap samples')
    result = {'candidate_minus_reference_cindex': difference,
              'paired_patient_bootstrap_95ci': np.percentile(samples, [2.5, 97.5]).tolist(),
              'n_patients': len(a), 'valid_bootstrap_replicates': len(samples), 'seed': args.seed}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
