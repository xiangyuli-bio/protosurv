"""Metrics implemented without a mandatory survival-analysis dependency."""

from __future__ import annotations

import numpy as np


def reverse_km_median(survival_time, event):
    """Median follow-up using censoring as the reverse-KM event; None if unreached."""
    time = np.asarray(survival_time, dtype=float)
    observed = np.asarray(event, dtype=int)
    survival = 1.0
    for t in np.unique(time):
        at_risk = np.count_nonzero(time >= t)
        censored = np.count_nonzero((time == t) & (observed == 0))
        survival *= 1.0 - censored / at_risk
        if survival <= 0.5:
            return float(t)
    return None


def concordance_index(survival_time, risk, event) -> float:
    """Harrell's C-index for risk scores where larger means higher risk."""

    time = np.asarray(survival_time, dtype=float).reshape(-1)
    scores = np.asarray(risk, dtype=float).reshape(-1)
    observed = np.asarray(event, dtype=int).reshape(-1)
    if not (len(time) == len(scores) == len(observed)):
        raise ValueError("survival_time, risk, and event must have the same length")
    comparable = 0.0
    concordant = 0.0
    for i in range(len(time)):
        if observed[i] != 1:
            continue
        for j in range(len(time)):
            if i == j or time[i] >= time[j]:
                continue
            comparable += 1.0
            if scores[i] > scores[j]:
                concordant += 1.0
            elif scores[i] == scores[j]:
                concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")


def bootstrap_cindex(
    survival_time, risk, event, *, n_bootstrap: int = 1000, seed: int = 1
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    time = np.asarray(survival_time)
    scores = np.asarray(risk)
    observed = np.asarray(event)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(time), len(time))
        value = concordance_index(time[indices], scores[indices], observed[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan"), float("nan")
    interval = np.percentile(values, [2.5, 97.5])
    return float(np.mean(values)), float(interval[0]), float(interval[1])
