"""Survival and prototype regularization losses."""

from __future__ import annotations

import torch
from torch import Tensor


def cox_partial_likelihood(risk: Tensor, survival_time: Tensor, event: Tensor) -> Tensor:
    """Negative Cox partial log-likelihood with exact risk sets.

    ``event`` uses 1 for an observed event and 0 for right censoring. Ties are
    handled by the Breslow denominator, which matches the risk-set convention
    used by the manuscript's patient-level objective.
    """

    risk = risk.reshape(-1)
    survival_time = survival_time.reshape(-1).to(risk.device, dtype=risk.dtype)
    event = event.reshape(-1).to(risk.device, dtype=risk.dtype)
    if not (risk.numel() == survival_time.numel() == event.numel()):
        raise ValueError("risk, survival_time, and event must have the same length")
    if risk.numel() == 0:
        raise ValueError("Cox loss received an empty batch")
    order = torch.argsort(survival_time, descending=True, stable=True)
    ordered_time = survival_time[order]
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumulative_hazard = torch.logcumsumexp(ordered_risk, dim=0)
    # Breslow ties use one denominator for all events at the same time.  A
    # row-wise log-cumulative denominator would incorrectly give tied rows
    # different risk sets.
    group_start = torch.cat(
        (
            torch.ones(1, dtype=torch.bool, device=risk.device),
            ordered_time[1:] != ordered_time[:-1],
        )
    )
    group_id = torch.cumsum(group_start.to(torch.long), dim=0) - 1
    n_groups = int(group_id[-1].item()) + 1
    group_event_count = torch.zeros(n_groups, dtype=risk.dtype, device=risk.device)
    group_numerator = torch.zeros(n_groups, dtype=risk.dtype, device=risk.device)
    group_event_count.scatter_add_(0, group_id, ordered_event)
    group_numerator.scatter_add_(0, group_id, ordered_risk * ordered_event)
    group_end = torch.nonzero(
        torch.cat((group_start[1:], torch.ones(1, dtype=torch.bool, device=risk.device))),
        as_tuple=False,
    ).reshape(-1)
    group_denominator = log_cumulative_hazard[group_end]
    event_count = group_event_count.sum()
    if event_count <= 0:
        return risk.sum() * 0.0
    log_likelihood = group_numerator - group_event_count * group_denominator
    return -log_likelihood.sum() / event_count


def entropy_confidence(priors: Tensor, epsilon: float = 1e-8) -> Tensor:
    if priors.ndim != 2:
        raise ValueError("priors must have shape [N, C]")
    classes = priors.shape[1]
    entropy = -(priors * torch.log(priors + epsilon)).sum(dim=-1)
    return 1.0 - entropy / torch.log(torch.tensor(float(classes), device=priors.device))


def combined_patient_loss(
    risk: Tensor,
    survival_time: Tensor,
    event: Tensor,
    alignment_losses: list[Tensor],
    orthogonality_losses: list[Tensor],
    *,
    alpha: float,
    beta: float,
) -> Tensor:
    cox = cox_partial_likelihood(risk, survival_time, event)
    alignment = torch.stack(alignment_losses).mean() if alignment_losses else cox * 0.0
    orthogonality = (
        torch.stack(orthogonality_losses).mean() if orthogonality_losses else cox * 0.0
    )
    return cox + float(alpha) * alignment + float(beta) * orthogonality
