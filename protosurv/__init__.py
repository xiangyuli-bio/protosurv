"""Reproducible prior-guided whole-slide survival modelling."""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig", "cox_partial_likelihood", "concordance_index"]


def __getattr__(name):
    if name == "cox_partial_likelihood":
        from .losses import cox_partial_likelihood

        return cox_partial_likelihood
    if name == "concordance_index":
        from .metrics import concordance_index

        return concordance_index
    raise AttributeError(name)
