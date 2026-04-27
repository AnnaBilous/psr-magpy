"""Compatibility wrapper for experiment lookup helpers."""

from psr_magpy.experiment.services.experiment_lookup_service import (
    experiment_matches_thD_fQ,
    find_experiment_numbers_by_thD_fQ,
    infer_experiment_count,
)

__all__ = [
    "experiment_matches_thD_fQ",
    "infer_experiment_count",
    "find_experiment_numbers_by_thD_fQ",
]
