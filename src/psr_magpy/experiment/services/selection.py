"""Shared selection helpers for experiment orchestration."""

from typing import List, Optional, Set, Tuple

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def parse_experiment_number(experiment_name: str) -> Optional[int]:
    """Parse trailing experiment index from ``{series}_{idx:06d}`` names."""
    try:
        name_parts = experiment_name.rsplit("_", 1)
        if len(name_parts) == 2:
            return int(name_parts[1])
    except (ValueError, IndexError):
        return None
    return None


def filter_experiments_by_numbers(
    experiments: List[ExperimentConfig], experiment_numbers: Optional[List[int]]
) -> Tuple[List[ExperimentConfig], Set[int], Set[int]]:
    """Filter experiments by requested numbers.

    Returns
    -------
    Tuple[List[ExperimentConfig], Set[int], Set[int]]
        (filtered_experiments, found_numbers, missing_numbers)
    """
    if experiment_numbers is None:
        return experiments, set(), set()

    requested_numbers = set(experiment_numbers)
    filtered_experiments: List[ExperimentConfig] = []
    found_numbers: Set[int] = set()

    for exp in experiments:
        exp_number = parse_experiment_number(exp.name)
        if exp_number is None:
            logger.warning(
                f"Could not parse experiment number from name: {exp.name}, skipping"
            )
            continue
        if exp_number in requested_numbers:
            filtered_experiments.append(exp)
            found_numbers.add(exp_number)

    missing_numbers = requested_numbers - found_numbers
    return filtered_experiments, found_numbers, missing_numbers
