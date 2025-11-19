"""Experiment metadata handling for experiment series.

This module provides utilities for creating and managing experiment metadata
and naming conventions for experiment series.
"""

from typing import Dict, Any
from psr_magpy.experiment.field_config import FieldConfig
from psr_magpy.experiment.grid_config import GridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def create_experiment_metadata(
    series_name: str,
    base_metadata: Dict[str, Any],
    experiment_number: int,
    field_config: Any,
    grid_config: Any,
    tracing_config: Any,
    template_config: Any,
) -> Dict[str, Any]:
    """Create minimal metadata for an experiment."""
    # Start with base metadata (which might contain user-defined fields)
    metadata = base_metadata.copy()

    # Only add experiment number if not already present
    if "experiment_number" not in metadata:
        metadata["experiment_number"] = experiment_number

    return metadata


def generate_experiment_name(series_name: str, experiment_count: int) -> str:
    """Generate a unique name for an experiment.

    Args:
        series_name: Name of the experiment series
        experiment_count: Sequential number of this experiment

    Returns:
        Unique experiment name in format: {series_name}_{experiment_count:03d}
    """
    return f"{series_name}_{experiment_count:03d}"
