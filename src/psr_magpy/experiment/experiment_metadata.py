"""Experiment metadata handling for experiment series.

This module provides utilities for creating and managing experiment metadata
and naming conventions for experiment series.
"""

from typing import Dict, Any
from psr_magpy.experiment.experiment_config import FieldConfig, GridConfig, TracingConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


def create_experiment_metadata(
    series_name: str,
    series_metadata: Dict[str, Any],
    experiment_count: int,
    field_config: FieldConfig,
    grid_config: GridConfig,
    tracing_config: TracingConfig,
    template_config: FieldLine
) -> Dict[str, Any]:
    """Create unique metadata for each experiment.
    
    Args:
        series_name: Name of the experiment series
        series_metadata: Base metadata from the series
        experiment_count: Sequential number of this experiment
        field_config: Field configuration for this experiment
        grid_config: Grid configuration for this experiment
        tracing_config: Tracing configuration for this experiment
        template_config: Template configuration for this experiment
        
    Returns:
        Dictionary containing experiment metadata
    """
    metadata = series_metadata.copy() if series_metadata else {}
    
    # Add experiment-specific metadata
    metadata.update({
        "experiment_number": experiment_count,
        "series_name": series_name,
        "field_components": len(field_config.components),
        "grid_type": grid_config.get_grid_type(),
        "template_properties": list(template_config.property_system.get_flags_dict().keys())
    })
    
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