"""Experiment metadata handling for experiment series.

This module provides utilities for creating and managing experiment metadata
and naming conventions for experiment series.
"""

from typing import Dict, Any
from psr_magpy.experiment.experiment_config import (
    FieldConfig,
    GridConfig,
    TracingConfig,
)
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
    """Create metadata for an experiment."""
    metadata = base_metadata.copy()

    # Add experiment number and series name
    metadata["experiment_number"] = experiment_number
    metadata["series_name"] = series_name

    # Add field metadata
    if hasattr(field_config, "components"):
        # Sort components by B_magnitude in descending order
        sorted_components = sorted(
            field_config.components, key=lambda x: x.B_magnitude, reverse=True
        )
        metadata["field_components"] = [
            {
                "type": comp.moment_type,
                "B": comp.B_magnitude,
                "lat": comp.moment_orientation_latitude_degree,
                "lon": comp.moment_orientation_longitude_degree,
            }
            for comp in sorted_components
        ]

    # Add grid metadata
    if hasattr(grid_config, "spherical") and grid_config.spherical is not None:
        metadata["grid_type"] = "spherical"
        metadata["grid_params"] = {
            "nlon": grid_config.spherical.nlon,
            "nlat": grid_config.spherical.nlat,
            "pole_lat": grid_config.spherical.pole_lat,
        }
    elif hasattr(grid_config, "healpix") and grid_config.healpix is not None:
        metadata["grid_type"] = "healpix"
        metadata["grid_params"] = {"nside": grid_config.healpix.nside}
    elif hasattr(grid_config, "random") and grid_config.random is not None:
        metadata["grid_type"] = "random"
        metadata["grid_params"] = {"nline_ori": grid_config.random.nline_ori}
    elif hasattr(grid_config, "custom") and grid_config.custom is not None:
        metadata["grid_type"] = "custom"
        metadata["grid_params"] = {"nline_ori": grid_config.custom.nline_ori}

    # Add tracing metadata
    if hasattr(tracing_config, "max_integration_radius"):
        metadata["tracing_params"] = {
            "max_integration_radius": tracing_config.max_integration_radius,
            "dt": tracing_config.dt,
            "n_cores": tracing_config.n_cores,
        }

    # Add template metadata
    if hasattr(template_config, "property_system"):
        metadata["template_properties"] = [
            prop.key_name for prop in template_config.property_system.enabled_properties
        ]

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
