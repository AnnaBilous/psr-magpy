"""Experiment module for managing field line calculations."""

from .experiment_config import (
    ExperimentConfig,
    FieldConfig,
    GridConfig,
    TracingConfig,
    FieldComponentConfig,
    SphericalGridConfig,
    RandomGridConfig,
    HealpixGridConfig,
    CustomGridConfig,
)
from .experiment_metadata import create_experiment_metadata
from .experiment_series import ExperimentSeriesConfig
from .experiment_series_runner import run_experiment_series
from .factory_configs import FieldFactoryConfig

__all__ = [
    "ExperimentConfig",
    "FieldConfig",
    "GridConfig",
    "TracingConfig",
    "FieldComponentConfig",
    "SphericalGridConfig",
    "RandomGridConfig",
    "HealpixGridConfig",
    "CustomGridConfig",
    "create_experiment_metadata",
    "ExperimentSeriesConfig",
    "run_experiment_series",
    "FieldFactoryConfig",
]
