"""Experiment module for managing field line calculations."""

from .experiment_config import ExperimentConfig
from .field_config import FieldConfig, FieldComponentConfig
from .grid_config import (
    GridConfig,
    SphericalGridConfig,
    RandomGridConfig,
    HealpixGridConfig,
    CustomGridConfig,
)
from .tracing_config import TracingConfig
from .experiment_metadata import create_experiment_metadata
from .experiment_series import ExperimentSeriesConfig
from .experiment_lookup import (
    experiment_matches_thD_fQ,
    infer_experiment_count,
    find_experiment_numbers_by_thD_fQ,
)
from .experiment_series_runner import (
    run_experiment_series,
    run_experiment_series_in_memory,
    run_experiment_series_in_memory_from_file,
    DeletionStrategy,
    OverwriteStrategy,
)
from .factory_configs import (
    FieldFactoryConfig,
    FieldComponentFactoryConfig,
    GridFactoryConfig,
    TracingFactoryConfig,
)

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
    "experiment_matches_thD_fQ",
    "infer_experiment_count",
    "find_experiment_numbers_by_thD_fQ",
    "run_experiment_series",
    "run_experiment_series_in_memory",
    "run_experiment_series_in_memory_from_file",
    "DeletionStrategy",
    "OverwriteStrategy",
    "FieldFactoryConfig",
    "FieldComponentFactoryConfig",
    "GridFactoryConfig",
    "TracingFactoryConfig",
]
