"""Magnetic field line tracing package for pulsar magnetosphere modeling."""

__version__ = "0.1.0"

# Field components
from .field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from .field.multipole import MultipoleField

# Grids
from .grid.grid import SphericalGrid, RandomGrid, HealpixGrid, CustomGrid

# Line tracing
from .line.calculator import FieldLineCalculator
from .line.tracing_parameters import TracingParameters, IntegrationParameters
from .line.field_line import FieldLine

# Utils
from .utils.logging_config import get_logger
from .utils.physical_conversions import get_R_LC, get_P_ms

# Experiment system
from .experiment import (
    ExperimentConfig,
    ExperimentSeriesConfig,
    FieldComponentConfig,
    CustomGridConfig,
    GridConfig,
    TracingConfig,
    FieldConfig,
    HealpixGridConfig,
)
from .experiment.factory_configs import (
    FieldComponentFactoryConfig,
    FieldFactoryConfig,
    GridFactoryConfig,
    HealpixGridFactoryConfig,
    SphericalGridFactoryConfig,
    TracingFactoryConfig,
    expand_grid_configs,
)
from .experiment.experiment_series_runner import (
    run_experiment_series,
    run_experiment_series_in_memory,
    run_experiment_series_in_memory_from_file,
    DeletionStrategy,
    OverwriteStrategy,
)

__all__ = [
    # Version
    "__version__",
    # Field components
    "DipoleFieldComponent",
    "QuadrupoleFieldComponent",
    "MultipoleField",
    # Grids
    "SphericalGrid",
    "RandomGrid",
    "HealpixGrid",
    "CustomGrid",
    # Line tracing
    "FieldLineCalculator",
    "TracingParameters",
    "IntegrationParameters",
    "FieldLine",
    # Utils
    "get_logger",
    "get_R_LC",
    "get_P_ms",
    # Experiment system
    "ExperimentConfig",
    "ExperimentSeriesConfig",
    "FieldComponentConfig",
    "CustomGridConfig",
    "GridConfig",
    "TracingConfig",
    "FieldConfig",
    "HealpixGridConfig",
    # Factory configs
    "FieldComponentFactoryConfig",
    "FieldFactoryConfig",
    "GridFactoryConfig",
    "HealpixGridFactoryConfig",
    "SphericalGridFactoryConfig",
    "TracingFactoryConfig",
    "expand_grid_configs",
    # Experiment runners
    "run_experiment_series",
    "run_experiment_series_in_memory",
    "run_experiment_series_in_memory_from_file",
    "DeletionStrategy",
    "OverwriteStrategy",
]
