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

# Experiment system
from .experiment import ExperimentConfig, ExperimentSeriesConfig, FieldComponentConfig, CustomGridConfig, GridConfig
from .experiment.experiment_config import TracingConfig, FieldConfig, HealpixGridConfig
from .experiment.factory_configs import (
    FieldComponentFactoryConfig,
    FieldFactoryConfig,
    GridFactoryConfig,
    SphericalGridFactoryConfig,
    TracingFactoryConfig,
    expand_grid_configs,
)
from .experiment.experiment_series_runner import run_experiment_series

# Common aliases for convenience
Dipole = DipoleFieldComponent
Quadrupole = QuadrupoleFieldComponent
Field = MultipoleField
Grid = SphericalGrid
Calculator = FieldLineCalculator
Params = TracingParameters
