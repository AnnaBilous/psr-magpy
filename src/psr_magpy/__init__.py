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

# Experiment system
from .experiment import ExperimentConfig, ExperimentSeriesConfig
from .experiment.factory_configs import (
    FieldComponentFactoryConfig,
    FieldFactoryConfig,
    GridFactoryConfig,
    SphericalGridFactoryConfig,
    TracingFactoryConfig,
)

# Common aliases for convenience
Dipole = DipoleFieldComponent
Quadrupole = QuadrupoleFieldComponent
Field = MultipoleField
Grid = SphericalGrid
Calculator = FieldLineCalculator
Params = TracingParameters
