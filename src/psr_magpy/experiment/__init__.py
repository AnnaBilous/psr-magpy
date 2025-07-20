# Simple experiment interface
from .experiment_config import (
    ExperimentConfig, FieldConfig, GridConfig, TracingConfig, FieldComponentConfig,
    SphericalGridConfig, RandomGridConfig, HealpixGridConfig, CustomGridConfig
)
from .experiment_runner import ExperimentRunner, run_experiment, run_experiment_from_file

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
    "ExperimentRunner",
    "run_experiment",
    "run_experiment_from_file"
]
