"""Experiment module for managing field line calculations."""

from .experiment_config import ExperimentConfig
from .field_config import FieldConfig, FieldComponentConfig
from .grid_config import (
    GridConfig,
    SphericalGridConfig,
    RandomGridConfig,
    HealpixGridConfig,
    CustomGridConfig,
    create_full_healpix_grid,
    create_full_healpix_grid_for_experiment,
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
from .services.batch_data_reader import BatchDataReader
from .services.experiment_executor import ExperimentExecutor
from .services.experiment_lookup_service import (
    experiment_matches_thD_fQ as experiment_matches_thD_fQ_service,
    infer_experiment_count as infer_experiment_count_service,
    find_experiment_numbers_by_thD_fQ as find_experiment_numbers_by_thD_fQ_service,
)
from .services.series_builder import SeriesBuilder
from .services.series_executor import SeriesExecutor
from .services.subset_workflow_service import SubsetWorkflowService
from .services.ofl_reintegration_data_product_service import (
    OFLReintegrationDataProductService,
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
    "create_full_healpix_grid",
    "create_full_healpix_grid_for_experiment",
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
    "BatchDataReader",
    "ExperimentExecutor",
    "SeriesBuilder",
    "SeriesExecutor",
    "SubsetWorkflowService",
    "OFLReintegrationDataProductService",
    "experiment_matches_thD_fQ_service",
    "infer_experiment_count_service",
    "find_experiment_numbers_by_thD_fQ_service",
]
