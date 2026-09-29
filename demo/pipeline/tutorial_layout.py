"""Shared tiny series used by the disk, memory, and post-processing tutorials."""

from pathlib import Path

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.factory_configs import (
    GridFactoryConfig,
    SphericalGridFactoryConfig,
)
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.workflow.line_batch_workflow_prep import (
    get_workflow_batch_load_template,
    merge_field_line_templates,
)

SERIES_NAME = "dipole_series"


def build_series(output_dir: Path) -> ExperimentSeriesConfig:
    """One dipole, two latitude counts, on a spherical grid.

    A list inside ``SphericalGridFactoryConfig`` is expanded by
    ``get_parameter_combinations``. Batches land in
    ``<output_dir>/dipole_series/lines``.
    """
    return ExperimentSeriesConfig(
        name=SERIES_NAME,
        output_dir=str(output_dir),
        description="Tiny dipole series for the tutorials",
        field=FieldConfig(
            components=[
                FieldComponentConfig(
                    moment_type="dipole",
                    B_magnitude=1.0,
                    moment_orientation_latitude_degree=30.0,
                )
            ]
        ),
        grid=GridFactoryConfig(
            grids=[SphericalGridFactoryConfig(nlon=4, nlat=[3, 4])]
        ),
        tracing=TracingConfig(
            max_integration_radius=20.0,
            save_results_to_disk=True,
            filename="dipole_series",
            max_memory_mb=100.0,
            skip_memory_estimation=True,
            max_step=1.0,
            verify_by_reversing=True,
        ),
        template=merge_field_line_templates(
            FieldLineTemplate.template_OFL_calc(),
            get_workflow_batch_load_template(),
        ),
    )
