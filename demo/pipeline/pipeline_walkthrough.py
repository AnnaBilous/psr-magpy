"""One dipole experiment on a tiny spherical grid.

Output goes to a new temporary directory printed at the end.
"""

import tempfile
from pathlib import Path

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_runner import run_experiment
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.grid_config import GridConfig, SphericalGridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_one_experiment_"))
    field = FieldConfig(
        components=[
            FieldComponentConfig(
                moment_type="dipole",
                moment_orientation_latitude_degree=30.0,
                moment_orientation_longitude_degree=0.0,
                B_magnitude=1.0,
            )
        ]
    )
    grid = GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3))
    tracing = TracingConfig(
        max_integration_radius=20.0,
        save_results_to_disk=True,
        filename=str(out / "dipole"),
        max_memory_mb=100.0,
        skip_memory_estimation=True,
        max_step=1.0,
    )
    config = ExperimentConfig(
        name="dipole",
        description="Tiny dipole experiment for the quickstart",
        field=field,
        grid=grid,
        tracing=tracing,
        template=FieldLineTemplate.template_OFL_calc(),
    )
    result = run_experiment(config, save_grids=False)
    print(f"output={out}")
    print(f"success={result.success} duration={result.duration:.1f}s")
    if result.error_message:
        print(f"error={result.error_message}")


if __name__ == "__main__":
    main()
