"""Expand mixed grids and fields, then run one saved series from its YAML file."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_series_runner import (
    run_experiment_series_from_file,
    run_experiment_series_in_memory_from_file,
)
from psr_magpy.experiment.factory_configs import (
    CustomGridFactoryConfig,
    FieldComponentFactoryConfig,
    FieldFactoryConfig,
    GridFactoryConfig,
    HealpixGridFactoryConfig,
    RandomGridFactoryConfig,
    SphericalGridFactoryConfig,
    TemplateFactoryConfig,
    TracingFactoryConfig,
    expand_grid_configs,
)
from psr_magpy.line.field_line_template import FieldLineTemplate

from tutorial_layout import build_series


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_factories_"))
    grid = GridFactoryConfig(
        grids=[
            SphericalGridFactoryConfig(nlon=4, nlat=3),
            HealpixGridFactoryConfig(nside=1),
            RandomGridFactoryConfig(nline_ori=4),
            CustomGridFactoryConfig(
                nline_ori=2,
                coordinates=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ),
        ]
    )
    expanded = expand_grid_configs(grid)
    series = ExperimentSeriesConfig(
        name="factory_series",
        output_dir=str(out),
        field=FieldFactoryConfig(
            components=[
                FieldComponentFactoryConfig(moment_type="dipole", B_magnitude=1.0),
                FieldComponentFactoryConfig(moment_type="quadrupole", B_magnitude=0.5),
            ]
        ),
        grid=grid,
        tracing=TracingFactoryConfig(
            max_integration_radius=20.0,
            skip_memory_estimation=True,
            save_results_to_disk=False,
        ),
        template=TemplateFactoryConfig(
            templates=[FieldLineTemplate.template_OFL_calc()]
        ),
    )
    experiments = series.create_experiment_series(
        save_configs=False,
        create_directories=False,
    )
    print(f"expanded_grids={len(expanded)}")
    print(f"experiments={len(experiments)}")

    disk = build_series(out / "disk")
    yaml_path = out / "dipole_series_config.yaml"
    disk.save(str(yaml_path))
    disk_result = run_experiment_series_from_file(
        yaml_path,
        input_directory=out / "disk",
        save_grids=False,
    )
    memory_result = run_experiment_series_in_memory_from_file(
        yaml_path,
        input_directory=out / "memory",
        max_memory_mb=200.0,
        save_grids=False,
    )
    print(f"output={out}")
    print(f"from_file={disk_result.successful_experiments}/{disk_result.total_experiments}")
    print(
        f"from_file_memory={memory_result.successful_experiments}/{memory_result.total_experiments}"
    )


if __name__ == "__main__":
    main()
