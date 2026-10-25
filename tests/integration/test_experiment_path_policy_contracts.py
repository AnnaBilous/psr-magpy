"""Contract tests for experiment path and filename policy."""

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.field_config import FieldConfig, FieldComponentConfig
from psr_magpy.experiment.grid_config import GridConfig, SphericalGridConfig
from psr_magpy.experiment.infrastructure.path_policy import (
    configs_dir,
    experiment_config_filename,
    grids_dir,
    lines_dir,
    logs_dir,
    results_summary_filename,
    series_config_filename,
    series_dir,
    subset_experiment_config_filename,
)
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate


def _make_series_config(output_dir: str = ".") -> ExperimentSeriesConfig:
    return ExperimentSeriesConfig(
        name="contract_series",
        field=FieldConfig(
            components=[
                FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            ]
        ),
        grid=GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3)),
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_all(),
        output_dir=output_dir,
    )


def test_filename_contracts_are_stable():
    assert series_config_filename("demo") == "demo_config.yaml"
    assert experiment_config_filename("demo", 0) == "demo_000000.yaml"
    assert experiment_config_filename("demo", 12) == "demo_000012.yaml"
    assert subset_experiment_config_filename("demo", 12) == "demo_000012_subset.yaml"
    assert results_summary_filename("demo") == "demo_results_summary.yaml"


def test_directory_contracts_with_output_dir():
    assert series_dir("/tmp/out", "demo").as_posix().endswith("/tmp/out/demo")
    assert configs_dir("/tmp/out", "demo").as_posix().endswith("/tmp/out/demo/configs")
    assert grids_dir("/tmp/out", "demo").as_posix().endswith("/tmp/out/demo/grids")
    assert lines_dir("/tmp/out", "demo").as_posix().endswith("/tmp/out/demo/lines")
    assert logs_dir("/tmp/out", "demo").as_posix().endswith("/tmp/out/demo/logs")


def test_directory_contracts_without_output_dir():
    assert series_dir(None, "demo").as_posix() == "demo"
    assert configs_dir(None, "demo").as_posix() == "demo/configs"
    assert grids_dir(None, "demo").as_posix() == "demo/grids"
    assert lines_dir(None, "demo").as_posix() == "demo/lines"
    assert logs_dir(None, "demo").as_posix() == "demo/logs"


def test_series_getters_align_with_path_policy():
    series = _make_series_config("/tmp/out")
    assert series.get_configs_dir() == configs_dir("/tmp/out", "contract_series")
    assert series.get_grids_dir() == grids_dir("/tmp/out", "contract_series")
    assert series.get_lines_dir() == lines_dir("/tmp/out", "contract_series")
    assert series.get_logs_dir() == logs_dir("/tmp/out", "contract_series")
