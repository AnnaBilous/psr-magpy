"""Contract tests for subset-vs-regular config precedence."""

import pytest

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.field_config import FieldConfig, FieldComponentConfig
from psr_magpy.experiment.grid_config import GridConfig, SphericalGridConfig
from psr_magpy.experiment.infrastructure.config_repository import ConfigRepository
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.exceptions import PSRMagPyIOError


def _make_series(output_dir: str) -> ExperimentSeriesConfig:
    return ExperimentSeriesConfig(
        name="subset_contract",
        field=FieldConfig(
            components=[FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)]
        ),
        grid=GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3)),
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_all(),
        output_dir=output_dir,
    )


def _variant(exp: ExperimentConfig, description: str) -> ExperimentConfig:
    return ExperimentConfig(
        name=exp.name,
        field=exp.field,
        grid=exp.grid,
        tracing=exp.tracing,
        template=exp.template,
        description=description,
        metadata=exp.metadata,
    )


@pytest.mark.filterwarnings("ignore:ExperimentSeriesConfig model I/O/workflow helpers.*:DeprecationWarning")
def test_load_experiments_prefers_regular_when_no_subset_flag(tmp_path):
    series = _make_series(str(tmp_path))
    exp = series.create_experiment_series(save_configs=False, create_directories=True)[0]
    repo = ConfigRepository()

    repo.save_experiment_config(series, _variant(exp, "regular"), subset=False, overwrite=True)
    repo.save_experiment_config(series, _variant(exp, "subset"), subset=True, overwrite=True)

    loaded = series.load_experiments()
    assert len(loaded) == 1
    assert loaded[0].description == "regular"


@pytest.mark.filterwarnings("ignore:ExperimentSeriesConfig model I/O/workflow helpers.*:DeprecationWarning")
def test_load_experiments_prefers_subset_when_subset_flag_enabled(tmp_path):
    series = _make_series(str(tmp_path))
    exp = series.create_experiment_series(save_configs=False, create_directories=True)[0]
    repo = ConfigRepository()

    repo.save_experiment_config(series, _variant(exp, "regular"), subset=False, overwrite=True)
    repo.save_experiment_config(series, _variant(exp, "subset"), subset=True, overwrite=True)

    series.metadata = {"has_subset_grids": True}
    loaded = series.load_experiments()
    assert len(loaded) == 1
    assert loaded[0].description == "subset"


@pytest.mark.filterwarnings("ignore:ExperimentSeriesConfig model I/O/workflow helpers.*:DeprecationWarning")
def test_load_experiments_raises_if_subset_expected_but_missing(tmp_path):
    series = _make_series(str(tmp_path))
    exp = series.create_experiment_series(save_configs=False, create_directories=True)[0]
    ConfigRepository().save_experiment_config(
        series, _variant(exp, "regular"), subset=False, overwrite=True
    )

    series.metadata = {"has_subset_grids": True}
    with pytest.raises(PSRMagPyIOError):
        series.load_experiments()
