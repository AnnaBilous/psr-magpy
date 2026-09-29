"""Calls for public functions the coverage audit showed the suite never entered."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_result import ExperimentResult, SeriesResult
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.grid_config import GridConfig, HealpixGridConfig, SphericalGridConfig
from psr_magpy.experiment.services.formatting import format_duration
from psr_magpy.experiment.services.subset_workflow_service import SubsetWorkflowService
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.field.components import DipoleFieldComponent
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.exceptions import (
    PSRMagPyExperimentError,
    PSRMagPyFieldError,
    PSRMagPyIOError,
)
from psr_magpy.viz.field_2d import plot_field_lines_2d
from psr_magpy.workflow.coarse_to_fine import create_fine_subset_workflow
from psr_magpy.workflow.extract_PA import extract_PA_processor, make_extract_PA
from psr_magpy.workflow.extract_ofl_map import (
    extract_ofl_map,
    find_OFL_regions_from_batch_files,
)
from psr_magpy.workflow.make_skymap import make_skymap


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _series(name: str, output_dir: Path) -> ExperimentSeriesConfig:
    return ExperimentSeriesConfig(
        name=name,
        field=FieldConfig(
            components=[FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)]
        ),
        grid=GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3)),
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_all(),
        output_dir=str(output_dir),
    )


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(
        name="pub_000",
        field=FieldConfig(
            components=[FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)]
        ),
        grid=GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3)),
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_all(),
    )


def test_format_duration_units():
    assert format_duration(10) == "10.0s"
    assert format_duration(90) == "1.5min"
    assert format_duration(3600) == "1.0h"
    assert format_duration(86400) == "1.0d"


def test_series_result_summary_names_the_series(capsys):
    result = SeriesResult(
        series_name="pub_series",
        total_experiments=1,
        experiments_in_this_session=1,
        successful_experiments=1,
        failed_experiments=0,
        total_duration=10.0,
        experiment_results=[
            ExperimentResult(experiment_name="pub_000", success=True, duration=10.0)
        ],
        output_directory=Path("/tmp/pub_series"),
    )
    result.print_summary(include_details=True)
    out = capsys.readouterr().out
    assert "pub_series" in out
    assert "pub_000" in out
    assert "10.0s" in out


def test_plot_field_lines_2d_labels_the_plane():
    field = MultipoleField([DipoleFieldComponent(B_magnitude=1.0)])
    ax = plot_field_lines_2d(
        field,
        plane="xy",
        fixed_coord=0.0,
        x_range=(1.0, 2.0),
        y_range=(1.0, 2.0),
        n_points=3,
    )
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    with pytest.raises(PSRMagPyFieldError, match="Invalid plane"):
        plot_field_lines_2d(
            field, plane="bad", fixed_coord=0.0, x_range=(1.0, 2.0), y_range=(1.0, 2.0)
        )


def test_extract_ofl_map_requires_lines_directory(tmp_path):
    series = _series("ofl_series", tmp_path)
    with pytest.raises(PSRMagPyIOError, match="Lines directory must exist"):
        extract_ofl_map(series, np.array([10.0]), save=False)


def test_find_ofl_regions_rejects_empty_lines_dir(tmp_path):
    lines = tmp_path / "lines"
    lines.mkdir()
    with pytest.raises(ValueError, match="No valid field line data"):
        find_OFL_regions_from_batch_files(
            _experiment(), lines, np.array([10.0]), save=False
        )


def test_create_fine_subset_requires_coarse_directories(tmp_path):
    coarse = _series("coarse", tmp_path / "coarse_root")
    fine = _series("fine", tmp_path / "fine_root")
    with pytest.raises(PSRMagPyExperimentError, match="must exist"):
        create_fine_subset_workflow(coarse, fine, R_LC=10.0, r_max=5.0)


def test_subset_workflow_service_requires_coarse_directories(tmp_path):
    coarse = _series("coarse", tmp_path / "coarse_root")
    fine = _series("fine", tmp_path / "fine_root")
    with pytest.raises(PSRMagPyExperimentError, match="must exist"):
        SubsetWorkflowService().create_subset_series_from_ofl(
            fine_series_config=fine,
            coarse_series_config=coarse,
            R_LC=10.0,
        )


def test_make_skymap_rejects_unknown_line_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown line_source"):
        make_skymap(
            _experiment(),
            tmp_path,
            "pub_series",
            "not-a-source",
            HealpixGridConfig(nside=1),
            np.array([10.0]),
            save=False,
        )


def test_make_extract_pa_requires_a_batch_location():
    with pytest.raises(PSRMagPyIOError, match="lines_dir"):
        make_extract_PA(
            _experiment(),
            np.array([10.0]),
            theta_tolerance=0.5,
            grid_theta=np.array([45.0]),
        )


def test_extract_pa_processor_empty_dir_returns_no_lines(tmp_path):
    lines = tmp_path / "lines"
    lines.mkdir()
    result = extract_PA_processor(
        _experiment(),
        r_lc=10.0,
        lines_dir=str(lines),
        theta_tolerance=0.5,
        grid_theta=np.array([45.0]),
        save=False,
    )
    assert result.n_field_lines == 0
