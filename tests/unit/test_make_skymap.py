"""Tests for skymap workflow helpers."""

import numpy as np
import pytest

from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.property_system import PropertyName
from psr_magpy.line.tracing_parameters import TracingDefaults
from psr_magpy.utils.exceptions import PSRMagPyPropertyError
from psr_magpy.workflow.line_batch_workflow_prep import (
    enrich_field_line_set_from_r_line,
    get_workflow_batch_load_template,
)
from psr_magpy.workflow.make_skymap import (
    DEFAULT_SKYMAP_OUTPUT_FILENAME_PATTERN,
    format_skymap_output_filename,
    make_skymap,
    skymap_pixel_counts_from_prepared_batch,
)
from psr_magpy.workflow.workflow_utils import create_healpix_grid


def test_format_skymap_output_filename_custom_pattern():
    from unittest.mock import MagicMock

    ec = MagicMock()
    ec.name = "healpix_nside_1024_grid_study_OFL_calc_main_009"
    grid = create_healpix_grid(HealpixGridConfig(nside=16))
    name = format_skymap_output_filename(
        "skymap_{experiment_number:06d}_RLC_{r_lc:.2f}_nside_{nside}.fits",
        experiment_config=ec,
        series_name="series",
        experiment_number=9,
        grid=grid,
        r_lc=23.85,
    )
    assert name == "skymap_000009_RLC_23.85_nside_16.fits"


def test_format_skymap_output_filename_default_pattern():
    from unittest.mock import MagicMock

    ec = MagicMock()
    ec.name = "healpix_nside_1024_grid_study_OFL_calc_main_009"
    name = format_skymap_output_filename(
        DEFAULT_SKYMAP_OUTPUT_FILENAME_PATTERN,
        experiment_config=ec,
        series_name="series",
        experiment_number=9,
        r_lc=23.85,
    )
    assert name == "skymap_000009_RLC_23.85.fits"


def test_make_skymap_threshold_matches_tracing_default():
    import inspect

    default = inspect.signature(make_skymap).parameters["threshold_line_distance"].default
    assert default == TracingDefaults.THRESHOLD_LINE_DISTANCE


def test_make_skymap_uses_full_sphere_for_subset_config(tmp_path):
    from types import SimpleNamespace

    cfg = HealpixGridConfig(
        nside=1,
        subset=True,
        grid_file_path="/nonexistent/subset_grid.fits",
    )
    skymap = make_skymap(
        SimpleNamespace(name="exp_000", metadata=None),
        tmp_path,
        "series",
        "lines",
        cfg,
        np.array([2.0]),
        save=False,
    )
    assert skymap.shape == (12,)
    assert np.all(skymap == 0)


def test_skymap_pixel_counts_raises_when_properties_missing():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    batch.n_field_lines = 1
    grid = create_healpix_grid(HealpixGridConfig(nside=1))
    with pytest.raises(PSRMagPyPropertyError, match="tangent_theta"):
        skymap_pixel_counts_from_prepared_batch(batch, grid)


def test_skymap_pixel_counts_from_none():
    grid = create_healpix_grid(HealpixGridConfig(nside=1))
    counts = skymap_pixel_counts_from_prepared_batch(None, grid)
    assert counts.shape == (12,)
    assert np.all(counts == 0)


def test_skymap_pixel_counts_nonempty():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    r_line = np.array(
        [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    batch.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([0], dtype=int)
    batch.property_system._properties[PropertyName.R_LINE] = [r_line.T]
    batch.n_field_lines = 1
    enrich_field_line_set_from_r_line(batch, template)

    grid = create_healpix_grid(HealpixGridConfig(nside=8))
    counts = skymap_pixel_counts_from_prepared_batch(batch, grid)
    assert counts.dtype == np.int64
    assert np.sum(counts) == 4
