"""Tests for :mod:`psr_magpy.viz.pa_panels`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.pa_theta_filter import filter_fieldline_points_by_theta
from psr_magpy.viz.pa import prepare_pa_scatter_data
from psr_magpy.viz.pa_panels import (
    filter_snapshot_for_theta,
    plot_snapshot_field_lines_panel,
    plot_snapshot_oflr_panel,
    plot_snapshot_pa_panel,
    plot_snapshot_skymap_panel,
    skymap_vmax_from_snapshot,
)
from psr_magpy.viz.field_line_style import FieldLinePlotConfig, NamedColorGenerator, PowerLawWidthGenerator
from psr_magpy.viz.pseudo3d import ProjectionConfig
from psr_magpy.viz.snapshot import SnapshotData

_PA_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_line=True,
    save_pa_line=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _line(footpoint_id, tangent_theta, tangent_phi, pa_line, r_line=None):
    fl = FieldLine.from_template(_PA_TEMPLATE)
    props = dict(
        footpoint_id=footpoint_id,
        tangent_theta=np.asarray(tangent_theta, dtype=float),
        tangent_phi=np.asarray(tangent_phi, dtype=float),
        pa_line=np.asarray(pa_line, dtype=float),
    )
    if r_line is not None:
        props["r_line"] = np.asarray(r_line, dtype=float)
    fl.set_property(**props)
    return fl


def _minimal_snapshot() -> SnapshotData:
    image_size = (8, 16)
    lon = np.linspace(-180, 180, image_size[1])
    lat = np.linspace(-90, 90, image_size[0])
    grid = HealpixGrid(nside=1)
    fls = FieldLineSet(
        [
            _line(
                0,
                [44, 45, 46],
                [0, 90, 180],
                [10, 20, 30],
                r_line=[[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            )
        ],
        copy_data=True,
    )
    series = MagicMock()
    series.name = "test_series"
    return SnapshotData(
        experiment_number=1,
        R_LC=4.77,
        series_config=series,
        skymap=np.linspace(0.0, 1.0, np.prod(image_size)).reshape(image_size),
        skymap_longitude_deg=lon,
        skymap_latitude_deg=lat,
        oflr_map=np.full(12, 4.77),
        grid_indices=np.arange(12),
        oflr_image=np.full(image_size, np.nan),
        oflr_longitude_deg=lon,
        oflr_latitude_deg=lat,
        field_line_set=fls,
        grid=grid,
        alpha_deg=60.0,
        BQ=0.5,
        period_ms=10.0,
        nside_skymap=16,
        nside_oflr=1,
        image_size=image_size,
    )


def test_skymap_vmax_from_snapshot():
    snap = _minimal_snapshot()
    vmax = skymap_vmax_from_snapshot(snap, percentile=100.0)
    assert vmax == pytest.approx(1.0)


def test_skymap_vmax_all_nan_returns_none():
    snap = _minimal_snapshot()
    snap.skymap[:] = np.nan
    assert skymap_vmax_from_snapshot(snap) is None


def test_plot_snapshot_skymap_panel_smoke():
    snap = _minimal_snapshot()
    fig, ax = plt.subplots()
    out_fig, out_ax, im = plot_snapshot_skymap_panel(
        snap, ax, theta_obs=45.0, vmax=0.5
    )
    assert out_fig is fig
    assert out_ax is ax
    assert im is not None
    assert ax.get_xlabel() == ""


def test_filter_snapshot_for_theta_matches_direct_filter():
    snap = _minimal_snapshot()
    filtered = filter_snapshot_for_theta(snap, 45.0, 0.5)
    expected = filter_fieldline_points_by_theta(
        snap.field_line_set, 45.0, 0.5, grid=snap.grid
    )
    assert len(filtered.pa_values) == len(expected.pa_values)


@patch("psr_magpy.viz.pa_panels.plot_OFLR_em_map")
@patch("psr_magpy.viz.pa_panels.plot_OFLR_map")
@patch("psr_magpy.viz.pa_panels.project_array_to_equirectangular")
def test_plot_snapshot_oflr_panel(mock_proj, mock_map, mock_em):
    snap = _minimal_snapshot()
    filtered = filter_snapshot_for_theta(snap, 45.0, 0.5)
    mock_map.return_value = (snap.skymap, plt.subplots()[1], None)
    mock_proj.return_value = (np.zeros(snap.image_size), snap.skymap_longitude_deg, snap.skymap_latitude_deg)

    fig, ax = plt.subplots()
    plot_snapshot_oflr_panel(snap, ax, 45.0, filtered)
    mock_map.assert_called_once()
    mock_proj.assert_called_once()
    mock_em.assert_called_once()


@patch("psr_magpy.viz.pa_panels.overlay_oflr_emission_scatter")
@patch("psr_magpy.viz.pa_panels.plot_OFLR_map")
def test_plot_snapshot_oflr_panel_scatter_overlay(mock_map, mock_scatter):
    snap = _minimal_snapshot()
    filtered = filter_snapshot_for_theta(snap, 45.0, 0.5)
    mock_map.return_value = (snap.skymap, plt.subplots()[1], None)

    fig, ax = plt.subplots()
    plot_snapshot_oflr_panel(snap, ax, 45.0, filtered, em_overlay="scatter")
    mock_map.assert_called_once()
    mock_scatter.assert_called_once()


def test_plot_snapshot_pa_panel_empty_message():
    snap = _minimal_snapshot()
    filtered = filter_fieldline_points_by_theta(
        snap.field_line_set, 200.0, 0.01, grid=snap.grid
    )
    scatter = prepare_pa_scatter_data(filtered, snap.R_LC)
    fig, ax = plt.subplots()
    plot_snapshot_pa_panel(snap, ax, 200.0, scatter)
    assert ax.texts
    assert "No PA data" in ax.texts[0].get_text()


def test_plot_snapshot_pa_panel_with_data():
    snap = _minimal_snapshot()
    filtered = filter_snapshot_for_theta(snap, 45.0, 0.5)
    scatter = prepare_pa_scatter_data(filtered, snap.R_LC)
    fig, ax = plt.subplots()
    plot_snapshot_pa_panel(snap, ax, 45.0, scatter, theta_tolerance=0.5)
    assert len(ax.collections) > 0 or len(ax.lines) > 0


@patch("psr_magpy.viz.pa_panels.plot_field_lines_on_axes")
def test_plot_snapshot_field_lines_panel(mock_fl):
    snap = _minimal_snapshot()
    filtered = filter_snapshot_for_theta(snap, 45.0, 0.5)
    scatter = prepare_pa_scatter_data(filtered, snap.R_LC)
    field_line_cfg = FieldLinePlotConfig(
        [NamedColorGenerator(colors=["gray"])],
        [PowerLawWidthGenerator(np.array([0.01]), np.array([0.1]), np.array([0.02]))],
    )
    projection = ProjectionConfig(
        XZview=5.0, YtoX=3.5, ZtoX=2.0, K=30.0, axis_half_extent=3.0
    )
    fig, axes = plt.subplots(1, 3)
    plot_snapshot_field_lines_panel(
        axes,
        snap,
        Path("/tmp/edges"),
        scatter,
        rot_ph0=[-90.0, 0.0, 90.0],
        incl=[-7.0, -7.0, -7.0],
        projection=projection,
        field_line_cfg=field_line_cfg,
    )
    mock_fl.assert_called_once()
