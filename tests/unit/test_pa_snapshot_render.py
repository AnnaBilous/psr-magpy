"""Tests for :mod:`psr_magpy.viz.pa_snapshot_render`."""

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
from psr_magpy.viz.pa_snapshot_render import (
    SnapshotRenderConfig,
    format_pa_snapshot_filename,
    format_pa_snapshot_title,
    prepare_pa_scatter_data,
    render_pa_snapshot_batch,
    render_pa_snapshot_panel,
    resolve_pa_snapshot_save_dir,
)
from psr_magpy.viz.presets import DEFAULT_THETA_TOLERANCE, pa_snapshot_projection_config
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
        experiment_number=3,
        R_LC=4.77,
        series_config=series,
        skymap=np.random.uniform(0.1, 1.0, image_size),
        skymap_longitude_deg=lon,
        skymap_latitude_deg=lat,
        oflr_map=np.full(12, 4.77),
        grid_indices=np.arange(12),
        oflr_image=np.full(image_size, 4.77),
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


def test_format_pa_snapshot_title():
    snap = _minimal_snapshot()
    title = format_pa_snapshot_title(snap, 45.0)
    assert r"\theta_\mathrm{obs}=45" in title
    assert "60" in title


def test_format_pa_snapshot_filename():
    snap = _minimal_snapshot()
    name = format_pa_snapshot_filename(snap, 45.0)
    assert name.endswith(".png")
    assert "003" in name


def test_prepare_pa_scatter_data_sorts_by_radius():
    snap = _minimal_snapshot()
    filtered = filter_fieldline_points_by_theta(
        snap.field_line_set, 45.0, 0.5, grid=snap.grid
    )
    scatter = prepare_pa_scatter_data(filtered, snap.R_LC)
    assert scatter.scatter_colors is not None
    radii = np.linalg.norm(scatter.r_line_stacked, axis=1)
    assert np.all(radii[:-1] <= radii[1:])


def test_resolve_pa_snapshot_save_dir_explicit():
    assert resolve_pa_snapshot_save_dir(Path("/tmp/out")) == Path("/tmp/out")


def test_resolve_pa_snapshot_save_dir_from_env(monkeypatch):
    monkeypatch.setenv("PA_SNAPSHOT_OUTPUT_DIR", "/tmp/from_env")
    assert resolve_pa_snapshot_save_dir() == Path("/tmp/from_env")


def test_snapshot_render_config_defaults():
    cfg = SnapshotRenderConfig()
    assert cfg.theta_tolerance == DEFAULT_THETA_TOLERANCE
    assert cfg.apply_rcparams is True


def test_snapshot_render_config_resolved_save_dir_explicit():
    cfg = SnapshotRenderConfig(save_dir=Path("/tmp/out"), resolve_save_dir_from_env=False)
    assert cfg.resolved_save_dir() == Path("/tmp/out")


def test_snapshot_render_config_projection_for():
    cfg = SnapshotRenderConfig()
    expected = pa_snapshot_projection_config(4.77)
    proj = cfg.projection_for(4.77)
    assert proj.XZview == pytest.approx(expected.XZview)


@patch("psr_magpy.viz.pa_snapshot_render.render_pa_snapshot_panel")
@patch("psr_magpy.viz.pa_snapshot_render.save_pa_snapshot_panel")
def test_render_pa_snapshot_batch_calls_panel_and_save(
    mock_save, mock_render, tmp_path
):
    snap = _minimal_snapshot()
    mock_axes = MagicMock()
    mock_render.return_value = mock_axes
    mock_save.return_value = tmp_path / "out.png"

    paths = render_pa_snapshot_batch(
        snap,
        [30.0, 45.0],
        Path("/tmp/oflr_edges"),
        config=SnapshotRenderConfig(
            save_dir=tmp_path,
            resolve_save_dir_from_env=False,
            apply_rcparams=False,
        ),
    )

    assert mock_render.call_count == 2
    assert mock_save.call_count == 2
    assert len(paths) == 2
    assert paths[0] == tmp_path / "out.png"


@patch("psr_magpy.viz.pa_snapshot_render.plot_field_lines_on_axes")
def test_render_pa_snapshot_panel_smoke(mock_fl_plot):
    snap = _minimal_snapshot()
    panels = render_pa_snapshot_panel(
        snap,
        45.0,
        Path("/tmp/oflr_edges"),
        config=SnapshotRenderConfig(theta_tolerance=0.5),
    )
    assert len(panels.field_lines) == 3
    assert panels.figure._suptitle is not None
    mock_fl_plot.assert_called_once()
