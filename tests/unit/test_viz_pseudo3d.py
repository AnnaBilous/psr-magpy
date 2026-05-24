"""Tests for :mod:`psr_magpy.viz.pseudo3d`."""

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.viz.field_line_style import (
    FieldLinePlotConfig,
    NamedColorGenerator,
    PowerLawWidthGenerator,
)
from psr_magpy.viz.pseudo3d import (
    ProjectionConfig,
    create_plot_figure,
    extract_field_lines_from_set,
    load_oflr_edge_lines_for_r_lc,
    plot_3d_axes_inset,
    plot_field_lines_for_r_lc,
    plot_field_lines_on_axes,
    plot_scattered_points,
)


_TEST_FIELD_LINE_CFG = FieldLinePlotConfig(
    [NamedColorGenerator(colors=["LightSlateGray", "SlateGray", "DarkSlateGray"])],
    [
        PowerLawWidthGenerator(
            np.array([0.002]), np.array([0.08]), np.array([0.01])
        )
    ],
    color_NS="PaleTurquoise",
    color_space="w",
)


def _test_field_line_cfg() -> FieldLinePlotConfig:
    return _TEST_FIELD_LINE_CFG


def test_projection_config_axis_limit_default():
    cfg = ProjectionConfig(XZview=10.0, YtoX=3.5, ZtoX=2.0, K=30)
    assert cfg.axis_limit() == 10.0


def test_projection_config_axis_limit_override():
    cfg = ProjectionConfig(
        XZview=10.0, YtoX=3.5, ZtoX=2.0, K=30, axis_half_extent=7.0
    )
    assert cfg.axis_limit() == 7.0


def test_extract_field_lines_from_set():
    fl = FieldLine(save_r_line=True, save_footpoint_id=True)
    fl.set_property(r_line=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    fls = FieldLineSet(fl, copy_data=True)
    lines = extract_field_lines_from_set(fls)
    assert len(lines) == 1
    assert lines[0].shape == (2, 3)


def _sample_field_line_set() -> FieldLineSet:
    fl = FieldLine(save_r_line=True, save_footpoint_id=True)
    fl.set_property(
        r_line=[
            [1.0, 0.0, 0.0],
            [1.5, 0.1, 0.0],
            [2.0, 0.2, 0.0],
        ]
    )
    return FieldLineSet(fl, copy_data=True)


def test_create_plot_figure_smoke():
    fig, ax, lines, zorder_layer = create_plot_figure(
        XZview=5.0,
        ZtoX=1.0,
        Nlayers=20,
        K=30.0,
        cfg=_test_field_line_cfg(),
        n_plots=1,
    )
    assert fig is not None
    assert ax is not None
    assert len(lines) == 1
    assert len(zorder_layer) == 20
    plt.close(fig)


def test_plot_field_lines_for_r_lc_smoke():
    fls = _sample_field_line_set()
    fig, ax, line_groups, zorder_layer = create_plot_figure(
        5.0, 1.0, 20, 30.0, _test_field_line_cfg(), 1
    )
    plot_field_lines_for_r_lc(
        fls,
        line_groups,
        rotph0=-90.0,
        incl=-7.0,
        XZview=5.0,
        YtoX=3.5,
        K=30.0,
        zorder_layer=zorder_layer,
        color_func=_test_field_line_cfg().color_functions[0],
        plot_idx=0,
    )
    assert len(ax.lines) > 0 or len(ax.collections) > 0
    plt.close(fig)


def test_load_oflr_edge_lines_missing_dir(tmp_path):
    assert (
        load_oflr_edge_lines_for_r_lc(tmp_path / "missing", 0, 4.77) is None
    )


def test_plot_field_lines_on_axes_no_edges(tmp_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    projection = ProjectionConfig(
        XZview=5.0, YtoX=3.5, ZtoX=1.0, K=30, axis_half_extent=4.0
    )
    with patch(
        "psr_magpy.viz.pseudo3d.load_oflr_edge_lines_for_r_lc", return_value=None
    ):
        plot_field_lines_on_axes(
            list(axes),
            oflr_edges_dir=tmp_path,
            experiment_number=0,
            R_LC=4.77,
            rotph0=[-90, 0, 90],
            incl=[-7, -7, -7],
            projection=projection,
            cfg=_test_field_line_cfg(),
        )
    plt.close(fig)


def test_plot_field_lines_on_axes_with_edge_set():
    fig, ax = plt.subplots()
    projection = ProjectionConfig(XZview=5.0, YtoX=3.5, ZtoX=1.0, K=30)
    fls = _sample_field_line_set()
    plot_field_lines_on_axes(
        [ax],
        oflr_edges_dir=Path("/unused"),
        experiment_number=0,
        R_LC=4.77,
        rotph0=[-90.0],
        incl=[-7.0],
        projection=projection,
        cfg=_test_field_line_cfg(),
        edge_field_line_set=fls,
        show_axes_inset=True,
    )
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_scattered_points_bad_shape():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="shape"):
        plot_scattered_points(
            np.zeros((2, 3)),
            np.array([0.0, 1.0]),
            ax,
            0,
            0,
            5.0,
            3.5,
            30.0,
            np.arange(10),
        )
    plt.close(fig)


def test_plot_scattered_points_smoke():
    fig, ax = plt.subplots()
    coords = np.array([[1, 2], [0, 0], [0, 0]], dtype=float)
    values = np.array([0.0, 1.0])
    zorder_layer = np.arange(10)
    plot_scattered_points(
        coords,
        values,
        ax,
        rotph0=-90.0,
        incl=-7.0,
        XZview=5.0,
        YtoX=3.5,
        K=30.0,
        zorder_layer=zorder_layer,
        colors=np.array([[1, 0, 0, 1], [0, 0, 1, 1]]),
    )
    plt.close(fig)


def test_plot_3d_axes_inset_smoke():
    fig, ax = plt.subplots(figsize=(3, 3))
    plot_3d_axes_inset(ax, rotph0=-90.0, incl=-7.0)
    assert len(fig.axes) >= 2
    plt.close(fig)
