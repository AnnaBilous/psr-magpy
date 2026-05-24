"""Tests for :mod:`psr_magpy.viz.pa_snapshot_layout`."""

import matplotlib

matplotlib.use("Agg")

from psr_magpy.viz.pa_snapshot_layout import PASnapshotLayout


def test_default_layout_panel_count():
    layout = PASnapshotLayout()
    axes = layout.create_axes()
    assert len(axes.field_lines) == 3
    assert axes.skymap is not axes.oflr
    assert axes.pa is not axes.skymap
    matplotlib.pyplot.close(axes.figure)


def test_field_line_slice_indices():
    layout = PASnapshotLayout()
    assert layout.field_line_slice(0) == (slice(0, 2), 1)
    assert layout.field_line_slice(2) == (slice(0, 2), 3)


def test_figure_size():
    layout = PASnapshotLayout(figure_size=(6.0, 5.0), constrained_layout=False)
    fig, _gs = layout.create_figure()
    assert fig.get_size_inches()[0] == 6.0
    matplotlib.pyplot.close(fig)
