"""Tests for PA snapshot data container."""

from unittest.mock import MagicMock

import numpy as np

import numpy as np

from psr_magpy.viz.snapshot import SnapshotData, categorize_oflr_map_for_r_lc


def test_snapshot_data_fields():
    series = MagicMock()
    series.name = "test_series"
    fls = MagicMock()
    grid = MagicMock()
    data = SnapshotData(
        experiment_number=3,
        R_LC=4.77,
        series_config=series,
        skymap=np.zeros((8, 16)),
        skymap_longitude_deg=np.linspace(-180, 180, 16),
        skymap_latitude_deg=np.linspace(-90, 90, 8),
        oflr_map=np.zeros(100),
        grid_indices=np.arange(100),
        oflr_image=np.zeros((8, 16)),
        oflr_longitude_deg=np.linspace(-180, 180, 16),
        oflr_latitude_deg=np.linspace(-90, 90, 8),
        field_line_set=fls,
        grid=grid,
        alpha_deg=60.0,
        BQ=0.5,
        period_ms=10.0,
        nside_skymap=16,
        nside_oflr=1024,
        image_size=(8, 16),
    )
    assert data.experiment_number == 3
    assert data.R_LC == 4.77
    assert data.skymap.shape == (8, 16)


def test_categorize_oflr_map_for_r_lc():
    oflr = np.array([2.0, 4.77, 5.0, np.nan])
    out = categorize_oflr_map_for_r_lc(oflr, 4.77)
    assert out[0] != out[0]  # nan
    assert out[1] == 1
    assert out[2] == 4.77
