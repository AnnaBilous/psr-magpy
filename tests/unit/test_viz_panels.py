"""Smoke tests for PA snapshot panel plotters (``psr_magpy.viz``)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from psr_magpy.viz.healpix import equirectangular_coords
from psr_magpy.viz.los import overlay_observer_los
from psr_magpy.viz.oflr import plot_OFLR_em_map, plot_OFLR_map, plot_healpix_scatter
from psr_magpy.viz.pa import plot_PA_vs_phi, _phi_tick_labels, _transform_scatter_phi
from psr_magpy.viz.skymap import plot_skymap


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def map_grids():
    image_size = (32, 64)
    lon, lat = equirectangular_coords(image_size)
    lon2d, lat2d = np.meshgrid(lon, lat)
    data = np.random.uniform(0.1, 1.0, image_size)
    return data, lon2d, lat2d


def test_plot_skymap_smoke(map_grids):
    data, lon, lat = map_grids
    fig, ax, im = plot_skymap(
        data, lon, lat, vmin=0, vmax=1, theta_obs=45.0, cmap="viridis"
    )
    assert fig is ax.figure
    assert im is not None
    assert ax.images


def test_plot_oflr_map_smoke(map_grids):
    data, lon, lat = map_grids
    oflr = np.where(data > 0.5, 4.77, np.nan)
    fig, ax, im = plot_OFLR_map(oflr, lon, lat, theta_obs=30.0)
    assert im is not None


def test_plot_oflr_em_map_smoke(map_grids):
    data, lon, lat = map_grids
    em = np.where(data > 0.4, 20.0, 0.0)
    fig, ax, im = plot_OFLR_em_map(em, lon, lat, theta_obs=60.0)
    assert im is not None


def test_plot_pa_vs_phi_smoke():
    phi = np.linspace(0, 300, 40)
    pa = 30 * np.sin(np.deg2rad(phi))
    fig, ax = plot_PA_vs_phi(pa, phi, target_theta=45.0, alpha_angle=20.0, s=4)
    assert len(ax.collections) >= 1
    assert len(ax.lines) >= 1


def test_phi_roll_deg_transforms():
    phi = np.array([0.0, 90.0, 180.0])
    rolled = _transform_scatter_phi(phi, phi_roll_deg=270)
    np.testing.assert_allclose(rolled, [270.0, 180.0, 90.0])

    ticks = np.arange(0, 360, 45)
    _, x_labels = _phi_tick_labels(ticks, 270)
    np.testing.assert_array_equal(
        x_labels,
        np.array([270, 315, 0, 45, 90, 135, 180, 225, 90]),
    )
    _, x_labels_zero = _phi_tick_labels(ticks, 0)
    np.testing.assert_array_equal(
        x_labels_zero,
        np.append(ticks, [90]),
    )


def test_plot_pa_vs_phi_phi_roll_deg_kwarg():
    phi = np.linspace(0, 300, 20)
    pa = np.zeros_like(phi)
    _, ax = plot_PA_vs_phi(
        pa, phi, target_theta=45.0, alpha_angle=20.0, phi_roll_deg=0, s=4
    )
    label_vals = [int(t.get_text().replace("$\\degree$", "")) for t in ax.get_xticklabels()]
    assert label_vals[:8] == [0, 45, 90, 135, 180, 225, 270, 315]


def test_overlay_observer_los():
    fig, ax = plt.subplots()
    overlay_observer_los(ax, 45.0, color="r", linewidth=2.0)
    assert len(ax.lines) == 1
    overlay_observer_los(ax, None)
    assert len(ax.lines) == 1


def test_plot_healpix_scatter_smoke():
    nside = 1
    npix = 12
    em_map = np.full(npix, np.nan)
    em_map[0] = 30.0
    em_map[3] = 60.0
    fig, ax, sc = plot_healpix_scatter(em_map, nside, s=5)
    assert sc is not None


def test_panel_savefig_non_empty(tmp_path, map_grids):
    """Saved panel PNG is non-trivial (catches empty/broken renders)."""
    data, lon, lat = map_grids
    fig, ax, _ = plot_skymap(data, lon, lat, theta_obs=90.0)
    path = tmp_path / "skymap_panel.png"
    fig.savefig(path, dpi=50)
    assert path.stat().st_size > 500
