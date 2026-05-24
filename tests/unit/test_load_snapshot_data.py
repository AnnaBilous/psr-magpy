"""Tests for :func:`~psr_magpy.viz.snapshot.load_snapshot_data`."""

from unittest.mock import MagicMock, patch

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits

from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.io.data_products import DataProductSpec
from psr_magpy.viz.snapshot import SnapshotData, load_snapshot_data
from psr_magpy.workflow.workflow_utils import create_healpix_grid

_TEST_PRODUCTS = {
    "skymap": DataProductSpec(
        "skymaps",
        "skymap_{experiment_number:03d}_RLC_{r_lc:05.2f}",
        requires_r_lc=True,
    ),
    "OFLR_map": DataProductSpec("OFLR_maps", "OFLR_map_{experiment_number:03d}"),
    "PA": DataProductSpec(
        "extracted_PA",
        "extract_PA_{experiment_number:03d}_RLC_{r_lc:05.2f}",
        requires_r_lc=True,
    ),
    "OFLR_edges": DataProductSpec("OFLR_edges", is_directory=True),
}

_SKYMAP_PATTERN = "skymap_{experiment_number:03d}_RLC_{r_lc:05.2f}.fits"
_NSIDE = 4
_NPIX = hp.nside2npix(_NSIDE)


def _write_skymap_fits(path, values: np.ndarray) -> None:
    col = fits.Column(name="SKYMAP_0", format="D", array=values)
    table_hdu = fits.BinTableHDU.from_columns([col])
    primary = fits.PrimaryHDU()
    primary.header["NSIDE"] = _NSIDE
    primary.header["NPIX"] = len(values)
    fits.HDUList([primary, table_hdu]).writeto(path, overwrite=True)


def _write_oflr_map_fits(path, values: np.ndarray) -> None:
    col = fits.Column(name="OFLR_MAP", format="D", array=values)
    table_hdu = fits.BinTableHDU.from_columns([col])
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path, overwrite=True)


def _mock_series_config(nside: int = _NSIDE) -> MagicMock:
    dipole = MagicMock()
    dipole.moment_type = "dipole"
    dipole.moment_orientation_latitude_degree = 45.0

    quadrupole = MagicMock()
    quadrupole.moment_type = "quadrupole"
    quadrupole.B_magnitude = 0.3

    exp_config = MagicMock()
    exp_config.name = "series_000"
    exp_config.field.components = [dipole, quadrupole]
    exp_config.grid.healpix.nside = nside

    series = MagicMock()
    series.name = "test_series"
    series.load_experiment_config.return_value = exp_config
    return series


def test_load_snapshot_data_returns_snapshot_data(tmp_path):
    data_dir = tmp_path
    skymap_path = tmp_path / "skymap.fits"
    skymap_values = np.linspace(0.1, 1.0, _NPIX)
    _write_skymap_fits(skymap_path, skymap_values)

    oflr_values = np.full(_NPIX, 4.77)
    oflr_values[0] = 2.0
    oflr_path = tmp_path / "oflr.fits"
    _write_oflr_map_fits(oflr_path, oflr_values)

    mock_fls = MagicMock()
    mock_fls.n_field_lines = 2
    grid = create_healpix_grid(HealpixGridConfig(nside=_NSIDE))

    series = _mock_series_config()

    with (
        patch(
            "psr_magpy.viz.snapshot.resolve_skymap_product_path",
            return_value=skymap_path,
        ),
        patch(
            "psr_magpy.viz.snapshot.load_footpoint_map_product",
            return_value=(oflr_values, np.arange(_NPIX, dtype=np.int64)),
        ),
        patch(
            "psr_magpy.viz.snapshot.load_extracted_pa_product",
            return_value=mock_fls,
        ),
        patch(
            "psr_magpy.viz.snapshot.create_full_healpix_grid_for_experiment",
            return_value=grid,
        ),
    ):
        snapshot = load_snapshot_data(
            0,
            4.77,
            series,
            data_dir,
            _TEST_PRODUCTS,
            nside_skymap=_NSIDE,
            nside_oflr=_NSIDE,
            image_size=(8, 16),
            skymap_filename_pattern=_SKYMAP_PATTERN,
        )

    assert isinstance(snapshot, SnapshotData)
    assert snapshot.experiment_number == 0
    assert snapshot.R_LC == 4.77
    assert snapshot.alpha_deg == 45.0
    assert snapshot.BQ == 0.3
    assert snapshot.skymap.shape == (8, 16)
    assert snapshot.oflr_image.shape == (8, 16)
    assert snapshot.field_line_set is mock_fls
    assert snapshot.grid.nside == _NSIDE
    assert snapshot.nside_skymap == _NSIDE


def test_load_snapshot_data_missing_skymap_raises(tmp_path):
    series = _mock_series_config()
    missing = tmp_path / "missing.fits"

    with (
        patch(
            "psr_magpy.viz.snapshot.resolve_skymap_product_path",
            return_value=missing,
        ),
        pytest.raises(FileNotFoundError, match="Skymap file not found"),
    ):
        load_snapshot_data(
            0,
            4.77,
            series,
            tmp_path,
            _TEST_PRODUCTS,
            nside_skymap=_NSIDE,
            nside_oflr=_NSIDE,
            skymap_filename_pattern=_SKYMAP_PATTERN,
        )
