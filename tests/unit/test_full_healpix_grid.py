"""Tests for full HEALPix grid creation."""

import healpy as hp

from psr_magpy.experiment.grid_config import (
    HealpixGridConfig,
    create_full_healpix_grid,
)


def test_create_full_healpix_grid_ignores_subset_file():
    cfg = HealpixGridConfig(
        nside=4,
        subset=True,
        grid_file_path="/nonexistent/subset_grid.fits",
    )
    grid = create_full_healpix_grid(cfg)
    assert grid.nside == 4
    assert grid.nline == hp.nside2npix(4)
