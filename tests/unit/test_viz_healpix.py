"""Tests for HEALPix equirectangular projection."""

import numpy as np
import healpy as hp
import pytest

from psr_magpy.viz.healpix import (
    equirectangular_coords,
    project_array_to_equirectangular,
)


def test_equirectangular_coords_shapes():
    lon, lat = equirectangular_coords((64, 128))
    assert lon.shape == (128,)
    assert lat.shape == (64,)


def test_project_array_shape_and_identity():
    nside = 8
    npix = hp.nside2npix(nside)
    data = np.arange(npix, dtype=float)
    grid_indices = np.arange(npix)
    image, lon, lat = project_array_to_equirectangular(
        data, grid_indices, nside, image_size=(32, 64)
    )
    assert image.shape == (32, 64)
    assert lon.shape == (64,)
    assert lat.shape == (32,)


def test_project_array_nan_propagation():
    nside = 4
    npix = hp.nside2npix(nside)
    data = np.full(npix, np.nan)
    data[0] = 1.0
    grid_indices = np.arange(npix)
    image, _, _ = project_array_to_equirectangular(
        data, grid_indices, nside, image_size=(16, 32)
    )
    assert np.any(np.isfinite(image))
    assert np.any(np.isnan(image))


def test_project_array_grid_indices_length_mismatch():
    nside = 4
    data = np.zeros(hp.nside2npix(nside))
    grid_indices = np.arange(10)
    with pytest.raises(ValueError, match="grid_indices length"):
        project_array_to_equirectangular(data, grid_indices, nside)
