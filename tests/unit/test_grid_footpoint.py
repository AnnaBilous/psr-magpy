"""Tests for footpoint grid module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from tests.fixtures.conftest import setup_test_logger
from psr_magpy.grid.grid import HealpixGrid, SphericalGrid, RandomGrid, GridType

logger = setup_test_logger(__name__)


@pytest.fixture  # type: ignore
def temp_fits_file():
    """Provide temporary file path for FITS testing."""
    temp_path = tempfile.mktemp(suffix=".fits")
    yield Path(temp_path)
    # Clean up
    Path(temp_path).unlink(missing_ok=True)


class TestHealpixGrid:
    """Test suite for HEALPix grid."""

    def test_init(self):
        """Test initialization with valid parameters."""
        logger.info("Testing HEALPix grid initialization")
        grid = HealpixGrid(nside=4)
        assert grid.nside == 4
        assert grid.nline == 192  # 12 * nside^2
        assert grid.grid_type == GridType.HEALPIX
        assert not grid.partial
        logger.debug("Grid initialized successfully")

    def test_invalid_nside(self):
        """Test initialization with invalid nside."""
        with pytest.raises(ValueError, match="nside must be a power of 2, got 3"):  # type: ignore
            HealpixGrid(nside=3)
        with pytest.raises(ValueError, match="Input should be greater than 0"):  # type: ignore
            HealpixGrid(nside=0)

    def test_coordinates(self):
        """Test coordinate generation."""
        grid = HealpixGrid(nside=4)
        assert grid.footpoint_coordinates is not None
        assert len(grid.footpoint_coordinates.xyz[0]) == grid.nline

        # Test unit vectors
        r = np.sqrt(
            grid.footpoint_coordinates.xyz[0]**2
            + grid.footpoint_coordinates.xyz[1]**2
            + grid.footpoint_coordinates.xyz[2]**2
        )
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)

    def test_save_load(self, temp_fits_file):
        """Test saving and loading from FITS."""
        original = HealpixGrid(nside=4)
        original.save_to_fits(temp_fits_file)
        loaded = HealpixGrid.load_from_fits(temp_fits_file)

        assert loaded.nside == original.nside
        assert loaded.nline == original.nline
        assert loaded.grid_type == original.grid_type
        assert loaded.footpoint_coordinates is not None
        assert original.footpoint_coordinates is not None
        np.testing.assert_array_equal(
            loaded.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value,
        )


class TestSphericalGrid:
    """Test suite for spherical grid."""

    def test_init(self):
        """Test initialization with valid parameters."""
        grid = SphericalGrid(nlon=12, nlat=6)
        assert grid.nlon == 12
        assert grid.nlat == 6
        assert grid.nline == 12 * (6 - 2) + 2
        assert grid.grid_type == GridType.SPHERICAL

    def test_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):  # type: ignore
            SphericalGrid(nlon=0, nlat=6)
        with pytest.raises(ValueError):  # type: ignore
            SphericalGrid(nlon=12, nlat=0)

    def test_pole_tilt(self):
        """Test grid with tilted pole."""
        grid = SphericalGrid(nlon=12, nlat=6, pole_lat=45.0)
        # Check that points exist at high latitudes
        lats = 90 - grid.colatitude
        assert np.any(lats > 80)  # Some points near north pole
        assert np.any(lats < -80)  # Some points near south pole


class TestRandomGrid:
    """Test suite for random grid."""

    def test_init(self):
        """Test initialization with valid parameters."""
        n_points = 100
        grid = RandomGrid(nline_ori=n_points)
        assert grid.nline == n_points
        assert grid.grid_type == GridType.RANDOM

    def test_invalid_npoints(self):
        """Test initialization with invalid number of points."""
        with pytest.raises(ValueError):  # type: ignore
            RandomGrid(nline_ori=0)


class TestGridSubsetting:
    """Test suite for grid subsetting functionality."""

    @pytest.mark.parametrize("GridClass", [HealpixGrid, SphericalGrid, RandomGrid])
    def test_subset(self, GridClass):
        """Test subsetting for all grid types."""
        if GridClass == HealpixGrid:
            grid = GridClass(nside=4)
        elif GridClass == SphericalGrid:
            grid = GridClass(nlon=12, nlat=6)
        else:
            grid = GridClass(nline_ori=100)

        n_subset = grid.nline // 2
        indices = np.random.choice(grid.nline, size=n_subset, replace=False)
        subset = grid.subset_grid(indices)

        assert subset.nline == n_subset
        assert subset.partial
        np.testing.assert_array_equal(subset.indices, indices)

        # Check coordinates match
        for i, idx in enumerate(indices):
            np.testing.assert_array_equal(
                subset.footpoint_coordinates.xyz.value[:, i],
                grid.footpoint_coordinates.xyz.value[:, idx],
            )


def test_grid_properties():
    """Test common grid properties."""
    grids = [
        HealpixGrid(nside=4),
        SphericalGrid(nlon=12, nlat=6),
        RandomGrid(nline_ori=100),
    ]

    for grid in grids:
        # Test coordinate ranges
        assert np.all(grid.longitude >= 0)
        assert np.all(grid.longitude <= 360)
        assert np.all(grid.colatitude >= 0)
        assert np.all(grid.colatitude <= 180)

        # Test coordinate conversions
        xyz = grid.footpoint_coordinates.xyz.value
        r = np.sqrt(np.sum(xyz**2, axis=0))
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)
