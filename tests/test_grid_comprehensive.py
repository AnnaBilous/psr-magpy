"""Comprehensive tests for grid module."""

import pytest
import numpy as np
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from astropy import units as u
from astropy.coordinates import CartesianRepresentation, PhysicsSphericalRepresentation
from astropy.io import fits as pf

from utils import setup_test_logger, temp_dir, rng
from psr_magpy.grid.grid import (
    FootpointGrid, GridType, SphericalGrid, HealpixGrid, 
    RandomGrid, CustomGrid, SphericalGridParams, HealpixGridParams,
    RandomGridParams, CustomGridParams
)

logger = setup_test_logger(__name__)


class TestGridType:
    """Test GridType enumeration."""
    
    def test_grid_type_values(self):
        """Test all grid type values."""
        assert GridType.SPHERICAL.value == "spherical"
        assert GridType.HEALPIX.value == "healpix"
        assert GridType.RANDOM.value == "random"
        assert GridType.CUSTOM.value == "custom"
    
    def test_grid_type_from_string(self):
        """Test creating GridType from string."""
        assert GridType("spherical") == GridType.SPHERICAL
        assert GridType("healpix") == GridType.HEALPIX
        assert GridType("random") == GridType.RANDOM
        assert GridType("custom") == GridType.CUSTOM


class TestSphericalGridComprehensive:
    """Comprehensive test suite for SphericalGrid."""
    
    def test_init_valid_params(self):
        """Test initialization with various valid parameters."""
        test_cases = [
            (8, 4, 0.0),
            (16, 8, 30.0),
            (32, 16, -45.0),
            (64, 32, 90.0),
        ]
        
        for nlon, nlat, pole_lat in test_cases:
            grid = SphericalGrid(nlon, nlat, pole_lat)
            assert grid.nlon == nlon
            assert grid.nlat == nlat
            assert abs(grid.pole_lat.deg - pole_lat) < 1e-10
            assert grid.grid_type == GridType.SPHERICAL
            assert not grid.partial
    
    def test_init_invalid_params(self):
        """Test initialization with invalid parameters."""
        # Invalid nlon
        with pytest.raises(ValueError):
            SphericalGrid(nlon=0, nlat=4)
        with pytest.raises(ValueError):
            SphericalGrid(nlon=-1, nlat=4)
        
        # Invalid nlat
        with pytest.raises(ValueError):
            SphericalGrid(nlon=8, nlat=0)
        with pytest.raises(ValueError):
            SphericalGrid(nlon=8, nlat=-1)
    
    def test_coordinate_generation(self):
        """Test coordinate generation and properties."""
        grid = SphericalGrid(nlon=12, nlat=6)
        
        # Check coordinate arrays
        assert grid.footpoint_coordinates is not None
        assert len(grid.footpoint_coordinates.x) == grid.nline
        assert len(grid.footpoint_coordinates.y) == grid.nline
        assert len(grid.footpoint_coordinates.z) == grid.nline
        
        # Check unit vectors
        xyz = grid.footpoint_coordinates.xyz.value
        r = np.sqrt(np.sum(xyz**2, axis=0))
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)
        
        # Check coordinate ranges
        assert np.all(grid.longitude >= 0)
        assert np.all(grid.longitude < 360)
        assert np.all(grid.colatitude >= 0)
        assert np.all(grid.colatitude <= 180)
    
    def test_pole_duplicate_removal(self):
        """Test that duplicate pole points are removed."""
        grid = SphericalGrid(nlon=8, nlat=4)
        
        # Count points at poles
        pole_mask = (grid.colatitude == 0) | (grid.colatitude == 180)
        pole_count = np.sum(pole_mask)
        
        # Should have exactly 2 pole points (north and south)
        assert pole_count == 2
    
    def test_pole_rotation(self):
        """Test grid rotation with pole_lat."""
        grid = SphericalGrid(nlon=12, nlat=6, pole_lat=45.0)
        
        # Check that rotation was applied
        # Points should exist in rotated coordinate system
        spherical = grid.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
        theta_rotated = spherical.theta.to(u.deg).value
        
        # Should have points at various latitudes after rotation
        assert np.any(theta_rotated > 80)
        assert np.any(theta_rotated < 100)
    
    def test_coordinate_properties(self):
        """Test coordinate property calculations."""
        grid = SphericalGrid(nlon=8, nlat=4)
        
        # Test colatitude property
        colat = grid.colatitude
        assert len(colat) == grid.nline
        assert np.all(colat >= 0)
        assert np.all(colat <= 180)
        
        # Test longitude property
        lon = grid.longitude
        assert len(lon) == grid.nline
        assert np.all(lon >= 0)
        assert np.all(lon < 360)
        
        # Test consistency with direct calculation
        spherical = grid.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
        np.testing.assert_allclose(colat, spherical.theta.to(u.deg).value)
        np.testing.assert_allclose(lon, spherical.phi.to(u.deg).value)
    
    def test_repr(self):
        """Test string representation."""
        grid = SphericalGrid(nlon=16, nlat=8, pole_lat=30.0)
        repr_str = repr(grid)
        assert "SphericalGrid" in repr_str
        assert "nlon=16" in repr_str
        assert "nlat=8" in repr_str
        assert "pole_lat=30.0" in repr_str
    
    def test_copy(self):
        """Test grid copying."""
        original = SphericalGrid(nlon=12, nlat=6, pole_lat=15.0)
        copied = original.copy()
        
        # Check attributes
        assert copied.nlon == original.nlon
        assert copied.nlat == original.nlat
        assert abs(copied.pole_lat.deg - original.pole_lat.deg) < 1e-10
        assert copied.grid_type == original.grid_type
        
        # Check coordinates
        np.testing.assert_allclose(
            copied.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value
        )
        
        # Check independence
        copied.nlon = 999
        assert original.nlon != copied.nlon


class TestHealpixGridComprehensive:
    """Comprehensive test suite for HealpixGrid."""
    
    def test_init_valid_nside(self):
        """Test initialization with valid nside values."""
        valid_nsides = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for nside in valid_nsides:
            grid = HealpixGrid(nside)
            assert grid.nside == nside
            assert grid.nline == 12 * nside**2
            assert grid.grid_type == GridType.HEALPIX
            assert not grid.partial
    
    def test_init_invalid_nside(self):
        """Test initialization with invalid nside values."""
        # nside <= 0
        with pytest.raises(ValueError, match="greater than 0"):
            HealpixGrid(0)

        # nside not a power of 2, but > 0
        invalid_nsides = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        for nside in invalid_nsides:
            with pytest.raises(ValueError, match="power of 2"):
                HealpixGrid(nside)
    
    def test_coordinate_generation(self):
        """Test HEALPix coordinate generation."""
        grid = HealpixGrid(nside=4)
        
        # Check coordinate arrays
        assert grid.footpoint_coordinates is not None
        assert len(grid.footpoint_coordinates.x) == grid.nline
        
        # Check unit vectors
        xyz = grid.footpoint_coordinates.xyz.value
        r = np.sqrt(np.sum(xyz**2, axis=0))
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)
        
        # Check coordinate ranges
        assert np.all(grid.longitude >= 0)
        assert np.all(grid.longitude < 360)
        assert np.all(grid.colatitude >= 0)
        assert np.all(grid.colatitude <= 180)
    
    def test_uniformity(self):
        """Test that HEALPix provides uniform coverage."""
        grid = HealpixGrid(nside=8)
        
        # Check that we have points at various latitudes
        colat = grid.colatitude
        assert np.any(colat < 30)  # Near poles
        assert np.any((colat >= 30) & (colat <= 150))  # Mid-latitudes
        assert np.any(colat > 150)  # Near poles
    
    def test_repr(self):
        """Test string representation."""
        grid = HealpixGrid(nside=16)
        repr_str = repr(grid)
        assert "HealpixGrid" in repr_str
        assert "nside=16" in repr_str
    
    def test_copy(self):
        """Test grid copying."""
        original = HealpixGrid(nside=8)
        copied = original.copy()
        
        assert copied.nside == original.nside
        assert copied.nline == original.nline
        assert copied.grid_type == original.grid_type
        
        np.testing.assert_allclose(
            copied.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value
        )


class TestRandomGridComprehensive:
    """Comprehensive test suite for RandomGrid."""
    
    def test_init_valid_npoints(self):
        """Test initialization with valid point counts."""
        test_npoints = [1, 10, 100, 1000, 10000]
        
        for npoints in test_npoints:
            grid = RandomGrid(nline_ori=npoints)
            assert grid.nline == npoints
            assert grid.grid_type == GridType.RANDOM
            assert not grid.partial
    
    def test_init_invalid_npoints(self):
        """Test initialization with invalid point counts."""
        invalid_npoints = [0, -1, -10]
        
        for npoints in invalid_npoints:
            with pytest.raises(ValueError):
                RandomGrid(nline_ori=npoints)
    
    def test_coordinate_generation(self):
        """Test random coordinate generation."""
        grid = RandomGrid(nline_ori=1000)
        
        # Check coordinate arrays
        assert grid.footpoint_coordinates is not None
        assert len(grid.footpoint_coordinates.x) == grid.nline
        
        # Check unit vectors
        xyz = grid.footpoint_coordinates.xyz.value
        r = np.sqrt(np.sum(xyz**2, axis=0))
        np.testing.assert_allclose(r, 1.0, rtol=1e-10)
        
        # Check coordinate ranges
        assert np.all(grid.longitude >= 0)
        assert np.all(grid.longitude < 360)
        assert np.all(grid.colatitude >= 0)
        assert np.all(grid.colatitude <= 180)
    
    def test_uniformity(self):
        """Test that random grid provides reasonable coverage."""
        grid = RandomGrid(nline_ori=10000)
        
        # Check latitude distribution (should be roughly uniform in cos(theta))
        colat = grid.colatitude
        cos_colat = np.cos(np.radians(colat))
        
        # Should have points across the range
        assert np.any(cos_colat > 0.8)  # Near poles
        assert np.any((cos_colat >= -0.5) & (cos_colat <= 0.5))  # Mid-latitudes
        assert np.any(cos_colat < -0.8)  # Near poles
    
    def test_repr(self):
        """Test string representation."""
        grid = RandomGrid(nline_ori=500)
        repr_str = repr(grid)
        assert "RandomGrid" in repr_str
        assert "nline_ori=500" in repr_str
    
    def test_copy(self):
        """Test grid copying."""
        original = RandomGrid(nline_ori=100)
        copied = original.copy()
        
        assert copied.nline == original.nline
        assert copied.grid_type == original.grid_type
        
        np.testing.assert_allclose(
            copied.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value
        )


class TestCustomGridComprehensive:
    """Comprehensive test suite for CustomGrid."""
    
    def test_init_with_coordinates(self):
        """Test initialization with provided coordinates."""
        npoints = 100
        coords = CartesianRepresentation(
            x=np.random.random(npoints),
            y=np.random.random(npoints),
            z=np.random.random(npoints)
        )
        
        grid = CustomGrid(nline_ori=npoints, coordinates=coords)
        assert grid.nline == npoints
        assert grid.grid_type == GridType.CUSTOM
        assert not grid.partial
        
        np.testing.assert_allclose(
            grid.footpoint_coordinates.xyz.value,
            coords.xyz.value
        )
    
    def test_init_without_coordinates(self):
        """Test initialization without coordinates (NaN default)."""
        npoints = 50
        grid = CustomGrid(nline_ori=npoints)
        
        assert grid.nline == npoints
        assert grid.grid_type == GridType.CUSTOM
        
        # Should have NaN coordinates
        coords = grid.footpoint_coordinates.xyz.value
        assert np.all(np.isnan(coords))
    
    def test_set_coordinates(self):
        """Test setting coordinates after initialization."""
        grid = CustomGrid(nline_ori=10)
        
        # Set valid coordinates
        new_coords = CartesianRepresentation(
            x=np.random.random(10),
            y=np.random.random(10),
            z=np.random.random(10)
        )
        grid.set_coordinates(new_coords)
        
        np.testing.assert_allclose(
            grid.footpoint_coordinates.xyz.value,
            new_coords.xyz.value
        )
    
    def test_set_coordinates_invalid_length(self):
        """Test setting coordinates with wrong length."""
        grid = CustomGrid(nline_ori=10)
        
        wrong_coords = CartesianRepresentation(
            x=np.random.random(5),  # Wrong length
            y=np.random.random(5),
            z=np.random.random(5)
        )
        
        with pytest.raises(ValueError, match="Coordinates length"):
            grid.set_coordinates(wrong_coords)
    
    def test_set_coordinates_none(self):
        """Test setting coordinates to None."""
        grid = CustomGrid(nline_ori=5)
        
        # Set some coordinates first
        coords = CartesianRepresentation(
            x=np.random.random(5),
            y=np.random.random(5),
            z=np.random.random(5)
        )
        grid.set_coordinates(coords)
        
        # Set to None
        grid.set_coordinates(None)
        
        # Should have NaN coordinates
        coords = grid.footpoint_coordinates.xyz.value
        assert np.all(np.isnan(coords))
    
    def test_repr(self):
        """Test string representation."""
        grid = CustomGrid(nline_ori=25)
        repr_str = repr(grid)
        assert "CustomGrid" in repr_str
        assert "nline_ori=25" in repr_str
    
    def test_copy(self):
        """Test grid copying."""
        original = CustomGrid(nline_ori=10)
        coords = CartesianRepresentation(
            x=np.random.random(10),
            y=np.random.random(10),
            z=np.random.random(10)
        )
        original.set_coordinates(coords)
        
        copied = original.copy()
        
        assert copied.nline == original.nline
        assert copied.grid_type == original.grid_type
        
        np.testing.assert_allclose(
            copied.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value
        )


class TestGridSubsettingComprehensive:
    """Comprehensive test suite for grid subsetting."""
    
    @pytest.mark.parametrize("GridClass,params", [
        (SphericalGrid, {"nlon": 12, "nlat": 6}),
        (HealpixGrid, {"nside": 4}),
        (RandomGrid, {"nline_ori": 100}),
        (CustomGrid, {"nline_ori": 50}),
    ])
    def test_subset_valid_indices(self, GridClass, params):
        """Test subsetting with valid indices."""
        if GridClass == CustomGrid:
            # Create custom grid with valid coordinates
            coords = CartesianRepresentation(
                x=np.random.random(params["nline_ori"]),
                y=np.random.random(params["nline_ori"]),
                z=np.random.random(params["nline_ori"])
            )
            grid = GridClass(**params, coordinates=coords)
        else:
            grid = GridClass(**params)
        
        # Create subset
        n_subset = grid.nline // 2
        indices = np.random.choice(grid.nline, size=n_subset, replace=False)
        subset = grid.subset_grid(indices)
        
        # Check subset properties
        assert subset.nline == n_subset
        assert subset.partial
        np.testing.assert_array_equal(subset.indices, indices)
        
        # Check coordinates match
        for i, idx in enumerate(indices):
            np.testing.assert_allclose(
                subset.footpoint_coordinates.xyz.value[:, i],
                grid.footpoint_coordinates.xyz.value[:, idx]
            )
    
    @pytest.mark.parametrize("GridClass,params", [
        (SphericalGrid, {"nlon": 8, "nlat": 4}),
        (HealpixGrid, {"nside": 2}),
        (RandomGrid, {"nline_ori": 20}),
    ])
    def test_subset_invalid_indices(self, GridClass, params):
        """Test subsetting with invalid indices."""
        grid = GridClass(**params)
        
        # Test out-of-bounds indices
        invalid_indices = [
            np.array([-1, 0, 1]),  # Negative index
            np.array([0, grid.nline, 1]),  # Index too large
            np.array([0, grid.nline + 1]),  # Index way too large
        ]
        
        for indices in invalid_indices:
            with pytest.raises(ValueError, match="Invalid indices"):
                grid.subset_grid(indices)
    
    def test_subset_empty(self):
        """Test subsetting with empty indices."""
        grid = SphericalGrid(nlon=8, nlat=4)
        
        with pytest.raises(ValueError, match="Cannot create subset with empty indices"):
            grid.subset_grid(np.array([]))
    
    def test_subset_full(self):
        """Test subsetting with all indices."""
        grid = SphericalGrid(nlon=8, nlat=4)
        all_indices = np.arange(grid.nline)
        
        full_subset = grid.subset_grid(all_indices)
        assert full_subset.nline == grid.nline
        assert full_subset.partial  # Still marked as partial
        np.testing.assert_array_equal(full_subset.indices, all_indices)


class TestGridIOTests:
    """Comprehensive test suite for grid I/O operations."""
    
    @pytest.mark.parametrize("GridClass,params", [
        (SphericalGrid, {"nlon": 12, "nlat": 6, "pole_lat": 30.0}),
        (HealpixGrid, {"nside": 4}),
        (RandomGrid, {"nline_ori": 100}),
        (CustomGrid, {"nline_ori": 50}),
    ])
    def test_save_load_roundtrip(self, GridClass, params, temp_dir):
        """Test save and load roundtrip for all grid types."""
        # Create grid
        if GridClass == CustomGrid:
            coords = CartesianRepresentation(
                x=np.random.random(params["nline_ori"]),
                y=np.random.random(params["nline_ori"]),
                z=np.random.random(params["nline_ori"])
            )
            original = GridClass(**params, coordinates=coords)
        else:
            original = GridClass(**params)
        
        # Save to file
        fits_file = temp_dir / "test_grid.fits"
        original.save_to_fits(str(fits_file))
        
        # Load from file
        loaded = FootpointGrid.load_from_fits(str(fits_file))
        
        # Check properties
        assert loaded.grid_type == original.grid_type
        assert loaded.nline == original.nline
        assert loaded.partial == original.partial
        
        # Check coordinates
        np.testing.assert_allclose(
            loaded.footpoint_coordinates.xyz.value,
            original.footpoint_coordinates.xyz.value,
            rtol=1e-10
        )
        
        # Check indices
        np.testing.assert_array_equal(loaded.indices, original.indices)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        from psr_magpy.utils.exceptions import PSRMagPyIOError
        with pytest.raises(PSRMagPyIOError, match="Validation failed for load operation"):
            FootpointGrid.load_from_fits("nonexistent_file.fits")
    
    def test_load_invalid_fits(self, temp_dir):
        """Test loading invalid FITS file."""
        # Create invalid FITS file
        invalid_file = temp_dir / "invalid.fits"
        hdu = pf.PrimaryHDU()
        hdu.writeto(invalid_file, overwrite=True)
        
        with pytest.raises(KeyError):  # Missing GRIDTYPE
            FootpointGrid.load_from_fits(str(invalid_file))
    
    def test_load_unknown_grid_type(self, temp_dir):
        """Test loading with unknown grid type."""
        # Create FITS file with unknown grid type
        hdu = pf.PrimaryHDU()
        hdu.header["GRIDTYPE"] = "unknown_type"
        hdu.header["NLINE"] = 10
        hdu.header["PARTIAL"] = False
        
        # Add required HDUs
        coords_data = np.random.random((3, 10))
        coords_hdu = pf.ImageHDU(coords_data, name="FTP_CRD")
        
        indices_data = np.column_stack([np.arange(10)])
        indices_hdu = pf.BinTableHDU.from_columns([
            pf.Column(name="footpoint_id", format="K", array=np.arange(10))
        ], name="FTP_ID")
        
        hdul = pf.HDUList([hdu, indices_hdu, coords_hdu])
        
        fits_file = temp_dir / "unknown_grid.fits"
        hdul.writeto(fits_file, overwrite=True)
        
        with pytest.raises(ValueError, match="not a valid GridType"):
            FootpointGrid.load_from_fits(str(fits_file))


class TestGridPerformance:
    """Performance tests for grid operations."""
    
    def test_large_spherical_grid_performance(self):
        """Test performance of large spherical grid creation."""
        start_time = time.time()
        grid = SphericalGrid(nlon=256, nlat=128)
        duration = time.time() - start_time
        
        logger.info(f"Large spherical grid creation: {duration:.3f}s")
        assert duration < 5.0  # Should complete within 5 seconds
        assert grid.nline > 30000
    
    def test_large_healpix_grid_performance(self):
        """Test performance of large HEALPix grid creation."""
        start_time = time.time()
        grid = HealpixGrid(nside=64)
        duration = time.time() - start_time
        
        logger.info(f"Large HEALPix grid creation: {duration:.3f}s")
        assert duration < 3.0  # Should complete within 3 seconds
        assert grid.nline == 49152
    
    def test_large_random_grid_performance(self):
        """Test performance of large random grid creation."""
        start_time = time.time()
        grid = RandomGrid(nline_ori=50000)
        duration = time.time() - start_time
        
        logger.info(f"Large random grid creation: {duration:.3f}s")
        assert len(grid.footpoint_coordinates.x) == 50000
        assert duration < 2.0  # Should complete within 2 seconds
    
    def test_subset_performance(self):
        """Test performance of subsetting large grid."""
        grid = SphericalGrid(nlon=128, nlat=64)
        
        start_time = time.time()
        indices = np.random.choice(grid.nline, size=grid.nline//4, replace=False)
        subset = grid.subset_grid(indices)
        duration = time.time() - start_time
        
        logger.info(f"Grid subsetting: {duration:.3f}s")
        assert duration < 1.0  # Should complete within 1 second
        assert subset.nline == grid.nline//4


class TestGridEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_spherical_grid(self):
        """Test minimal spherical grid."""
        grid = SphericalGrid(nlon=2, nlat=2)
        assert grid.nline >= 2  # At least north and south poles
    
    def test_minimal_healpix_grid(self):
        """Test minimal HEALPix grid."""
        grid = HealpixGrid(nside=1)
        assert grid.nline == 12
    
    def test_single_point_random_grid(self):
        """Test random grid with single point."""
        grid = RandomGrid(nline_ori=1)
        assert grid.nline == 1
        assert grid.footpoint_coordinates is not None
    
    def test_single_point_custom_grid(self):
        """Test custom grid with single point."""
        coords = CartesianRepresentation(x=[1], y=[0], z=[0])
        grid = CustomGrid(nline_ori=1, coordinates=coords)
        assert grid.nline == 1
    
    def test_coordinate_precision(self):
        """Test coordinate precision and numerical stability."""
        grid = SphericalGrid(nlon=64, nlat=32)
        
        # Check that coordinates are unit vectors to high precision
        xyz = grid.footpoint_coordinates.xyz.value
        r = np.sqrt(np.sum(xyz**2, axis=0))
        np.testing.assert_allclose(r, 1.0, rtol=1e-12, atol=1e-12)
    
    def test_pole_lat_extreme_values(self):
        """Test extreme pole latitude values."""
        extreme_values = [-180, -90, 90, 180]
        
        for pole_lat in extreme_values:
            grid = SphericalGrid(nlon=8, nlat=4, pole_lat=pole_lat)
            assert grid.pole_lat.deg == pole_lat
            assert grid.footpoint_coordinates is not None


class TestGridIntegration:
    """Integration tests for grid functionality."""
    
    def test_grid_type_consistency(self):
        """Test that grid types are consistent across operations."""
        grids = [
            SphericalGrid(nlon=12, nlat=6),
            HealpixGrid(nside=4),
            RandomGrid(nline_ori=100),
        ]
        
        for grid in grids:
            # Test that grid type is preserved through operations
            assert grid.grid_type is not None
            
            # Test that copy preserves grid type
            copied = grid.copy()
            assert copied.grid_type == grid.grid_type
            
            # Test that subset preserves grid type
            if grid.nline > 1:
                subset = grid.subset_grid([0, 1])
                assert subset.grid_type == grid.grid_type
    
    def test_coordinate_system_consistency(self):
        """Test that coordinate systems are consistent."""
        grids = [
            SphericalGrid(nlon=12, nlat=6),
            HealpixGrid(nside=4),
            RandomGrid(nline_ori=100),
        ]
        
        for grid in grids:
            # Test that longitude and colatitude properties work
            lon = grid.longitude
            colat = grid.colatitude
            
            assert len(lon) == grid.nline
            assert len(colat) == grid.nline
            
            # Test coordinate conversion consistency
            spherical = grid.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)
            np.testing.assert_allclose(lon, spherical.phi.to(u.deg).value)
            np.testing.assert_allclose(colat, spherical.theta.to(u.deg).value)
    
    def test_grid_operations_chain(self):
        """Test chaining of grid operations."""
        # Create grid
        grid = SphericalGrid(nlon=16, nlat=8)
        
        # Create subset
        subset = grid.subset_grid(np.arange(10))
        
        # Copy subset
        copied_subset = subset.copy()
        
        # Test that all operations work together
        assert copied_subset.nline == 10
        assert copied_subset.partial
        assert copied_subset.grid_type == GridType.SPHERICAL
        
        # Test coordinate consistency
        np.testing.assert_allclose(
            copied_subset.footpoint_coordinates.xyz.value,
            subset.footpoint_coordinates.xyz.value
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 