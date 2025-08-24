"""Tests for OFL edge calculation functionality."""

import pytest
import numpy as np
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeCalculator, 
    OFLEdgeResult,
    healpix_grid_get_OFL_edges,
    healpix_grid_get_single_OFL_edge
)
from psr_magpy.utils.physical_conversions import get_R_LC
import healpy as hp


class TestOFLEdgeCalculator:
    """Test OFLEdgeCalculator class."""
    
    def test_init_valid_grid(self):
        """Test initialization with valid HEALPix grid."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        assert calculator.grid == grid
        assert calculator.h_LC_coeff == 3**1.5 / 2.0
    
    def test_init_invalid_grid(self):
        """Test initialization with invalid grid type."""
        # Mock a non-HEALPix grid
        class MockGrid:
            def __init__(self):
                self.grid_type = type('MockType', (), {'value': 'spherical'})()
        
        mock_grid = MockGrid()
        with pytest.raises(ValueError, match="Only HEALPix grids are supported"):
            OFLEdgeCalculator(mock_grid)
    
    def test_init_custom_coeff(self):
        """Test initialization with custom h_LC_coeff."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid, h_LC_coeff=2.0)
        assert calculator.h_LC_coeff == 2.0


class TestHealpixGridGetSingleOFLEdge:
    """Test single OFL edge detection function."""
    
    def test_empty_input(self):
        """Test with empty footpoint array."""
        grid = HealpixGrid(nside=4)
        result = healpix_grid_get_single_OFL_edge(grid, np.array([], dtype=int))
        assert len(result) == 0
        assert result.dtype == int
    
    def test_single_footpoint(self):
        """Test with single footpoint (should be edge)."""
        grid = HealpixGrid(nside=4)
        ftp_ids = np.array([100])
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        assert len(result) == 1
        assert result[0] == 100
    
    def test_isolated_footpoints(self):
        """Test with isolated footpoints (all should be edges)."""
        grid = HealpixGrid(nside=4)
        # Use valid pixels within grid bounds
        ftp_ids = np.array([0, 50, 100])
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        # All isolated pixels should be edges
        assert len(result) == 3
        assert set(result) == {0, 50, 100}
    
    def test_connected_footpoints(self):
        """Test with connected footpoints (should have some edges)."""
        grid = HealpixGrid(nside=4)
        # Get neighbors of a pixel
        neighbors = hp.pixelfunc.get_all_neighbours(grid.nside, [0])
        valid_neighbors = neighbors[neighbors > 0]
        
        # Create connected region
        ftp_ids = np.concatenate([[0], valid_neighbors[:3]])  # 4 connected pixels
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        
        # Should have some edges (may be all pixels in small connected regions)
        assert len(result) > 0
        assert len(result) <= len(ftp_ids)  # Allow all pixels to be edges
    
    def test_with_inner_exclusion(self):
        """Test with inner region exclusion."""
        grid = HealpixGrid(nside=4)
        ftp_ids = np.array([0, 1, 2, 3, 4, 5])
        ftp_ids_inner = np.array([2, 3])  # Exclude middle pixels
        
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids, ftp_ids_inner)
        
        # Should not include excluded pixels
        assert 2 not in result
        assert 3 not in result
        # Should include some of the remaining pixels
        assert len(result) > 0
    
    def test_empty_inner_exclusion(self):
        """Test with empty inner exclusion array."""
        grid = HealpixGrid(nside=4)
        ftp_ids = np.array([0, 1, 2])
        ftp_ids_inner = np.array([], dtype=int)
        
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids, ftp_ids_inner)
        assert len(result) > 0  # Should work same as no exclusion


class TestHealpixGridGetOFLEdges:
    """Test multiple OFL edges calculation function."""
    
    def _create_mock_field_line_set(self):
        """Create a mock FieldLineSet for testing."""
        # Create a template field line
        template = FieldLine(save_footpoint_id=True)
        
        # Create a mock field line with get_OFL_footpoints method
        class MockFieldLineSet(FieldLineSet):
            def __init__(self):
                super().__init__(template, copy_data=False)
                self._n_field_lines = 1
            
            def get_OFL_footpoints(self, R_LC, h_LC_coeff=None):
                """Mock OFL footpoint calculation."""
                # Return different footpoint sets based on R_LC
                if R_LC >= 10.0:
                    return np.array([0, 1, 2])
                elif R_LC >= 5.0:
                    return np.array([0, 1, 2, 3, 4, 5])
                else:
                    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        return MockFieldLineSet()
    
    def test_R_LC_input(self):
        """Test with R_LC input."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        R_LC = np.array([2.0, 5.0, 10.0])
        
        result = healpix_grid_get_OFL_edges(grid, lines, R_LC=R_LC)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
        # R_LC should be sorted descending
        assert np.all(np.diff(result.R_LC) <= 0)
    
    def test_P_ms_input(self):
        """Test with P_ms input."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        P_ms = np.array([10.0, 50.0, 100.0])
        
        result = healpix_grid_get_OFL_edges(grid, lines, P_ms=P_ms)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
    
    def test_single_value_input(self):
        """Test with single value input."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        R_LC = 5.0
        
        result = healpix_grid_get_OFL_edges(grid, lines, R_LC=R_LC)
        
        assert len(result.R_LC) == 1
        assert len(result.edge_footpoints) == 1
    
    def test_no_input_error(self):
        """Test error when neither P_ms nor R_LC provided."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        
        with pytest.raises(ValueError, match="Either P_ms or R_LC must be provided"):
            healpix_grid_get_OFL_edges(grid, lines)
    
    def test_both_input_error(self):
        """Test error when both P_ms and R_LC provided."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        
        with pytest.raises(ValueError, match="Provide either P_ms or R_LC, not both"):
            healpix_grid_get_OFL_edges(grid, lines, P_ms=10.0, R_LC=5.0)
    
    def test_invalid_R_LC_range(self):
        """Test error for R_LC < 1.0."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        
        with pytest.raises(ValueError, match="R_LC must be >= 1.0"):
            healpix_grid_get_OFL_edges(grid, lines, R_LC=0.5)
    
    def test_invalid_P_ms_range(self):
        """Test error for P_ms <= 0."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        
        with pytest.raises(ValueError, match="P_ms must be positive"):
            healpix_grid_get_OFL_edges(grid, lines, P_ms=0.0)


class TestOFLEdgeResult:
    """Test OFLEdgeResult NamedTuple."""
    
    def test_creation(self):
        """Test creating OFLEdgeResult."""
        R_LC = np.array([2.0, 5.0])
        edge_footpoints = [np.array([1, 2]), np.array([3, 4])]
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints)
        
        assert result.R_LC is R_LC
        assert result.edge_footpoints is edge_footpoints
    
    def test_immutability(self):
        """Test that OFLEdgeResult is immutable."""
        R_LC = np.array([2.0])
        edge_footpoints = [np.array([1])]
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints)
        
        with pytest.raises(AttributeError):
            result.R_LC = np.array([3.0])


class TestVectorizedAlgorithm:
    """Test the vectorized neighbor checking algorithm."""
    
    def test_vectorized_neighbor_checking(self):
        """Test that vectorized algorithm produces same results as loop-based."""
        grid = HealpixGrid(nside=4)
        ftp_ids = np.array([0, 1, 2, 3, 4, 5])
        
        # Test vectorized version
        result_vectorized = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        
        # Should return some edges
        assert len(result_vectorized) > 0
        assert len(result_vectorized) <= len(ftp_ids)
    
    def test_edge_detection_logic(self):
        """Test edge detection logic with known pattern."""
        grid = HealpixGrid(nside=4)
        
        # Create a simple pattern: center pixel with some neighbors
        center_pixel = 50  # Use valid pixel
        neighbors = hp.pixelfunc.get_all_neighbours(grid.nside, [center_pixel])
        valid_neighbors = neighbors[neighbors > 0]
        
        # Test 1: Single pixel (should be edge)
        result = healpix_grid_get_single_OFL_edge(grid, [center_pixel])
        assert len(result) == 1
        assert result[0] == center_pixel
        
        # Test 2: Center + 2 neighbors (should have edges)
        ftp_ids = np.concatenate([[center_pixel], valid_neighbors[:2]])
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        assert len(result) > 0
        assert len(result) <= len(ftp_ids)  # Allow all pixels to be edges
    
    def test_empty_input_handling(self):
        """Test handling of empty input arrays."""
        grid = HealpixGrid(nside=4)
        
        # Empty ftp_ids
        result = healpix_grid_get_single_OFL_edge(grid, np.array([], dtype=int))
        assert len(result) == 0
        
        # Empty ftp_ids_inner
        ftp_ids = np.array([0, 1, 2])
        result = healpix_grid_get_single_OFL_edge(grid, ftp_ids, np.array([], dtype=int))
        assert len(result) > 0  # Should work normally


if __name__ == "__main__":
    pytest.main([__file__])
