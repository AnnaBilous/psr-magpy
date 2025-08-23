"""Tests for OFL edge calculation functionality."""

import pytest
import numpy as np
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeCalculator, 
    OFLEdgeResult,
    healpix_grid_get_OFL_edges,
    healpix_grid_get_single_OFL_edge
)


class TestOFLEdgeCalculator:
    """Test OFLEdgeCalculator class."""
    
    def test_init_valid_grid(self):
        """Test calculator initialization with valid HEALPix grid."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        
        assert calculator.grid is grid
        assert calculator.h_LC_coeff == 3**1.5 / 2.0
    
    def test_init_custom_h_lc_coeff(self):
        """Test calculator initialization with custom h_LC_coeff."""
        grid = HealpixGrid(nside=4)
        h_lc_custom = 2.5
        calculator = OFLEdgeCalculator(grid, h_LC_coeff=h_lc_custom)
        
        assert calculator.h_LC_coeff == h_lc_custom
    
    def test_init_invalid_grid_type(self):
        """Test calculator initialization with invalid grid type."""
        # Create a mock grid with wrong type
        class MockGrid:
            def __init__(self):
                self.grid_type = None
        
        with pytest.raises(ValueError, match="Only HEALPix grids are supported"):
            OFLEdgeCalculator(MockGrid())
    
    def test_calculate_edges_r_lc_input(self):
        """Test edge calculation with R_LC input."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        
        # Create mock FieldLineSet
        mock_lines = self._create_mock_field_line_set()
        
        R_LC_values = np.array([1.0, 2.0])
        result = calculator.calculate_edges(mock_lines, R_LC=R_LC_values)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 2
        assert len(result.edge_footpoints) == 2
        np.testing.assert_array_equal(result.R_LC, [2.0, 1.0])  # Sorted descending
    
    def test_calculate_edges_p_ms_input(self):
        """Test edge calculation with P_ms input."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        
        mock_lines = self._create_mock_field_line_set()
        
        P_ms_values = np.array([10.0, 20.0])
        result = calculator.calculate_edges(mock_lines, P_ms=P_ms_values)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 2
        assert len(result.edge_footpoints) == 2
    
    def test_calculate_single_edge(self):
        """Test single edge calculation."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        
        # Simple test with a few footpoint IDs
        ftp_ids = np.array([0, 1, 2, 3, 4])
        edge_ids = calculator.calculate_single_edge(ftp_ids)
        
        assert isinstance(edge_ids, np.ndarray)
        assert len(edge_ids) <= len(ftp_ids)
    
    def test_consistency_with_functions(self):
        """Test that class methods produce same results as functions."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        
        mock_lines = self._create_mock_field_line_set()
        R_LC_values = np.array([1.0, 2.0])
        
        # Compare class and function results
        class_result = calculator.calculate_edges(mock_lines, R_LC=R_LC_values)
        func_result = healpix_grid_get_OFL_edges(grid, mock_lines, R_LC=R_LC_values)
        
        np.testing.assert_array_equal(class_result.R_LC, func_result.R_LC)
        assert len(class_result.edge_footpoints) == len(func_result.edge_footpoints)
        
        for i in range(len(class_result.edge_footpoints)):
            np.testing.assert_array_equal(
                class_result.edge_footpoints[i], 
                func_result.edge_footpoints[i]
            )
    
    def _create_mock_field_line_set(self):
        """Create a mock FieldLineSet for testing."""
        class MockFieldLineSet:
            def get_OFL_footpoints(self, R_LC, h_LC_coeff=None):
                # Return different footpoint sets based on R_LC
                if R_LC >= 2.0:
                    return np.array([0, 1, 2])
                else:
                    return np.array([0, 1, 2, 3, 4, 5])
        
        return MockFieldLineSet()


class TestOFLEdgeResult:
    """Test OFLEdgeResult structure."""
    
    def test_create_result(self):
        """Test creating OFLEdgeResult."""
        R_LC = np.array([2.0, 1.0])
        edge_footpoints = [np.array([0, 1]), np.array([2, 3, 4])]
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints)
        
        np.testing.assert_array_equal(result.R_LC, R_LC)
        assert len(result.edge_footpoints) == 2
        np.testing.assert_array_equal(result.edge_footpoints[0], [0, 1])
        np.testing.assert_array_equal(result.edge_footpoints[1], [2, 3, 4])
    
    def test_result_immutable(self):
        """Test that OFLEdgeResult is immutable."""
        R_LC = np.array([1.0])
        edge_footpoints = [np.array([0])]
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints)
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            result.R_LC = np.array([2.0])


class TestInputValidation:
    """Test input validation for OFL edge functions."""
    
    def test_r_lc_validation(self):
        """Test R_LC parameter validation."""
        grid = HealpixGrid(nside=4)
        mock_lines = MockFieldLineSet()
        
        # Test R_LC < 1.0
        with pytest.raises(ValueError, match="R_LC must be >= 1.0"):
            healpix_grid_get_OFL_edges(grid, mock_lines, R_LC=0.5)
        
        # Test negative R_LC
        with pytest.raises(ValueError, match="R_LC must be >= 1.0"):
            healpix_grid_get_OFL_edges(grid, mock_lines, R_LC=-1.0)
    
    def test_p_ms_validation(self):
        """Test P_ms parameter validation."""
        grid = HealpixGrid(nside=4)
        mock_lines = MockFieldLineSet()
        
        # Test P_ms <= 0
        with pytest.raises(ValueError, match="P_ms must be positive"):
            healpix_grid_get_OFL_edges(grid, mock_lines, P_ms=0.0)
        
        # Test negative P_ms
        with pytest.raises(ValueError, match="P_ms must be positive"):
            healpix_grid_get_OFL_edges(grid, mock_lines, P_ms=-5.0)
    
    def test_both_parameters_provided(self):
        """Test error when both P_ms and R_LC provided."""
        grid = HealpixGrid(nside=4)
        mock_lines = MockFieldLineSet()
        
        with pytest.raises(ValueError, match="Provide either P_ms or R_LC, not both"):
            healpix_grid_get_OFL_edges(grid, mock_lines, P_ms=10.0, R_LC=1.5)
    
    def test_no_parameters_provided(self):
        """Test error when neither P_ms nor R_LC provided."""
        grid = HealpixGrid(nside=4)
        mock_lines = MockFieldLineSet()
        
        with pytest.raises(ValueError, match="Either P_ms or R_LC must be provided"):
            healpix_grid_get_OFL_edges(grid, mock_lines)


class TestVectorizedAlgorithm:
    """Test vectorized neighbor checking algorithm."""
    
    def test_single_edge_detection(self):
        """Test single edge detection with known configuration."""
        grid = HealpixGrid(nside=4)
        
        # Test with specific footpoint configuration
        ftp_ids = np.array([0, 1, 2, 3])
        edge_ids = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        
        assert isinstance(edge_ids, np.ndarray)
        assert all(edge_id in ftp_ids for edge_id in edge_ids)
    
    def test_empty_input(self):
        """Test edge detection with empty input."""
        grid = HealpixGrid(nside=4)
        
        ftp_ids = np.array([])
        edge_ids = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        
        assert len(edge_ids) == 0
    
    def test_single_footpoint(self):
        """Test edge detection with single footpoint."""
        grid = HealpixGrid(nside=4)
        
        ftp_ids = np.array([0])
        edge_ids = healpix_grid_get_single_OFL_edge(grid, ftp_ids)
        
        # Single isolated footpoint should be an edge
        np.testing.assert_array_equal(edge_ids, [0])


class MockFieldLineSet:
    """Mock FieldLineSet for testing."""
    
    def get_OFL_footpoints(self, R_LC, h_LC_coeff=None):
        """Mock OFL footpoint calculation."""
        # Return different sets based on R_LC for testing
        if R_LC >= 2.0:
            return np.array([0, 1, 2, 5])
        elif R_LC >= 1.5:
            return np.array([0, 1, 2, 3, 4, 5, 8, 9])
        else:
            return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


if __name__ == "__main__":
    pytest.main([__file__])
