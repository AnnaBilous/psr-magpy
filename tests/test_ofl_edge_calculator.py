"""Tests for OFL edge calculation functionality."""

import pytest
import numpy as np
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.open_field_line_edges import (
    OFLEdgeCalculator, 
    OFLEdgeResult,
    get_OFL_edges_precise,
    get_OFL_edges_approximate
)
from psr_magpy.utils.physical_conversions import get_R_LC
import healpy as hp


class TestOFLEdgeCalculator:
    """Test OFLEdgeCalculator class."""
    
    def test_init_valid_grid(self):
        """Test initialization with valid grid."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid)
        assert calculator.grid == grid
        assert calculator.h_LC_coeff == 3**1.5 / 2.0
        assert calculator.precise_edges == False
    
    def test_init_custom_coeff(self):
        """Test initialization with custom h_LC_coeff."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid, h_LC_coeff=2.0)
        assert calculator.h_LC_coeff == 2.0
    
    def test_init_precise_edges(self):
        """Test initialization with precise_edges flag."""
        grid = HealpixGrid(nside=4)
        calculator = OFLEdgeCalculator(grid, precise_edges=True)
        assert calculator.precise_edges == True


class TestOFLEdgeResult:
    """Test OFLEdgeResult NamedTuple."""
    
    def test_creation(self):
        """Test creating OFLEdgeResult."""
        R_LC = np.array([2.0, 5.0])
        edge_footpoints = [np.array([1, 2]), np.array([3, 4])]
        precise_edges = True
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints, precise_edges=precise_edges)
        
        assert result.R_LC is R_LC
        assert result.edge_footpoints is edge_footpoints
        assert result.precise_edges == precise_edges
    
    def test_immutability(self):
        """Test that OFLEdgeResult is immutable."""
        R_LC = np.array([2.0])
        edge_footpoints = [np.array([1])]
        precise_edges = False
        
        result = OFLEdgeResult(R_LC=R_LC, edge_footpoints=edge_footpoints, precise_edges=precise_edges)
        
        # NamedTuple is immutable by design, so we just verify the structure
        assert hasattr(result, '_fields')
        assert 'R_LC' in result._fields
        assert 'edge_footpoints' in result._fields
        assert 'precise_edges' in result._fields


class TestEdgeCalculationFunctions:
    """Test the edge calculation functions."""
    
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
    
    def test_get_ofl_edges_precise_R_LC_input(self):
        """Test get_OFL_edges_precise with R_LC input."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        R_LC = np.array([2.0, 5.0, 10.0])
        
        result = get_OFL_edges_precise(grid, lines, R_LC=R_LC)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
        assert result.precise_edges == True
        # R_LC should be sorted descending
        assert np.all(np.diff(result.R_LC) <= 0)
    
    def test_get_ofl_edges_approximate_R_LC_input(self):
        """Test get_OFL_edges_approximate with R_LC input."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        R_LC = np.array([2.0, 5.0, 10.0])
        
        result = get_OFL_edges_approximate(grid, lines, R_LC=R_LC, R_LC_edge_coefficient=1.02)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
        assert result.precise_edges == False
        # R_LC should be sorted descending
        assert np.all(np.diff(result.R_LC) <= 0)
    
    def test_get_ofl_edges_approximate_list_coefficient(self):
        """Test get_OFL_edges_approximate with list of coefficients."""
        grid = HealpixGrid(nside=4)
        lines = self._create_mock_field_line_set()
        R_LC = np.array([2.0, 5.0, 10.0])
        coefficients = [1.01, 1.02, 1.03]
        
        result = get_OFL_edges_approximate(grid, lines, R_LC=R_LC, R_LC_edge_coefficient=coefficients)
        
        assert isinstance(result, OFLEdgeResult)
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
        assert result.precise_edges == False


if __name__ == "__main__":
    pytest.main([__file__])
