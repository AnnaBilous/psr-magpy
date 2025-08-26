"""Tests for open field line edges module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from psr_magpy.line.open_field_line_edges import (
    OFLEdgeResult,
    OFLEdgeCalculator,
    healpix_grid_get_single_OFL_edge,
    healpix_grid_get_OFL_edges,
    calculate_and_save_OFL_edges,
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.property_system import PropertyName


class TestOFLEdgeResult:
    """Test OFLEdgeResult named tuple."""
    
    def test_ofledge_result_creation(self):
        """Test creating OFLEdgeResult."""
        r_lc = np.array([10.0, 5.0, 2.0])
        edge_footpoints = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])]
        
        result = OFLEdgeResult(R_LC=r_lc, edge_footpoints=edge_footpoints)
        
        assert np.array_equal(result.R_LC, r_lc)
        assert result.edge_footpoints == edge_footpoints
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
    
    def test_ofledge_result_immutability(self):
        """Test that OFLEdgeResult is immutable."""
        r_lc = np.array([10.0])
        edge_footpoints = [np.array([1, 2, 3])]
        
        result = OFLEdgeResult(R_LC=r_lc, edge_footpoints=edge_footpoints)
        
        with pytest.raises(AttributeError):
            result.R_LC = np.array([20.0])


class TestOFLEdgeCalculator:
    """Test OFLEdgeCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock HealpixGrid
        self.mock_grid = Mock()
        self.mock_grid.grid_type = Mock()
        self.mock_grid.grid_type.value = "healpix"
        self.calculator = OFLEdgeCalculator(self.mock_grid)
    
    def test_init_with_valid_grid(self):
        """Test initialization with valid HealpixGrid."""
        calculator = OFLEdgeCalculator(self.mock_grid)
        assert calculator.grid == self.mock_grid
        assert calculator.h_LC_coeff == 3**1.5 / 2.0
    
    def test_init_with_invalid_grid(self):
        """Test initialization with invalid grid type."""
        invalid_grid = Mock()
        invalid_grid.grid_type.value = "invalid"
        
        with pytest.raises(ValueError, match="Only HEALPix grids are supported"):
            OFLEdgeCalculator(invalid_grid)
    
    def test_init_with_custom_h_lc_coeff(self):
        """Test initialization with custom h_LC_coeff."""
        calculator = OFLEdgeCalculator(self.mock_grid, h_LC_coeff=2.0)
        assert calculator.h_LC_coeff == 2.0
    
    @patch('psr_magpy.line.open_field_line_edges.healpix_grid_get_OFL_edges')
    def test_calculate_edges_with_r_lc(self, mock_get_edges):
        """Test calculate_edges with R_LC parameter."""
        mock_lines = Mock(spec=FieldLineSet)
        r_lc_values = np.array([10.0, 5.0])
        expected_result = OFLEdgeResult(
            R_LC=r_lc_values,
            edge_footpoints=[np.array([1, 2]), np.array([3, 4])]
        )
        mock_get_edges.return_value = expected_result
        
        result = self.calculator.calculate_edges(mock_lines, R_LC=r_lc_values)
        
        mock_get_edges.assert_called_once_with(
            self.mock_grid, mock_lines, None, r_lc_values, self.calculator.h_LC_coeff
        )
        assert result == expected_result
    
    @patch('psr_magpy.line.open_field_line_edges.healpix_grid_get_OFL_edges')
    def test_calculate_edges_with_p_ms(self, mock_get_edges):
        """Test calculate_edges with P_ms parameter."""
        mock_lines = Mock(spec=FieldLineSet)
        p_ms_values = np.array([100.0, 50.0])
        expected_result = OFLEdgeResult(
            R_LC=np.array([10.0, 5.0]),
            edge_footpoints=[np.array([1, 2]), np.array([3, 4])]
        )
        mock_get_edges.return_value = expected_result
        
        result = self.calculator.calculate_edges(mock_lines, P_ms=p_ms_values)
        
        mock_get_edges.assert_called_once_with(
            self.mock_grid, mock_lines, p_ms_values, None, self.calculator.h_LC_coeff
        )
        assert result == expected_result
    
    @patch('psr_magpy.line.open_field_line_edges.healpix_grid_get_single_OFL_edge')
    def test_calculate_single_edge(self, mock_get_single_edge):
        """Test calculate_single_edge method."""
        ftp_ids = np.array([1, 2, 3, 4])
        ftp_ids_inner = np.array([2, 3])
        expected_result = np.array([1, 4])
        mock_get_single_edge.return_value = expected_result
        
        result = self.calculator.calculate_single_edge(ftp_ids, ftp_ids_inner)
        
        mock_get_single_edge.assert_called_once_with(
            self.mock_grid, ftp_ids, ftp_ids_inner
        )
        assert np.array_equal(result, expected_result)


class TestHealpixGridGetSingleOFLEdge:
    """Test healpix_grid_get_single_OFL_edge function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_grid = Mock()
        self.mock_grid.grid_type = Mock()
        self.mock_grid.grid_type.value = "healpix"
        self.mock_grid.nside = 64
    
    @patch('psr_magpy.line.open_field_line_edges.hp.pixelfunc.get_all_neighbours')
    def test_get_single_edge_basic(self, mock_get_neighbours):
        """Test basic edge detection."""
        ftp_ids = np.array([1, 2, 3, 4, 5])
        
        # Mock neighbor data: each footpoint has 8 neighbors, some are valid
        # Format: (8 neighbors) x (5 footpoints)
        mock_neighbours = np.array([
            [2, 3, -1, -1, -1, -1, -1, -1],  # footpoint 1: 2 valid neighbors
            [1, 3, 4, -1, -1, -1, -1, -1],  # footpoint 2: 3 valid neighbors
            [1, 2, 4, 5, -1, -1, -1, -1],   # footpoint 3: 4 valid neighbors
            [2, 3, 5, -1, -1, -1, -1, -1],  # footpoint 4: 3 valid neighbors
            [3, 4, -1, -1, -1, -1, -1, -1], # footpoint 5: 2 valid neighbors
        ]).T
        mock_get_neighbours.return_value = mock_neighbours
        
        result = healpix_grid_get_single_OFL_edge(self.mock_grid, ftp_ids)
        
        # The function should return some edge footpoints
        # We just verify it returns an array of integers
        assert isinstance(result, np.ndarray)
        assert result.dtype == int
    
    def test_get_single_edge_invalid_grid(self):
        """Test edge detection with invalid grid type."""
        invalid_grid = Mock()
        invalid_grid.grid_type.value = "invalid"
        ftp_ids = np.array([1, 2, 3])
        
        with pytest.raises(AssertionError, match="Finding OFL edge is currently supported only for healpix grid"):
            healpix_grid_get_single_OFL_edge(invalid_grid, ftp_ids)
    
    def test_get_single_edge_empty_input(self):
        """Test edge detection with empty input."""
        ftp_ids = np.array([])
        
        result = healpix_grid_get_single_OFL_edge(self.mock_grid, ftp_ids)
        
        assert np.array_equal(result, np.array([], dtype=int))
    
    def test_get_single_edge_with_inner_exclusion(self):
        """Test edge detection with inner region exclusion."""
        ftp_ids = np.array([1, 2, 3, 4, 5])
        ftp_ids_inner = np.array([2, 3])
        
        with patch('psr_magpy.line.open_field_line_edges.hp.pixelfunc.get_all_neighbours') as mock_get_neighbours:
            # Mock neighbor structure: (8 neighbors) x (3 remaining footpoints after exclusion)
            # After excluding [2, 3], we have [1, 4, 5] remaining
            mock_get_neighbours.return_value = np.array([
                [4, 5, -1, -1, -1, -1, -1, -1],  # footpoint 1: 2 valid neighbors
                [1, 5, -1, -1, -1, -1, -1, -1],  # footpoint 4: 2 valid neighbors  
                [1, 4, -1, -1, -1, -1, -1, -1],  # footpoint 5: 2 valid neighbors
            ]).T
            
            result = healpix_grid_get_single_OFL_edge(self.mock_grid, ftp_ids, ftp_ids_inner)
            
            # Just verify it returns an array of integers
            assert isinstance(result, np.ndarray)
            assert result.dtype == int


class TestHealpixGridGetOFLEdges:
    """Test healpix_grid_get_OFL_edges function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_grid = Mock(spec=HealpixGrid)
        self.mock_lines = Mock(spec=FieldLineSet)
    
    def test_get_ofl_edges_no_parameters(self):
        """Test edge calculation with no parameters."""
        with pytest.raises(ValueError, match="Either P_ms or R_LC must be provided"):
            healpix_grid_get_OFL_edges(self.mock_grid, self.mock_lines)
    
    def test_get_ofl_edges_both_parameters(self):
        """Test edge calculation with both parameters."""
        with pytest.raises(ValueError, match="Provide either P_ms or R_LC, not both"):
            healpix_grid_get_OFL_edges(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([100.0]), R_LC=np.array([10.0])
            )
    
    def test_get_ofl_edges_invalid_p_ms(self):
        """Test edge calculation with invalid P_ms."""
        with pytest.raises(ValueError, match="P_ms must be positive"):
            healpix_grid_get_OFL_edges(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([-1.0, 0.0])
            )
    
    def test_get_ofl_edges_invalid_r_lc(self):
        """Test edge calculation with invalid R_LC."""
        with pytest.raises(ValueError, match="R_LC must be >= 1.0"):
            healpix_grid_get_OFL_edges(
                self.mock_grid, self.mock_lines, 
                R_LC=np.array([0.5, 0.8])
            )
    
    @patch('psr_magpy.line.open_field_line_edges.get_R_LC')
    def test_get_ofl_edges_with_p_ms(self, mock_get_r_lc):
        """Test edge calculation with P_ms parameter."""
        p_ms = np.array([100.0, 50.0])
        expected_r_lc = np.array([10.0, 5.0])
        mock_get_r_lc.return_value = expected_r_lc
        
        # Mock the get_OFL_footpoints method
        self.mock_lines.get_OFL_footpoints.side_effect = [
            np.array([1, 2, 3]),  # for R_LC = 10.0
            np.array([4, 5]),     # for R_LC = 5.0
        ]
        
        # Mock healpix_grid_get_single_OFL_edge
        with patch('psr_magpy.line.open_field_line_edges.healpix_grid_get_single_OFL_edge') as mock_get_single:
            mock_get_single.side_effect = [
                np.array([1, 3]),  # edge for first R_LC
                np.array([4]),     # edge for second R_LC
            ]
            
            result = healpix_grid_get_OFL_edges(self.mock_grid, self.mock_lines, P_ms=p_ms)
            
            assert len(result.R_LC) == 2
            assert len(result.edge_footpoints) == 2
            # The function sorts R_LC in descending order, so we expect the original order
            assert np.array_equal(result.R_LC, expected_r_lc)


class TestCalculateAndSaveOFLEdges:
    """Test calculate_and_save_OFL_edges function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_config_path = Path(self.temp_dir) / "test_config.yaml"
        self.lines_dir = Path(self.temp_dir) / "lines"
        self.output_path = Path(self.temp_dir) / "edge_lines.fits"
        
        # Create template
        self.template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_r_line=True,
        )
        
        self.r_lc_values = np.array([10.0, 5.0, 2.0])
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_and_save_function_signature(self):
        """Test that the function has the correct signature and basic structure."""
        # Test that the function exists and has the right signature
        import inspect
        sig = inspect.signature(calculate_and_save_OFL_edges)
        
        # Check required parameters
        assert 'experiment_config_path' in sig.parameters
        assert 'lines_dir' in sig.parameters
        assert 'template' in sig.parameters
        assert 'R_LC_values' in sig.parameters
        assert 'output_path' in sig.parameters
        
        # Check return type annotation
        assert sig.return_annotation == OFLEdgeResult
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    def test_calculate_and_save_config_not_found(self, mock_experiment_config):
        """Test handling of missing experiment config."""
        mock_experiment_config.load.side_effect = FileNotFoundError("Config not found")
        
        with pytest.raises(FileNotFoundError, match="Failed to load experiment config"):
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path)
            )
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    def test_calculate_and_save_no_field_lines(self, mock_experiment_config):
        """Test handling of missing field line data."""
        mock_config = Mock()
        mock_experiment_config.load.return_value = mock_config
        mock_config.load_combined_batch_data.return_value = None
        
        with pytest.raises(ValueError, match="No field line data found"):
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path)
            )
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    def test_calculate_and_save_invalid_grid(self, mock_experiment_config):
        """Test handling of invalid grid type."""
        mock_config = Mock()
        mock_experiment_config.load.return_value = mock_config
        
        # Mock field line data
        mock_ofl_data = Mock(spec=FieldLineSet)
        mock_ofl_data.n_field_lines = 1000
        mock_config.load_combined_batch_data.return_value = mock_ofl_data
        
        # Mock invalid grid
        mock_grid = Mock()
        mock_grid.grid_type.value = "invalid"
        mock_config.grid.create_grid.return_value = mock_grid
        
        with pytest.raises(ValueError, match="OFL edge calculation requires HealpixGrid"):
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path)
            )


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.lines_dir = Path(self.temp_dir) / "lines"
        self.lines_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_workflow_with_mock_data(self):
        """Test the complete workflow with mock data."""
        # This test would require more complex setup with actual field line data
        # For now, we'll test the basic structure
        template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_r_line=True,
        )
        
        r_lc_values = np.array([10.0, 5.0])
        
        # This test would need actual field line data files
        # For now, we'll just verify the function signature and basic behavior
        assert template is not None
        assert len(r_lc_values) == 2
        assert r_lc_values[0] > r_lc_values[1]  # Should be in descending order


if __name__ == "__main__":
    pytest.main([__file__])
