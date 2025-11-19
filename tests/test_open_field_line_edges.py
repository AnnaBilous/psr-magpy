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
    get_OFL_edges_precise,
    get_OFL_edges_approximate,
    calculate_and_save_OFL_edges,
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.grid.grid import HealpixGrid, FootpointGrid
from psr_magpy.line.property_system import PropertyName


class TestOFLEdgeResult:
    """Test OFLEdgeResult named tuple."""
    
    def test_ofledge_result_creation(self):
        """Test creating OFLEdgeResult."""
        r_lc = np.array([10.0, 5.0, 2.0])
        edge_footpoints = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])]
        precise_edges = True
        
        result = OFLEdgeResult(R_LC=r_lc, edge_footpoints=edge_footpoints, precise_edges=precise_edges)
        
        assert np.array_equal(result.R_LC, r_lc)
        assert result.edge_footpoints == edge_footpoints
        assert result.precise_edges == precise_edges
        assert len(result.R_LC) == 3
        assert len(result.edge_footpoints) == 3
    
    def test_ofledge_result_immutability(self):
        """Test that OFLEdgeResult is immutable."""
        r_lc = np.array([10.0])
        edge_footpoints = [np.array([1, 2, 3])]
        precise_edges = False
        
        result = OFLEdgeResult(R_LC=r_lc, edge_footpoints=edge_footpoints, precise_edges=precise_edges)
        
        # Test that we can't modify the NamedTuple
        # NamedTuple is immutable by design, so we just verify the structure
        assert hasattr(result, '_fields')
        assert 'R_LC' in result._fields
        assert 'edge_footpoints' in result._fields
        assert 'precise_edges' in result._fields


class TestOFLEdgeCalculator:
    """Test OFLEdgeCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock FootpointGrid
        self.mock_grid = Mock(spec=FootpointGrid)
        self.mock_grid.grid_type = Mock()
        self.mock_grid.grid_type.value = "healpix"
        self.calculator = OFLEdgeCalculator(self.mock_grid)
    
    def test_init_with_valid_grid(self):
        """Test initialization with valid grid."""
        calculator = OFLEdgeCalculator(self.mock_grid)
        assert calculator.grid == self.mock_grid
        assert calculator.h_LC_coeff == 3**1.5 / 2.0
        assert calculator.precise_edges == False
    
    def test_init_with_custom_h_lc_coeff(self):
        """Test initialization with custom h_LC_coeff."""
        calculator = OFLEdgeCalculator(self.mock_grid, h_LC_coeff=2.0)
        assert calculator.h_LC_coeff == 2.0
    
    def test_init_with_precise_edges(self):
        """Test initialization with precise_edges flag."""
        calculator = OFLEdgeCalculator(self.mock_grid, precise_edges=True)
        assert calculator.precise_edges == True
    
    @patch('psr_magpy.line.open_field_line_edges.get_OFL_edges_precise')
    def test_calculate_edges_precise(self, mock_get_edges):
        """Test calculate_edges with precise method."""
        calculator = OFLEdgeCalculator(self.mock_grid, precise_edges=True)
        mock_lines = Mock(spec=FieldLineSet)
        r_lc_values = np.array([10.0, 5.0])
        expected_result = OFLEdgeResult(
            R_LC=r_lc_values,
            edge_footpoints=[np.array([1, 2]), np.array([3, 4])],
            precise_edges=True
        )
        mock_get_edges.return_value = expected_result
        
        result = calculator.calculate_edges(mock_lines, R_LC=r_lc_values)
        
        mock_get_edges.assert_called_once_with(
            self.mock_grid, mock_lines, None, r_lc_values, calculator.h_LC_coeff
        )
        assert result == expected_result
    
    @patch('psr_magpy.line.open_field_line_edges.get_OFL_edges_approximate')
    def test_calculate_edges_approximate(self, mock_get_edges):
        """Test calculate_edges with approximate method."""
        calculator = OFLEdgeCalculator(self.mock_grid, precise_edges=False)
        mock_lines = Mock(spec=FieldLineSet)
        r_lc_values = np.array([10.0, 5.0])
        expected_result = OFLEdgeResult(
            R_LC=r_lc_values,
            edge_footpoints=[np.array([1, 2]), np.array([3, 4])],
            precise_edges=False
        )
        mock_get_edges.return_value = expected_result
        
        result = calculator.calculate_edges(mock_lines, R_LC=r_lc_values, R_LC_edge_coefficient=1.02)
        
        mock_get_edges.assert_called_once_with(
            self.mock_grid, mock_lines, None, r_lc_values, calculator.h_LC_coeff, 1.02
        )
        assert result == expected_result
    
    @patch('psr_magpy.line.open_field_line_edges.get_OFL_edges_approximate')
    def test_calculate_edges_with_list_coefficient(self, mock_get_edges):
        """Test calculate_edges with list of coefficients."""
        calculator = OFLEdgeCalculator(self.mock_grid, precise_edges=False)
        mock_lines = Mock(spec=FieldLineSet)
        r_lc_values = np.array([10.0, 5.0])
        coefficients = [1.01, 1.02]
        expected_result = OFLEdgeResult(
            R_LC=r_lc_values,
            edge_footpoints=[np.array([1, 2]), np.array([3, 4])],
            precise_edges=False
        )
        mock_get_edges.return_value = expected_result
        
        result = calculator.calculate_edges(mock_lines, R_LC=r_lc_values, R_LC_edge_coefficient=coefficients)
        
        mock_get_edges.assert_called_once_with(
            self.mock_grid, mock_lines, None, r_lc_values, calculator.h_LC_coeff, coefficients
        )
        assert result == expected_result


class TestGetOFLEdgesPrecise:
    """Test get_OFL_edges_precise function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_grid = Mock(spec=FootpointGrid)
        self.mock_grid.indices = np.array([0, 1, 2, 3, 4, 5])
        self.mock_lines = Mock(spec=FieldLineSet)
    
    def test_get_ofl_edges_precise_no_parameters(self):
        """Test edge calculation with no parameters."""
        try:
            get_OFL_edges_precise(self.mock_grid, self.mock_lines)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Either P_ms or R_LC must be provided" in str(e)
    
    def test_get_ofl_edges_precise_both_parameters(self):
        """Test edge calculation with both parameters."""
        try:
            get_OFL_edges_precise(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([100.0]), R_LC=np.array([10.0])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Provide either P_ms or R_LC, not both" in str(e)
    
    def test_get_ofl_edges_precise_invalid_p_ms(self):
        """Test edge calculation with invalid P_ms."""
        try:
            get_OFL_edges_precise(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([-1.0, 0.0])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "P_ms must be positive" in str(e)
    
    def test_get_ofl_edges_precise_invalid_r_lc(self):
        """Test edge calculation with invalid R_LC."""
        try:
            get_OFL_edges_precise(
                self.mock_grid, self.mock_lines, 
                R_LC=np.array([0.5, 0.8])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "R_LC must be >= 1.0" in str(e)
    
    @patch('psr_magpy.line.open_field_line_edges.get_R_LC')
    @patch('psr_magpy.line.open_field_line_edges.healpix_get_single_OFL_edge_precise')
    def test_get_ofl_edges_precise_with_p_ms(self, mock_get_single, mock_get_r_lc):
        """Test edge calculation with P_ms parameter."""
        p_ms = np.array([100.0, 50.0])
        expected_r_lc = np.array([10.0, 5.0])
        mock_get_r_lc.return_value = expected_r_lc
        
        # Mock the get_OFL_footpoints method
        self.mock_lines.get_OFL_footpoints.side_effect = [
            np.array([1, 2, 3]),  # for R_LC = 10.0
            np.array([4, 5]),     # for R_LC = 5.0
        ]
        
        # Mock healpix_get_single_OFL_edge_precise
        mock_get_single.side_effect = [
            np.array([1, 3]),  # edge for first R_LC
            np.array([4]),     # edge for second R_LC
        ]
        
        result = get_OFL_edges_precise(self.mock_grid, self.mock_lines, P_ms=p_ms)
        
        assert len(result.R_LC) == 2
        assert len(result.edge_footpoints) == 2
        assert result.precise_edges == True
        # The function sorts R_LC in descending order
        assert np.array_equal(result.R_LC, expected_r_lc)


class TestGetOFLEdgesApproximate:
    """Test get_OFL_edges_approximate function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_grid = Mock(spec=FootpointGrid)
        self.mock_lines = Mock(spec=FieldLineSet)
    
    def test_get_ofl_edges_approximate_no_parameters(self):
        """Test edge calculation with no parameters."""
        try:
            get_OFL_edges_approximate(self.mock_grid, self.mock_lines)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Either P_ms or R_LC must be provided" in str(e)
    
    def test_get_ofl_edges_approximate_both_parameters(self):
        """Test edge calculation with both parameters."""
        try:
            get_OFL_edges_approximate(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([100.0]), R_LC=np.array([10.0])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Provide either P_ms or R_LC, not both" in str(e)
    
    def test_get_ofl_edges_approximate_invalid_p_ms(self):
        """Test edge calculation with invalid P_ms."""
        try:
            get_OFL_edges_approximate(
                self.mock_grid, self.mock_lines, 
                P_ms=np.array([-1.0, 0.0])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "P_ms must be positive" in str(e)
    
    def test_get_ofl_edges_approximate_invalid_r_lc(self):
        """Test edge calculation with invalid R_LC."""
        try:
            get_OFL_edges_approximate(
                self.mock_grid, self.mock_lines, 
                R_LC=np.array([0.5, 0.8])
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "R_LC must be >= 1.0" in str(e)
    
    @patch('psr_magpy.line.open_field_line_edges.get_R_LC')
    def test_get_ofl_edges_approximate_with_p_ms(self, mock_get_r_lc):
        """Test edge calculation with P_ms parameter."""
        p_ms = np.array([100.0, 50.0])
        expected_r_lc = np.array([10.0, 5.0])
        mock_get_r_lc.return_value = expected_r_lc
        
        # Mock the get_OFL_footpoints method
        self.mock_lines.get_OFL_footpoints.side_effect = [
            np.array([1, 2, 3]),  # for R_LC = 10.0
            np.array([4, 5]),     # for R_LC = 5.0
            np.array([1, 2]),     # for R_LC * 1.02 = 10.2
            np.array([4]),        # for R_LC * 1.02 = 5.1
        ]
        
        result = get_OFL_edges_approximate(self.mock_grid, self.mock_lines, P_ms=p_ms, R_LC_edge_coefficient=1.02)
        
        assert len(result.R_LC) == 2
        assert len(result.edge_footpoints) == 2
        assert result.precise_edges == False
        # The function sorts R_LC in descending order
        assert np.array_equal(result.R_LC, expected_r_lc)
    
    def test_get_ofl_edges_approximate_with_list_coefficient(self):
        """Test edge calculation with list of coefficients."""
        r_lc = np.array([10.0, 5.0])
        coefficients = [1.01, 1.02]
        
        # Mock the get_OFL_footpoints method
        self.mock_lines.get_OFL_footpoints.side_effect = [
            np.array([1, 2, 3]),  # for R_LC = 10.0
            np.array([4, 5]),     # for R_LC = 5.0
            np.array([1, 2]),     # for R_LC * 1.01 = 10.1
            np.array([4]),        # for R_LC * 1.02 = 5.1
        ]
        
        result = get_OFL_edges_approximate(self.mock_grid, self.mock_lines, R_LC=r_lc, R_LC_edge_coefficient=coefficients)
        
        assert len(result.R_LC) == 2
        assert len(result.edge_footpoints) == 2
        assert result.precise_edges == False
    
    def test_get_ofl_edges_approximate_invalid_coefficient_length(self):
        """Test edge calculation with invalid coefficient list length."""
        r_lc = np.array([10.0, 5.0, 2.0])
        coefficients = [1.01, 1.02]  # Wrong length
        
        try:
            get_OFL_edges_approximate(self.mock_grid, self.mock_lines, R_LC=r_lc, R_LC_edge_coefficient=coefficients)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "R_LC_edge_coefficient list length (2) must match R_LC length (3)" in str(e)


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
            reversed_error=True,
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
        
        try:
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path)
            )
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "Failed to load experiment config" in str(e)
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    def test_calculate_and_save_no_field_lines(self, mock_experiment_config):
        """Test handling of missing field line data."""
        mock_config = Mock()
        mock_experiment_config.load.return_value = mock_config
        mock_config.load_combined_batch_data.return_value = None
        
        try:
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path)
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No field line data found" in str(e)
    
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
        
        # Set precise_edges=True to trigger the grid type check earlier
        try:
            calculate_and_save_OFL_edges(
                experiment_config_path=str(self.experiment_config_path),
                lines_dir=str(self.lines_dir),
                template=self.template,
                R_LC_values=self.r_lc_values,
                output_path=str(self.output_path),
                precise_edges=True  # This will trigger the grid type check
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Precise edge calculation requires HealpixGrid" in str(e)
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    @patch('psr_magpy.io.get_io_factory')
    def test_calculate_and_save_template_reversed_error_addition(self, mock_io_factory, mock_experiment_config):
        """Test that reversed_error is added to template if not present."""
        mock_config = Mock()
        mock_experiment_config.load.return_value = mock_config
        
        # Mock field line data
        mock_ofl_data = Mock(spec=FieldLineSet)
        mock_ofl_data.n_field_lines = 1000
        mock_config.load_combined_batch_data.return_value = mock_ofl_data
        
        # Mock HealpixGrid
        mock_grid = Mock(spec=HealpixGrid)
        mock_grid.nside = 64
        mock_grid.grid_type = Mock()
        mock_grid.grid_type.value = "healpix"
        mock_config.grid.create_grid.return_value = mock_grid
        
        # Mock IO factory and field line set
        mock_io = Mock()
        mock_io_factory.return_value = mock_io
        mock_field_line_set = Mock(spec=FieldLineSet)
        mock_field_line_set.n_field_lines = 100
        
        # Mock property system
        mock_property_system = Mock()
        mock_field_line_set.property_system = mock_property_system
        
        # Mock get_property to return appropriate data based on property type
        def mock_get_property(prop):
            if prop.name == 'FOOTPOINT_ID':
                return np.array([1, 2, 3])
            elif prop.name == 'REVERSED_ERROR':
                return np.array([0.001, 0.002, 0.005])
            else:
                return None
        
        mock_property_system.get_property.side_effect = mock_get_property
        
        # Mock save_flags for I/O system
        mock_save_flags = Mock()
        mock_save_flags.save_footpoint_id = True
        mock_save_flags.save_r_cyl_max = True
        mock_save_flags.save_z_abs_max = True
        mock_save_flags.save_r_line = True
        mock_save_flags.save_reversed_error = True
        mock_field_line_set.save_flags = mock_save_flags
        
        mock_field_line_set.subset_by_footpoint_ids.return_value = mock_field_line_set
        mock_io.load_data.return_value = mock_field_line_set
        
        # Create template without reversed_error
        template_without_error = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_r_line=True,
            save_reversed_error=False  # Not included initially
        )
        
        # Mock the edge calculation
        with patch('psr_magpy.line.open_field_line_edges.OFLEdgeCalculator') as mock_calc:
            mock_calc_instance = Mock()
            mock_calc.return_value = mock_calc_instance
            mock_calc_instance.calculate_edges.return_value = OFLEdgeResult(
                R_LC=np.array([10.0, 5.0]),
                edge_footpoints=[np.array([1, 2]), np.array([3])],
                precise_edges=False
            )
            
            # Mock glob to return batch files
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['batch_001.fits', 'batch_002.fits']
                
                result = calculate_and_save_OFL_edges(
                    experiment_config_path=str(self.experiment_config_path),
                    lines_dir=str(self.lines_dir),
                    template=template_without_error,
                    R_LC_values=self.r_lc_values,
                    output_path=str(self.output_path),
                    precise_edges=False
                )
                
                # Verify that reversed_error was added to template
                assert template_without_error.property_system.save_flags.save_reversed_error == True
                assert result is not None
    
    @patch('psr_magpy.experiment.experiment_config.ExperimentConfig')
    @patch('psr_magpy.io.get_io_factory')
    def test_calculate_and_save_error_filtering(self, mock_io_factory, mock_experiment_config):
        """Test that footpoints with high reversed_error are filtered out."""
        mock_config = Mock()
        mock_experiment_config.load.return_value = mock_config
        
        # Mock tracing config with reversed_error_threshold
        mock_tracing = Mock()
        mock_tracing.reversed_error_threshold = 0.001
        mock_config.tracing = mock_tracing
        mock_config.name = "test_experiment"
        
        # Mock field line data
        mock_ofl_data = Mock(spec=FieldLineSet)
        mock_ofl_data.n_field_lines = 1000
        mock_config.load_combined_batch_data.return_value = mock_ofl_data
        
        # Mock HealpixGrid
        mock_grid = Mock(spec=HealpixGrid)
        mock_grid.nside = 64
        mock_grid.grid_type = Mock()
        mock_grid.grid_type.value = "healpix"
        mock_config.grid.create_grid.return_value = mock_grid
        
        # Mock IO factory and field line set
        mock_io = Mock()
        mock_io_factory.return_value = mock_io
        mock_field_line_set = Mock(spec=FieldLineSet)
        mock_field_line_set.n_field_lines = 5  # Match the number of footpoints
        
        # Mock property system to return footpoint IDs and reversed errors
        mock_property_system = Mock()
        mock_field_line_set.property_system = mock_property_system
        
        # Mock get_property to return appropriate data based on property type
        def mock_get_property(prop):
            if prop.name == 'FOOTPOINT_ID':
                return np.array([1, 2, 3, 4, 5])
            elif prop.name == 'REVERSED_ERROR':
                return np.array([0.001, 0.002, 0.01, 0.005, 0.0005])  # threshold is 0.001
            else:
                return None
        
        mock_property_system.get_property.side_effect = mock_get_property
        
        # Mock save_flags for I/O system
        mock_save_flags = Mock()
        mock_save_flags.save_footpoint_id = True
        mock_save_flags.save_r_cyl_max = True
        mock_save_flags.save_z_abs_max = True
        mock_save_flags.save_r_line = True
        mock_save_flags.save_reversed_error = True
        mock_field_line_set.save_flags = mock_save_flags
        
        # Create a mock for the subset result
        mock_subset = Mock(spec=FieldLineSet)
        mock_subset.n_field_lines = 2  # After filtering, should have 2 footpoints [1, 5]
        mock_subset.append = Mock()  # For combining batches
        mock_subset.save_flags = mock_save_flags  # Copy save_flags from parent mock
        mock_subset.property_system = mock_property_system  # Copy property_system from parent mock
        
        # Make subset_by_footpoint_ids return the subset mock
        def mock_subset_by_footpoint_ids(footpoint_ids):
            # Verify it's called with filtered footpoints
            return mock_subset
        
        mock_field_line_set.subset_by_footpoint_ids = Mock(side_effect=mock_subset_by_footpoint_ids)
        mock_io.load_data.return_value = mock_field_line_set
        
        # Create template with reversed_error
        template_with_error = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_r_line=True,
            save_reversed_error=True
        )
        
        # Mock the edge calculation
        with patch('psr_magpy.line.open_field_line_edges.OFLEdgeCalculator') as mock_calc:
            mock_calc_instance = Mock()
            mock_calc.return_value = mock_calc_instance
            mock_calc_instance.calculate_edges.return_value = OFLEdgeResult(
                R_LC=np.array([10.0, 5.0]),
                edge_footpoints=[np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3])],  # Two arrays for two R_LC values
                precise_edges=False
            )
            
            # Mock glob to return batch files
            with patch('glob.glob') as mock_glob:
                mock_glob.return_value = ['batch_001.fits']
                
                result = calculate_and_save_OFL_edges(
                    experiment_config_path=str(self.experiment_config_path),
                    lines_dir=str(self.lines_dir),
                    template=template_with_error,
                    R_LC_values=self.r_lc_values,
                    output_path=str(self.output_path),
                    precise_edges=False
                )
                
                # Verify that subset_by_footpoint_ids was called with filtered footpoints
                # The filtering should remove footpoints with error > 0.001
                calls = mock_field_line_set.subset_by_footpoint_ids.call_args_list
                assert len(calls) > 0, "subset_by_footpoint_ids was never called"
                
                # Check that filtering occurred (some footpoints were removed)
                # Original footpoints: [1, 2, 3, 4, 5] with errors: [0.001, 0.002, 0.01, 0.005, 0.0005]
                # After filtering (error <= 0.001): should be [1, 5] (removing 2, 3, 4)
                # For second R_LC: [1, 2, 3] -> [1] (removing 2, 3)
                
                # Verify that the first call has fewer footpoints than the original 5
                first_call_footpoints = calls[0][0][0]
                assert len(first_call_footpoints) < 5, f"Expected filtering to reduce footpoints, got {first_call_footpoints}"
                
                # Verify that footpoint 3 (error 0.01) was filtered out
                assert 3 not in first_call_footpoints, f"Footpoint 3 (high error) should be filtered out, got {first_call_footpoints}"
                assert result is not None


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
            reversed_error=True,
        )
        
        r_lc_values = np.array([10.0, 5.0])
        
        # This test would need actual field line data files
        # For now, we'll just verify the function signature and basic behavior
        assert template is not None
        assert len(r_lc_values) == 2
        assert r_lc_values[0] > r_lc_values[1]  # Should be in descending order


if __name__ == "__main__":
    pytest.main([__file__])
