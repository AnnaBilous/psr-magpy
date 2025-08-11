#!/usr/bin/env python3
"""
Comprehensive tests for FieldLineCalculator module.

Tests calculator functionality including memory estimation logic,
batch processing, and field line calculations.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.field.components import DipoleFieldComponent
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid
from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.tracing_parameters import (
    TracingParameters, 
    IntegrationParameters, 
    OutputParameters, 
    PerformanceParameters
)
from tests.utils import setup_test_logger


class TestFieldLineCalculator:
    """Test FieldLineCalculator class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_calculator")
        
        # Create field and grid
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.field = MultipoleField([self.dipole])
        self.grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        
        # Default parameters
        self.params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5),
            output=OutputParameters(save_results_to_disk=False, keep_results_in_memory=True)
        )
    
    def test_initialization(self):
        """Test calculator initialization."""
        calculator = FieldLineCalculator(self.field, self.grid, self.params)
        
        assert calculator.context is not None
        assert calculator.context.field is self.field
        assert calculator.context.grid is self.grid
        assert calculator.context.params is self.params
    
    def test_get_field_lines_basic(self):
        """Test basic field line calculation."""
        calculator = FieldLineCalculator(self.field, self.grid, self.params)
        template = FieldLine.template_OFL_calc()
        
        results = calculator.get_field_lines(template)
        
        assert results is not None
        assert results.n_field_lines > 0
        assert hasattr(results, 'estimate_size')


class TestMemoryEstimationSkipping:
    """Test memory estimation skipping logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_memory_estimation")
        
        # Create field
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.field = MultipoleField([self.dipole])
    
    def test_can_skip_when_user_enables(self):
        """Test skipping when user explicitly enables it."""
        grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Should skip: user requested skip_memory_estimation=True
        assert calculator._can_skip_memory_estimation() is True
    
    def test_cannot_skip_when_user_disables(self):
        """Test not skipping when user disables it (default behavior)."""
        grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=25.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Should not skip: user requested skip_memory_estimation=False
        assert calculator._can_skip_memory_estimation() is False
    
    def test_default_behavior_no_skip(self):
        """Test default behavior doesn't skip memory estimation."""
        grid = SphericalGrid(nlon=40, nlat=40, pole_lat=0.0)  # ~1400 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=50),  # Default skip_memory_estimation=False
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Should not skip: default behavior is skip_memory_estimation=False
        assert calculator._can_skip_memory_estimation() is False
    
    def test_skip_when_user_enables_even_with_saving(self):
        """Test skipping when user enables it even when saving results."""
        grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=True, filename="test.fits")
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Should skip: user requested skip_memory_estimation=True (overrides save_results_to_disk)
        assert calculator._can_skip_memory_estimation() is True
    
    def test_user_control_overrides_all_conditions(self):
        """Test that user control overrides all other conditions."""
        grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        
        # Test skip=True with various conditions
        params_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=100.0),  # Large R_max
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=True, filename="test.fits")  # Saving results
        )
        calculator_skip = FieldLineCalculator(self.field, grid, params_skip)
        assert calculator_skip._can_skip_memory_estimation() is True
        
        # Test skip=False with various conditions  
        params_no_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=2.0),  # Small R_max but > threshold_line_distance
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)  # Not saving results
        )
        calculator_no_skip = FieldLineCalculator(self.field, grid, params_no_skip)
        assert calculator_no_skip._can_skip_memory_estimation() is False
    
    def test_different_grid_sizes_user_control(self):
        """Test user control works with different grid sizes."""
        # Test with different grid sizes
        grid_small = SphericalGrid(nlon=5, nlat=5, pole_lat=0.0)  # Small grid (~17 points)
        grid_large = SphericalGrid(nlon=35, nlat=35, pole_lat=0.0)  # Large grid (~1157 points)
        
        # Test skip=True with both grids
        params_skip_small = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=True),  # Small n_trial for small grid
            output=OutputParameters(save_results_to_disk=False)
        )
        
        params_skip_large = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=100, skip_memory_estimation=True),  # Larger n_trial for large grid
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_small_skip = FieldLineCalculator(self.field, grid_small, params_skip_small)
        assert calculator_small_skip._can_skip_memory_estimation() is True
        
        calculator_large_skip = FieldLineCalculator(self.field, grid_large, params_skip_large)
        assert calculator_large_skip._can_skip_memory_estimation() is True
        
        # Test skip=False with both grids
        params_no_skip_small = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        params_no_skip_large = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=100, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_small_no_skip = FieldLineCalculator(self.field, grid_small, params_no_skip_small)
        assert calculator_small_no_skip._can_skip_memory_estimation() is False
        
        calculator_large_no_skip = FieldLineCalculator(self.field, grid_large, params_no_skip_large)
        assert calculator_large_no_skip._can_skip_memory_estimation() is False


class TestCalculatorBehavior:
    """Test calculator behavior with and without memory estimation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_calculator_behavior")
        
        # Create field
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.field = MultipoleField([self.dipole])
        self.template = FieldLine.template_OFL_calc()
    
    def test_skipping_memory_estimation_produces_results(self):
        """Test that skipping memory estimation still produces valid results."""
        grid = SphericalGrid(nlon=5, nlat=5, pole_lat=0.0)  # ~17 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Verify it would skip memory estimation
        assert calculator._can_skip_memory_estimation() is True
        
        # Should still produce results
        results = calculator.get_field_lines(self.template)
        assert results is not None
        assert results.n_field_lines > 0
    
    def test_not_skipping_memory_estimation_produces_results(self):
        """Test that not skipping memory estimation still produces valid results."""
        grid = SphericalGrid(nlon=5, nlat=5, pole_lat=0.0)  # ~17 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=25.0),  # Large R_max
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        
        # Verify it would not skip memory estimation
        assert calculator._can_skip_memory_estimation() is False
        
        # Should still produce results
        results = calculator.get_field_lines(self.template)
        assert results is not None
        assert results.n_field_lines > 0
    
    def test_results_consistency_skip_vs_no_skip(self):
        """Test that results are consistent whether memory estimation is skipped or not."""
        grid = SphericalGrid(nlon=5, nlat=5, pole_lat=0.0)  # ~17 points
        
        # Configuration that skips memory estimation
        params_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        # Configuration that doesn't skip memory estimation
        params_no_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=10, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_skip = FieldLineCalculator(self.field, grid, params_skip)
        calculator_no_skip = FieldLineCalculator(self.field, grid, params_no_skip)
        
        # Verify skip behavior
        assert calculator_skip._can_skip_memory_estimation() is True
        assert calculator_no_skip._can_skip_memory_estimation() is False
        
        # Get results
        results_skip = calculator_skip.get_field_lines(self.template)
        results_no_skip = calculator_no_skip.get_field_lines(self.template)
        
        # Both should produce results
        assert results_skip is not None
        assert results_no_skip is not None
        assert results_skip.n_field_lines == results_no_skip.n_field_lines
        
        # Results should be consistent (same number of field lines)
        assert results_skip.n_field_lines == grid.nline
        assert results_no_skip.n_field_lines == grid.nline


class TestCalculatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_calculator_edge_cases")
        
        # Create field
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.field = MultipoleField([self.dipole])
    
    def test_empty_grid(self):
        """Test behavior with minimal grid."""
        # Use very small grid
        grid = SphericalGrid(nlon=2, nlat=2, pole_lat=0.0)  # 2 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=2),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, grid, params)
        template = FieldLine.template_OFL_calc()
        
        # Should still work
        results = calculator.get_field_lines(template)
        assert results is not None
        assert results.n_field_lines == 2
    
    def test_memory_estimation_with_small_grid(self):
        """Test memory estimation logic with very small grid."""
        # Use very small grid (edge case but realistic)
        grid = SphericalGrid(nlon=2, nlat=2, pole_lat=0.0)  # 2 points
        
        # Test with skip=True
        params_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=2, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_skip = FieldLineCalculator(self.field, grid, params_skip)
        assert calculator_skip._can_skip_memory_estimation() is True
        
        # Test with skip=False (default)
        params_no_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0),
            performance=PerformanceParameters(n_trial=2, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_no_skip = FieldLineCalculator(self.field, grid, params_no_skip)
        assert calculator_no_skip._can_skip_memory_estimation() is False 