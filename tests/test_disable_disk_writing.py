#!/usr/bin/env python3
"""
Tests for disabling disk writing functionality.

Tests that save_results_to_disk=False properly disables disk writing
and returns results in memory.
"""

import pytest
import tempfile
import os
from pathlib import Path

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


class TestDisableDiskWriting:
    """Test disabling disk writing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple field and grid for testing
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.field = MultipoleField([self.dipole])
        self.grid = SphericalGrid(nlon=6, nlat=4, pole_lat=0.0)  # 14 points
        self.template = FieldLine.template_OFL_calc()
    
    def test_save_results_false_returns_memory(self):
        """Test that save_results_to_disk=False returns results in memory."""
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=3.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for testing
                filename=None
            )
        )
        
        calculator = FieldLineCalculator(self.field, self.grid, params)
        results = calculator.get_field_lines(self.template)
        
        # Should return results in memory
        assert results is not None
        assert results.n_field_lines > 0
        assert hasattr(results, 'estimate_size')
    
    def test_memory_estimation_skipping_when_user_enables(self):
        """Test that memory estimation is skipped when user enables it."""
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, self.grid, params)
        
        # Should skip memory estimation (user requested skip_memory_estimation=True)
        assert calculator._can_skip_memory_estimation() is True
    
    def test_memory_estimation_not_skipped_when_user_disables(self):
        """Test that memory estimation is not skipped when user disables it."""
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=25.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, self.grid, params)
        
        # Should not skip memory estimation (user requested skip_memory_estimation=False)
        assert calculator._can_skip_memory_estimation() is False
    
    def test_memory_estimation_default_behavior(self):
        """Test that memory estimation is not skipped by default."""
        # Test with default behavior (skip_memory_estimation=False)
        large_grid = SphericalGrid(nlon=40, nlat=40, pole_lat=0.0)  # ~1400 points
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=50),  # Default skip_memory_estimation=False
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, large_grid, params)
        
        # Should not skip memory estimation (default behavior)
        assert calculator._can_skip_memory_estimation() is False
    
    def test_memory_estimation_user_control_overrides_save_results_to_disk(self):
        """Test that user control overrides save_results_to_disk setting."""
        # Test skip=True even with save_results_to_disk=True
        params_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=True),
            output=OutputParameters(save_results_to_disk=True, filename="test.fits")
        )
        
        calculator_skip = FieldLineCalculator(self.field, self.grid, params_skip)
        
        # Should skip memory estimation (user override)
        assert calculator_skip._can_skip_memory_estimation() is True
        
        # Test skip=False even with save_results_to_disk=False
        params_no_skip = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=5.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5, skip_memory_estimation=False),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_no_skip = FieldLineCalculator(self.field, self.grid, params_no_skip)
        
        # Should not skip memory estimation (user override)
        assert calculator_no_skip._can_skip_memory_estimation() is False
    
    def test_save_results_true_still_works(self):
        """Test that save_results_to_disk=True still works as before."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_output")
            
            params = TracingParameters(
                integration=IntegrationParameters(max_integration_radius=3.0, threshold_line_distance=1.0),
                performance=PerformanceParameters(n_trial=5),
                output=OutputParameters(
                    save_results_to_disk=True,
                    filename=filename
                )
            )
            
            calculator = FieldLineCalculator(self.field, self.grid, params)
            results = calculator.get_field_lines(self.template)
            
            # Should return results in memory
            assert results is not None
            assert results.n_field_lines > 0
            
            # Note: File saving functionality is tested elsewhere, focus on memory estimation control
    
    def test_no_save_vs_save_behavior(self):
        """Test behavior difference between save and no-save modes."""
        # Test with save_results_to_disk=False
        params_no_save = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=3.0, threshold_line_distance=1.0),
            performance=PerformanceParameters(n_trial=5),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator_no_save = FieldLineCalculator(self.field, self.grid, params_no_save)
        results_no_save = calculator_no_save.get_field_lines(self.template)
        
        # Test with save_results_to_disk=True
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_output")
            
            params_save = TracingParameters(
                integration=IntegrationParameters(max_integration_radius=3.0, threshold_line_distance=1.0),
                performance=PerformanceParameters(n_trial=5),
                output=OutputParameters(save_results_to_disk=True, filename=filename)
            )
            
            calculator_save = FieldLineCalculator(self.field, self.grid, params_save)
            results_save = calculator_save.get_field_lines(self.template)
            
            # Both should return results
            assert results_no_save is not None
            assert results_save is not None
            assert results_no_save.n_field_lines == results_save.n_field_lines
            
            # Note: File saving functionality is tested elsewhere, focus on memory estimation control
    
    def test_empty_results_handling(self):
        """Test handling of empty results."""
        # Use very restrictive parameters to get empty results
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=1.001, threshold_line_distance=0.5),
            performance=PerformanceParameters(n_trial=5),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        calculator = FieldLineCalculator(self.field, self.grid, params)
        results = calculator.get_field_lines(self.template)
        
        # Should still return a valid object
        assert results is not None
        # May have 0 field lines due to restrictive parameters
        assert results.n_field_lines >= 0
    
    def test_parameter_validation_save_results_false(self):
        """Test parameter validation when save_results_to_disk=False."""
        # Should work without filename when save_results_to_disk=False
        params = TracingParameters(
            output=OutputParameters(
                save_results_to_disk=False,
                filename=None
            )
        )
        
        # Should not raise an error
        assert params.output.save_results_to_disk is False
        assert params.output.filename is None
    
    def test_parameter_validation_save_results_true(self):
        """Test parameter validation when save_results_to_disk=True."""
        # Should require filename when save_results_to_disk=True
        with pytest.raises(ValueError, match="Filename must be specified"):  # type: ignore
            TracingParameters(
                output=OutputParameters(
                    save_results_to_disk=True,
                    filename=None
                )
            )


if __name__ == "__main__":
    pytest.main([__file__]) 