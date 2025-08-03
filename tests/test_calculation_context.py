"""
Tests for CalculationContext class.
"""

import pytest
import numpy as np
from astropy.coordinates import CartesianRepresentation

from psr_magpy.field.components import DipoleFieldComponent
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters, PerformanceParameters
from psr_magpy.line.field_line import FieldLine


class TestCalculationContext:
    """Test CalculationContext functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create field and grid
        self.dipole = DipoleFieldComponent()
        self.field = MultipoleField([self.dipole])
        self.grid = SphericalGrid(nlon=10, nlat=5, pole_lat=0.0)
        
        # Create parameters with safe t_span and n_trial <= grid.nline
        self.params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 100.0)),
            output=OutputParameters(save_results=False)
        )
        self.params.performance.n_trial = 10
    
    def test_initialization(self):
        """Test basic initialization."""
        context = CalculationContext(self.field, self.grid, self.params)
        
        assert context.field is self.field
        assert context.grid is self.grid
        assert context.params is self.params
        assert context.tracer is not None
        assert context.memory_manager is not None

        assert context.progress_tracker is not None
    
    def test_initialization_with_default_params(self):
        """Test initialization with default parameters (but output=OutputParameters(save_results=False))."""
        params = TracingParameters(output=OutputParameters(save_results=False))
        params.performance.n_trial = 10
        context = CalculationContext(
            self.field,
            self.grid,
            params
        )
        assert context.field is self.field
        assert context.grid is self.grid
        assert context.params is not None
        assert context.tracer is not None
    
    def test_trace_single_line(self):
        """Test tracing a single field line."""
        context = CalculationContext(self.field, self.grid, self.params)
        template = FieldLine()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        result = context.tracer.trace_single_line(r_initial, 1, template)
        
        assert result is not None
        assert result.get_property('footpoint_id') == 1
    
    def test_memory_estimation(self):
        """Test memory estimation and file-size-based flushing."""
        # Create context with small max_memory_mb to force flushing
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 100.0)),
            output=OutputParameters(save_results=False),
            performance=PerformanceParameters(max_memory_mb=0.001)  # Very small memory limit
        )
        params.performance.n_trial = 10
        
        context = CalculationContext(self.field, self.grid, params)
        template = FieldLine.template_OFL_calc()
        
        trial_size, completion_time = context.estimate_memory_requirements(template)
        assert trial_size > 0
        assert completion_time > 0
        
        # Add enough results to trigger flush
        # Use a safe coordinate since grid should have coordinates after init
        safe_coord = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        batch = context.tracer.trace_single_line(safe_coord, 0, template)
        from psr_magpy.line.field_line_set import FieldLineSet
        batch_set = FieldLineSet([batch] * 10)
        context.add_results(batch_set, "test_flush", 0)
        # After flush, in-memory buffer should be reset
        assert context.memory_manager.results_in_memory is None or context.memory_manager.memory_usage_mb < 0.001
    
    def test_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        context = CalculationContext(self.field, self.grid, self.params)
        
        template = FieldLine.template_OFL_calc()
        
        trial_size, _ = context.estimate_memory_requirements(template)
        # Test batch size calculation
        trial_size = 1024 * 1024  # 1 MB
        batch_size = context.memory_manager.calculate_batch_size(trial_size)
        assert batch_size > 0
    
    def test_validation_error(self):
        """Test that validation errors are raised."""
        # Create parameters with n_trial > grid size
        invalid_params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 100.0)),
            output=OutputParameters(save_results=False)
        )
        invalid_params.performance.n_trial = 1000  # More than grid size (10)
        
        with pytest.raises(ValueError, match="n_trial.*cannot exceed grid size"):  # type: ignore
            CalculationContext(self.field, self.grid, invalid_params)
    
    def test_repr(self):
        """Test string representation."""
        context = CalculationContext(self.field, self.grid, self.params)
        repr_str = repr(context)
        
        assert "CalculationContext" in repr_str
        assert "MultipoleField" in repr_str
        assert "SphericalGrid" in repr_str 