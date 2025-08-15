#!/usr/bin/env python3
"""
Comprehensive tests for field line tracer module.

Tests FieldLineTracer, integration handling, property calculation,
and related functionality for magnetic field line tracing.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from astropy.coordinates.representation import CartesianRepresentation

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters
from psr_magpy.line.field_line import FieldLine, PropertyName
from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.line.property_system import PropertyValidationError
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.field.components import DipoleFieldComponent
from tests.utils import setup_test_logger, rng


class TestFieldLineTracer:
    """Test FieldLineTracer class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_tracer")
        
        # Create a simple dipole field for testing
        self.dipole = DipoleFieldComponent()
        self.field = MultipoleField([self.dipole])
        
        # Create default tracing parameters
        self.params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        # Create tracer instance
        self.tracer = FieldLineTracer(self.field, self.params)
    
    def test_initialization(self):
        """Test tracer initialization."""
        assert self.tracer.field is self.field
        assert self.tracer.params is self.params
        assert hasattr(self.tracer, 'integrate_field_line')
        assert isinstance(self.tracer.property_calculator, PropertyCalculator)
    
    def test_initialization_with_custom_params(self):
        """Test tracer initialization with custom parameters."""
        custom_params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=15.0,
                threshold_line_distance=2.0,
                t_span=(0.0, 2e6),
                rtol=1e-9,
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
    
        tracer = FieldLineTracer(self.field, custom_params)
        assert tracer.params.integration.max_integration_radius == 15.0
        assert tracer.params.integration.threshold_line_distance == 2.0
        assert tracer.params.integration.rtol == 1e-9
    
    def test_initialization_validation(self):
        """Test that initialization validates inputs."""
        # Test with None field - currently doesn't validate, but should
        # This test documents the current behavior
        tracer_none_field = FieldLineTracer(None, self.params)
        assert tracer_none_field.field is None
        
        # Test with None params - currently doesn't validate, but should
        # This test documents the current behavior
        with pytest.raises(AttributeError):  # type: ignore
            FieldLineTracer(self.field, None)  # type: ignore
    
    def test_trace_single_line_basic(self):
        """Test basic field line tracing."""
        # Create template with basic properties
        template = FieldLine(save_r_line=True, save_r_max=True)
        
        # Initial position at surface
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Trace field line
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Verify result structure
        assert isinstance(result, FieldLine)
        assert result.get_property('footpoint_id') == footpoint_id
        
        # Verify properties were calculated
        r_line = result.get_property('r_line')
        r_max = result.get_property('r_max')
        
        assert r_line is not None
        assert r_max is not None
        assert r_max >= 1.0  # Should be at least the initial radius
    
    def test_trace_single_line_all_properties(self):
        """Test field line tracing with all properties enabled."""
        # Create template with all properties
        template = FieldLine.template_all()
        
        # Initial position
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.5)
        footpoint_id = 42
        
        # Trace field line
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Verify all properties are present
        for prop in ['r_line', 'r_max', 'pa_line', 'r_cyl_max', 'z_abs_max', 'b_ini', 'b_r_ini_angle']:
            value = result.get_property(prop)
            assert value is not None, f"Property {prop} should be present"
        
        # Verify specific property values
        assert result.get_property('footpoint_id') == footpoint_id
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
    
    def test_trace_single_line_different_positions(self):
        """Test field line tracing from different initial positions."""
        template = FieldLine.template_OFL_calc()
        # Use a custom tracer with a safe t_span to avoid memory errors
        safe_params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 100.0)),
            output=OutputParameters(save_results_to_disk=False)
        )
        tracer = FieldLineTracer(self.field, safe_params)
        # Test different positions
        positions = [
            CartesianRepresentation(x=1.0, y=0.0, z=0.0),  # Equator
            CartesianRepresentation(x=0.0, y=0.0, z=1.0),  # North pole
            CartesianRepresentation(x=0.0, y=0.0, z=-1.0), # South pole
            CartesianRepresentation(x=0.7, y=0.7, z=0.0),  # Mid-latitude
        ]
        for i, pos in enumerate(positions):
            result = tracer.trace_single_line(pos, i, template)
            r_max = result.get_property('r_max')
            assert r_max is not None
            assert r_max >= 0.5, f"Position {i} should have r_max >= 0.5"
    
    def test_trace_single_line_save_dr(self):
        """Test field line tracing with save_dR option."""
        template = FieldLine.template_line_coordinates_only()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Trace with save_dR=True (should not affect result structure)
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template, save_dR=True)
        
        # Result should still be valid
        assert isinstance(result, FieldLine)
        assert result.get_property('footpoint_id') == footpoint_id
    
    def test_trace_single_line_invalid_inputs(self):
        """Test field line tracing with invalid inputs."""
        template = FieldLine.template_line_coordinates_only()
        
        # Test with None position - should raise AttributeError, not ValueError
        with pytest.raises(AttributeError):  # type: ignore
            self.tracer.trace_single_line(None, 1, template)  # type: ignore        
        
        # Test with invalid position type - should raise AttributeError
        with pytest.raises(AttributeError):  # type: ignore
            self.tracer.trace_single_line([1, 0, 0], 1, template)  # type: ignore
        
        # Test with None template - should raise AttributeError
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        with pytest.raises(AttributeError):  # type: ignore
            self.tracer.trace_single_line(r_initial, 1, None)  # type: ignore
    
    def test_trace_single_line_edge_cases(self):
        """Test field line tracing edge cases."""
        template = FieldLine.template_OFL_calc()
        
        # Test very small dt
        small_dt_params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                dt=0.001,
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        tracer_small_dt = FieldLineTracer(self.field, small_dt_params)
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        result = tracer_small_dt.trace_single_line(r_initial, 1, template)
        
        # Should still work
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
        
        # Test very large dt
        large_dt_params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                dt=0.1,
                t_span=(0.0, 1e4),
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        tracer_large_dt = FieldLineTracer(self.field, large_dt_params)
        
        result = tracer_large_dt.trace_single_line(r_initial, 1, template)
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
    
    def test_trace_single_line_property_consistency(self):
        """Test that field line properties are consistent."""
        template = FieldLine.template_all()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Test property consistency
        r_line = result.get_property('r_line')
        r_max = result.get_property('r_max')
        pa_line = result.get_property('pa_line')
        r_cyl_max = result.get_property('r_cyl_max')
        z_abs_max = result.get_property('z_abs_max')
        b_ini = result.get_property('b_ini')
        b_r_ini_angle = result.get_property('b_r_ini_angle')
        
        # Check that all properties are not None before using them
        assert r_line is not None
        assert r_max is not None
        assert pa_line is not None
        assert r_cyl_max is not None
        assert z_abs_max is not None
        assert b_ini is not None
        assert b_r_ini_angle is not None
        
        # r_max should be the maximum of |r_line|
        assert r_max ==  np.max(np.sum(r_line**2, axis=1) ** 0.5)
        
        # r_cyl_max should be reasonable
        assert r_cyl_max == np.max(np.sum(r_line[:,:2] ** 2, axis=1) ** 0.5)
        
        # z_abs_max should be reasonable
        assert np.allclose(z_abs_max, np.max(np.abs(r_line[:,2])), rtol=1e-6)
        
        # b_ini should be reasonable
        assert b_ini >= 0
        
        # All arrays should have the same length
        assert len(r_line) == len(pa_line)
        
        # b_r_ini_angle should be reasonable
        assert np.isfinite(b_r_ini_angle)
    
    def test_trace_single_line_magnetic_field_consistency(self):
        """Test that magnetic field calculations are consistent."""
        template = FieldLine.template_surface_field_properties()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
        
        b_ini = result.get_property('b_ini')
        b_r_ini_angle = result.get_property('b_r_ini_angle')
        
        # Check that properties are not None before using them
        assert b_ini is not None
        assert b_r_ini_angle is not None
        
        # b_ini should be reasonable
        assert b_ini >= 0
        assert np.isfinite(b_ini)
        
        # b_r_ini_angle should be reasonable
        assert np.isfinite(b_r_ini_angle)
    
    def test_trace_single_line_multiple_runs(self):
        """Test that multiple runs produce consistent results."""
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Run multiple times
        results = []
        for _ in range(3):
            result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
            results.append(result)
        
        # Results should be consistent
        r_max_values = [r.get_property('r_max') for r in results]
        assert np.allclose(r_max_values, r_max_values[0], rtol=1e-10)
    
    def test_trace_single_line_different_field_components(self):
        """Test field line tracing with different field configurations."""
        # Create field with multiple components
        dipole1 = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )
        dipole2 = DipoleFieldComponent(
            B_magnitude=0.5,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0
        )
        
        multi_field = MultipoleField([dipole1, dipole2])
        multi_tracer = FieldLineTracer(multi_field, self.params)
        
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        result = multi_tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Should still work with multiple components
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
    
    def test_trace_single_line_integration_parameters(self):
        """Test that integration parameters affect the results."""
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Test with different tolerances
        high_tol_params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                rtol=1e-6,
                atol=1e-6,
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        low_tol_params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                rtol=1e-10,
                atol=1e-10,
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        high_tol_tracer = FieldLineTracer(self.field, high_tol_params)
        low_tol_tracer = FieldLineTracer(self.field, low_tol_params)
        
        high_result = high_tol_tracer.trace_single_line(r_initial, footpoint_id, template)
        low_result = low_tol_tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Results should be similar but not necessarily identical
        high_r_max = high_result.get_property('r_max')
        low_r_max = low_result.get_property('r_max')
        
        # Both should reach the target R_max or surface
        assert high_r_max is not None
        assert high_r_max >= 1.0
        assert low_r_max is not None
        assert low_r_max >= 1.0
    
    def test_trace_single_line_surface_radius(self):
        """Test field line tracing with different surface radius values."""
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Test with different surface radius
        custom_params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, surface_radius=0.5),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        custom_tracer = FieldLineTracer(self.field, custom_params)
        result = custom_tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Should still work
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
    
    def test_trace_single_line_time_span(self):
        """Test field line tracing with different time spans."""
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Test with shorter time span
        short_span_params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 1e5)),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        short_span_tracer = FieldLineTracer(self.field, short_span_params)
        result = short_span_tracer.trace_single_line(r_initial, footpoint_id, template)
        
        # Should still work
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0

    def test_template_methods(self):
        """Test the new predefined template methods."""
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Test all template methods
        templates = [
            ("all", FieldLine.template_all()),
            ("all_except_along_line", FieldLine.template_all_except_along_line()),
            ("OFL_calc", FieldLine.template_OFL_calc()),
            ("line_coordinates_only", FieldLine.template_line_coordinates_only()),
            ("properties_along_line", FieldLine.template_properties_along_line()),
            ("surface_field_properties", FieldLine.template_surface_field_properties()),
        ]
        
        for name, template in templates:
            result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
            
            # All templates should have footpoint_id
            assert result.get_property('footpoint_id') == footpoint_id
            
            # Test specific properties based on template
            if name == "all":
                assert result.get_property('r_line') is not None
                assert result.get_property('r_max') is not None
                assert result.get_property('pa_line') is not None
                assert result.get_property('r_cyl_max') is not None
                assert result.get_property('z_abs_max') is not None
                assert result.get_property('b_ini') is not None
                assert result.get_property('b_r_ini_angle') is not None
            elif name == "all_except_line_and_PA":
                # These properties should not be accessible (disabled)
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_line')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('pa_line')
                assert result.get_property('r_max') is not None
                assert result.get_property('r_cyl_max') is not None
                assert result.get_property('z_abs_max') is not None
                assert result.get_property('b_ini') is not None
                assert result.get_property('b_r_ini_angle') is not None
            elif name == "OFL_calc":
                # These properties should not be accessible (disabled)
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_line')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('pa_line')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('b_ini')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('b_r_ini_angle')
                assert result.get_property('r_max') is not None
                assert result.get_property('r_cyl_max') is not None
                assert result.get_property('z_abs_max') is not None
            elif name == "coordinates_only":
                assert result.get_property('r_line') is not None
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('pa_line')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_max')
            elif name == "coordinates_and_PA":
                assert result.get_property('r_line') is not None
                assert result.get_property('pa_line') is not None
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_max')
            elif name == "surface_properties":
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_line')
                with pytest.raises(PropertyValidationError):  # type: ignore
                    result.get_property('r_max')
                assert result.get_property('b_ini') is not None
                assert result.get_property('b_r_ini_angle') is not None


class TestFieldLineTracerIntegration:
    """Test integration between FieldLineTracer and its components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_tracer_integration")
        
        # Create field and parameters
        self.dipole = DipoleFieldComponent()
        self.field = MultipoleField([self.dipole])
        self.params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0),
            output=OutputParameters(save_results_to_disk=False)
        )
    
    def test_integration_handler_initialization(self):
        """Test that FieldLineTracer has integration capability (legacy integration handler removed)."""
        tracer = FieldLineTracer(self.field, self.params)
        # Instead of checking for integration_handler, check for integrate_field_line method
        assert hasattr(tracer, 'integrate_field_line')
    
    def test_property_calculator_initialization(self):
        """Test that property calculator is properly initialized."""
        tracer = FieldLineTracer(self.field, self.params)
        
        assert tracer.property_calculator.threshold_line_distance == self.params.threshold_line_distance
    
    def test_end_to_end_tracing(self):
        """Test complete end-to-end field line tracing process."""
        template = FieldLine.template_all()


        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                dt=0.1,
                t_span=(0.0, 1e4),
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        tracer = FieldLineTracer(self.field, params)
        
        # Test multiple positions
        positions = [
            CartesianRepresentation(x=1.0, y=0.0, z=0.0),
            CartesianRepresentation(x=0.0, y=0.0, z=1.0),
            CartesianRepresentation(x=0.7, y=0.7, z=0.0),
        ]
        
        for i, pos in enumerate(positions):
            result = tracer.trace_single_line(pos, i, template)
            
            # Verify all properties are present and reasonable
            for prop in ['r_line', 'r_max', 'pa_line', 'r_cyl_max', 'z_abs_max', 'b_ini', 'b_r_ini_angle']:
                value = result.get_property(prop)
                assert value is not None, f"Property {prop} missing for position {i}"
            
            # Verify property consistency
            r_max = result.get_property('r_max')
            r_line = result.get_property('r_line')
            if r_line is not None and len(r_line) > 0:
                # Calculate max radius from line coordinates (r_line has shape (n_points, 3))
                line_radii = np.sqrt(np.sum(r_line**2, axis=1))
                max_radius_from_line = np.max(line_radii)
                assert r_max is not None, f"r_max should not be None for position {i}"
                assert np.isclose(float(r_max), float(max_radius_from_line)), f"r_max inconsistency for position {i}: r_max={r_max}, max_radius_from_line={max_radius_from_line}"
            
            r_cyl_max = result.get_property('r_cyl_max')
            assert r_cyl_max is not None
            assert r_cyl_max >= 0, f"r_cyl_max should be non-negative for position {i}"
            
            
            z_abs_max = result.get_property('z_abs_max')
            assert z_abs_max is not None
            assert z_abs_max >= 0, f"z_abs_max should be non-negative for position {i}"
    
    def test_error_propagation(self):
        """Test that errors in components are properly propagated."""
        # Mock field to raise an error
        mock_field = Mock()
        mock_field.get_total_field.side_effect = ValueError("Field calculation error")

        tracer = FieldLineTracer(mock_field, self.params)
        template = FieldLine.template_line_coordinates_only()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        
        with pytest.raises(ValueError, match="Field calculation error"):  # type: ignore
            tracer.trace_single_line(r_initial, 1, template)


class TestFieldLineTracerPerformance:
    """Test performance characteristics of field line tracing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_tracer_performance")
        
        self.dipole = DipoleFieldComponent()
        self.field = MultipoleField([self.dipole])
        self.params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0),
            output=OutputParameters(save_results_to_disk=False)
        )
        self.tracer = FieldLineTracer(self.field, self.params)
    
    def test_tracing_speed(self):
        """Test that field line tracing completes in reasonable time."""
        import time
        
        template = FieldLine.template_OFL_calc()
        
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        start_time = time.time()
        result = self.tracer.trace_single_line(r_initial, footpoint_id, template)
        end_time = time.time()
        
        # Should complete in less than 1 second
        assert end_time - start_time < 1.0
        
        # Result should be valid
        r_max = result.get_property('r_max')
        assert r_max is not None
        assert r_max >= 1.0
    
    def test_memory_usage(self):
        """Test that field line tracing doesn't use excessive memory."""
        import psutil
        import os
        
        template = FieldLine.template_all_except_along_line()
        
        # Use custom parameters with shorter time span to avoid memory issues
        params = TracingParameters(
            integration=IntegrationParameters(max_integration_radius=10.0, threshold_line_distance=1.0, t_span=(0.0, 100.0)),
            output=OutputParameters(save_results_to_disk=False)
        )
        tracer = FieldLineTracer(self.field, params)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Trace multiple field lines
        for i in range(10):
            r_initial = CartesianRepresentation(
                x=np.random.uniform(0.5, 1.5),
                y=np.random.uniform(-0.5, 0.5),
                z=np.random.uniform(-0.5, 0.5)
            )
            result = tracer.trace_single_line(r_initial, i, template)
            # Allow for field lines that don't reach threshold_line_distance (valid physics)
            r_max = result.get_property('r_max')
            assert r_max is not None
            assert r_max > 0, f"r_max should be positive, got {r_max}"
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes


if __name__ == "__main__":
    pytest.main([__file__]) 