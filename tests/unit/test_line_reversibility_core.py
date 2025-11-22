#!/usr/bin/env python3
"""
Unit tests for field line reversibility core functionality.

Tests the fundamental reversibility properties demonstrated in the example:
- Time reversal approach
- Forward/backward tracing accuracy 
- Endpoint validation
- Position comparison statistics
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from astropy.coordinates import CartesianRepresentation

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid, CustomGrid, RandomGrid
from psr_magpy.line.calculator import FieldLineCalculator
from psr_magpy.line.calculation_context import CalculationContext
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters, PerformanceParameters
import logging
from tests.fixtures.conftest import setup_test_logger


class TestFieldLineReversibilityCore:
    """Core unit tests for field line reversibility."""

    def setup_method(self):
        """Set up test fixtures."""
        setup_test_logger("test_reversibility_core")
        
        # Standard field components from example
        self.dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=65.0,
            moment_orientation_longitude_degree=0.0
        )
        
        self.quadrupole = QuadrupoleFieldComponent(
            B_magnitude=0.9,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )
        
        self.field = MultipoleField([self.dipole, self.quadrupole])
        
        # Standard parameters from example
        self.R_max = 10.0
        self.t_span_forward = (0.0, 1e8)
        self.t_span_backward = (-1e8, 0.0)
        


    def _trace_field_lines_time_reversal(self, field, grid, R_max, t_span, description="", verify_by_reversing=False):
        """
        Trace field lines using time reversal approach like in demo.
        
        Parameters
        ----------
        verify_by_reversing : bool
            If True, enable automatic reversibility verification via verify_by_reversing flag.
        """
        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=R_max,
                threshold_line_distance=1.0,
                t_span=t_span,
                dt=0.1,
                trace_open_lines_backwards=True,
                verify_by_reversing=verify_by_reversing,
                field_backend="numba",
                solver_backend="cyrk"
            ),
            performance=PerformanceParameters(
                n_trial=grid.nline if hasattr(grid, 'nline') and grid.nline else 50,
                max_memory_mb=100.0,
                n_cores=1
            ),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for reversibility testing
                filename=None
            )
        )
        
        # Create new calculator for each tracing operation
        context = CalculationContext(field, grid, params)
        calculator = FieldLineCalculator(context)
        template = FieldLineTemplate.template_line_coordinates_only()
        
        return calculator.get_field_lines(template)

    def _extract_endpoints(self, field_line_results):
        """Extract endpoints from field line results."""
        if field_line_results is None or field_line_results.r_line is None:
            return np.array([])
        
        endpoints = []
        for r_line in field_line_results.r_line:
            if r_line is not None and len(r_line) > 0:
                # Use same logic as example
                endpoint = r_line[-1]
                if np.linalg.norm(endpoint) < 1.001:
                    endpoints.append(endpoint)
                else:
                    endpoints.append(endpoint)
        
        # Ensure float64 dtype for CyRK solver compatibility
        return np.asarray(endpoints, dtype=np.float64)

    def _create_custom_grid_from_endpoints(self, endpoints):
        """Create custom grid from endpoint coordinates."""
        if len(endpoints) == 0:
            return None
        
        # Ensure float64 dtype for CyRK solver compatibility
        endpoints = np.asarray(endpoints, dtype=np.float64)
            
        cart_coords = CartesianRepresentation(
            x=endpoints[:, 0],
            y=endpoints[:, 1], 
            z=endpoints[:, 2]
        )
        
        return CustomGrid(nline_ori=len(endpoints), coordinates=cart_coords)

    def _calculate_position_differences(self, original_grid, final_endpoints):
        """Calculate position differences between original and final points."""
        if len(final_endpoints) == 0:
            return np.array([])
            
        original_positions = original_grid.footpoint_coordinates.xyz.value.T
        
        # Handle size mismatches
        min_count = min(len(original_positions), len(final_endpoints))
        original_positions = original_positions[:min_count]
        final_endpoints = final_endpoints[:min_count]
        
        return np.linalg.norm(original_positions - final_endpoints, axis=1)

    def _check_xz_plane_confinement(self, field_line_results, tolerance=1e-10):
        """Check if field lines are confined to XZ plane."""
        if field_line_results is None or field_line_results.r_line is None:
            return False
            
        max_y_values = []
        for r_line in field_line_results.r_line:
            if r_line is not None and len(r_line) > 0:
                y_coords = r_line[:, 1]
                max_y = np.max(np.abs(y_coords))
                max_y_values.append(max_y)
        
        return len(max_y_values) > 0 and np.max(max_y_values) < tolerance

    def test_forward_tracing_basic(self):
        """Test basic forward field line tracing."""
        grid = SphericalGrid(nlon=2, nlat=10, pole_lat=0.0)
        
        results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, "Forward tracing test")
        
        assert results is not None
        assert results.r_line is not None
        assert len(results.r_line) > 0
        
        # Check that all field lines have data
        valid_lines = [r for r in results.r_line if r is not None and len(r) > 0]
        assert len(valid_lines) > 0

    def test_backward_tracing_basic(self):
        """Test basic backward field line tracing."""
        # Create grid from known endpoints
        endpoints = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
        custom_grid = self._create_custom_grid_from_endpoints(endpoints)
        
        results = self._trace_field_lines_time_reversal(self.field, custom_grid, self.R_max, self.t_span_backward, "Backward tracing test")
        
        assert results is not None
        assert results.r_line is not None
        assert len(results.r_line) > 0

    def test_endpoint_extraction_accuracy(self):
        """Test that endpoints are extracted correctly."""
        grid = SphericalGrid(nlon=2, nlat=5, pole_lat=0.0)
        
        results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, "Forward trace")
        endpoints = self._extract_endpoints(results)
        
        assert len(endpoints) > 0
        assert endpoints.shape[1] == 3  # x, y, z coordinates
        
        # Check that endpoints are at reasonable distances
        distances = np.linalg.norm(endpoints, axis=1)
        assert np.all(distances > 0.5)  # Not too close to center
        assert np.all(distances < 20.0)  # Not too far away

    def test_custom_grid_creation(self):
        """Test creating custom grid from endpoint coordinates."""
        test_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        custom_grid = self._create_custom_grid_from_endpoints(test_points)
        
        assert custom_grid is not None
        assert custom_grid.nline == 3
        
        # Check that coordinates are preserved
        assert custom_grid.footpoint_coordinates is not None
        grid_coords = custom_grid.footpoint_coordinates.xyz.value.T
        np.testing.assert_allclose(grid_coords, test_points)

    def test_position_difference_calculation(self):
        """Test calculation of position differences."""
        # Create simple grid
        coords = CartesianRepresentation(x=[1.0, 0.0], y=[0.0, 1.0], z=[0.0, 0.0])
        grid = CustomGrid(nline_ori=2, coordinates=coords)
        
        # Test with same endpoints (perfect match)
        same_endpoints = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        distances = self._calculate_position_differences(grid, same_endpoints)
        
        assert len(distances) == 2
        np.testing.assert_allclose(distances, [0.0, 0.0])
        
        # Test with different endpoints
        diff_endpoints = np.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0]])
        distances = self._calculate_position_differences(grid, diff_endpoints)
        
        assert len(distances) == 2
        np.testing.assert_allclose(distances, [0.1, 0.1])

    def test_xz_plane_confinement(self):
        """Test XZ plane confinement checking."""
        grid = SphericalGrid(nlon=2, nlat=8, pole_lat=0.0)
        
        results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, "XZ plane test")
        
        # Should be confined to XZ plane for symmetric field
        is_confined = self._check_xz_plane_confinement(results)
        assert is_confined, "Field lines should be confined to XZ plane"

    def test_reversibility_simple_case(self):
        """Test basic reversibility with simple grid."""
        grid = SphericalGrid(nlon=2, nlat=5, pole_lat=0.0)
        
        # Forward trace
        forward_results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, "Forward trace")
        endpoints = self._extract_endpoints(forward_results)
        
        # Backward trace
        custom_grid = self._create_custom_grid_from_endpoints(endpoints)
        backward_results = self._trace_field_lines_time_reversal(self.field, custom_grid, self.R_max, self.t_span_forward, "Backward trace")
        final_endpoints = self._extract_endpoints(backward_results)
        
        # Check reversibility
        distances = self._calculate_position_differences(grid, final_endpoints)
        
        assert len(distances) > 0
        assert np.mean(distances) < 1.0  # Reasonable accuracy
        
        # Count successful reversals (within tolerance)
        tolerance = 1e-5
        successful = np.sum(distances < tolerance)
        success_rate = successful / len(distances)
        
        assert success_rate == 1.0  # All lines should reverse well

    def test_reversibility_with_inclined_dipole(self):
        """Test reversibility with inclined dipole (65° like example)."""
        grid = SphericalGrid(nlon=2, nlat=8, pole_lat=0.0)
        
        # Use only inclined dipole
        dipole_field = MultipoleField([self.dipole])
        
        # Forward trace
        forward_results = self._trace_field_lines_time_reversal(dipole_field, grid, self.R_max, self.t_span_forward, "Forward dipole")
        endpoints = self._extract_endpoints(forward_results)
        
        assert len(endpoints) > 0
        
        # Backward trace
        custom_grid = self._create_custom_grid_from_endpoints(endpoints)
        backward_results = self._trace_field_lines_time_reversal(dipole_field, custom_grid, self.R_max, self.t_span_forward, "Backward dipole")
        final_endpoints = self._extract_endpoints(backward_results)
        
        # Check reversibility
        distances = self._calculate_position_differences(grid, final_endpoints)
        print("Distances", distances)
        # Dipole should have better reversibility
        tolerance = 5e-4
        successful = np.sum(distances < tolerance)
        success_rate = successful / len(distances)
        
        assert success_rate == 1.0  

    def test_reversibility_statistics(self):
        """Test comprehensive reversibility statistics."""
        grid = SphericalGrid(nlon=2, nlat=10, pole_lat=0.0)
        
        # Forward trace
        forward_results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, "Forward stats")
        endpoints = self._extract_endpoints(forward_results)
        
        # Backward trace
        custom_grid = self._create_custom_grid_from_endpoints(endpoints)
        backward_results = self._trace_field_lines_time_reversal(self.field, custom_grid, self.R_max, self.t_span_forward, "Backward stats")
        final_endpoints = self._extract_endpoints(backward_results)
        
        # Calculate statistics
        distances = self._calculate_position_differences(grid, final_endpoints)
        
        if len(distances) > 0:
            max_distance = np.max(distances)
            mean_distance = np.mean(distances)
            median_distance = np.median(distances)
            
            # Basic sanity checks
            assert max_distance >= mean_distance
            assert max_distance >= median_distance
            assert mean_distance >= 0
            assert median_distance >= 0
            
            # Test different tolerance levels
            tolerances = [1e-3, 1e-2]
            for tol in tolerances:
                within_tolerance = np.sum(distances < tol)
                success_rate = within_tolerance / len(distances)
                # Allow for small numerical errors (at least 90% should pass)
                assert success_rate >= 0.9, f"Success rate {success_rate} below 0.9 for tolerance {tol}"

    def test_different_grid_sizes(self):
        """Test reversibility with different grid sizes."""
        grid_configs = [
            (2, 5),   # Small grid
            (2, 10),  # Medium grid  
            (2, 20),  # Large grid
        ]
        
        for nlon, nlat in grid_configs:
            grid = SphericalGrid(nlon=nlon, nlat=nlat, pole_lat=0.0)
            
            # Forward trace
            forward_results = self._trace_field_lines_time_reversal(self.field, grid, self.R_max, self.t_span_forward, f"Forward {nlon}x{nlat}")
            
            if forward_results is not None:
                endpoints = self._extract_endpoints(forward_results)
                
                if len(endpoints) > 0:
                    # Should be able to create custom grid
                    custom_grid = self._create_custom_grid_from_endpoints(endpoints)
                    assert custom_grid is not None
                    
                    # Should be able to trace backward
                    backward_results = self._trace_field_lines_time_reversal(self.field, custom_grid, self.R_max, self.t_span_backward, f"Backward {nlon}x{nlat}")
                    assert backward_results is not None

    def test_field_strength_ratios(self):
        """Test different dipole/quadrupole strength ratios using verify_by_reversing."""
        grid = SphericalGrid(nlon=2, nlat=5, pole_lat=0.0)
        
        ratios = [0, 0.1, 0.5, 0.9, 1.5]  # quadrupole/dipole ratios
        
        for ratio in ratios:
            quad = QuadrupoleFieldComponent(
                B_magnitude=ratio,
                moment_orientation_latitude_degree=45.0,
                moment_orientation_longitude_degree=0.0
            )
            
            field = MultipoleField([self.dipole, quad])
            
            # Forward trace with verify_by_reversing enabled
            forward_results = self._trace_field_lines_time_reversal(
                field, grid, self.R_max, self.t_span_forward, 
                f"Forward ratio {ratio}", verify_by_reversing=True
            )
            
            if forward_results is not None:
                # Check that reversed_error property exists
                reversed_errors = forward_results.get_property('reversed_error')
                
                if reversed_errors is not None and len(reversed_errors) > 0:
                    # Filter out infinite errors (failed reversals)
                    finite_errors = reversed_errors[np.isfinite(reversed_errors)]
                    
                    if len(finite_errors) > 0:
                        max_error = np.max(finite_errors)
                        mean_error = np.mean(finite_errors)
                        min_error = np.min(finite_errors)
                        print(f"\nRatio {ratio} reversed_error:")
                        print(f"  All errors: {reversed_errors}")
                        print(f"  Finite errors: {finite_errors}")
                        print(f"  Min: {min_error:.6e}, Mean: {mean_error:.6e}, Max: {max_error:.6e}")
                        print(f"  Within 1e-3 tolerance: {np.sum(finite_errors < 1e-3)}/{len(finite_errors)}")
                        
                        # Verify reversibility: reversed_error should be within 1e-3 tolerance
                        assert np.all(finite_errors < 1e-3), f"All reversed_error values should be within 1e-3 tolerance for ratio {ratio}, got max error {max_error:.6f}"

    def test_edge_cases(self):
        """Test edge cases for reversibility."""
        # Single point grid
        coords = CartesianRepresentation(x=[0.0], y=[0.0], z=[1.0])
        single_grid = CustomGrid(nline_ori=1, coordinates=coords)
        
        # Should be able to trace single field line
        results = self._trace_field_lines_time_reversal(self.field, single_grid, self.R_max, self.t_span_forward, "Single point")
        
        if results is not None:
            endpoints = self._extract_endpoints(results)
            assert len(endpoints) <= 1  # At most one endpoint
            
            if len(endpoints) == 1:
                # Should be able to create custom grid from single point
                custom_grid = self._create_custom_grid_from_endpoints(endpoints)
                assert custom_grid is not None
                assert custom_grid.nline == 1

    def test_empty_results_handling(self):
        """Test handling of empty or None results."""
        # Test with empty endpoints
        empty_endpoints = np.array([])
        custom_grid = self._create_custom_grid_from_endpoints(empty_endpoints)
        assert custom_grid is None
        
        # Test position differences with empty endpoints
        coords = CartesianRepresentation(x=[1.0], y=[0.0], z=[0.0])
        grid = CustomGrid(nline_ori=1, coordinates=coords)
        
        distances = self._calculate_position_differences(grid, empty_endpoints)
        assert len(distances) == 0
        
        # Test XZ plane check with None results
        is_confined = self._check_xz_plane_confinement(None)
        assert not is_confined


class TestReversedErrorFunctionality:
    """Tests for the reversed_error property and verification functionality."""

    def setup_method(self):
        """Set up test fixtures for reversed_error tests."""
        setup_test_logger("test_reversed_error")
        
        # Create field components
        self.dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )
        
        self.quadrupole = QuadrupoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )
        
        self.field = MultipoleField([self.dipole, self.quadrupole])
        
        # Create small grid for testing
        self.grid = SphericalGrid(nlon=2, nlat=5, pole_lat=0.0)

    def test_reversed_error_enabled(self):
        """Test that reversed_error is calculated when verify_by_reversing is enabled."""
        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                t_span=(0.0, 1e6),
                dt=0.1,
                trace_open_lines_backwards=True,
                verify_by_reversing=True,
                field_backend="numba",
                solver_backend="cyrk"
            ),
            performance=PerformanceParameters(
                n_trial=5,
                max_memory_mb=10.0,
                n_cores=1
            ),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for reversibility testing
                filename=None
            )
        )
        
        context = CalculationContext(self.field, self.grid, params)
        calculator = FieldLineCalculator(context)
        template = FieldLineTemplate.template_line_coordinates_only()
        
        results = calculator.get_field_lines(template)
        
        assert results is not None
        assert results.n_field_lines > 0
        
        # Check that reversed_error property exists
        reversed_error = results.get_property('reversed_error')
        assert reversed_error is not None
        assert len(reversed_error) == results.n_field_lines

    def test_reversed_error_disabled(self):
        """Test that reversed_error is not calculated when verify_by_reversing is disabled."""
        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                t_span=(0.0, 1e6),
                dt=0.1,
                trace_open_lines_backwards=True,
                verify_by_reversing=False,
                field_backend="numba",
                solver_backend="cyrk"
            ),
            performance=PerformanceParameters(
                n_trial=5,
                max_memory_mb=10.0,
                n_cores=1
            ),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for reversibility testing
                filename=None
            )
        )
        
        context = CalculationContext(self.field, self.grid, params)
        calculator = FieldLineCalculator(context)
        template = FieldLineTemplate.template_line_coordinates_only()
        
        results = calculator.get_field_lines(template)
        
        assert results is not None
        assert results.n_field_lines > 0
        
        # Check that reversed_error property does not exist
        reversed_error = results.get_property('reversed_error')
        assert reversed_error is None

    def test_reversed_error_values_range(self):
        """Test that reversed_error values are within expected range."""
        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                t_span=(0.0, 1e6),
                dt=0.1,
                trace_open_lines_backwards=True,
                verify_by_reversing=True,
                field_backend="numba",
                solver_backend="cyrk"
            ),
            performance=PerformanceParameters(
                n_trial=5,
                max_memory_mb=10.0,
                n_cores=1
            ),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for reversibility testing
                filename=None
            )
        )
        
        context = CalculationContext(self.field, self.grid, params)
        calculator = FieldLineCalculator(context)
        template = FieldLineTemplate.template_line_coordinates_only()
        
        results = calculator.get_field_lines(template)
        reversed_error = results.get_property('reversed_error') if results is not None else None
        
        # Check value ranges
        if reversed_error is not None:
            for error in reversed_error:
                assert error >= 0.0  # Errors should be non-negative
                assert not np.isnan(error)  # Should not be NaN
                # Note: error can be inf if reversed integration fails

    def test_reversed_error_with_different_max_steps(self):
        """Test reversed_error calculation with different max_step values."""
        max_steps = [0.5, 1.0, 5.0]
        
        for max_step in max_steps:
            params = TracingParameters(
                integration=IntegrationParameters(
                    max_integration_radius=10.0,
                    threshold_line_distance=1.0,
                    t_span=(0.0, 1e6),
                    dt=0.1,
                    trace_open_lines_backwards=True,
                    verify_by_reversing=True,
                    max_step=max_step,
                    field_backend="numba",
                    solver_backend="cyrk"       
                ),
                performance=PerformanceParameters(
                    n_trial=5,
                    max_memory_mb=10.0,
                    n_cores=1
                ),
                output=OutputParameters(
                    save_results_to_disk=False,
                    keep_results_in_memory=True,  # Keep full results for reversibility testing
                    filename=None
                )
            )
            
            context = CalculationContext(self.field, self.grid, params)
            calculator = FieldLineCalculator(context)
            template = FieldLineTemplate.template_line_coordinates_only()
            
            results = calculator.get_field_lines(template)
            
            assert results is not None
            assert results.n_field_lines > 0

    def test_reversed_error_statistics(self):
        """Test statistical properties of reversed_error values."""
        params = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                t_span=(0.0, 1e6),
                dt=0.1,
                trace_open_lines_backwards=True,
                verify_by_reversing=True,
            ),
            performance=PerformanceParameters(
                n_trial=5,
                max_memory_mb=10.0,
                n_cores=1
            ),
            output=OutputParameters(
                save_results_to_disk=False,
                keep_results_in_memory=True,  # Keep full results for reversibility testing
                filename=None
            )
        )
        
        context = CalculationContext(self.field, self.grid, params)
        calculator = FieldLineCalculator(context)
        template = FieldLineTemplate.template_line_coordinates_only()
        
        results = calculator.get_field_lines(template)
        
        assert results is not None
        assert results.n_field_lines > 0
        
        # Check that reversed_error property exists and has reasonable statistics
        reversed_error = results.get_property('reversed_error')
        if reversed_error is not None:
            assert len(reversed_error) == results.n_field_lines
            assert np.all(np.isfinite(reversed_error) | np.isinf(reversed_error))  # Should be finite or inf
            assert np.all(reversed_error >= 0.0)  # Should be non-negative

