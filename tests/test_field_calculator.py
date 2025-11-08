#!/usr/bin/env python3
"""
Tests for FieldCalculator module.

Tests PythonFieldCalculator to ensure identical results to MultipoleField.get_total_field().
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path (same as other tests)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.line.field_calculator import PythonFieldCalculator, NumbaFieldCalculator
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from psr_magpy.line.tracer import FieldLineTracer
from psr_magpy.line.tracing_parameters import TracingParameters, IntegrationParameters, OutputParameters
from psr_magpy.line.field_line import FieldLine
from astropy.coordinates import CartesianRepresentation


class TestPythonFieldCalculator:
    """Test PythonFieldCalculator against MultipoleField.get_total_field()."""
    
    def test_single_dipole_normalized(self):
        """Test single dipole field with normalization."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        # Test positions
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
            np.array([1.5, 1.5, 1.5], dtype=np.float64),
        ]
        
        for r in positions:
            # Convert to float32 for MultipoleField.get_total_field()
            r_f32 = r.astype(np.float32)
            
            # Get results
            result_calc = calculator.get_field(r, K=1, normalize=True)
            result_field = field.get_total_field(r_f32, K=1, normalize=True)
            
            # Compare (convert field result to float64 for comparison)
            result_field_f64 = result_field.astype(np.float64)
            
            np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7,
                                     err_msg=f"Position {r} failed")
    
    def test_single_dipole_non_normalized(self):
        """Test single dipole field without normalization."""
        dipole = DipoleFieldComponent(B_magnitude=2.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        result_calc = calculator.get_field(r, K=1, normalize=False)
        result_field = field.get_total_field(r_f32, K=1, normalize=False)
        result_field_f64 = result_field.astype(np.float64)
        
        np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7)
    
    def test_single_dipole_negative_K(self):
        """Test single dipole field with K=-1."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        result_calc = calculator.get_field(r, K=-1, normalize=True)
        result_field = field.get_total_field(r_f32, K=-1, normalize=True)
        result_field_f64 = result_field.astype(np.float64)
        
        np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7)
    
    def test_multiple_dipoles(self):
        """Test multiple dipole components."""
        dipole1 = DipoleFieldComponent(B_magnitude=1.0)
        dipole2 = DipoleFieldComponent(
            B_magnitude=0.5,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0
        )
        field = MultipoleField([dipole1, dipole2])
        calculator = PythonFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.5, 1.5, 1.5], dtype=np.float64),
            np.array([0.0, 0.0, 3.0], dtype=np.float64),
        ]
        
        for r in positions:
            r_f32 = r.astype(np.float32)
            
            result_calc = calculator.get_field(r, K=1, normalize=True)
            result_field = field.get_total_field(r_f32, K=1, normalize=True)
            result_field_f64 = result_field.astype(np.float64)
            
            np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7,
                                     err_msg=f"Position {r} failed")
    
    def test_dipole_quadrupole(self):
        """Test dipole + quadrupole combination."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        quadrupole = QuadrupoleFieldComponent(B_magnitude=0.3)
        field = MultipoleField([dipole, quadrupole])
        calculator = PythonFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.5], dtype=np.float64),
        ]
        
        for r in positions:
            r_f32 = r.astype(np.float32)
            
            result_calc = calculator.get_field(r, K=1, normalize=True)
            result_field = field.get_total_field(r_f32, K=1, normalize=True)
            result_field_f64 = result_field.astype(np.float64)
            
            np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7,
                                     err_msg=f"Position {r} failed")
    
    def test_different_K_values(self):
        """Test different K values."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([2.0, 1.0, 1.5], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        for K in [1, -1]:
            result_calc = calculator.get_field(r, K=K, normalize=True)
            result_field = field.get_total_field(r_f32, K=K, normalize=True)
            result_field_f64 = result_field.astype(np.float64)
            
            np.testing.assert_allclose(result_calc, result_field_f64, rtol=1e-6, atol=1e-7,
                                     err_msg=f"K={K} failed")
    
    def test_normalize_on_off(self):
        """Test normalize=True vs normalize=False."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([2.0, 1.0, 1.5], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        # Normalized
        result_calc_norm = calculator.get_field(r, K=1, normalize=True)
        result_field_norm = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
        np.testing.assert_allclose(result_calc_norm, result_field_norm, rtol=1e-6, atol=1e-7)
        
        # Non-normalized
        result_calc_nonnorm = calculator.get_field(r, K=1, normalize=False)
        result_field_nonnorm = field.get_total_field(r_f32, K=1, normalize=False).astype(np.float64)
        np.testing.assert_allclose(result_calc_nonnorm, result_field_nonnorm, rtol=1e-6, atol=1e-7)
        
        # Verify normalized result has unit magnitude (if field is non-zero)
        field_mag = np.linalg.norm(result_field_nonnorm)
        norm_mag = np.linalg.norm(result_calc_norm)
        if field_mag > 1e-10:  # Only check if field is non-zero
            assert np.allclose(norm_mag, 1.0, rtol=1e-6), \
                f"Normalized field should have unit magnitude, got {norm_mag}"
        
        # Verify non-normalized is different from normalized (unless field magnitude happens to be 1)
        if not np.allclose(field_mag, 1.0, rtol=1e-3):
            assert not np.allclose(result_calc_norm, result_calc_nonnorm, rtol=1e-3)
    
    def test_edge_case_near_origin(self):
        """Test position near origin."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        result_calc = calculator.get_field(r, K=1, normalize=True)
        result_field = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
        
        np.testing.assert_allclose(result_calc, result_field, rtol=1e-6, atol=1e-7)
    
    def test_edge_case_large_radius(self):
        """Test position at large radius."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        r = np.array([100.0, 50.0, 75.0], dtype=np.float64)
        r_f32 = r.astype(np.float32)
        
        result_calc = calculator.get_field(r, K=1, normalize=True)
        result_field = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
        
        np.testing.assert_allclose(result_calc, result_field, rtol=1e-6, atol=1e-7)
    
    def test_inclined_dipole(self):
        """Test inclined dipole (non-aligned with z-axis)."""
        dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0
        )
        field = MultipoleField([dipole])
        calculator = PythonFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
        ]
        
        for r in positions:
            r_f32 = r.astype(np.float32)
            
            result_calc = calculator.get_field(r, K=1, normalize=True)
            result_field = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
            
            np.testing.assert_allclose(result_calc, result_field, rtol=1e-6, atol=1e-7,
                                     err_msg=f"Position {r} failed")


class TestNumbaFieldCalculator:
    """Test NumbaFieldCalculator against PythonFieldCalculator."""
    
    def test_single_dipole_normalized(self):
        """Test single dipole field with normalization."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
            np.array([1.5, 1.5, 1.5], dtype=np.float64),
        ]
        
        for r in positions:
            result_python = calc_python.get_field(r, K=1, normalize=True)
            result_numba = calc_numba.get_field(r, K=1, normalize=True)
            
            np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Position {r} failed")
    
    def test_single_dipole_non_normalized(self):
        """Test single dipole field without normalization."""
        dipole = DipoleFieldComponent(B_magnitude=2.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        
        result_python = calc_python.get_field(r, K=1, normalize=False)
        result_numba = calc_numba.get_field(r, K=1, normalize=False)
        
        np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12)
    
    def test_single_dipole_negative_K(self):
        """Test single dipole field with K=-1."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        
        result_python = calc_python.get_field(r, K=-1, normalize=True)
        result_numba = calc_numba.get_field(r, K=-1, normalize=True)
        
        np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12)
    
    def test_multiple_dipoles(self):
        """Test multiple dipole components."""
        dipole1 = DipoleFieldComponent(B_magnitude=1.0)
        dipole2 = DipoleFieldComponent(
            B_magnitude=0.5,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0
        )
        field = MultipoleField([dipole1, dipole2])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.5, 1.5, 1.5], dtype=np.float64),
            np.array([0.0, 0.0, 3.0], dtype=np.float64),
        ]
        
        for r in positions:
            result_python = calc_python.get_field(r, K=1, normalize=True)
            result_numba = calc_numba.get_field(r, K=1, normalize=True)
            
            np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Position {r} failed")
    
    def test_dipole_quadrupole(self):
        """Test dipole + quadrupole combination."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        quadrupole = QuadrupoleFieldComponent(B_magnitude=0.3)
        field = MultipoleField([dipole, quadrupole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.5], dtype=np.float64),
        ]
        
        for r in positions:
            result_python = calc_python.get_field(r, K=1, normalize=True)
            result_numba = calc_numba.get_field(r, K=1, normalize=True)
            
            np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Position {r} failed")
    
    def test_different_K_values(self):
        """Test different K values."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([2.0, 1.0, 1.5], dtype=np.float64)
        
        for K in [1, -1]:
            result_python = calc_python.get_field(r, K=K, normalize=True)
            result_numba = calc_numba.get_field(r, K=K, normalize=True)
            
            np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12,
                                     err_msg=f"K={K} failed")
    
    def test_normalize_on_off(self):
        """Test normalize=True vs normalize=False."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([2.0, 1.0, 1.5], dtype=np.float64)
        
        # Normalized
        result_python_norm = calc_python.get_field(r, K=1, normalize=True)
        result_numba_norm = calc_numba.get_field(r, K=1, normalize=True)
        np.testing.assert_allclose(result_numba_norm, result_python_norm, rtol=1e-10, atol=1e-12)
        
        # Non-normalized
        result_python_nonnorm = calc_python.get_field(r, K=1, normalize=False)
        result_numba_nonnorm = calc_numba.get_field(r, K=1, normalize=False)
        np.testing.assert_allclose(result_numba_nonnorm, result_python_nonnorm, rtol=1e-10, atol=1e-12)
        
        # Verify normalized result has unit magnitude (if field is non-zero)
        field_mag = np.linalg.norm(result_python_nonnorm)
        norm_mag = np.linalg.norm(result_numba_norm)
        if field_mag > 1e-10:  # Only check if field is non-zero
            assert np.allclose(norm_mag, 1.0, rtol=1e-10), \
                f"Normalized field should have unit magnitude, got {norm_mag}"
        
        # Verify non-normalized is different from normalized (unless field magnitude happens to be 1)
        if not np.allclose(field_mag, 1.0, rtol=1e-3):
            assert not np.allclose(result_numba_norm, result_numba_nonnorm, rtol=1e-3)
    
    def test_edge_case_near_origin(self):
        """Test position near origin."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        
        result_python = calc_python.get_field(r, K=1, normalize=True)
        result_numba = calc_numba.get_field(r, K=1, normalize=True)
        
        np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12)
    
    def test_edge_case_large_radius(self):
        """Test position at large radius."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        r = np.array([100.0, 50.0, 75.0], dtype=np.float64)
        
        result_python = calc_python.get_field(r, K=1, normalize=True)
        result_numba = calc_numba.get_field(r, K=1, normalize=True)
        
        np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12)
    
    def test_inclined_dipole(self):
        """Test inclined dipole (non-aligned with z-axis)."""
        dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0
        )
        field = MultipoleField([dipole])
        calc_python = PythonFieldCalculator(field)
        calc_numba = NumbaFieldCalculator(field)
        
        positions = [
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 2.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
        ]
        
        for r in positions:
            result_python = calc_python.get_field(r, K=1, normalize=True)
            result_numba = calc_numba.get_field(r, K=1, normalize=True)
            
            np.testing.assert_allclose(result_numba, result_python, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Position {r} failed")


class TestFieldLineTracerBackendComparison:
    """Integration test: FieldLineTracer with field_backend='numba' vs 'python'."""
    
    def test_trace_single_line_python_vs_numba(self):
        """Test that FieldLineTracer produces identical results with both backends."""
        # Create field
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        
        # Create parameters with explicit field_backend
        params_python = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                field_backend="python"
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        params_numba = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                field_backend="numba"
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        # Create tracers
        tracer_python = FieldLineTracer(field, params_python)
        tracer_numba = FieldLineTracer(field, params_numba)
        
        # Create template
        template = FieldLine(save_r_line=True, save_r_max=True)
        
        # Initial position at surface
        r_initial = CartesianRepresentation(x=1.0, y=0.0, z=0.0)
        footpoint_id = 1
        
        # Trace with both backends
        result_python = tracer_python.trace_single_line(r_initial, footpoint_id, template)
        result_numba = tracer_numba.trace_single_line(r_initial, footpoint_id, template)
        
        # Compare results
        r_line_python = result_python.get_property('r_line')
        r_line_numba = result_numba.get_property('r_line')
        r_max_python = result_python.get_property('r_max')
        r_max_numba = result_numba.get_property('r_max')
        
        # Both should have results
        assert r_line_python is not None
        assert r_line_numba is not None
        assert r_max_python is not None
        assert r_max_numba is not None
        
        # Compare r_line (field line trajectory)
        # r_line is already a numpy array with shape (N, 3)
        r_line_python_arr = np.asarray(r_line_python)
        r_line_numba_arr = np.asarray(r_line_numba)
        
        # Check same length
        assert len(r_line_python_arr) == len(r_line_numba_arr), \
            f"Field lines have different lengths: python={len(r_line_python_arr)}, numba={len(r_line_numba_arr)}"
        
        # Compare trajectories (allow for small numerical differences)
        np.testing.assert_allclose(r_line_numba_arr, r_line_python_arr, rtol=1e-6, atol=1e-7,
                                 err_msg="Field line trajectories differ between backends")
        
        # Compare r_max
        np.testing.assert_allclose(r_max_numba, r_max_python, rtol=1e-6, atol=1e-7,
                                 err_msg="r_max differs between backends")
    
    def test_trace_multiple_positions_python_vs_numba(self):
        """Test multiple field line traces with both backends."""
        # Create field with multiple components
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        quadrupole = QuadrupoleFieldComponent(B_magnitude=0.3)
        field = MultipoleField([dipole, quadrupole])
        
        # Create parameters
        params_python = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                field_backend="python"
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        params_numba = TracingParameters(
            integration=IntegrationParameters(
                max_integration_radius=10.0,
                threshold_line_distance=1.0,
                field_backend="numba"
            ),
            output=OutputParameters(save_results_to_disk=False)
        )
        
        # Create tracers
        tracer_python = FieldLineTracer(field, params_python)
        tracer_numba = FieldLineTracer(field, params_numba)
        
        # Test multiple positions
        positions = [
            CartesianRepresentation(x=1.0, y=0.0, z=0.0),
            CartesianRepresentation(x=0.0, y=1.0, z=0.0),
            CartesianRepresentation(x=0.0, y=0.0, z=1.0),
            CartesianRepresentation(x=0.707, y=0.707, z=0.0),
        ]
        
        template = FieldLine(save_r_line=True, save_r_max=True)
        
        for i, r_initial in enumerate(positions):
            footpoint_id = i + 1
            
            # Trace with both backends
            result_python = tracer_python.trace_single_line(r_initial, footpoint_id, template)
            result_numba = tracer_numba.trace_single_line(r_initial, footpoint_id, template)
            
            # Compare r_max
            r_max_python = result_python.get_property('r_max')
            r_max_numba = result_numba.get_property('r_max')
            
            np.testing.assert_allclose(r_max_numba, r_max_python, rtol=1e-6, atol=1e-7,
                                     err_msg=f"r_max differs for position {i+1}: {r_initial}")
            
            # Compare r_line trajectories
            r_line_python = result_python.get_property('r_line')
            r_line_numba = result_numba.get_property('r_line')
            
            # r_line is already a numpy array with shape (N, 3)
            r_line_python_arr = np.asarray(r_line_python)
            r_line_numba_arr = np.asarray(r_line_numba)
            
            # Check same length
            assert len(r_line_python_arr) == len(r_line_numba_arr), \
                f"Field line {i+1} has different lengths: python={len(r_line_python_arr)}, numba={len(r_line_numba_arr)}"
            
            # Compare trajectories
            np.testing.assert_allclose(r_line_numba_arr, r_line_python_arr, rtol=1e-6, atol=1e-7,
                                     err_msg=f"Field line trajectory differs for position {i+1}: {r_initial}")
