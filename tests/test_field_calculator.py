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

from psr_magpy.line.field_calculator import PythonFieldCalculator
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent


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

