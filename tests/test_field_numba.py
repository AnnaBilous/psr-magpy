#!/usr/bin/env python3
"""
Tests for Numba-compiled field calculation functions.

Tests Numba implementations against Python implementations for accuracy.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.line.field_numba import (
    _calculate_dipole_field_numba,
    _rotate_field_to_spin_frame_numba,
)
from psr_magpy.field.components import DipoleFieldComponent


class TestNumbaDipoleField:
    """Test Numba-compiled dipole field calculation against Python implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.B_magnitude = 1.0
    
    def test_dipole_field_on_axis(self):
        """Test dipole field calculation on magnetic axis."""
        # Test positions in magnetic frame
        positions = [
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.5], dtype=np.float64),
        ]
        
        for r_magn in positions:
            # Python: _calculate_field takes r_magn_frame and returns field in spin frame
            B_python_spin = self.dipole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_dipole_field_numba(r_magn, self.B_magnitude)
            rotation_matrix = self.dipole.numba_compatible_rotation_matrix
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            # Compare in spin frame
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_dipole_field_equatorial_plane(self):
        """Test dipole field in equatorial plane (z=0)."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0, 0.0], dtype=np.float64),
        ]
        
        rotation_matrix = self.dipole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: returns field in spin frame
            B_python_spin = self.dipole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_dipole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_dipole_field_off_axis(self):
        """Test dipole field at off-axis positions."""
        positions = [
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.0, 1.5], dtype=np.float64),
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
        ]
        
        rotation_matrix = self.dipole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: returns field in spin frame
            B_python_spin = self.dipole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_dipole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_dipole_field_magnitude_scaling(self):
        """Test that field scales correctly with B_magnitude."""
        r_magn = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        rotation_matrix = self.dipole.numba_compatible_rotation_matrix
        
        B_magnitudes = [0.5, 1.0, 2.0, 5.0]
        python_results = []
        numba_results = []
        
        for B_mag in B_magnitudes:
            # Python
            dipole = DipoleFieldComponent(B_magnitude=B_mag)
            B_python_spin = dipole._calculate_field(r_magn)
            python_results.append(B_python_spin)
            
            # Numba
            B_numba_magn = _calculate_dipole_field_numba(r_magn, B_mag)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            numba_results.append(B_numba_spin)
        
        # Compare Python vs Numba for each magnitude
        for i, B_mag in enumerate(B_magnitudes):
            np.testing.assert_allclose(
                numba_results[i], python_results[i], rtol=1e-10, atol=1e-12,
                err_msg=f"B_magnitude {B_mag} comparison failed"
            )
    
    def test_dipole_field_origin_singularity(self):
        """Test handling of field near origin."""
        # Very close to origin
        positions = [
            np.array([1e-11, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1e-11, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1e-11], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ]
        
        rotation_matrix = self.dipole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: should return zero field
            B_python_spin = self.dipole._calculate_field(r_magn)
            
            # Numba: calculate and rotate
            B_numba_magn = _calculate_dipole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            # Both should return zero field
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Origin singularity handling failed for {r_magn}"
            )
            np.testing.assert_allclose(
                B_python_spin, np.zeros(3, dtype=np.float64), atol=1e-12,
                err_msg=f"Python should return zero for {r_magn}"
            )
    
    def test_dipole_field_radial_falloff(self):
        """Test 1/r³ radial falloff of dipole field magnitude."""
        # Positions along z-axis
        distances = [1.0, 2.0, 3.0, 4.0]
        magnitudes = []
        
        for r in distances:
            r_magn = np.array([0.0, 0.0, r], dtype=np.float64)
            B_numba = _calculate_dipole_field_numba(r_magn, self.B_magnitude)
            magnitudes.append(np.linalg.norm(B_numba))
        
        # Check 1/r³ falloff (field magnitude should scale as 1/r³)
        # On z-axis, field is purely in z-direction with magnitude ~1/r³
        expected_ratios = [1.0, 1.0/8.0, 1.0/27.0, 1.0/64.0]
        actual_ratios = [m/magnitudes[0] for m in magnitudes]
        
        for expected, actual in zip(expected_ratios, actual_ratios):
            assert np.abs(expected - actual) < 0.01, \
                f"Radial falloff test failed: expected {expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

