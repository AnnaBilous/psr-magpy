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
    _calculate_quadrupole_field_numba,
    _transform_to_magnetic_frame_numba,
    _rotate_field_to_spin_frame_numba,
    _get_component_field_numba,
    get_total_field_numba,
)
from psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
)
from psr_magpy.field.multipole import MultipoleField


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


class TestNumbaQuadrupoleField:
    """Test Numba-compiled quadrupole field calculation against Python implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quadrupole = QuadrupoleFieldComponent(B_magnitude=1.0)
        self.B_magnitude = 1.0
    
    def test_quadrupole_field_on_axis(self):
        """Test quadrupole field calculation on magnetic axis."""
        positions = [
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.5], dtype=np.float64),
        ]
        
        for r_magn in positions:
            # Python: _calculate_field takes r_magn_frame and returns field in spin frame
            B_python_spin = self.quadrupole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, self.B_magnitude)
            rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            # Compare in spin frame
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_quadrupole_field_equatorial_plane(self):
        """Test quadrupole field in equatorial plane (z=0)."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([2.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 1.0, 0.0], dtype=np.float64),
        ]
        
        rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: returns field in spin frame
            B_python_spin = self.quadrupole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_quadrupole_field_off_axis(self):
        """Test quadrupole field at off-axis positions."""
        positions = [
            np.array([1.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.0, 1.5], dtype=np.float64),
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
        ]
        
        rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: returns field in spin frame
            B_python_spin = self.quadrupole._calculate_field(r_magn)
            
            # Numba: calculate in magnetic frame, then rotate to spin frame
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            
            np.testing.assert_allclose(
                B_numba_spin, B_python_spin, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_magn} failed"
            )
    
    def test_quadrupole_field_magnitude_scaling(self):
        """Test that field scales correctly with B_magnitude."""
        r_magn = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
        
        B_magnitudes = [0.5, 1.0, 2.0, 5.0]
        python_results = []
        numba_results = []
        
        for B_mag in B_magnitudes:
            # Python
            quadrupole = QuadrupoleFieldComponent(B_magnitude=B_mag)
            B_python_spin = quadrupole._calculate_field(r_magn)
            python_results.append(B_python_spin)
            
            # Numba
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, B_mag)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            numba_results.append(B_numba_spin)
        
        # Compare Python vs Numba for each magnitude
        for i, B_mag in enumerate(B_magnitudes):
            np.testing.assert_allclose(
                numba_results[i], python_results[i], rtol=1e-10, atol=1e-12,
                err_msg=f"B_magnitude {B_mag} comparison failed"
            )
    
    def test_quadrupole_field_origin_singularity(self):
        """Test handling of field near origin."""
        positions = [
            np.array([1e-11, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1e-11, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1e-11], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ]
        
        rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
        
        for r_magn in positions:
            # Python: should return zero field
            B_python_spin = self.quadrupole._calculate_field(r_magn)
            
            # Numba: calculate and rotate
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, self.B_magnitude)
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
    
    def test_quadrupole_field_radial_falloff(self):
        """Test 1/r⁴ radial falloff of quadrupole field magnitude."""
        # Positions along z-axis
        distances = [1.0, 2.0, 3.0, 4.0]
        magnitudes = []
        
        rotation_matrix = self.quadrupole.numba_compatible_rotation_matrix
        
        for r in distances:
            r_magn = np.array([0.0, 0.0, r], dtype=np.float64)
            B_numba_magn = _calculate_quadrupole_field_numba(r_magn, self.B_magnitude)
            B_numba_spin = _rotate_field_to_spin_frame_numba(B_numba_magn, rotation_matrix)
            magnitudes.append(np.linalg.norm(B_numba_spin))
        
        # Check 1/r⁴ falloff (quadrupole field magnitude should scale as 1/r⁴)
        expected_ratios = [1.0, 1.0/16.0, 1.0/81.0, 1.0/256.0]
        actual_ratios = [m/magnitudes[0] for m in magnitudes]
        
        for expected, actual in zip(expected_ratios, actual_ratios):
            assert np.abs(expected - actual) < 0.01, \
                f"Radial falloff test failed: expected {expected}, got {actual}"


class TestNumbaTransform:
    """Test Numba-compiled coordinate transform against scipy Rotation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create components with different orientations and offsets
        self.dipole_aligned = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0,
            offset_r_ns=[0.0, 0.0, 0.0],
        )
        self.dipole_inclined = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=30.0,
            offset_r_ns=[0.5, 0.3, 0.2],
        )
        self.quadrupole_offset = QuadrupoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=60.0,
            moment_orientation_longitude_degree=90.0,
            offset_r_ns=[1.0, 0.5, 0.0],
        )
    
    def test_transform_aligned_dipole(self):
        """Test transform for aligned dipole (no rotation, no offset)."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
        ]
        
        rotation_matrix_inv = self.dipole_aligned.numba_compatible_rotation_matrix_inv
        offset_vector = self.dipole_aligned.numba_compatible_offset_vector
        
        for r_spin in positions:
            # Python: _transform_to_magnetic_frame
            r_python_magn = self.dipole_aligned._transform_to_magnetic_frame(r_spin)
            
            # Numba: transform
            r_numba_magn = _transform_to_magnetic_frame_numba(
                r_spin, rotation_matrix_inv, offset_vector
            )
            
            np.testing.assert_allclose(
                r_numba_magn, r_python_magn, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_spin} failed"
            )
    
    def test_transform_inclined_dipole(self):
        """Test transform for inclined dipole with offset."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
        ]
        
        rotation_matrix_inv = self.dipole_inclined.numba_compatible_rotation_matrix_inv
        offset_vector = self.dipole_inclined.numba_compatible_offset_vector
        
        for r_spin in positions:
            # Python: _transform_to_magnetic_frame
            r_python_magn = self.dipole_inclined._transform_to_magnetic_frame(r_spin)
            
            # Numba: transform
            r_numba_magn = _transform_to_magnetic_frame_numba(
                r_spin, rotation_matrix_inv, offset_vector
            )
            
            np.testing.assert_allclose(
                r_numba_magn, r_python_magn, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_spin} failed"
            )
    
    def test_transform_quadrupole_with_offset(self):
        """Test transform for quadrupole with offset."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
        ]
        
        rotation_matrix_inv = self.quadrupole_offset.numba_compatible_rotation_matrix_inv
        offset_vector = self.quadrupole_offset.numba_compatible_offset_vector
        
        for r_spin in positions:
            # Python: _transform_to_magnetic_frame
            r_python_magn = self.quadrupole_offset._transform_to_magnetic_frame(r_spin)
            
            # Numba: transform
            r_numba_magn = _transform_to_magnetic_frame_numba(
                r_spin, rotation_matrix_inv, offset_vector
            )
            
            np.testing.assert_allclose(
                r_numba_magn, r_python_magn, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_spin} failed"
            )
    
    def test_transform_multiple_orientations(self):
        """Test transform with various orientations."""
        orientations = [
            (0.0, 0.0),      # Aligned
            (45.0, 0.0),     # 45° latitude
            (90.0, 0.0),     # Equatorial
            (45.0, 90.0),    # 45° lat, 90° lon
            (30.0, 60.0),    # 30° lat, 60° lon
        ]
        
        r_spin = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        for lat, lon in orientations:
            dipole = DipoleFieldComponent(
                B_magnitude=1.0,
                moment_orientation_latitude_degree=lat,
                moment_orientation_longitude_degree=lon,
                offset_r_ns=[0.1, 0.2, 0.3],
            )
            
            rotation_matrix_inv = dipole.numba_compatible_rotation_matrix_inv
            offset_vector = dipole.numba_compatible_offset_vector
            
            # Python
            r_python_magn = dipole._transform_to_magnetic_frame(r_spin)
            
            # Numba
            r_numba_magn = _transform_to_magnetic_frame_numba(
                r_spin, rotation_matrix_inv, offset_vector
            )
            
            np.testing.assert_allclose(
                r_numba_magn, r_python_magn, rtol=1e-10, atol=1e-12,
                err_msg=f"Orientation (lat={lat}, lon={lon}) failed"
            )
    
    def test_transform_offset_only(self):
        """Test transform with offset but no rotation (aligned dipole)."""
        dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0,
            offset_r_ns=[0.5, 0.3, 0.2],
        )
        
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        
        rotation_matrix_inv = dipole.numba_compatible_rotation_matrix_inv
        offset_vector = dipole.numba_compatible_offset_vector
        
        for r_spin in positions:
            # Python
            r_python_magn = dipole._transform_to_magnetic_frame(r_spin)
            
            # Numba
            r_numba_magn = _transform_to_magnetic_frame_numba(
                r_spin, rotation_matrix_inv, offset_vector
            )
            
            np.testing.assert_allclose(
                r_numba_magn, r_python_magn, rtol=1e-10, atol=1e-12,
                err_msg=f"Position {r_spin} with offset failed"
            )


class TestNumbaComponentField:
    """Test Numba-compiled component field calculation against Python FieldComponent.get_field()."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=30.0,
            offset_r_ns=[0.5, 0.3, 0.2],
        )
        self.quadrupole = QuadrupoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=60.0,
            moment_orientation_longitude_degree=90.0,
            offset_r_ns=[1.0, 0.5, 0.0],
        )
    
    def _get_moment_type_int(self, moment_type_str: str) -> int:
        """Convert moment_type string to int for Numba function."""
        if moment_type_str == "dipole":
            return 0
        elif moment_type_str == "quadrupole":
            return 1
        else:
            raise ValueError(f"Unknown moment_type: {moment_type_str}")
    
    def test_dipole_component_field(self):
        """Test dipole component field calculation."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
        ]
        
        params = self.dipole.get_numba_params()
        moment_type_int = self._get_moment_type_int(params["moment_type"])
        
        for r_spin in positions:
            # Python: get_field returns field in spin frame
            B_python = self.dipole.get_field(r_spin)
            
            # Numba: component field calculation
            B_numba = _get_component_field_numba(
                r_spin,
                params["B_magnitude"],
                params["rotation_matrix"],
                params["rotation_matrix_inv"],
                params["offset_vector"],
                moment_type_int,
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-10, atol=1e-12,
                err_msg=f"Dipole position {r_spin} failed"
            )
    
    def test_quadrupole_component_field(self):
        """Test quadrupole component field calculation."""
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
        ]
        
        params = self.quadrupole.get_numba_params()
        moment_type_int = self._get_moment_type_int(params["moment_type"])
        
        for r_spin in positions:
            # Python: get_field returns field in spin frame
            B_python = self.quadrupole.get_field(r_spin)
            
            # Numba: component field calculation
            B_numba = _get_component_field_numba(
                r_spin,
                params["B_magnitude"],
                params["rotation_matrix"],
                params["rotation_matrix_inv"],
                params["offset_vector"],
                moment_type_int,
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-10, atol=1e-12,
                err_msg=f"Quadrupole position {r_spin} failed"
            )
    
    def test_dipole_aligned_no_offset(self):
        """Test aligned dipole with no offset."""
        dipole = DipoleFieldComponent(
            B_magnitude=1.0,
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0,
            offset_r_ns=[0.0, 0.0, 0.0],
        )
        
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        
        params = dipole.get_numba_params()
        moment_type_int = self._get_moment_type_int(params["moment_type"])
        
        for r_spin in positions:
            B_python = dipole.get_field(r_spin)
            B_numba = _get_component_field_numba(
                r_spin,
                params["B_magnitude"],
                params["rotation_matrix"],
                params["rotation_matrix_inv"],
                params["offset_vector"],
                moment_type_int,
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-10, atol=1e-12,
                err_msg=f"Aligned dipole position {r_spin} failed"
            )
    
    def test_quadrupole_various_orientations(self):
        """Test quadrupole with various orientations."""
        orientations = [
            (0.0, 0.0, [0.0, 0.0, 0.0]),      # Aligned, no offset
            (45.0, 0.0, [0.5, 0.0, 0.0]),     # 45° lat, offset
            (90.0, 90.0, [0.0, 0.5, 0.0]),    # Equatorial, offset
        ]
        
        r_spin = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        
        for lat, lon, offset in orientations:
            quadrupole = QuadrupoleFieldComponent(
                B_magnitude=1.0,
                moment_orientation_latitude_degree=lat,
                moment_orientation_longitude_degree=lon,
                offset_r_ns=offset,
            )
            
            params = quadrupole.get_numba_params()
            moment_type_int = self._get_moment_type_int(params["moment_type"])
            
            B_python = quadrupole.get_field(r_spin)
            B_numba = _get_component_field_numba(
                r_spin,
                params["B_magnitude"],
                params["rotation_matrix"],
                params["rotation_matrix_inv"],
                params["offset_vector"],
                moment_type_int,
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-10, atol=1e-12,
                err_msg=f"Quadrupole (lat={lat}, lon={lon}, offset={offset}) failed"
            )
    
    def test_dipole_magnitude_scaling(self):
        """Test that field scales correctly with B_magnitude."""
        r_spin = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        B_magnitudes = [0.5, 1.0, 2.0, 5.0]
        
        for B_mag in B_magnitudes:
            dipole = DipoleFieldComponent(B_magnitude=B_mag)
            params = dipole.get_numba_params()
            moment_type_int = self._get_moment_type_int(params["moment_type"])
            
            B_python = dipole.get_field(r_spin)
            B_numba = _get_component_field_numba(
                r_spin,
                params["B_magnitude"],
                params["rotation_matrix"],
                params["rotation_matrix_inv"],
                params["offset_vector"],
                moment_type_int,
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-10, atol=1e-12,
                err_msg=f"B_magnitude {B_mag} failed"
            )


class TestNumbaTotalField:
    """Test Numba-compiled total field calculation against MultipoleField.get_total_field()."""
    
    def _convert_config_to_arrays(self, config_list):
        """Convert list of component param dicts to arrays for Numba function."""
        n_components = len(config_list)
        
        B_magnitudes = np.array([p["B_magnitude"] for p in config_list], dtype=np.float64)
        
        rotation_matrices = np.array(
            [p["rotation_matrix"] for p in config_list], dtype=np.float64
        )
        
        rotation_matrices_inv = np.array(
            [p["rotation_matrix_inv"] for p in config_list], dtype=np.float64
        )
        
        offset_vectors = np.array(
            [p["offset_vector"] for p in config_list], dtype=np.float64
        )
        
        moment_types = np.array(
            [0 if p["moment_type"] == "dipole" else 1 for p in config_list],
            dtype=np.int32
        )
        
        return n_components, B_magnitudes, rotation_matrices, rotation_matrices_inv, offset_vectors, moment_types
    
    def test_single_dipole(self):
        """Test total field with single dipole component."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
        ]
        
        config = field.get_numba_config()
        n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types = self._convert_config_to_arrays(config)
        
        for r in positions:
            # Python: get_total_field (takes float32, returns float32)
            r_f32 = r.astype(np.float32)
            B_python = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
            
            # Numba: get_total_field_numba
            B_numba = get_total_field_numba(
                r, 1, True, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-6, atol=1e-7,
                err_msg=f"Single dipole position {r} failed"
            )
    
    def test_multiple_dipoles(self):
        """Test total field with multiple dipole components."""
        dipoles = [
            DipoleFieldComponent(B_magnitude=1.0),
            DipoleFieldComponent(
                B_magnitude=0.5,
                moment_orientation_latitude_degree=45.0,
                moment_orientation_longitude_degree=30.0,
                offset_r_ns=[0.5, 0.0, 0.0],
            ),
        ]
        field = MultipoleField(dipoles)
        
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
        ]
        
        config = field.get_numba_config()
        n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types = self._convert_config_to_arrays(config)
        
        for r in positions:
            r_f32 = r.astype(np.float32)
            B_python = field.get_total_field(r_f32, K=1, normalize=False).astype(np.float64)
            
            B_numba = get_total_field_numba(
                r, 1, False, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-6, atol=1e-7,
                err_msg=f"Multiple dipoles position {r} failed"
            )
    
    def test_dipole_quadrupole(self):
        """Test total field with dipole and quadrupole components."""
        components = [
            DipoleFieldComponent(B_magnitude=1.0),
            QuadrupoleFieldComponent(
                B_magnitude=0.5,
                moment_orientation_latitude_degree=60.0,
                moment_orientation_longitude_degree=90.0,
                offset_r_ns=[1.0, 0.5, 0.0],
            ),
        ]
        field = MultipoleField(components)
        
        positions = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([2.0, 1.5, 1.0], dtype=np.float64),
        ]
        
        config = field.get_numba_config()
        n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types = self._convert_config_to_arrays(config)
        
        for r in positions:
            r_f32 = r.astype(np.float32)
            B_python = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
            
            B_numba = get_total_field_numba(
                r, 1, True, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
            )
            
            np.testing.assert_allclose(
                B_numba, B_python, rtol=1e-6, atol=1e-7,
                err_msg=f"Dipole+quadrupole position {r} failed"
            )
    
    def test_k_multiplier(self):
        """Test K multiplier (field direction)."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        
        r = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        config = field.get_numba_config()
        n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types = self._convert_config_to_arrays(config)
        
        # Test K=1
        r_f32 = r.astype(np.float32)
        B_python_1 = field.get_total_field(r_f32, K=1, normalize=False).astype(np.float64)
        B_numba_1 = get_total_field_numba(
            r, 1, False, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
        )
        np.testing.assert_allclose(B_numba_1, B_python_1, rtol=1e-6, atol=1e-7)
        
        # Test K=-1
        B_python_neg1 = field.get_total_field(r_f32, K=-1, normalize=False).astype(np.float64)
        B_numba_neg1 = get_total_field_numba(
            r, -1, False, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
        )
        np.testing.assert_allclose(B_numba_neg1, B_python_neg1, rtol=1e-6, atol=1e-7)
        
        # Verify K=-1 is negative of K=1
        np.testing.assert_allclose(B_numba_neg1, -B_numba_1, rtol=1e-10, atol=1e-12)
    
    def test_normalize_on_off(self):
        """Test normalization flag."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        field = MultipoleField([dipole])
        
        r = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        config = field.get_numba_config()
        n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types = self._convert_config_to_arrays(config)
        
        # Test normalize=False
        r_f32 = r.astype(np.float32)
        B_python_no_norm = field.get_total_field(r_f32, K=1, normalize=False).astype(np.float64)
        B_numba_no_norm = get_total_field_numba(
            r, 1, False, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
        )
        np.testing.assert_allclose(B_numba_no_norm, B_python_no_norm, rtol=1e-6, atol=1e-7)
        
        # Test normalize=True
        B_python_norm = field.get_total_field(r_f32, K=1, normalize=True).astype(np.float64)
        B_numba_norm = get_total_field_numba(
            r, 1, True, n_comp, B_mags, rot_mats, rot_mats_inv, offsets, moment_types
        )
        np.testing.assert_allclose(B_numba_norm, B_python_norm, rtol=1e-6, atol=1e-7)
        
        # Verify normalized field has unit magnitude
        norm_mag = np.linalg.norm(B_numba_norm)
        assert np.abs(norm_mag - 1.0) < 1e-10, f"Normalized field magnitude should be 1.0, got {norm_mag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

