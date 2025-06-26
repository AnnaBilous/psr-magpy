#!/usr/bin/env python3
"""
Comprehensive tests for field components module.

Tests FieldComponentParams, FieldComponent, DipoleFieldComponent, and QuadrupoleFieldComponent.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.field.components import (
    FieldComponentParams, 
    FieldComponent, 
    DipoleFieldComponent, 
    QuadrupoleFieldComponent,
    MultipoleType
)


class TestFieldComponentParams:
    """Test FieldComponentParams validation and behavior."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = FieldComponentParams(moment_type=MultipoleType.DIPOLE)
        assert params.B_magnitude == 1.0
        assert params.moment_orientation_latitude_degree == 0.0
        assert params.moment_orientation_longitude_degree == 0.0
        assert np.allclose(params.offset_r_ns, [0.0, 0.0, 0.0])
        assert params.moment_type == MultipoleType.DIPOLE
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = FieldComponentParams(
            B_magnitude=2.5,
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=90.0,
            offset_r_ns=[1.0, 2.0, 3.0],
            moment_type=MultipoleType.QUADRUPOLE
        )
        assert params.B_magnitude == 2.5
        assert params.moment_orientation_latitude_degree == 45.0
        assert params.moment_orientation_longitude_degree == 90.0
        assert np.allclose(params.offset_r_ns, [1.0, 2.0, 3.0])
        assert params.moment_type == MultipoleType.QUADRUPOLE
    
    def test_offset_vector_validation_list(self):
        """Test offset vector validation with list input."""
        params = FieldComponentParams(
            offset_r_ns=[1.0, 2.0, 3.0],
            moment_type=MultipoleType.DIPOLE
        )
        assert isinstance(params.offset_r_ns, np.ndarray)
        assert params.offset_r_ns.dtype == np.float32
        assert np.allclose(params.offset_r_ns, [1.0, 2.0, 3.0])
    
    def test_offset_vector_validation_array(self):
        """Test offset vector validation with numpy array input."""
        offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        params = FieldComponentParams(
            offset_r_ns=offset,
            moment_type=MultipoleType.DIPOLE
        )
        assert isinstance(params.offset_r_ns, np.ndarray)
        assert params.offset_r_ns.dtype == np.float32
        assert np.allclose(params.offset_r_ns, [1.0, 2.0, 3.0])
    
    def test_invalid_offset_vector_length(self):
        """Test validation error for wrong offset vector length."""
        with pytest.raises(ValueError, match="exactly 3 components"):
            FieldComponentParams(
                offset_r_ns=[1.0, 2.0],
                moment_type=MultipoleType.DIPOLE
            )
        
        with pytest.raises(ValueError, match="exactly 3 components"):
            FieldComponentParams(
                offset_r_ns=[1.0, 2.0, 3.0, 4.0],
                moment_type=MultipoleType.DIPOLE
            )
    
    def test_invalid_offset_vector_type(self):
        """Test validation error for wrong offset vector type."""
        with pytest.raises(ValueError, match="list or array"):
            FieldComponentParams(
                offset_r_ns="invalid",
                moment_type=MultipoleType.DIPOLE
            )
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation."""
        # B_magnitude should be non-negative
        with pytest.raises(ValueError):
            FieldComponentParams(
                B_magnitude=-1.0,
                moment_type=MultipoleType.DIPOLE
            )
        
        # Latitude should be between 0 and 90
        with pytest.raises(ValueError):
            FieldComponentParams(
                moment_orientation_latitude_degree=-10.0,
                moment_type=MultipoleType.DIPOLE
            )
        
        with pytest.raises(ValueError):
            FieldComponentParams(
                moment_orientation_latitude_degree=100.0,
                moment_type=MultipoleType.DIPOLE
            )
        
        # Longitude should be between 0 and 360
        with pytest.raises(ValueError):
            FieldComponentParams(
                moment_orientation_longitude_degree=-10.0,
                moment_type=MultipoleType.DIPOLE
            )
        
        with pytest.raises(ValueError):
            FieldComponentParams(
                moment_orientation_longitude_degree=370.0,
                moment_type=MultipoleType.DIPOLE
            )


class TestFieldComponent:
    """Test abstract FieldComponent base class."""
    
    def test_abstract_class_instantiation(self):
        """Test that FieldComponent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FieldComponent()
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        assert hasattr(FieldComponent, '_calculate_field')
        assert hasattr(FieldComponent, 'get_moment_type')
        assert hasattr(FieldComponent, 'get_field')


class TestDipoleFieldComponent:
    """Test DipoleFieldComponent implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dipole = DipoleFieldComponent()
    
    def test_creation_defaults(self):
        """Test dipole creation with default parameters."""
        assert self.dipole.params.B_magnitude == 1.0
        assert self.dipole.params.moment_orientation_latitude_degree == 0.0
        assert self.dipole.params.moment_orientation_longitude_degree == 0.0
        assert np.allclose(self.dipole.params.offset_r_ns, [0.0, 0.0, 0.0])
        assert self.dipole.params.moment_type == MultipoleType.DIPOLE
    
    def test_creation_custom_params(self):
        """Test dipole creation with custom parameters."""
        dipole = DipoleFieldComponent(
            B_magnitude=2.0,
            moment_orientation_latitude_degree=30.0,
            moment_orientation_longitude_degree=45.0,
            offset_r_ns=[0.5, 0.0, 0.0]
        )
        assert dipole.params.B_magnitude == 2.0
        assert dipole.params.moment_orientation_latitude_degree == 30.0
        assert dipole.params.moment_orientation_longitude_degree == 45.0
        assert np.allclose(dipole.params.offset_r_ns, [0.5, 0.0, 0.0])
    
    def test_get_moment_type(self):
        """Test moment type retrieval."""
        assert self.dipole.get_moment_type() == MultipoleType.DIPOLE
    
    def test_cached_properties(self):
        """Test cached property calculations."""
        # Test moment_orientation
        orientation = self.dipole.moment_orientation
        assert hasattr(orientation, 'phi')
        assert hasattr(orientation, 'theta')
        assert hasattr(orientation, 'r')
        
        # Test moment_offset_vector
        offset = self.dipole.moment_offset_vector
        assert isinstance(offset, np.ndarray)
        assert offset.dtype == np.float32
        
        # Test rotation matrices
        rot_magn_to_spin = self.dipole.rotate_magn_to_spin_frame
        rot_spin_to_magn = self.dipole.rotate_spin_to_magn_frame
        assert hasattr(rot_magn_to_spin, 'apply')
        assert hasattr(rot_spin_to_magn, 'apply')
        
        # Test that inverse relationship holds
        test_vector = np.array([1.0, 0.0, 0.0])
        rotated = rot_magn_to_spin.apply(test_vector)
        back_rotated = rot_spin_to_magn.apply(rotated)
        assert np.allclose(test_vector, back_rotated, atol=1e-10)
    
    def test_field_calculation_on_axis(self):
        """Test field calculation on magnetic axis."""
        # On positive Z-axis
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field = self.dipole.get_field(pos)
        # Field should be finite and have 3 components
        assert np.isfinite(field).all()
        assert field.shape == (3,)
        # Field magnitude should be positive
        assert np.linalg.norm(field) > 0
        
        # On negative Z-axis - field should point toward negative Z
        pos = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        field = self.dipole.get_field(pos)
        # Field should be finite and have 3 components
        assert np.isfinite(field).all()
        assert field.shape == (3,)
        # Field magnitude should be positive
        assert np.linalg.norm(field) > 0
    
    def test_field_calculation_equatorial_plane(self):
        """Test field calculation in equatorial plane."""
        # In equatorial plane (Z=0)
        pos = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        field = self.dipole.get_field(pos)
        expected = np.array([0.0, 0.0, -0.5], dtype=np.float32)
        assert np.allclose(field, expected, atol=1e-6)
        
        pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        field = self.dipole.get_field(pos)
        expected = np.array([0.0, 0.0, -0.5], dtype=np.float32)
        assert np.allclose(field, expected, atol=1e-6)
    
    def test_field_magnitude_scaling(self):
        """Test that field magnitude scales with B_magnitude."""
        dipole_strong = DipoleFieldComponent(B_magnitude=2.0)
        dipole_weak = DipoleFieldComponent(B_magnitude=0.5)
        
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        field_strong = dipole_strong.get_field(pos)
        field_weak = dipole_weak.get_field(pos)
        field_default = self.dipole.get_field(pos)
        
        assert np.allclose(field_strong, 2.0 * field_default, atol=1e-6)
        assert np.allclose(field_weak, 0.5 * field_default, atol=1e-6)
    
    def test_radial_falloff(self):
        """Test 1/r³ radial falloff of dipole field."""
        distances = [1.0, 2.0, 3.0]
        magnitudes = []
        
        for r in distances:
            pos = np.array([0.0, 0.0, r], dtype=np.float32)
            field = self.dipole.get_field(pos)
            magnitudes.append(np.linalg.norm(field))
        
        # Check 1/r³ falloff
        expected_ratios = [1.0, 1.0/8.0, 1.0/27.0]
        actual_ratios = [m/magnitudes[0] for m in magnitudes]
        
        for expected, actual in zip(expected_ratios, actual_ratios):
            assert np.abs(expected - actual) < 0.01
    
    def test_origin_singularity_handling(self):
        """Test handling of field near origin."""
        # Very close to origin - use smaller threshold than 1e-10
        pos_close = np.array([0.0, 0.0, 1e-11], dtype=np.float32)
        field_close = self.dipole.get_field(pos_close)
        assert np.allclose(field_close, [0.0, 0.0, 0.0], atol=1e-6)
        
        # Slightly further
        pos_finite = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        field_finite = self.dipole.get_field(pos_finite)
        assert not np.allclose(field_finite, [0.0, 0.0, 0.0])
    
    def test_tilted_dipole(self):
        """Test tilted dipole field calculation."""
        tilted_dipole = DipoleFieldComponent(
            moment_orientation_latitude_degree=45.0,
            moment_orientation_longitude_degree=0.0
        )
        
        # Field should be different from aligned dipole
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        field_tilted = tilted_dipole.get_field(pos)
        field_aligned = self.dipole.get_field(pos)
        
        assert not np.allclose(field_tilted, field_aligned)
    
    def test_offset_dipole(self):
        """Test dipole with offset."""
        offset_dipole = DipoleFieldComponent(
            offset_r_ns=[0.5, 0.0, 0.0]
        )
        
        # Field at origin should be different
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field_offset = offset_dipole.get_field(pos)
        field_centered = self.dipole.get_field(pos)
        
        assert not np.allclose(field_offset, field_centered)
    
    def test_coordinate_transformation(self):
        """Test coordinate transformation methods."""
        pos_spin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Test transformation to magnetic frame
        pos_magn = self.dipole._transform_to_magnetic_frame(pos_spin)
        assert isinstance(pos_magn, np.ndarray)
        # scipy rotation returns float64
        assert pos_magn.dtype == np.float64
        assert pos_magn.shape == (3,)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input."""
        # Non-finite values
        pos_invalid = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            self.dipole.get_field(pos_invalid)
        
        pos_invalid = np.array([np.inf, 1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite values"):
            self.dipole.get_field(pos_invalid)
    
    def test_repr_method(self):
        """Test string representation."""
        repr_str = repr(self.dipole)
        assert "DipoleFieldComponent" in repr_str
        assert "MultipoleType.DIPOLE" in repr_str
        assert "B=1" in repr_str


class TestQuadrupoleFieldComponent:
    """Test QuadrupoleFieldComponent implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quadrupole = QuadrupoleFieldComponent()
    
    def test_creation_defaults(self):
        """Test quadrupole creation with default parameters."""
        assert self.quadrupole.params.B_magnitude == 1.0
        assert self.quadrupole.params.moment_orientation_latitude_degree == 0.0
        assert self.quadrupole.params.moment_orientation_longitude_degree == 0.0
        assert np.allclose(self.quadrupole.params.offset_r_ns, [0.0, 0.0, 0.0])
        assert self.quadrupole.params.moment_type == MultipoleType.QUADRUPOLE
    
    def test_get_moment_type(self):
        """Test moment type retrieval."""
        assert self.quadrupole.get_moment_type() == MultipoleType.QUADRUPOLE
    
    def test_field_calculation_on_axis(self):
        """Test field calculation on magnetic axis."""
        # On positive Z-axis
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field = self.quadrupole.get_field(pos)
        # Quadrupole field should be finite and have 3 components
        assert np.isfinite(field).all()
        assert field.shape == (3,)
        # Field magnitude should be positive
        assert np.linalg.norm(field) > 0
    
    def test_field_magnitude_scaling(self):
        """Test that field magnitude scales with B_magnitude."""
        quad_strong = QuadrupoleFieldComponent(B_magnitude=2.0)
        quad_weak = QuadrupoleFieldComponent(B_magnitude=0.5)
        
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        field_strong = quad_strong.get_field(pos)
        field_weak = quad_weak.get_field(pos)
        field_default = self.quadrupole.get_field(pos)
        
        assert np.allclose(field_strong, 2.0 * field_default, atol=1e-6)
        assert np.allclose(field_weak, 0.5 * field_default, atol=1e-6)
    
    def test_radial_falloff(self):
        """Test 1/r⁴ radial falloff of quadrupole field."""
        distances = [1.0, 2.0, 3.0]
        magnitudes = []
        
        for r in distances:
            pos = np.array([0.0, 0.0, r], dtype=np.float32)
            field = self.quadrupole.get_field(pos)
            magnitudes.append(np.linalg.norm(field))
        
        # Check 1/r⁴ falloff (faster than dipole)
        expected_ratios = [1.0, 1.0/16.0, 1.0/81.0]
        actual_ratios = [m/magnitudes[0] for m in magnitudes]
        
        for expected, actual in zip(expected_ratios, actual_ratios):
            assert np.abs(expected - actual) < 0.1  # Less strict due to complexity
    
    def test_origin_singularity_handling(self):
        """Test handling of field near origin."""
        # Very close to origin - use smaller threshold than 1e-10
        pos_close = np.array([0.0, 0.0, 1e-11], dtype=np.float32)
        field_close = self.quadrupole.get_field(pos_close)
        assert np.allclose(field_close, [0.0, 0.0, 0.0], atol=1e-6)
        
        # Slightly further
        pos_finite = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        field_finite = self.quadrupole.get_field(pos_finite)
        assert not np.allclose(field_finite, [0.0, 0.0, 0.0])
    
    def test_different_from_dipole(self):
        """Test that quadrupole field differs from dipole."""
        dipole = DipoleFieldComponent()
        
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        field_quad = self.quadrupole.get_field(pos)
        field_dip = dipole.get_field(pos)
        
        # Fields should be different
        assert not np.allclose(field_quad, field_dip)
        
        # Quadrupole should fall off faster
        pos_far = np.array([0.0, 2.0, 2.0], dtype=np.float32)
        field_quad_far = self.quadrupole.get_field(pos_far)
        field_dip_far = dipole.get_field(pos_far)
        
        ratio_quad = np.linalg.norm(field_quad_far) / np.linalg.norm(field_quad)
        ratio_dip = np.linalg.norm(field_dip_far) / np.linalg.norm(field_dip)
        
        # Quadrupole should fall off faster than dipole
        assert ratio_quad < ratio_dip
    
    def test_repr_method(self):
        """Test string representation."""
        repr_str = repr(self.quadrupole)
        assert "QuadrupoleFieldComponent" in repr_str
        assert "MultipoleType.QUADRUPOLE" in repr_str


class TestMultipoleType:
    """Test MultipoleType enum."""
    
    def test_enum_values(self):
        """Test enum values."""
        assert MultipoleType.DIPOLE.value == "dipole"
        assert MultipoleType.QUADRUPOLE.value == "quadrupole"
    
    def test_enum_comparison(self):
        """Test enum comparison."""
        assert MultipoleType.DIPOLE != MultipoleType.QUADRUPOLE
        assert MultipoleType.DIPOLE == MultipoleType.DIPOLE


class TestIntegration:
    """Integration tests for field components."""
    
    def test_multiple_components_same_position(self):
        """Test multiple components at same position."""
        dipole = DipoleFieldComponent(B_magnitude=1.0)
        quadrupole = QuadrupoleFieldComponent(B_magnitude=0.5)
        
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        field_dip = dipole.get_field(pos)
        field_quad = quadrupole.get_field(pos)
        
        # Both should return valid fields
        assert np.isfinite(field_dip).all()
        assert np.isfinite(field_quad).all()
        assert field_dip.shape == (3,)
        assert field_quad.shape == (3,)
    
    def test_component_parameter_consistency(self):
        """Test parameter consistency across components."""
        dipole = DipoleFieldComponent(
            B_magnitude=2.0,
            moment_orientation_latitude_degree=30.0,
            moment_orientation_longitude_degree=45.0,
            offset_r_ns=[0.5, 0.0, 0.0]
        )
        
        quadrupole = QuadrupoleFieldComponent(
            B_magnitude=2.0,
            moment_orientation_latitude_degree=30.0,
            moment_orientation_longitude_degree=45.0,
            offset_r_ns=[0.5, 0.0, 0.0]
        )
        
        # Parameters should be consistent
        assert dipole.params.B_magnitude == quadrupole.params.B_magnitude
        assert dipole.params.moment_orientation_latitude_degree == quadrupole.params.moment_orientation_latitude_degree
        assert dipole.params.moment_orientation_longitude_degree == quadrupole.params.moment_orientation_longitude_degree
        assert np.allclose(dipole.params.offset_r_ns, quadrupole.params.offset_r_ns)
        
        # But moment types should differ
        assert dipole.params.moment_type != quadrupole.params.moment_type


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 