#!/usr/bin/env python3
"""
Tests for MultipoleField module.

Tests the refactored MultipoleField class with Pydantic validation.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from psr_magpy.field.multipole import MultipoleField, MultipoleFieldParams
from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent


class TestMultipoleFieldParams:
    """Test MultipoleFieldParams validation."""
    
    def test_valid_components(self):
        """Test validation with valid field components."""
        dipole = DipoleFieldComponent()
        quadrupole = QuadrupoleFieldComponent()
        
        params = MultipoleFieldParams(field_components=[dipole, quadrupole])
        assert len(params.field_components) == 2
        assert params.field_components[0] == dipole
        assert params.field_components[1] == quadrupole
    
    def test_empty_components_list(self):
        """Test validation error for empty components list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultipoleFieldParams(field_components=[])
    
    def test_invalid_component_type(self):
        """Test validation error for invalid component type."""
        invalid_component = "not a field component"
        
        with pytest.raises(ValueError, match="Input should be an instance of FieldComponent"):
            MultipoleFieldParams(field_components=[invalid_component])
    
    def test_mixed_valid_invalid_components(self):
        """Test validation error when mixing valid and invalid components."""
        dipole = DipoleFieldComponent()
        invalid_component = 42
        
        with pytest.raises(ValueError, match="Input should be an instance of FieldComponent"):
            MultipoleFieldParams(field_components=[dipole, invalid_component])


class TestMultipoleField:
    """Test MultipoleField class. Assumes get_total_field does not validate input for speed."""
    # NOTE: get_total_field does not perform input validation. Only valid float32 input is tested.
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dipole = DipoleFieldComponent(B_magnitude=1.0)
        self.quadrupole = QuadrupoleFieldComponent(B_magnitude=0.5)
        self.multipole = MultipoleField([self.dipole, self.quadrupole])
    
    def test_initialization_valid_components(self):
        """Test initialization with valid components."""
        assert self.multipole.n_components == 2
        assert self.multipole.field_components[0] == self.dipole
        assert self.multipole.field_components[1] == self.quadrupole
    
    def test_initialization_empty_list(self):
        """Test initialization error with empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MultipoleField([])
    
    def test_initialization_invalid_component(self):
        """Test initialization error with invalid component."""
        with pytest.raises(ValueError, match="Input should be an instance of FieldComponent"):
            MultipoleField([self.dipole, "invalid"])
    
    def test_n_components_property(self):
        """Test n_components property."""
        assert self.multipole.n_components == 2
        
        single_component = MultipoleField([self.dipole])
        assert single_component.n_components == 1
    
    def test_get_total_field_basic(self):
        """Test basic field calculation with valid float32 input."""
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field = self.multipole.get_total_field(pos)
        
        assert isinstance(field, np.ndarray)
        assert field.shape == (3,)
        assert field.dtype == np.float32
        assert np.isfinite(field).all()
    
    def test_get_total_field_with_direction(self):
        """Test field calculation with different K values."""
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Forward direction (K=1)
        field_forward = self.multipole.get_total_field(pos, K=1)
        
        # Reverse direction (K=-1)
        field_reverse = self.multipole.get_total_field(pos, K=-1)
        
        # Fields should be opposite
        assert np.allclose(field_forward, -field_reverse)
    
    def test_get_total_field_normalization(self):
        """Test field normalization."""
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # With normalization
        field_normalized = self.multipole.get_total_field(pos, normalize=True)
        magnitude_normalized = np.linalg.norm(field_normalized)
        assert np.abs(magnitude_normalized - 1.0) < 1e-6
        
        # Without normalization
        field_not_normalized = self.multipole.get_total_field(pos, normalize=False)
        magnitude_not_normalized = np.linalg.norm(field_not_normalized)
        assert magnitude_not_normalized > 0
        assert magnitude_not_normalized != 1.0
    
    def test_field_superposition(self):
        """Test that total field is sum of component fields."""
        pos = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        
        # Individual component fields
        field_dipole = self.dipole.get_field(pos)
        field_quadrupole = self.quadrupole.get_field(pos)
        
        # Total field without normalization
        field_total = self.multipole.get_total_field(pos, normalize=False)
        
        # Should be sum of components
        expected_total = field_dipole + field_quadrupole
        assert np.allclose(field_total, expected_total)
    
    def test_repr_method(self):
        """Test string representation."""
        repr_str = repr(self.multipole)
        assert "MultipoleField" in repr_str
        assert "DipoleFieldComponent" in repr_str
        assert "QuadrupoleFieldComponent" in repr_str
        assert "n_components=2" in repr_str


class TestIntegration:
    """Integration tests for MultipoleField."""
    
    def test_single_component_field(self):
        """Test field calculation with single component."""
        dipole = DipoleFieldComponent(B_magnitude=2.0)
        multipole = MultipoleField([dipole])
        
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field_multipole = multipole.get_total_field(pos, normalize=False)
        field_direct = dipole.get_field(pos)
        
        # Should be identical
        assert np.allclose(field_multipole, field_direct)
    
    def test_multiple_components_scaling(self):
        """Test that field scales with component strength."""
        # Weak dipole
        dipole_weak = DipoleFieldComponent(B_magnitude=0.5)
        multipole_weak = MultipoleField([dipole_weak])
        
        # Strong dipole
        dipole_strong = DipoleFieldComponent(B_magnitude=2.0)
        multipole_strong = MultipoleField([dipole_strong])
        
        pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        field_weak = multipole_weak.get_total_field(pos, normalize=False)
        field_strong = multipole_strong.get_total_field(pos, normalize=False)
        
        # Strong field should be 4x stronger
        ratio = np.linalg.norm(field_strong) / np.linalg.norm(field_weak)
        assert np.abs(ratio - 4.0) < 0.1
    
    def test_component_parameter_access(self):
        """Test access to component parameters through multipole."""
        dipole = DipoleFieldComponent(B_magnitude=1.5)
        quadrupole = QuadrupoleFieldComponent(B_magnitude=0.3)
        multipole = MultipoleField([dipole, quadrupole])
        
        # Access component parameters
        assert multipole.field_components[0].params.B_magnitude == 1.5
        assert multipole.field_components[1].params.B_magnitude == 0.3
        assert multipole.field_components[0].params.moment_type.value == "dipole"
        assert multipole.field_components[1].params.moment_type.value == "quadrupole"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 