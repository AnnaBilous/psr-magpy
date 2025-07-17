#!/usr/bin/env python3
"""
Comprehensive tests for FieldLineSet module.

Tests FieldLineSet initialization, append operations, property management,
memory estimation, and edge cases.
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

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine, PropertyName
from psr_magpy.line.property_system import PropertyValidationError
from tests.utils import setup_test_logger, rng


class TestFieldLineSetInitialization:
    """Test FieldLineSet initialization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set")
        
        # Create sample field lines for testing
        self.template_basic = FieldLine(save_r_line=True, save_r_max=True)
        self.template_all = FieldLine.template_all()
        self.template_coords = FieldLine.template_coordinates_only()
        
        # Create sample field line data
        self.sample_field_line = self._create_sample_field_line(self.template_basic)
        self.sample_field_lines = [
            self._create_sample_field_line(self.template_basic, i)
            for i in range(3)
        ]
    
    def _create_sample_field_line(self, template: FieldLine, footpoint_id: int = 1) -> FieldLine:
        """Create a sample field line with test data."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            field_line.set_property(r_line=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        
        if template.property_system.is_property_enabled('pa_line'):
            field_line.set_property(pa_line=np.array([0.0, 0.1]))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=2.0)
        
        if template.property_system.is_property_enabled('r_cyl_max'):
            field_line.set_property(r_cyl_max=2.0)
        
        if template.property_system.is_property_enabled('z_abs_max'):
            field_line.set_property(z_abs_max=0.5)
        
        if template.property_system.is_property_enabled('b_ini'):
            field_line.set_property(b_ini=1.0)
        
        if template.property_system.is_property_enabled('b_r_ini_angle'):
            field_line.set_property(b_r_ini_angle=0.0)
        
        return field_line
    
    def test_initialization_single_field_line(self):
        """Test initialization with a single field line."""
        field_line_set = FieldLineSet(self.sample_field_line, copy_data=True)
        
        assert field_line_set.n_field_lines == 1
        assert field_line_set.property_system.get_flags_dict() == self.template_basic.property_system.get_flags_dict()
        assert field_line_set.property_system.get_property('footpoint_id') == 1
    
    def test_initialization_list_field_lines(self):
        """Test initialization with a list of field lines."""
        field_line_set = FieldLineSet(self.sample_field_lines, copy_data=True)
        
        assert field_line_set.n_field_lines == 3
        assert field_line_set.property_system.get_flags_dict() == self.template_basic.property_system.get_flags_dict()
        
        # Check that all footpoint IDs are present
        footpoint_ids = field_line_set.property_system.get_property('footpoint_id')
        assert footpoint_ids is not None
        assert np.array_equal(footpoint_ids, np.array([0, 1, 2]))
    
    def test_initialization_no_copy_data(self):
        """Test initialization without copying data."""
        field_line_set = FieldLineSet(self.sample_field_line, copy_data=False)
        
        assert field_line_set.n_field_lines == 0
        assert field_line_set.property_system.get_flags_dict() == self.template_basic.property_system.get_flags_dict()
    
    def test_initialization_empty_list(self):
        """Test initialization with empty list should raise error."""
        with pytest.raises(ValueError, match="Empty field_lines list provided"):  # type: ignore
            FieldLineSet([], copy_data=True)
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid type should raise error."""
        with pytest.raises(ValueError, match="Expected FieldLine or list of FieldLine"):  # type: ignore
            FieldLineSet("invalid", copy_data=True)  # type: ignore
    
    def test_initialization_incompatible_templates(self):
        """Test initialization with incompatible field line templates."""
        # Create field lines with different save flags
        template1 = FieldLine(save_r_line=True, save_r_max=True)
        template2 = FieldLine(save_r_line=True, save_pa_line=True)
        
        field_line1 = self._create_sample_field_line(template1, 1)
        field_line2 = self._create_sample_field_line(template2, 2)
        
        # This should fail because field lines have incompatible save flags
        with pytest.raises(ValueError, match="All field lines must have the same save flags configuration"):  # type: ignore
            FieldLineSet([field_line1, field_line2], copy_data=True)


class TestFieldLineSetAppend:
    """Test FieldLineSet append functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set_append")
        
        # Create templates and sample data
        self.template = FieldLine(save_r_line=True, save_r_max=True, save_pa_line=True)
        self.field_line_set = FieldLineSet(self._create_sample_field_line(self.template), copy_data=True)
        
        # Create additional field lines for appending
        self.additional_field_lines = [
            self._create_sample_field_line(self.template, i + 10)
            for i in range(2)
        ]
        
        # Create another field line set for appending
        self.another_field_line_set = FieldLineSet(
            [self._create_sample_field_line(self.template, i + 20) for i in range(2)],
            copy_data=True
        )
    
    def _create_sample_field_line(self, template: FieldLine, footpoint_id: int = 1) -> FieldLine:
        """Create a sample field line with test data."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            field_line.set_property(r_line=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        
        if template.property_system.is_property_enabled('pa_line'):
            field_line.set_property(pa_line=np.array([0.0, 0.1]))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=2.0)
        
        return field_line
    
    def test_append_single_field_line(self):
        """Test appending a single field line."""
        initial_count = self.field_line_set.n_field_lines
        
        new_field_line = self._create_sample_field_line(self.template, 100)
        self.field_line_set.append(new_field_line)
        
        assert self.field_line_set.n_field_lines == initial_count + 1
        footpoint_ids = self.field_line_set.property_system.get_property('footpoint_id')
        assert footpoint_ids is not None
        assert 100 in footpoint_ids
    
    def test_append_list_field_lines(self):
        """Test appending a list of field lines."""
        initial_count = self.field_line_set.n_field_lines
        
        self.field_line_set.append(self.additional_field_lines)
        
        assert self.field_line_set.n_field_lines == initial_count + 2
        footpoint_ids = self.field_line_set.property_system.get_property('footpoint_id')
        assert footpoint_ids is not None
        assert all(fid in footpoint_ids for fid in [10, 11])
    
    def test_append_field_line_set(self):
        """Test appending another FieldLineSet."""
        initial_count = self.field_line_set.n_field_lines
        
        self.field_line_set.append(self.another_field_line_set)
        
        assert self.field_line_set.n_field_lines == initial_count + 2
        footpoint_ids = self.field_line_set.property_system.get_property('footpoint_id')
        assert footpoint_ids is not None
        assert all(fid in footpoint_ids for fid in [20, 21])
    
    def test_append_incompatible_field_line_set(self):
        """Test appending FieldLineSet with incompatible save flags."""
        # Create field line set with different template
        different_template = FieldLine(save_r_line=True, save_b_ini=True)
        incompatible_set = FieldLineSet(
            self._create_sample_field_line(different_template),
            copy_data=True
        )
        
        with pytest.raises(ValueError, match="incompatible save flags"):  # type: ignore
            self.field_line_set.append(incompatible_set)
    
    def test_append_incompatible_field_lines(self):
        """Test appending field lines with incompatible save flags."""
        # Create field line with different template
        different_template = FieldLine(save_r_line=True, save_b_ini=True)
        incompatible_field_line = self._create_sample_field_line(different_template)
        
        with pytest.raises(ValueError, match="All field lines must have the same save flags configuration"):  # type: ignore
            self.field_line_set.append([incompatible_field_line])
    
    def test_append_array_list_properties(self):
        """Test appending with array list properties (r_line, pa_line)."""
        # Create field line set with array list properties
        template = FieldLine(save_r_line=True, save_pa_line=True)
        field_line_set = FieldLineSet(self._create_sample_field_line(template), copy_data=True)
        
        # Append another field line
        new_field_line = self._create_sample_field_line(template, 100)
        field_line_set.append(new_field_line)
        
        # Check that array list properties are properly extended
        r_line_data = field_line_set.property_system.get_property('r_line')
        pa_line_data = field_line_set.property_system.get_property('pa_line')
        
        assert r_line_data is not None
        assert pa_line_data is not None
        assert len(r_line_data) == 2  # Two field lines
        assert len(pa_line_data) == 2  # Two field lines
        assert isinstance(r_line_data, list)
        assert isinstance(pa_line_data, list)
    
    def test_append_none_values(self):
        """Test appending field lines with None values."""
        # Create field line with some None values
        template = FieldLine(save_r_line=True, save_r_max=True)
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=100)
        field_line.set_property(r_line=np.array([[1.0, 0.0, 0.0]]))
        # r_max intentionally not set (will be None)
        
        field_line_set = FieldLineSet(self._create_sample_field_line(template), copy_data=True)
        field_line_set.append(field_line)
        
        # Should handle None values gracefully - None values are filtered out
        assert field_line_set.n_field_lines == 2
        r_max_values = field_line_set.property_system.get_property('r_max')
        # Only the first field line has r_max, the second has None which is filtered out
        assert r_max_values is not None
        assert len(r_max_values) == 1


class TestFieldLineSetProperties:
    """Test FieldLineSet property management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set_properties")
        
        # Create field line set with all properties
        self.template = FieldLine.template_all()
        self.field_line_set = FieldLineSet(
            self._create_sample_field_line(self.template),
            copy_data=True
        )
    
    def _create_sample_field_line(self, template: FieldLine, footpoint_id: int = 1) -> FieldLine:
        """Create a sample field line with test data."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            field_line.set_property(r_line=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        
        if template.property_system.is_property_enabled('pa_line'):
            field_line.set_property(pa_line=np.array([0.0, 0.1]))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=2.0)
        
        if template.property_system.is_property_enabled('r_cyl_max'):
            field_line.set_property(r_cyl_max=2.0)
        
        if template.property_system.is_property_enabled('z_abs_max'):
            field_line.set_property(z_abs_max=0.5)
        
        if template.property_system.is_property_enabled('b_ini'):
            field_line.set_property(b_ini=1.0)
        
        if template.property_system.is_property_enabled('b_r_ini_angle'):
            field_line.set_property(b_r_ini_angle=0.0)
        
        return field_line
    
    def test_property_access(self):
        """Test accessing properties through PropertySystem."""
        # Test that all enabled properties are accessible
        for prop in self.field_line_set.property_system.get_enabled_properties():
            value = self.field_line_set.property_system.get_property(prop)
            assert value is not None, f"Property {prop.key_name} should be accessible"
    
    def test_property_consistency(self):
        """Test that properties are consistent across field lines."""
        # Add another field line
        new_field_line = self._create_sample_field_line(self.template, 2)
        self.field_line_set.append(new_field_line)
        
        # Check that all properties have the expected number of values
        for prop in self.field_line_set.property_system.get_enabled_properties():
            value = self.field_line_set.property_system.get_property(prop)
            
            if prop in FieldLineSet.ARRAY_LIST_PROPERTIES:
                assert value is not None
                assert len(value) == 2, f"Array list property {prop.key_name} should have 2 items"
            else:
                assert value is not None
                assert len(value) == 2, f"Property {prop.key_name} should have 2 values"
    
    def test_array_list_properties(self):
        """Test array list properties (r_line, pa_line)."""
        r_line_data = self.field_line_set.property_system.get_property('r_line')
        pa_line_data = self.field_line_set.property_system.get_property('pa_line')
        
        assert isinstance(r_line_data, list)
        assert isinstance(pa_line_data, list)
        assert len(r_line_data) == 1
        assert len(pa_line_data) == 1
        assert isinstance(r_line_data[0], np.ndarray)
        assert isinstance(pa_line_data[0], np.ndarray)
    
    def test_numeric_properties(self):
        """Test numeric properties."""
        r_max = self.field_line_set.property_system.get_property('r_max')
        r_cyl_max = self.field_line_set.property_system.get_property('r_cyl_max')
        z_abs_max = self.field_line_set.property_system.get_property('z_abs_max')
        b_ini = self.field_line_set.property_system.get_property('b_ini')
        b_r_ini_angle = self.field_line_set.property_system.get_property('b_r_ini_angle')
        
        assert isinstance(r_max, np.ndarray)
        assert isinstance(r_cyl_max, np.ndarray)
        assert isinstance(z_abs_max, np.ndarray)
        assert isinstance(b_ini, np.ndarray)
        assert isinstance(b_r_ini_angle, np.ndarray)
        
        assert len(r_max) == 1
        assert len(r_cyl_max) == 1
        assert len(z_abs_max) == 1
        assert len(b_ini) == 1
        assert len(b_r_ini_angle) == 1

    def test_get_property_method_with_string(self):
        """Test get_property method with string parameter."""
        # Test with enabled property using key_name format
        r_max = self.field_line_set.get_property(PropertyName.R_MAX.key_name)
        assert r_max is not None
        assert isinstance(r_max, np.ndarray)
        
        # Test with enabled array list property
        r_line = self.field_line_set.get_property(PropertyName.R_LINE.key_name) 
        assert r_line is not None
        assert isinstance(r_line, list)
        
        # Test case insensitive with enum name
        r_max_upper = self.field_line_set.get_property('R_MAX')
        assert r_max_upper is not None
        assert np.array_equal(r_max, r_max_upper)
        
        # Test with disabled property
        template_limited = FieldLine(save_r_max=True)  # Only r_max enabled
        field_line_limited = self._create_sample_field_line(template_limited, 1)
        field_line_set_limited = FieldLineSet(field_line_limited, copy_data=True)
        
        # r_line should be None since it's not enabled
        r_line_disabled = field_line_set_limited.get_property(PropertyName.R_LINE.key_name)
        assert r_line_disabled is None
    
    def test_get_property_method_with_enum(self):
        """Test get_property method with PropertyName enum parameter."""
        # Test with enabled property
        r_max = self.field_line_set.get_property(PropertyName.R_MAX)
        assert r_max is not None
        assert isinstance(r_max, np.ndarray)
        
        # Test with enabled array list property
        r_line = self.field_line_set.get_property(PropertyName.R_LINE)
        assert r_line is not None
        assert isinstance(r_line, list)
        
        # Test with disabled property
        template_limited = FieldLine(save_r_max=True)  # Only r_max enabled
        field_line_limited = self._create_sample_field_line(template_limited, 1)
        field_line_set_limited = FieldLineSet(field_line_limited, copy_data=True)
        
        # r_line should be None since it's not enabled
        r_line_disabled = field_line_set_limited.get_property(PropertyName.R_LINE)
        assert r_line_disabled is None
    
    def test_get_property_method_invalid_name(self):
        """Test get_property method with invalid property name."""
        # Test with invalid string
        result = self.field_line_set.get_property('invalid_property')
        assert result is None
        
        # Test with empty string
        result = self.field_line_set.get_property('')
        assert result is None
    
    def test_dynamic_property_access(self):
        """Test dynamic property access through __getattr__."""
        # Test with enabled properties using key_name format
        r_max = self.field_line_set.r_max
        assert r_max is not None
        assert isinstance(r_max, np.ndarray)
        
        # Test with enabled array list property
        r_line = self.field_line_set.r_line
        assert r_line is not None
        assert isinstance(r_line, list)
        
        # Test case sensitivity (should be case sensitive)
        pa_line = self.field_line_set.pa_line
        assert pa_line is not None
        assert isinstance(pa_line, list)
        
        # Test that results match get_property method
        r_max_explicit = self.field_line_set.get_property(PropertyName.R_MAX.key_name)
        assert r_max_explicit is not None
        assert np.array_equal(r_max, r_max_explicit)
    
    def test_dynamic_property_access_disabled_property(self):
        """Test dynamic property access with disabled property."""
        # Create field line set with limited properties
        template_limited = FieldLine(save_r_max=True)  # Only r_max enabled
        field_line_limited = self._create_sample_field_line(template_limited, 1)
        field_line_set_limited = FieldLineSet(field_line_limited, copy_data=True)
        
        # Accessing enabled property should work
        r_max = field_line_set_limited.r_max
        assert r_max is not None
        
        # Accessing disabled property should raise AttributeError
        with pytest.raises(AttributeError, match="'FieldLineSet' object has no attribute 'r_line'"):
            _ = field_line_set_limited.r_line
    
    def test_dynamic_property_access_invalid_property(self):
        """Test dynamic property access with invalid property name."""
        # Test with invalid property name
        with pytest.raises(AttributeError, match="'FieldLineSet' object has no attribute 'invalid_property'"):
            _ = self.field_line_set.invalid_property
        
        # Test with existing method/attribute that's not a property
        # This should work normally since n_field_lines is a real attribute
        n_lines = self.field_line_set.n_field_lines
        assert n_lines == 1


class TestFieldLineSetMemoryEstimation:
    """Test FieldLineSet memory estimation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set_memory")
        
        # Create field line set with various data sizes
        self.template = FieldLine.template_all()
        self.field_line_set = FieldLineSet(
            self._create_large_sample_field_line(self.template),
            copy_data=True
        )
    
    def _create_large_sample_field_line(self, template: FieldLine, footpoint_id: int = 1) -> FieldLine:
        """Create a sample field line with large test data."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            # Create larger coordinate array
            field_line.set_property(r_line=np.random.rand(100, 3))
        
        if template.property_system.is_property_enabled('pa_line'):
            field_line.set_property(pa_line=np.random.rand(100))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=2.0)
        
        if template.property_system.is_property_enabled('r_cyl_max'):
            field_line.set_property(r_cyl_max=2.0)
        
        if template.property_system.is_property_enabled('z_abs_max'):
            field_line.set_property(z_abs_max=0.5)
        
        if template.property_system.is_property_enabled('b_ini'):
            field_line.set_property(b_ini=1.0)
        
        if template.property_system.is_property_enabled('b_r_ini_angle'):
            field_line.set_property(b_r_ini_angle=0.0)
        
        return field_line
    
    def test_estimate_size_basic(self):
        """Test basic memory size estimation."""
        size = self.field_line_set.estimate_size()
        
        assert isinstance(size, float)
        assert size > 0
        assert size < 1e9  # Should be less than 1GB for test data
    
    def test_estimate_size_with_multiple_field_lines(self):
        """Test memory size estimation with multiple field lines."""
        # Add more field lines
        for i in range(5):
            new_field_line = self._create_large_sample_field_line(self.template, i + 2)
            self.field_line_set.append(new_field_line)
        
        size = self.field_line_set.estimate_size()
        
        assert isinstance(size, float)
        assert size > 0
        # Size should be larger with more field lines
        assert size > 1000  # At least 1KB
    
    def test_estimate_size_array_list_properties(self):
        """Test memory size estimation with array list properties."""
        # Create field line set with array list properties
        template = FieldLine(save_r_line=True, save_pa_line=True)
        field_line_set = FieldLineSet(
            self._create_large_sample_field_line(template),
            copy_data=True
        )
        
        size = field_line_set.estimate_size()
        
        assert isinstance(size, float)
        assert size > 0
    
    def test_estimate_size_none_properties(self):
        """Test memory size estimation with None properties."""
        # Create field line set with minimal properties
        template = FieldLine(save_r_max=True)  # Only r_max enabled
        field_line_set = FieldLineSet(
            self._create_large_sample_field_line(template),
            copy_data=True
        )
        
        size = field_line_set.estimate_size()
        
        assert isinstance(size, float)
        assert size > 0


class TestFieldLineSetEdgeCases:
    """Test FieldLineSet edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set_edge_cases")
        
        self.template = FieldLine(save_r_line=True, save_r_max=True)
    
    def _create_sample_field_line(self, template: FieldLine, footpoint_id: int = 1) -> FieldLine:
        """Create a sample field line with test data."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            field_line.set_property(r_line=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=2.0)
        
        return field_line
    
    def test_repr_string(self):
        """Test string representation."""
        field_line_set = FieldLineSet(self._create_sample_field_line(self.template), copy_data=True)
        repr_str = repr(field_line_set)
        
        assert "FieldLineSet" in repr_str
        assert "n_lines=1" in repr_str
        assert "enabled_properties" in repr_str
    
    def test_large_number_of_field_lines(self):
        """Test handling large number of field lines."""
        # Create many field lines
        field_lines = [self._create_sample_field_line(self.template, i) for i in range(100)]
        field_line_set = FieldLineSet(field_lines[0], copy_data=True)
        
        # Append the rest
        field_line_set.append(field_lines[1:])
        
        assert field_line_set.n_field_lines == 100
        footpoint_ids = field_line_set.property_system.get_property('footpoint_id')
        assert footpoint_ids is not None
        assert len(footpoint_ids) == 100
        assert footpoint_ids[0] == 0
        assert footpoint_ids[-1] == 99
    
    def test_empty_properties(self):
        """Test handling of empty property values."""
        # Create field line with minimal data
        template = FieldLine(save_r_line=True)
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=1)
        # Don't set r_line (will be None)
        
        field_line_set = FieldLineSet(field_line, copy_data=True)
        
        # Should handle None values gracefully
        assert field_line_set.n_field_lines == 1
        r_line_data = field_line_set.property_system.get_property('r_line')
        # For array list properties, None values become [None]
        assert r_line_data == [None]
    
    def test_mixed_none_and_valid_values(self):
        """Test handling of mixed None and valid values."""
        template = FieldLine(save_r_line=True, save_r_max=True)
        
        # Create field lines with mixed data
        field_line1 = template.copy_structure()
        field_line1.set_property(footpoint_id=1, r_line=np.array([[1.0, 0.0, 0.0]]), r_max=2.0)
        
        field_line2 = template.copy_structure()
        field_line2.set_property(footpoint_id=2, r_max=3.0)  # r_line intentionally None
        
        field_line_set = FieldLineSet(field_line1, copy_data=True)
        field_line_set.append(field_line2)
        
        assert field_line_set.n_field_lines == 2
        # Check that all properties have the expected number of values
        r_max_values = field_line_set.property_system.get_property('r_max')
        assert r_max_values is not None
        assert len(r_max_values) == 2
        assert r_max_values[0] == 2.0
        assert r_max_values[1] == 3.0
        
        # r_line should have mixed data
        r_line_data = field_line_set.property_system.get_property('r_line')
        assert r_line_data is not None
        assert len(r_line_data) == 2
        assert isinstance(r_line_data[0], np.ndarray)
        assert r_line_data[1] is None


class TestFieldLineSetIntegration:
    """Integration tests for FieldLineSet."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = setup_test_logger("test_field_line_set_integration")
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from creation to multiple appends."""
        # Create initial field line set
        template = FieldLine.template_all()
        initial_field_line = self._create_complex_field_line(template, 1)
        field_line_set = FieldLineSet(initial_field_line, copy_data=True)
        
        # Append individual field lines
        for i in range(2, 5):
            field_line = self._create_complex_field_line(template, i)
            field_line_set.append(field_line)
        
        # Append list of field lines
        additional_lines = [self._create_complex_field_line(template, i) for i in range(5, 8)]
        field_line_set.append(additional_lines)
        
        # Append another field line set
        another_set = FieldLineSet(
            [self._create_complex_field_line(template, i) for i in range(8, 11)],
            copy_data=True
        )
        field_line_set.append(another_set)
        
        # Verify final state
        assert field_line_set.n_field_lines == 10
        
        # Check all properties
        for prop in field_line_set.property_system.get_enabled_properties():
            value = field_line_set.property_system.get_property(prop)
            assert value is not None
            if prop in FieldLineSet.ARRAY_LIST_PROPERTIES:
                assert len(value) == 10
            else:
                assert len(value) == 10
        
        # Check memory estimation
        size = field_line_set.estimate_size()
        assert size > 0
    
    def _create_complex_field_line(self, template: FieldLine, footpoint_id: int) -> FieldLine:
        """Create a complex field line with all properties."""
        field_line = template.copy_structure()
        field_line.set_property(footpoint_id=footpoint_id)
        
        if template.property_system.is_property_enabled('r_line'):
            field_line.set_property(r_line=np.random.rand(10, 3))
        
        if template.property_system.is_property_enabled('pa_line'):
            field_line.set_property(pa_line=np.random.rand(10))
        
        if template.property_system.is_property_enabled('r_max'):
            field_line.set_property(r_max=float(np.random.rand()))
        
        if template.property_system.is_property_enabled('r_cyl_max'):
            field_line.set_property(r_cyl_max=float(np.random.rand()))
        
        if template.property_system.is_property_enabled('z_abs_max'):
            field_line.set_property(z_abs_max=float(np.random.rand()))
        
        if template.property_system.is_property_enabled('b_ini'):
            field_line.set_property(b_ini=float(np.random.rand()))
        
        if template.property_system.is_property_enabled('b_r_ini_angle'):
            field_line.set_property(b_r_ini_angle=float(np.random.rand()))
        
        return field_line 