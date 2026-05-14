"""Tests for new properties: B_line and curvature_line."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from psr_magpy.line.property_system import PropertySystem, PropertyName, SaveFlags
from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.line.field_line import FieldLine


class TestNewProperties:
    """Test the new B_line and curvature_line properties."""

    def test_property_system_new_properties(self):
        """Test that new properties are properly registered in PropertySystem."""
        # Test with default flags (both should be False)
        ps = PropertySystem()
        assert not ps.is_property_enabled("b_line")
        assert not ps.is_property_enabled("curvature_line")
        
        # Test with enabled flags
        ps = PropertySystem(save_b_line=True, save_curvature_line=True)
        assert ps.is_property_enabled("b_line")
        assert ps.is_property_enabled("curvature_line")
        
        # Test property names
        assert "b_line" in ps.get_enabled_property_names()
        assert "curvature_line" in ps.get_enabled_property_names()

    def test_property_validation_b_line(self):
        """Test validation of B_line property values."""
        ps = PropertySystem(save_b_line=True)
        
        # Valid: array-like object
        valid_b_line = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ps.set_property(b_line=valid_b_line)
        
        # Invalid: not array-like
        with pytest.raises(ValueError):
            ps.set_property(b_line="invalid")

    def test_property_validation_curvature_line(self):
        """Test validation of curvature_line property values."""
        ps = PropertySystem(save_curvature_line=True)
        
        # Valid: array-like object
        valid_curvature = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        ps.set_property(curvature_line=valid_curvature)
        
        # Invalid: not array-like
        with pytest.raises(ValueError):
            ps.set_property(curvature_line="invalid")

    def test_property_calculator_b_line(self):
        """Test B_line calculation in PropertyCalculator."""
        # Mock field object
        mock_field = Mock()
        mock_field.get_total_field.return_value = np.array([1.0, 2.0, 3.0])
        
        # Create calculator with field
        calculator = PropertyCalculator(threshold_line_distance=1.0, field=mock_field)
        
        # Create template with B_line enabled
        template = FieldLine()
        template.save_flags.save_b_line = True
        
        # Test data
        r = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3 points
        footpoint_id = 1
        B_initial = np.array([1.0, 0.0, 0.0])
        dotprod = 0.5
        dt = 0.1
        
        # Calculate properties
        props = calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, dt, template
        )
        
        # Check that B_line was calculated
        assert "b_line" in props
        assert isinstance(props["b_line"], np.ndarray)
        assert props["b_line"].dtype == np.float32
        assert len(props["b_line"]) == 3  # 3 points
        
        # Check that field.get_total_field was called for each point
        assert mock_field.get_total_field.call_count == 3

    def test_property_calculator_curvature_line(self):
        """Test curvature_line calculation in PropertyCalculator."""
        # Create calculator without field (curvature doesn't need field)
        calculator = PropertyCalculator(threshold_line_distance=1.0)
        
        # Create template with curvature_line enabled
        template = FieldLine()
        template.save_flags.save_curvature_line = True
        
        # Test data - create a simple curve
        t = np.linspace(0, 2*np.pi, 10)
        r = np.array([
            np.cos(t),  # x = cos(t)
            np.sin(t),  # y = sin(t)
            np.zeros_like(t)  # z = 0
        ])
        footpoint_id = 1
        B_initial = np.array([1.0, 0.0, 0.0])
        dotprod = 0.5
        dt = 0.1
        
        # Calculate properties
        props = calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, dt, template
        )
        
        # Check that curvature_line was calculated
        assert "curvature_line" in props
        assert isinstance(props["curvature_line"], np.ndarray)
        assert props["curvature_line"].dtype == np.float32
        assert len(props["curvature_line"]) == 10  # 10 points

    def test_property_calculator_no_field(self):
        """Test that properties are None when field is not provided."""
        # Create calculator without field
        calculator = PropertyCalculator(threshold_line_distance=1.0)
        
        # Create template with both properties enabled
        template = FieldLine()
        template.save_flags.save_b_line = True
        template.save_flags.save_curvature_line = True
        
        # Test data - need at least 3 points for curvature calculation with edge_order=2
        r = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
        footpoint_id = 1
        B_initial = np.array([1.0, 0.0, 0.0])
        dotprod = 0.5
        dt = 0.1
        
        # Calculate properties
        props = calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, dt, template
        )
        
        # B_line should be None (no field provided)
        assert "b_line" not in props or props["b_line"] is None
        
        # Curvature_line should still be calculated (doesn't need field)
        assert "curvature_line" in props
        assert isinstance(props["curvature_line"], np.ndarray)

    def test_property_calculator_disabled_properties(self):
        """Test that properties are not calculated when disabled."""
        # Mock field object
        mock_field = Mock()
        mock_field.get_total_field.return_value = np.array([1.0, 2.0, 3.0])
        
        # Create calculator with field
        calculator = PropertyCalculator(threshold_line_distance=1.0, field=mock_field)
        
        # Create template with properties disabled
        template = FieldLine()
        template.save_flags.save_b_line = False
        template.save_flags.save_curvature_line = False
        
        # Test data - need at least 3 points for curvature calculation with edge_order=2
        r = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
        footpoint_id = 1
        B_initial = np.array([1.0, 0.0, 0.0])
        dotprod = 0.5
        dt = 0.1
        
        # Calculate properties
        props = calculator.calculate_line_properties(
            footpoint_id, r, B_initial, dotprod, dt, template
        )
        
        # Properties should not be in results
        assert "b_line" not in props
        assert "curvature_line" not in props
        
        # Field should not be called
        mock_field.get_total_field.assert_not_called()

    def test_validation_summary_includes_new_properties(self):
        """Test that validation summary includes new properties."""
        ps = PropertySystem()
        summary = ps.get_validation_summary()
        
        # Check that new properties are in validation rules
        assert "b_line" in summary["validation_rules"]
        assert "curvature_line" in summary["validation_rules"]
        assert summary["validation_rules"]["b_line"] == "array-like object"
        assert summary["validation_rules"]["curvature_line"] == "array-like object"
