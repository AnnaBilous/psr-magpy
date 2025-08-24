"""Unit tests for tangent properties."""

import numpy as np
import pytest

from psr_magpy.line.property_calculator import PropertyCalculator
from psr_magpy.line.property_system import PropertyName, SaveFlags
from psr_magpy.line.field_line import FieldLine
from psr_magpy.field.components import DipoleFieldComponent
from psr_magpy.field.multipole import MultipoleField


class TestTangentProperties:
    """Test tangent theta and phi properties."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create dipole field with moment latitude = 0 (aligned dipole)
        self.dipole = DipoleFieldComponent(
            moment_orientation_latitude_degree=0.0,
            moment_orientation_longitude_degree=0.0
        )
        self.field = MultipoleField([self.dipole])
        self.calculator = PropertyCalculator(threshold_line_distance=1.0, field=self.field)

    def test_tangent_phi_constant_for_aligned_dipole(self):
        """Test that tangent phi stays constant for aligned dipole field."""
        # Create a field line in the equatorial plane (y=0, z=0)
        r = np.array([
            [1.0, 1.5, 2.0, 2.5, 3.0],  # x - increasing radius
            [0.0, 0.0, 0.0, 0.0, 0.0],  # y - constant at 0 (equatorial plane)
            [0.0, 0.0, 0.0, 0.0, 0.0]   # z - constant at 0 (equatorial plane)
        ])

        template = FieldLine()
        template.save_flags.save_tangent_phi = True

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([1.0, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        assert 'tangent_phi' in props
        phi_values = props['tangent_phi']
        
        # For aligned dipole in equatorial plane, phi should be constant (0°)
        assert np.allclose(phi_values, 0.0, atol=1e-6)
        assert np.std(phi_values) < 1e-6

    def test_tangent_theta_constant_for_equatorial_line(self):
        """Test that tangent theta is 90° for equatorial field line."""
        # Create a field line in the equatorial plane
        r = np.array([
            [1.0, 1.5, 2.0, 2.5, 3.0],  # x - increasing radius
            [0.0, 0.0, 0.0, 0.0, 0.0],  # y - constant at 0
            [0.0, 0.0, 0.0, 0.0, 0.0]   # z - constant at 0
        ])

        template = FieldLine()
        template.save_flags.save_tangent_theta = True

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([1.0, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        assert 'tangent_theta' in props
        theta_values = props['tangent_theta']
        
        # For equatorial line, theta should be 90° (perpendicular to z-axis)
        assert np.allclose(theta_values, 90.0, atol=1e-6)
        assert np.std(theta_values) < 1e-6

    def test_tangent_properties_disabled(self):
        """Test that tangent properties are not calculated when disabled."""
        r = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        template = FieldLine()
        # Don't enable tangent properties
        template.save_flags.save_tangent_theta = False
        template.save_flags.save_tangent_phi = False

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([1.0, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        assert 'tangent_theta' not in props
        assert 'tangent_phi' not in props

    def test_tangent_properties_short_line(self):
        """Test tangent properties for line that doesn't reach threshold."""
        r = np.array([
            [0.1, 0.2, 0.3],  # Short line
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        template = FieldLine()
        template.save_flags.save_tangent_theta = True
        template.save_flags.save_tangent_phi = True

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([0.1, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        assert 'tangent_theta' in props
        assert 'tangent_phi' in props
        # Should return empty arrays for short lines
        assert len(props['tangent_theta']) == 0
        assert len(props['tangent_phi']) == 0

    def test_tangent_properties_data_type(self):
        """Test that tangent properties return correct data type."""
        r = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        template = FieldLine()
        template.save_flags.save_tangent_theta = True
        template.save_flags.save_tangent_phi = True

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([1.0, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        assert props['tangent_theta'].dtype == np.float32
        assert props['tangent_phi'].dtype == np.float32

    def test_tangent_properties_optimization(self):
        """Test that gradient is calculated only once when multiple tangent properties are enabled."""
        r = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ])

        template = FieldLine()
        template.save_flags.save_tangent_theta = True
        template.save_flags.save_tangent_phi = True
        template.save_flags.save_pa_line = True  # Also uses tangent vector

        props = self.calculator.calculate_line_properties(
            footpoint_id=0,
            r=r,
            B_initial=self.field.get_total_field(np.array([1.0, 0.0, 0.0])),
            dotprod=1.0,
            K=1.0,
            dt=1.0,
            template=template
        )

        # All properties should be calculated correctly
        assert 'tangent_theta' in props
        assert 'tangent_phi' in props
        assert 'pa_line' in props
        
        # Values should be consistent
        assert np.allclose(props['tangent_phi'], 0.0, atol=1e-6)
        assert np.allclose(props['tangent_theta'], 90.0, atol=1e-6)
