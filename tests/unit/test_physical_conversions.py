"""Tests for psr_magpy.utils.physical_conversions."""

import numpy as np
import pytest

from psr_magpy.utils.physical_conversions import tangent_phi_to_observer_phi
from psr_magpy.viz.pa import _transform_scatter_phi


class TestTangentPhiToObserverPhi:
    def test_scalar(self):
        assert tangent_phi_to_observer_phi(30.0) == pytest.approx(330.0)

    def test_negative_tangent_phi(self):
        assert tangent_phi_to_observer_phi(-30.0) == pytest.approx(30.0)

    def test_zero(self):
        assert tangent_phi_to_observer_phi(0.0) == pytest.approx(0.0)

    def test_array(self):
        phi = np.array([0.0, 30.0, -30.0, 180.0])
        expected = np.array([0.0, 330.0, 30.0, 180.0])
        np.testing.assert_allclose(tangent_phi_to_observer_phi(phi), expected)

    def test_matches_scatter_transform_without_roll(self):
        phi = np.array([0.0, 30.0, -30.0, 90.0, 180.0, 360.0])
        np.testing.assert_allclose(
            tangent_phi_to_observer_phi(phi),
            _transform_scatter_phi(phi, phi_roll_deg=0.0),
        )
