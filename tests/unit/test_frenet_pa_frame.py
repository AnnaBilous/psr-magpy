"""Tests for Frenet PA frame helpers."""

import numpy as np
import pytest

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.property_calculator import (
    calculate_frenet_pa_frame,
    calculate_pa_line,
    compute_frenet_frames_field_line_set,
)


def _helix_r(n: int = 40) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return np.vstack([np.cos(t), np.sin(t), 0.1 * t])


class TestCalculateFrenetPaFrame:
    def test_shapes_and_finite_pa(self):
        r = _helix_r()
        frame = calculate_frenet_pa_frame(r)

        assert frame.pa_deg.shape == (r.shape[1],)
        assert frame.omega_p.shape == (3, r.shape[1])
        assert frame.b_norm.shape == (3, r.shape[1])
        assert frame.n_norm.shape == (3, r.shape[1])
        assert np.all(np.isfinite(frame.pa_deg))

    def test_empty_line(self):
        frame = calculate_frenet_pa_frame(np.empty((3, 0)))

        assert frame.pa_deg.size == 0
        assert frame.omega_p.shape == (3, 0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match=r"must have shape"):
            calculate_frenet_pa_frame(np.zeros((2, 5)))


class TestCalculatePaLineParity:
    def test_matches_frenet_frame(self):
        r = _helix_r()
        template = FieldLine()
        template.save_flags.save_pa_line = True
        line_r_max = float(np.max(np.linalg.norm(r, axis=0)))

        pa = calculate_pa_line(r, line_r_max, threshold_line_distance=0.0, template=template)
        frame = calculate_frenet_pa_frame(r)

        np.testing.assert_array_equal(pa, frame.pa_deg)


class TestComputeFrenetFramesFieldLineSet:
    def test_batch_on_field_line_set(self):
        r_line = _helix_r().T.astype(np.float32)
        template = FieldLine(save_r_line=True, save_pa_line=True)
        line1 = template.copy_structure()
        line1.set_property(r_line=r_line)
        line2 = template.copy_structure()
        line2.set_property(r_line=np.empty((0, 3), dtype=np.float32))
        fls = FieldLineSet([line1, line2], copy_data=True)

        out, omega_p_list, b_norm_list, n_norm_list = compute_frenet_frames_field_line_set(
            fls
        )

        pa_line = out.get_property("pa_line")
        assert len(pa_line) == 2
        assert pa_line[0].shape == (r_line.shape[0],)
        assert pa_line[1].size == 0
        assert omega_p_list[0].shape == (3, r_line.shape[0])
        assert omega_p_list[1] is None
        assert b_norm_list[1] is None
        assert n_norm_list[1] is None

        np.testing.assert_array_equal(
            pa_line[0],
            calculate_frenet_pa_frame(r_line.T.astype(np.float64)).pa_deg,
        )
