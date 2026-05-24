"""Tests for :mod:`psr_magpy.line.pa_theta_filter`."""

import numpy as np
import pytest

from psr_magpy.grid.grid import HealpixGrid
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.pa_theta_filter import (
    filter_field_line_set_by_theta,
    filter_fieldline_points_by_theta,
    theta_point_mask,
)

_PA_TEMPLATE = FieldLineTemplate(
    save_footpoint_id=True,
    save_r_line=True,
    save_pa_line=True,
    save_tangent_theta=True,
    save_tangent_phi=True,
)


def _line(
    footpoint_id: int,
    tangent_theta,
    tangent_phi,
    pa_line,
    r_line=None,
) -> FieldLine:
    fl = FieldLine.from_template(_PA_TEMPLATE)
    props = dict(
        footpoint_id=footpoint_id,
        tangent_theta=np.asarray(tangent_theta, dtype=float),
        tangent_phi=np.asarray(tangent_phi, dtype=float),
        pa_line=np.asarray(pa_line, dtype=float),
    )
    if r_line is not None:
        props["r_line"] = np.asarray(r_line, dtype=float)
    fl.set_property(**props)
    return fl


def _set(*lines: FieldLine) -> FieldLineSet:
    return FieldLineSet(list(lines), copy_data=True)


class TestThetaPointMask:
    def test_scalar_target_strict_less_than_tolerance(self):
        theta = np.array([44.0, 45.0, 46.0])
        mask = theta_point_mask(theta, 45.0, 0.5)
        assert mask.tolist() == [False, True, False]

    def test_scalar_wider_tolerance(self):
        theta = np.array([44.0, 45.0, 46.0])
        mask = theta_point_mask(theta, 45.0, 1.01)
        assert mask.tolist() == [True, True, True]

    def test_scalar_excludes_outside_band(self):
        theta = np.array([40.0, 45.0, 50.0])
        mask = theta_point_mask(theta, 45.0, 0.2)
        assert mask.tolist() == [False, True, False]

    def test_grid_matches_any_reference(self):
        theta = np.array([4.9, 45.0, 100.0])
        grid = np.array([5.0, 45.0, 90.0])
        mask = theta_point_mask(theta, grid, 0.5)
        assert mask.tolist() == [True, True, False]

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            theta_point_mask([45.0], [], 0.2)


class TestFilterFieldlinePointsByTheta:
    def test_flat_output_without_grid(self):
        fls = _set(
            _line(0, [44, 45, 46], [0, 1, 2], [10, 20, 30]),
            _line(1, [10, 20], [3, 4], [1, 2]),
        )
        out = filter_fieldline_points_by_theta(fls, 45.0, 0.5)
        assert np.allclose(out.pa_values, [20])
        assert np.allclose(out.phi_values, [1])
        assert out.colatitude_values is None
        assert out.oflr_em_map is None
        assert out.r_line_stacked is None

    def test_with_grid_fills_oflr_em_map(self):
        grid = HealpixGrid(nside=1)
        fp = 3
        colat = float(grid.colatitude[fp])
        lon = float(grid.longitude[fp])
        r_line = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        fls = _set(_line(fp, [44, 45, 46], [0, 1, 2], [10, 20, 30], r_line=r_line))

        out = filter_fieldline_points_by_theta(fls, 45.0, 0.5, grid=grid)

        assert len(out.oflr_em_map) == grid.nline
        assert np.isnan(out.oflr_em_map).sum() == grid.nline - 1
        assert out.oflr_em_map[fp] == colat
        assert np.all(out.colatitude_values == colat)
        assert np.all(out.longitude_values == lon)
        assert out.r_line_stacked.shape == (1, 3)

    def test_no_matches_returns_empty(self):
        fls = _set(_line(0, [10, 20], [0, 1], [1, 2]))
        out = filter_fieldline_points_by_theta(fls, 90.0, 0.1)
        assert out.pa_values.size == 0
        assert out.phi_values.size == 0

    def test_missing_pa_raises(self):
        fl = FieldLine(save_footpoint_id=True, save_tangent_theta=True, save_tangent_phi=True)
        fl.set_property(
            footpoint_id=0,
            tangent_theta=[45.0],
            tangent_phi=[0.0],
        )
        with pytest.raises(ValueError, match="pa_line"):
            filter_fieldline_points_by_theta(_set(fl), 45.0, 0.2)

    def test_grid_without_footpoint_id_raises(self):
        fl = FieldLine(
            save_tangent_theta=True,
            save_tangent_phi=True,
            save_pa_line=True,
        )
        fl.set_property(
            tangent_theta=[45.0],
            tangent_phi=[0.0],
            pa_line=[1.0],
        )
        grid = HealpixGrid(nside=1)
        with pytest.raises(ValueError, match="footpoint_id"):
            filter_fieldline_points_by_theta(_set(fl), 45.0, 0.2, grid=grid)


class TestFilterFieldLineSetByTheta:
    def test_scalar_reference_keeps_one_line(self):
        fls = _set(
            _line(
                7,
                [44, 45, 46],
                [0, 1, 2],
                [10, 20, 30],
                r_line=[[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            ),
            _line(8, [5, 6], [3, 4], [1, 2], r_line=[[0, 1, 0], [0, 2, 0]]),
        )
        lines = filter_field_line_set_by_theta(fls, 45.0, 0.5, _PA_TEMPLATE)
        assert len(lines) == 1
        assert lines[0].get_property("footpoint_id") == 7
        assert len(lines[0].get_property("tangent_theta")) == 1
        assert np.allclose(lines[0].get_property("pa_line"), [20])

    def test_grid_reference_union_of_bands(self):
        fls = _set(
            _line(
                0,
                [5.0, 45.0, 90.0],
                [0, 1, 2],
                [1, 2, 3],
                r_line=[[1, 0, 0], [2, 0, 0], [3, 0, 0]],
            ),
        )
        grid_theta = np.array([5.0, 45.0, 90.0])
        lines = filter_field_line_set_by_theta(fls, grid_theta, 0.5, _PA_TEMPLATE)
        assert len(lines) == 1
        assert len(lines[0].get_property("tangent_theta")) == 3

    def test_grid_excludes_line_with_no_match(self):
        fls = _set(
            _line(0, [10, 20], [0, 1], [1, 2], r_line=[[1, 0, 0], [2, 0, 0]]),
        )
        lines = filter_field_line_set_by_theta(
            fls, np.array([5.0, 45.0]), 0.2, _PA_TEMPLATE
        )
        assert lines == []
