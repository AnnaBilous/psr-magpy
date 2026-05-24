"""Tests for :mod:`psr_magpy.viz.presets`."""

from psr_magpy.viz.field_line_plotting import FieldLinePlotConfig
from psr_magpy.viz.presets import (
    PA_cfg_large_RLC,
    PA_cfg_small_RLC,
    pa_field_line_cfg_for_r_lc,
)


def test_pa_cfg_types():
    assert isinstance(PA_cfg_small_RLC, FieldLinePlotConfig)
    assert isinstance(PA_cfg_large_RLC, FieldLinePlotConfig)


def test_pa_field_line_cfg_for_r_lc_threshold():
    assert pa_field_line_cfg_for_r_lc(4.77) is PA_cfg_small_RLC
    assert pa_field_line_cfg_for_r_lc(25.0) is PA_cfg_large_RLC
