"""Tests for skipping empty sentinel batch FITS files."""

from pathlib import Path

import numpy as np
import pytest

from psr_magpy.experiment.services.batch_data_reader import BatchDataReader
from psr_magpy.io import save_data
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.property_system import PropertyName
from psr_magpy.workflow.extract_PA import _PA_SAVE_TEMPLATE


def _write_nonempty_batch(path: Path, footpoint_ids: np.ndarray) -> None:
    template = FieldLineTemplate(
        save_footpoint_id=True,
        save_r_line=True,
        save_pa_line=True,
        save_tangent_theta=True,
        save_tangent_phi=True,
    )
    fls = FieldLineSet(template, copy_data=False)
    n = len(footpoint_ids)
    fls.property_system._properties[PropertyName.FOOTPOINT_ID] = footpoint_ids
    fls.property_system._properties[PropertyName.R_LINE] = [
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) for _ in range(n)
    ]
    fls.property_system._properties[PropertyName.PA_LINE] = [
        np.array([0.1, 0.2]) for _ in range(n)
    ]
    fls.property_system._properties[PropertyName.TANGENT_THETA] = [
        np.array([10.0, 20.0]) for _ in range(n)
    ]
    fls.property_system._properties[PropertyName.TANGENT_PHI] = [
        np.array([30.0, 40.0]) for _ in range(n)
    ]
    fls.n_field_lines = n
    save_data(fls, path)


def test_load_combined_skips_empty_sentinel_batch(tmp_path):
    """Empty sentinel batches (0 rows, no columns) must not break combined load."""
    lines_dir = tmp_path / "extracted_PA"
    lines_dir.mkdir()
    prefix = "extract_PA_030_RLC_04.77"

    _write_nonempty_batch(
        lines_dir / f"{prefix}_batch_001_out_of_002.fits",
        np.array([1, 2]),
    )
    # Sentinel: same as extract_PA flush when filter yields nothing.
    save_data(
        FieldLineSet(FieldLine.from_template(_PA_SAVE_TEMPLATE), copy_data=False),
        lines_dir / f"{prefix}_batch_002_out_of_002.fits",
    )

    template = FieldLine.from_template(_PA_SAVE_TEMPLATE)
    combined = BatchDataReader(experiment_name=prefix).load_combined_batch_data(
        template=template,
        lines_dir=str(lines_dir),
        batch_filename_prefix=prefix,
    )
    assert combined is not None
    assert combined.n_field_lines == 2
    assert list(combined.footpoint_id) == [1, 2]
