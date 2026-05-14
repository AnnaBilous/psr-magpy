"""Tests for workflow line batch preparation helpers."""

import numpy as np

from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.property_system import PropertyName
from psr_magpy.workflow.line_batch_workflow_prep import (
    enrich_field_line_set_from_r_line,
    get_workflow_batch_load_template,
    prepare_line_batch_after_load,
    resolve_oflr_lines_file_path,
)


def test_enrich_field_line_set_from_r_line_fills_missing():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    # Single line: radius along x
    r_line = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    batch.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([0], dtype=int)
    batch.property_system._properties[PropertyName.R_LINE] = [r_line.T]
    batch.n_field_lines = 1

    enrich_field_line_set_from_r_line(batch, template)

    th = batch.get_property("tangent_theta")[0]
    ph = batch.get_property("tangent_phi")[0]
    assert th is not None and ph is not None
    assert len(th) == 4 and len(ph) == 4
    assert np.all(np.isfinite(th)) and np.all(np.isfinite(ph))


def test_prepare_line_batch_data_products_trim_only():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    r_line = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    batch.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([0], dtype=int)
    batch.property_system._properties[PropertyName.R_LINE] = [r_line.T]
    batch.n_field_lines = 1

    out = prepare_line_batch_after_load(
        batch,
        source="data_products_oflr",
        r_lc=10.0,
        trim_type="sphere",
        max_radius=2.5,
        reversed_error_threshold=0.01,
    )
    assert out is not None
    rl = out.get_property("r_line")[0]
    assert rl is not None
    pts = np.asarray(rl).reshape(-1, 3)
    radii = np.sqrt(np.sum(pts**2, axis=1))
    assert np.all(radii < 2.5 + 1e-6)


def test_resolve_oflr_lines_file_path():
    p = resolve_oflr_lines_file_path(
        "/data/heavy", "series_A", 93, 10.0, data_products_subdir="OFLR"
    )
    assert p.name == "OFLR_lines_093_RLC_10.00.fits"
    assert "data_products" in str(p)
    assert p.parent.name == "OFLR"


def test_prepare_line_batch_r_min_trims_leading_points():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    # Radii 0.5, 1.0, 2.0 along x — with r_min=1.0 keep last two samples
    r_line = np.array([[0.5, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    batch.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([0], dtype=int)
    batch.property_system._properties[PropertyName.R_LINE] = [r_line.T]
    batch.n_field_lines = 1
    enrich_field_line_set_from_r_line(batch, template)

    out = prepare_line_batch_after_load(
        batch,
        source="data_products_oflr",
        r_lc=10.0,
        trim_type="sphere",
        max_radius=None,
        r_min=1.0,
        reversed_error_threshold=0.01,
    )
    assert out is not None
    rl = out.get_property("r_line")[0]
    pts = np.asarray(rl).reshape(-1, 3)
    assert pts.shape[0] == 2
    assert np.all(np.linalg.norm(pts, axis=1) >= 1.0 - 1e-6)
