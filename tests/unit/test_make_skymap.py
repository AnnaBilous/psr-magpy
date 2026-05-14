"""Tests for skymap workflow helpers."""

import numpy as np

from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.property_system import PropertyName
from psr_magpy.workflow.line_batch_workflow_prep import (
    enrich_field_line_set_from_r_line,
    get_workflow_batch_load_template,
)
from psr_magpy.workflow.make_skymap import skymap_pixel_counts_from_prepared_batch
from psr_magpy.workflow.workflow_utils import create_healpix_grid


def test_skymap_pixel_counts_from_none():
    grid = create_healpix_grid(HealpixGridConfig(nside=1))
    counts = skymap_pixel_counts_from_prepared_batch(None, grid)
    assert counts.shape == (12,)
    assert np.all(counts == 0)


def test_skymap_pixel_counts_nonempty():
    template = get_workflow_batch_load_template()
    batch = FieldLineSet(template, copy_data=False)
    r_line = np.array(
        [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    batch.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([0], dtype=int)
    batch.property_system._properties[PropertyName.R_LINE] = [r_line.T]
    batch.n_field_lines = 1
    enrich_field_line_set_from_r_line(batch, template)

    grid = create_healpix_grid(HealpixGridConfig(nside=8))
    counts = skymap_pixel_counts_from_prepared_batch(batch, grid)
    assert counts.dtype == np.int64
    assert np.sum(counts) == 4
