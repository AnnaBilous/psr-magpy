"""Tests for R_LC / r_max alignment in OFL reintegration service."""

import numpy as np

from psr_magpy.experiment.services.ofl_reintegration_data_product_service import (
    OFLReintegrationDataProductService,
)


def test_r_max_aligned_after_r_lc_sort():
    """Edge footpoint calc sorts R_LC descending; r_max must follow by value."""
    service = OFLReintegrationDataProductService()
    input_r_lc = np.array([4.77, 23.85])
    r_max = np.array([3.39, 16.93])

    lookup = service._build_r_max_lookup(r_max, input_r_lc)
    sorted_r_lc = np.sort(input_r_lc)[::-1]
    aligned = service._align_r_max_to_r_lc(sorted_r_lc, lookup)

    assert aligned == [16.93, 3.39]
