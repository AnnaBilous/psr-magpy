"""Tests for R_LC / r_max alignment in OFL reintegration service."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from psr_magpy.experiment.services.ofl_reintegration_data_product_service import (
    OFLReintegrationDataProductService,
)
from psr_magpy.grid.grid import CustomGrid, HealpixGrid
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.open_field_line_edges import OFLEdgeResult


def test_r_max_aligned_after_r_lc_sort():
    """Edge footpoint calc sorts R_LC descending; r_max must follow by value."""
    service = OFLReintegrationDataProductService()
    input_r_lc = np.array([4.77, 23.85])
    r_max = np.array([3.39, 16.93])

    lookup = service._build_r_max_lookup(r_max, input_r_lc)
    sorted_r_lc = np.sort(input_r_lc)[::-1]
    aligned = service._align_r_max_to_r_lc(sorted_r_lc, lookup)

    assert aligned == [16.93, 3.39]


@patch(
    "psr_magpy.experiment.services.ofl_reintegration_data_product_service.OFLEdgeCalculator"
)
@patch(
    "psr_magpy.experiment.services.ofl_reintegration_data_product_service._load_experiment_setup"
)
def test_calculate_footpoints_passes_use_compound_grid(mock_setup, mock_calc):
    """use_compound_grid is forwarded to OFLEdgeCalculator in edge mode."""
    service = OFLReintegrationDataProductService()
    grid = Mock(spec=HealpixGrid)
    ofl_data = Mock(spec=FieldLineSet)
    mock_setup.return_value = (None, grid, ofl_data, None)

    calc_instance = Mock()
    calc_instance.calculate_edges.return_value = OFLEdgeResult(
        R_LC=np.array([10.0]), edge_footpoints=[np.array([1, 2])], precise_edges=True
    )
    mock_calc.return_value = calc_instance

    service._calculate_footpoints_only(
        experiment_config_path=Path("dummy.yaml"),
        lines_dir=Path("lines"),
        r_lc_values=np.array([10.0]),
        h_lc_coeff=3**1.5 / 2.0,
        selection_mode="edge",
        use_compound_grid=True,
    )

    assert mock_calc.call_args.kwargs["use_compound_grid"] is True


def test_downsample_edges_requires_edge_mode():
    """downsample_edges is rejected for non-edge selection modes."""
    import pytest

    from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig

    service = OFLReintegrationDataProductService()
    series = Mock(spec=ExperimentSeriesConfig)
    series.output_dir = "out"
    with pytest.raises(ValueError, match="downsample_edges"):
        service.run(
            series_config=series,
            experiment_numbers=[0],
            r_lc_values=[10.0],
            selection_mode="region",
            downsample_edges=True,
        )


@patch(
    "psr_magpy.experiment.services.ofl_reintegration_data_product_service.FieldLineCalculator"
)
@patch(
    "psr_magpy.experiment.services.ofl_reintegration_data_product_service.CalculationContext"
)
@patch(
    "psr_magpy.experiment.services.ofl_reintegration_data_product_service.ExperimentConfig"
)
def test_integrate_downsample_builds_custom_grid(mock_exp_cfg, mock_context, mock_calc):
    """With downsample_edges, integration traces from a CustomGrid (cardinal ids)."""
    import healpy as hp

    service = OFLReintegrationDataProductService()

    base_grid = HealpixGrid(nside=8)
    config = Mock()
    config.name = "series_000"
    config.metadata = {"experiment_number": 0}
    config.grid.create_grid.return_value = base_grid
    config.field.create_field.return_value = Mock()
    config.tracing.model_copy.return_value.create_tracing_parameters.return_value = Mock()
    mock_exp_cfg.load.return_value = config

    # Edge footpoints: all fine pixels of one coarse pixel -> one CustomGrid point.
    coarse_pix = 5
    fine_ids = np.where(
        hp.vec2pix(8 // 8, *hp.pix2vec(8, np.arange(hp.nside2npix(8)))) == coarse_pix
    )[0]

    service._integrate_selected_footpoints(
        experiment_config_path=Path("dummy.yaml"),
        selection_mode="edge",
        output_dir=Path("out"),
        r_lc_values=np.array([10.0]),
        selected_footpoints=[fine_ids],
        template=Mock(),
        r_max=None,
        downsample_edges=True,
        downsample_factor=8,
    )

    grid_arg = mock_context.call_args[0][1]
    assert isinstance(grid_arg, CustomGrid)
    # Cardinal footpoint IDs 0..M-1.
    assert grid_arg.indices.tolist() == list(range(grid_arg.nline_ori))
